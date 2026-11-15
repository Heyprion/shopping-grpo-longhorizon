"""Opt-in, no-optimizer resource replay through the pinned GRPO runtime."""

import functools
import json
import os
import time
from pathlib import Path


class ResourceProbeComplete(Exception):
    """Stop the pilot after profiling its first useful batch."""


def install():
    from shopping_grpo.training.grpo.compat import install_torch_padding_fallback

    install_torch_padding_fallback()
    destination = os.environ.get("SHOPPING_RESOURCE_PROBE_DIR")
    if not destination:
        raise RuntimeError("Resource probe requires SHOPPING_RESOURCE_PROBE_DIR")

    import torch
    import verl.utils.tensordict_utils as tu
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.workers.engine_workers import TrainingWorker
    from verl.workers.utils.losses import sft_loss

    if getattr(TrainingWorker.train_batch, "_shopping_resource_probe", False):
        return
    original_train = TrainingWorker.train_batch

    @functools.wraps(original_train)
    def train(self, data):
        engine = self.engine
        rank = torch.distributed.get_rank()
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        layers = [m for m in engine.module.modules() if any(
            c.__name__ == "Qwen3_5DecoderLayer" for c in type(m).__mro__)]
        if len(layers) != 24:
            raise RuntimeError(f"Expected 24 decoder layers, found {len(layers)}")
        original_flags = [m.gradient_checkpointing for m in layers]
        step = engine.optimizer.step
        optimizer_step = engine.optimizer_step
        engine.optimizer.step = lambda *args, **kwargs: None
        rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        reference = {}
        current = {}
        results = []

        def capture_step():
            norm = optimizer_step()
            current["grad_norm"] = float(norm)
            if current.get("compare"):
                numerator = torch.zeros((), device=torch.cuda.current_device())
                denominator = numerator.clone()
                for name, param in engine.module.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad = param.grad
                        if hasattr(grad, "to_local"):
                            grad = grad.to_local()
                        value = grad.detach().float()
                        if current["name"] == "real_micro2_full":
                            reference[name] = value.clone()
                        elif name in reference:
                            numerator += (value-reference[name]).square().sum()
                            denominator += reference[name].square().sum()
                torch.distributed.all_reduce(numerator)
                torch.distributed.all_reduce(denominator)
                current["gradient_relative_l2"] = float(
                    (numerator/denominator.clamp_min(1e-30)).sqrt())
            return norm

        engine.optimizer_step = capture_step

        def run(name, micro, checkpointed, stress=False, warmup=False):
            for i, layer in enumerate(layers):
                # Spread the layers without recomputation across the decoder.
                layer.gradient_checkpointing = i not in {
                    (j+1)*24//(24-checkpointed)-1 for j in range(24-checkpointed)
                } if checkpointed < 24 else True
            torch.set_rng_state(rng)
            torch.cuda.set_rng_state(cuda_rng)
            engine.optimizer_zero_grad()
            torch.cuda.empty_cache()
            torch.distributed.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            current.clear()
            current.update(name=name, compare=not stress and not warmup)
            started = time.perf_counter()
            if stress:
                count, length = 4, 24576
                values = data["input_ids"].values()
                ids = values.repeat((length*count+values.numel()-1)//values.numel())[:length*count]
                offsets = torch.arange(count+1, device=ids.device, dtype=torch.int64)*length
                nested = torch.nested.nested_tensor_from_jagged
                stress_data = tu.get_tensordict(tensor_dict={
                    "input_ids":nested(ids, offsets),
                    "position_ids":nested(torch.arange(length,device=ids.device).repeat(count), offsets),
                    "loss_mask":nested(torch.ones(length*count,device=ids.device), offsets),
                }, non_tensor_dict={
                    "temperature":0.7, "pad_token_id":0,
                    "use_remove_padding":False, "use_dynamic_bsz":False,
                    "micro_batch_size_per_gpu":micro, "max_token_len_per_gpu":24576,
                    "use_fused_kernels":True, "calculate_entropy":True,
                })
                with engine.train_mode(disable_auto_offload=True):
                    engine.forward_backward_batch(
                        stress_data, loss_function=functools.partial(sft_loss, None))
                    engine.optimizer_step()
                    engine.optimizer_zero_grad()
                output = None
            else:
                tu.assign_non_tensor(data, micro_batch_size_per_gpu=micro, update_lr_scheduler=False)
                output = original_train(self, data)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            peak = torch.tensor(torch.cuda.max_memory_allocated(), device=torch.cuda.current_device())
            torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
            record = {"name":name, "rank":rank, "micro_batch":micro,
                      "checkpointed_layers":checkpointed, "stress_length":24576 if stress else None,
                      "seconds":elapsed, "allocated_gib":float(peak)/2**30,
                      "reserved_gib":torch.cuda.max_memory_reserved()/2**30,
                      "optimizer_updated":False, "scheduler_advanced":False,
                      "unix":time.time(), **current}
            with (path/f"rank{rank}.jsonl").open("a") as handle:
                handle.write(json.dumps(record)+"\n")
            print("SHOPPING_RESOURCE_PROBE "+json.dumps(record), flush=True)
            results.append(record)
            return output, record

        try:
            run("warmup",2,24,warmup=True)
            run("real_micro2_full",2,24)
            output, _ = run("real_micro4_full",4,24)
            if os.environ.get("SHOPPING_RESOURCE_PROBE_MODE") == "confirm":
                run("real_micro4_full_warm",4,24)
                run("real_micro4_cp20_warmup",4,20)
                output, _ = run("real_micro4_cp20",4,20)
                run("stress_micro4_cp20",4,20,stress=True)
                return output
            _, full = run("stress_micro4_full",4,24,stress=True)
            chosen = 24
            if full["allocated_gib"] < 60:
                _, partial = run("stress_micro4_cp20",4,20,stress=True)
                chosen = 20
                if partial["allocated_gib"] < 65:
                    _, partial = run("stress_micro4_cp18",4,18,stress=True)
                    chosen = 18
            if chosen < 24:
                output, _ = run(f"real_micro4_cp{chosen}",4,chosen)
            return output
        finally:
            engine.optimizer.step = step
            engine.optimizer_step = optimizer_step
            for layer, flag in zip(layers,original_flags,strict=True):
                layer.gradient_checkpointing = flag
            engine.optimizer_zero_grad()
            torch.set_rng_state(rng)
            torch.cuda.set_rng_state(cuda_rng)

    train._shopping_resource_probe = True
    TrainingWorker.train_batch = train
    original_update = RayPPOTrainer._update_actor
    original_fit = RayPPOTrainer.fit

    def update(self, batch):
        original_update(self,batch)
        raise ResourceProbeComplete()

    def fit(self,*args,**kwargs):
        try:
            return original_fit(self,*args,**kwargs)
        except ResourceProbeComplete:
            import swanlab

            metrics = {}
            for file in Path(destination).glob("rank*.jsonl"):
                for line in file.read_text().splitlines():
                    row = json.loads(line)
                    prefix = f"probe/rank{row['rank']}/{row['name']}"
                    for key in ["seconds","allocated_gib","reserved_gib","grad_norm","gradient_relative_l2"]:
                        if key in row:
                            metrics[f"{prefix}/{key}"] = row[key]
            swanlab.log(metrics,step=0)
            swanlab.finish()
            Path(destination,"completed.json").write_text(json.dumps({
                "resource_only":True,"optimizer_updates":0,"validation_runs":0}))

    RayPPOTrainer._update_actor = update
    RayPPOTrainer.fit = fit
