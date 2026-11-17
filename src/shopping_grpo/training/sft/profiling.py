"""Small distributed measurements for the existing one-stage SFT trainer."""

import hashlib
import json
import math
import time
from pathlib import Path


def unique_validation_samples(samples):
    """Distributed sampler padding must not overweight its repeated last sample."""
    return list({sample["input_sha256"]: sample for sample in samples}.values())


def instrument_trainer(base, torch, *, longest_first=False):
    class MeasuredTrainer(base):
        sft_profile = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sft_profile = {"steps": [], "evaluations": []}
            self.sft_counts = [0, 0, 0, 0]
            self.sft_eval_counts = [0.0] * 6
            self.sft_eval_samples = []

        def _get_train_sampler(self, train_dataset=None):
            if longest_first:
                return torch.utils.data.SequentialSampler(
                    train_dataset if train_dataset is not None else self.train_dataset
                )
            return super()._get_train_sampler(train_dataset)

        def training_step(self, model, inputs, num_items_in_batch=None):
            self.sft_counts[0] += int(inputs["attention_mask"].sum().item())
            self.sft_counts[1] += int((inputs["labels"][..., 1:] != -100).sum().item())
            self.sft_counts[2] += inputs["input_ids"].numel()
            self.sft_counts[3] += inputs["input_ids"].shape[0]
            loss = super().training_step(model, inputs, num_items_in_batch)
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError("Non-finite SFT loss")
            return loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            result = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
            loss = result[0]
            if loss is not None and inputs.get("labels") is not None:
                if not bool(torch.isfinite(loss).all()):
                    raise FloatingPointError("Non-finite SFT validation loss")
                # Exact token weighting for eval batch=1; batches of equal-length examples also work.
                count = int((inputs["labels"][..., 1:] != -100).sum().item())
                value = float(loss.item()) * count
                self.sft_eval_counts[0] += value
                self.sft_eval_counts[1] += count
                index = 4 if inputs["input_ids"].shape[-1] >= 8192 else 2
                self.sft_eval_counts[index] += value
                self.sft_eval_counts[index + 1] += count
                if inputs["input_ids"].shape[0] == 1:
                    self.sft_eval_samples.append({
                        "input_sha256": hashlib.sha256(
                            inputs["input_ids"][0].detach().cpu().numpy().tobytes()
                        ).hexdigest(),
                        "loss": float(loss.item()), "supervised_tokens": count,
                        "input_tokens": int(inputs["attention_mask"].sum().item()),
                    })
            return result

        def evaluate(self, *args, **kwargs):
            self.sft_eval_counts = [0.0] * 6
            self.sft_eval_samples = []
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            metrics = super().evaluate(*args, **kwargs)
            torch.cuda.synchronize()
            values = torch.tensor(self.sft_eval_counts, dtype=torch.float64, device=self.args.device)
            values = self.accelerator.reduce(values, reduction="sum").tolist()
            from accelerate.utils import gather_object
            samples = gather_object(self.sft_eval_samples)
            if self.args.per_device_eval_batch_size == 1:
                samples = unique_validation_samples(samples)
                values = [0.0] * 6
                for sample in samples:
                    count = sample["supervised_tokens"]
                    loss_sum = sample["loss"] * count
                    values[0] += loss_sum
                    values[1] += count
                    index = 4 if sample["input_tokens"] >= 8192 else 2
                    values[index] += loss_sum
                    values[index + 1] += count
                metrics["eval_unique_examples"] = len(samples)
            memory = self.accelerator.gather(torch.tensor([
                torch.cuda.max_memory_allocated() / 1024**3,
                torch.cuda.max_memory_reserved() / 1024**3,
            ], device=self.args.device)).reshape(-1, 2).tolist()
            for name, index in [("assistant_nll", 0), ("short_nll", 2), ("long_nll", 4)]:
                if values[index + 1]:
                    metrics[f"eval_{name}"] = values[index] / values[index + 1]
            metrics["eval_supervised_tokens"] = values[1]
            sample_file = f"validation_epoch_{self.state.epoch:.4g}.json"
            if self.is_world_process_zero() and samples:
                Path(self.args.output_dir, sample_file).write_text(json.dumps(samples, indent=2))
            self.sft_profile["evaluations"].append({
                "epoch": self.state.epoch, "metrics": dict(metrics), "memory_gib": memory,
                "wall_seconds": time.perf_counter() - started,
                "per_example_file": sample_file if samples else None,
            })
            self.log({key: value for key, value in metrics.items() if key not in {
                "eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"
            }})
            return metrics

    return MeasuredTrainer


def make_profile_callback(base, torch, cli_args):
    class ProfileCallback(base):
        profile_trainer = None

        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.started = time.perf_counter()
            self.profile_trainer.sft_counts = [0, 0, 0, 0]

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            trainer = self.profile_trainer
            elapsed = time.perf_counter() - self.started
            free, total = torch.cuda.mem_get_info()
            gathered = trainer.accelerator.gather(torch.tensor([
                elapsed, torch.cuda.max_memory_allocated() / 1024**3,
                torch.cuda.max_memory_reserved() / 1024**3,
                (total - free) / 1024**3, *trainer.sft_counts,
            ], dtype=torch.float64, device=args.device)).reshape(-1, 8).tolist()
            duration = max(row[0] for row in gathered)
            counts = [sum(row[index] for row in gathered) for index in range(4, 8)]
            record = {
                "step": state.global_step, "epoch": state.epoch, "seconds": duration,
                "tokens": counts[0], "supervised_tokens": counts[1],
                "padded_tokens": counts[2], "examples": counts[3],
                "tokens_per_second": counts[0] / duration,
                "supervised_tokens_per_second": counts[1] / duration,
                "padding_fraction": 1 - counts[0] / counts[2],
                "ranks": [{"seconds": row[0], "allocated_gib": row[1],
                           "reserved_gib": row[2], "device_used_gib": row[3]} for row in gathered],
            }
            trainer.sft_profile["steps"].append(record)
            if state.is_world_process_zero:
                with (cli_args.output / "step_profile.jsonl").open("a") as file:
                    file.write(json.dumps(record) + "\n")
                if cli_args.swanlab:
                    import swanlab
                    swanlab.log({
                        "profile/tokens_per_second": record["tokens_per_second"],
                        "profile/step_seconds": duration,
                        "profile/padding_fraction": record["padding_fraction"],
                        **{f"profile/gpu{rank}_peak_allocated_gib": row[1]
                           for rank, row in enumerate(gathered)},
                        **{f"profile/gpu{rank}_peak_reserved_gib": row[2]
                           for rank, row in enumerate(gathered)},
                    }, step=state.global_step)
            if (getattr(cli_args, "stop_after_steps", None) is not None
                    and state.global_step >= cli_args.stop_after_steps):
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "grad_norm" in logs and not math.isfinite(float(logs["grad_norm"])):
                raise FloatingPointError("Non-finite SFT gradient norm")
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            if cli_args.stop_after_epoch is not None and state.epoch >= cli_args.stop_after_epoch:
                control.should_training_stop = True
            return control

    return ProfileCallback()
