"""Opt-in checks of the real actor-to-vLLM path during a bounded GRPO pilot."""

import hashlib
import json
import os
import time
from functools import wraps
from pathlib import Path

SUFFIXES = (
    "layers.0.linear_attn.out_proj.weight",
    "layers.3.self_attn.o_proj.weight",
    "layers.12.linear_attn.out_proj.weight",
    "layers.23.self_attn.o_proj.weight",
    "layers.0.mlp.down_proj.weight",
    "layers.23.mlp.down_proj.weight",
)


def selected(name):
    return any(name.endswith(suffix) for suffix in SUFFIXES)


def fingerprint(tensor):
    import torch

    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def record(kind, **payload):
    directory = Path(os.environ["SHOPPING_SYNC_PROBE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    row = {"event": kind, "time": time.time(), "pid": os.getpid(), **payload}
    with (directory / f"events-{os.getpid()}.jsonl").open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def install():
    from shopping_grpo.training.grpo.compat import install_torch_padding_fallback

    install_torch_padding_fallback()
    if not os.environ.get("SHOPPING_SYNC_PROBE_DIR"):
        raise RuntimeError("A sync probe requires SHOPPING_SYNC_PROBE_DIR")

    import torch
    import torch.distributed as dist
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    original = FSDPEngine.get_per_tensor_param
    if getattr(original, "_shopping_sync_probe", False):
        return

    def full(value):
        return value.full_tensor().detach() if hasattr(value, "full_tensor") else value.detach()

    @wraps(original)
    def get_params(self, *args, **kwargs):
        if not self.model_config.lora.get("merge", False):
            return original(self, *args, **kwargs)
        sequence = getattr(self, "_shopping_sync_index", -1) + 1
        self._shopping_sync_index = sequence
        parameters = dict(self.module.named_parameters())
        expected = {}
        bases = {}
        for name, param in parameters.items():
            clean = name.replace("base_model.model.", "").replace(".base_layer", "")
            if not selected(clean):
                continue
            module_name = name.removesuffix(".base_layer.weight")
            a = full(parameters[module_name + ".lora_A.default.weight"]).float()
            b = full(parameters[module_name + ".lora_B.default.weight"]).float()
            base = full(param).float()
            delta = b @ a * (self.model_config.lora_alpha / self.model_config.lora_rank)
            expected[clean] = (base + delta).to(torch.bfloat16)
            bases[clean] = base.to(torch.bfloat16)
        if len(expected) != len(SUFFIXES):
            raise RuntimeError(f"Expected six actor probe tensors, found {list(expected)}")
        started = time.monotonic()
        iterator, peft_config = original(self, *args, **kwargs)

        def checked():
            total = 0
            tensors = []
            for name, value in iterator:
                total += 1
                if name in expected:
                    equal = bool(torch.equal(value, expected[name]))
                    if not equal:
                        raise RuntimeError(f"Actor payload differs from base+LoRA: {name}")
                    tensors.append({"name": name, "hash": fingerprint(value),
                                    "matches_expected": equal,
                                    "changed_from_base": int((value != bases[name]).sum()),
                                    "elements": value.numel()})
                yield name, value
            torch.cuda.synchronize()
            record("actor_payload", rank=dist.get_rank(), sync_index=sequence,
                   tensors=tensors, total_tensors=total,
                   snapshot_and_transfer_seconds=time.monotonic()-started,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)

        return checked(), peft_config

    get_params._shopping_sync_probe = True
    FSDPEngine.get_per_tensor_param = get_params
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    original_generate = vLLMHttpServer.generate

    @wraps(original_generate)
    async def generate(self, *args, **kwargs):
        seen = getattr(self, "_shopping_probe_versions", set())
        if self.global_steps not in seen:
            record("generation_uses_version", replica=self.replica_rank,
                   global_step=self.global_steps)
            seen.add(self.global_steps)
            self._shopping_probe_versions = seen
        return await original_generate(self, *args, **kwargs)

    vLLMHttpServer.generate = generate


# The extension is imported in vLLM's engine subprocess, independently of Ray's hook.
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension


class SyncProbeWorkerExtension(vLLMColocateWorkerExtension):
    def update_weights_from_ipc(self, **kwargs):
        self._shopping_sync_index = getattr(self, "_shopping_sync_index", -1) + 1
        self._shopping_received = []
        result = super().update_weights_from_ipc(**kwargs)
        record("vllm_sync_complete", replica=int(os.environ.get("VERL_REPLICA_RANK", "0")),
               sync_index=self._shopping_sync_index, tensors=self._shopping_received)
        return result

    def _update_weights(self, weights, peft_config, base_sync_done):
        import torch

        probes = [(name, value.detach().clone()) for name, value in weights if selected(name)]
        result = super()._update_weights(weights, peft_config, base_sync_done)
        model = self.model_runner.model
        parameters = dict(model.named_parameters())
        for name, sent in probes:
            mapped = model.hf_to_vllm_mapper._map_name(name)
            actual = parameters.get(mapped)
            if actual is None:
                raise RuntimeError(f"Cannot inspect loaded vLLM tensor: {name} -> {mapped}")
            equal = bool(torch.equal(actual, sent))
            if not equal:
                raise RuntimeError(f"vLLM loaded value differs from received payload: {name}")
            self._shopping_received.append({"name": name, "hash": fingerprint(actual),
                                             "matches_received": equal})
        return result
