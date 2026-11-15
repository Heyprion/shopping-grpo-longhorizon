#!/usr/bin/env python3
"""Check two-GPU FSDP2 prerequisites without loading a model or starting training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_config(config):
    if (
        config.get("distributed_type") != "FSDP"
        or config.get("num_processes") != 2
        or config.get("num_machines") != 1
        or config.get("mixed_precision") != "bf16"
    ):
        raise ValueError("FSDP2 requires one node, two processes and BF16")
    fsdp = config.get("fsdp_config", {})
    for key, expected in {
        "fsdp_version": 2,
        "fsdp_reshard_after_forward": True,
        "fsdp_activation_checkpointing": True,
        "fsdp_cpu_ram_efficient_loading": False,
        "fsdp_state_dict_type": "FULL_STATE_DICT",
    }.items():
        if fsdp.get(key) != expected:
            raise ValueError(f"{key} must be {expected!r}")


def check_runtime():
    import torch
    import yaml
    from accelerate.commands.config.config_args import load_config_from_file
    from accelerate.utils import FullyShardedDataParallelPlugin
    from torch.distributed.fsdp import fully_shard  # noqa: F401

    from scripts.train_lora_sft import _training_dependencies

    config_path = ROOT / "configs/fsdp2.yaml"
    validate_config(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    load_config_from_file(str(config_path))
    _training_dependencies()
    plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2, reshard_after_forward=True, state_dict_type="FULL_STATE_DICT"
    )
    if plugin.fsdp_version != 2:
        raise SystemExit("Accelerate did not select FSDP2")
    if not torch.distributed.is_nccl_available() or torch.cuda.device_count() < 2:
        raise SystemExit("FSDP2 requires NCCL and two visible CUDA GPUs")
    devices = []
    for index in range(2):
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():
                raise SystemExit(f"GPU {index} does not support BF16")
        devices.append(torch.cuda.get_device_name(index))
    print(json.dumps({"fsdp_version": 2, "devices": devices, "torch": torch.__version__,
                      "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version()}, indent=2))


def check_collectives():
    import torch
    import torch.distributed as dist
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    from scripts.train_lora_sft import _validate_fsdp2_launch

    _validate_fsdp2_launch(argparse.Namespace(fsdp2=True, qlora=False))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=60))]
    )
    try:
        plugin = accelerator.state.fsdp_plugin
        if plugin is None or plugin.fsdp_version != 2 or not plugin.activation_checkpointing:
            raise RuntimeError("Accelerate did not enable the requested FSDP2 configuration")
        if dist.get_world_size() != 2:
            raise RuntimeError("Expected exactly two ranks")
        value = torch.tensor([dist.get_rank() + 1.0], device="cuda")
        dist.all_reduce(value)
        torch.testing.assert_close(value, torch.tensor([3.0], device="cuda"))
        mesh = init_device_mesh("cuda", (2,))
        # Only allocate and shard tiny tensors. No forward, backward or optimizer.
        module = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).cuda()
        module[0].requires_grad_(False)
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.fill_(1.0)
        fully_shard(module, mesh=mesh, reshard_after_forward=True)
        for parameter in module.parameters():
            if not isinstance(parameter, DTensor):
                raise TypeError("FSDP2 did not shard parameters as DTensors")
            if parameter.to_local().numel() * 2 != parameter.numel():
                raise RuntimeError("Parameter was not sharded across both GPUs")
            gathered = parameter.full_tensor()
            torch.testing.assert_close(gathered, torch.ones_like(gathered))
        dist.barrier()
        if dist.get_rank() == 0:
            print("PASS: two-GPU NCCL all-reduce, FSDP2 mixed frozen/trainable parameter "
                  "sharding and full-tensor gathering; no training executed")
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distributed", action="store_true", help="also check two-GPU collectives")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        check_collectives()
        return
    check_runtime()
    if args.distributed:
        subprocess.run([
            sys.executable, "-m", "accelerate.commands.launch",
            "--config_file", str(ROOT / "configs/fsdp2.yaml"),
            str(Path(__file__).resolve()), "--worker",
        ], check=True)


if __name__ == "__main__":
    # Also allow direct execution from outside the repository root.
    sys.path.insert(0, str(ROOT))
    main()
