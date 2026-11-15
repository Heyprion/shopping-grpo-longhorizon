"""Dry runs must expose the complete recipe without invoking Python or a GPU."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("balanced", [False, True])
def test_sft_dry_run_never_executes_training_and_preserves_effective_batch(tmp_path, balanced):
    forbidden = tmp_path / "forbidden-python"
    forbidden.write_text("#!/bin/sh\nexit 99\n")
    forbidden.chmod(0o755)
    output = tmp_path / "output with spaces"
    root = Path(__file__).resolve().parents[1]
    command = ["bash", "scripts/sft.sh", "--dry-run"]
    if balanced:
        command.append("--balanced")
    env = {**os.environ, "SFT_PYTHON": str(forbidden), "SFT_ADAPTER_DIR": str(output)}
    result = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, check=True)
    args = shlex.split(result.stdout)
    for flag, value in [("--seed", "42"), ("--per-device-train-batch-size", "1"),
                        ("--gradient-accumulation-steps", "4"), ("--epochs", "3"),
                        ("--output", str(output)), ("--max-length", "24576")]:
        assert args[args.index(flag) + 1] == value
    assert "--max-steps" not in args and "--stop-after-steps" not in args
    assert args[args.index("--lora-r") + 1] == ("32" if balanced else "16")
    assert args[args.index("--lora-alpha") + 1] == ("64" if balanced else "32")
    assert ("--no-fsdp-activation-checkpointing" in args) == balanced
    if balanced:
        assert args[args.index("--fsdp-checkpoint-stride") + 1] == "2"
        assert any(item.startswith("PYTORCH_ALLOC_CONF=") for item in args)
    assert not output.exists()
