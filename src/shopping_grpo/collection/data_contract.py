"""Reject stale SFT conversations after the task selection has changed."""

from __future__ import annotations

import json
from pathlib import Path

from shopping_grpo.environment.manifest import sha256_file
from shopping_grpo.environment.reward import REWARD_VERSION


def validate_current_sft(root, train=None, validation=None):
    root = Path(root)
    manifest_path = root / "data/splits/manifest.json"
    if not manifest_path.is_file():
        raise ValueError("缺少固定任务 manifest，请先执行 prepare_data_splits.py build")
    manifest = json.loads(manifest_path.read_text())
    metadata_path = root / "data/sft/metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if metadata.get("reward") != REWARD_VERSION:
        raise ValueError(
            "SFT 任务已经固定，但当前示范仍是旧数据；请按 data/sft/train_tasks.jsonl 和 validation_tasks.jsonl 采集并构建新的 Strict Reward 示范，再启动训练。"
        )
    for role, path in (
        ("train", train or root / "data/sft/train.jsonl"),
        ("validation", validation or root / "data/sft/validation.jsonl"),
    ):
        path = Path(path)
        spec = manifest["initial"]["sft_" + role]
        task_path = root / spec["path"]
        if sha256_file(task_path) != spec["sha256"]:
            raise ValueError("SFT 固定任务名单校验和不符")
        expected = {json.loads(line)["task_id"] for line in task_path.read_text().splitlines()}
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ids = [r.get("task_id") for r in rows]
        if len(ids) != len(expected) or set(ids) != expected:
            raise ValueError(f"SFT {role} 示范与已固定任务不一致，不能回退使用旧划分")
        if any(not isinstance(r.get("messages"), list) for r in rows):
            raise ValueError("SFT 输入必须是示范对话，不能直接使用任务 ID 列表")
        if metadata.get(role, {}).get("sha256") != sha256_file(path):
            raise ValueError("SFT 示范与 metadata 校验和不符")
    return {
        "status": "verified",
        "train": manifest["initial"]["sft_train"]["tasks"],
        "validation": manifest["initial"]["sft_validation"]["tasks"],
    }
