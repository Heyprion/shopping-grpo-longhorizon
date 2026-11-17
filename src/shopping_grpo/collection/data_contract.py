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
    if (
        manifest.get("sft_demonstrations", {}).get("split_status")
        == "deferred_until_after_quality_screening"
    ):
        raise ValueError(
            "SFT 当前只采集统一的 1,000 条合格轨迹；训练/验证划分须等后续质量筛选完成，不能沿用旧 800/200 名单启动训练。"
        )
    metadata_path = root / "data/sft/metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if metadata.get("reward") != REWARD_VERSION:
        raise ValueError(
            "SFT 任务已经固定，但当前示范仍是旧数据；请按 data/sft/train_tasks.jsonl 和 validation_tasks.jsonl 采集并构建新的 Strict Reward 示范，再启动训练。"
        )
    final = manifest.get("sft_final")
    if final:
        quality = metadata.get("quality_screening", {})
        if (
            quality.get("version") != "shopping-sft-stage1-v1"
            or quality.get("completed") is not True
            or quality.get("stage2_applied") is not False
            or metadata.get("split", {}).get("finalized") is not True
        ):
            raise ValueError("SFT 第一阶段质量筛选或正式划分未完成")
        if sha256_file(metadata_path) != manifest["files"]["data/sft/metadata.json"]:
            raise ValueError("SFT metadata 与 manifest 校验和不符")
        quality_path = root / quality["path"]
        if sha256_file(quality_path) != quality["sha256"]:
            raise ValueError("SFT 质量筛选记录校验和不符")
        report = json.loads(quality_path.read_text())
        retained = {r["task_id"]: r for r in report["retained"]}
        if any(
            not 0 < r["student_tokens"] <= 24576 or not r["assistant_loss_tokens"]
            for r in retained.values()
        ):
            raise ValueError("SFT 保留轨迹不满足上下文或监督标签要求")
    role_rows = {}
    for role, path in (
        ("train", train or root / "data/sft/train.jsonl"),
        ("validation", validation or root / "data/sft/validation.jsonl"),
    ):
        path = Path(path)
        spec = final[role] if final else manifest["initial"]["sft_" + role]
        task_path = root / spec["path"]
        if sha256_file(task_path) != spec["sha256"]:
            raise ValueError("SFT 固定任务名单校验和不符")
        expected = {json.loads(line)["task_id"] for line in task_path.read_text().splitlines()}
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if final and (len(expected) != spec["tasks"] or len(rows) != metadata[role]["rows"]):
            raise ValueError("SFT 正式名单或示范数量不一致")
        ids = [r.get("task_id") for r in rows]
        if len(ids) != len(expected) or set(ids) != expected:
            raise ValueError(f"SFT {role} 示范与已固定任务不一致，不能回退使用旧划分")
        if any(not isinstance(r.get("messages"), list) for r in rows):
            raise ValueError("SFT 输入必须是示范对话，不能直接使用任务 ID 列表")
        if metadata.get(role, {}).get("sha256") != sha256_file(path):
            raise ValueError("SFT 示范与 metadata 校验和不符")
        role_rows[role] = rows
    if final:
        from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS

        labels = {
            r["task_id"]: r
            for r in map(json.loads, (root / "data/splits/labels.jsonl").read_text().splitlines())
        }
        candidates = {
            r["task_id"]
            for r in map(
                json.loads, (root / "data/sft/candidate_tasks.jsonl").read_text().splitlines()
            )
        }
        train_ids = {r["task_id"] for r in role_rows["train"]}
        val_ids = {r["task_id"] for r in role_rows["validation"]}
        union = train_ids | val_ids
        protected = {r["group_id"] for r in labels.values() if r["source_tag"] == "eval"}
        if train_ids & val_ids or union != set(retained) or len(union) != final["total"]:
            raise ValueError("SFT 训练/验证重叠或与筛选名单不一致")
        if not union <= candidates or any(
            labels[i]["source_tag"] != "train" or labels[i]["group_id"] in protected for i in union
        ):
            raise ValueError("SFT 来源与候选池或 held-out 边界不一致")
        if {labels[i]["group_id"] for i in train_ids} & {labels[i]["group_id"] for i in val_ids}:
            raise ValueError("SFT 重复指令组跨训练与验证")
        if len(val_ids) != round(len(union) * 0.1) or metadata["split"]["seed"] != 42:
            raise ValueError("SFT 划分不满足固定 seed 与 1:9 比例")
        spec = metadata["trajectories"]
        if sha256_file(root / spec["path"]) != spec["sha256"]:
            raise ValueError("SFT 完整轨迹集合校验和不符")
        all_rows = {
            r["task_id"]: r for r in map(json.loads, (root / spec["path"]).read_text().splitlines())
        }
        if set(all_rows) != union:
            raise ValueError("SFT 正式集合与划分名单不一致")
        for rows in role_rows.values():
            for row in rows:
                if row != all_rows[row["task_id"]] or row.get("tools") != SHOP_TOOL_SCHEMAS:
                    raise ValueError("SFT 对话或工具 schema 与正式集合不一致")
                if not row["messages"] or any(
                    "reasoning_content" in m or m.get("runtime_action_guard")
                    for m in row["messages"]
                ):
                    raise ValueError("SFT 输入包含未格式化教师轨迹")
    return {
        "status": "verified",
        "train": final["train"]["tasks"] if final else manifest["initial"]["sft_train"]["tasks"],
        "validation": final["validation"]["tasks"]
        if final
        else manifest["initial"]["sft_validation"]["tasks"],
    }
