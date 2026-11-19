#!/usr/bin/env python3
"""Freeze stage-one-only SFT demonstrations and the seed-42 90/10 split."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from scripts.audit_sft_trajectories import structure_issues
from shopping_grpo.collection.data_contract import validate_current_sft
from shopping_grpo.collection.sft import build_sft_row
from shopping_grpo.collection.stratification import distribution, select_tasks, stable_rank
from shopping_grpo.environment.manifest import sha256_file
from shopping_grpo.environment.reward import REWARD_VERSION
from shopping_grpo.evaluation.artifacts import write_json_atomic, write_jsonl_atomic

ROOT = Path(__file__).resolve().parents[1]
VERSION = "shopping-sft-stage1-v1"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare_rows(source, metrics, labels):
    """Use only stage-one flags; behavior diagnostics never affect membership."""
    by_id = {r["task_id"]: r for r in metrics}
    if len(by_id) != len(metrics) or {t["task_id"] for t in source} != set(by_id):
        raise ValueError("Audit/source task membership mismatch")
    kept, rejected = [], []
    for t in source:
        m = by_id[t["task_id"]]
        if m["trajectory_id"] != t["trajectory_id"]:
            raise ValueError("Audit trajectory identity mismatch")
        if m["stage1_issues"]:
            rejected.append(
                {
                    "task_id": t["task_id"],
                    "trajectory_id": t["trajectory_id"],
                    "reasons": m["stage1_issues"],
                    "student_tokens": m.get("student_tokens"),
                }
            )
            continue
        row = build_sft_row(t)
        if (
            structure_issues(t, row)
            or not 0 < m["student_tokens"] <= 24576
            or not m["assistant_loss_tokens"]
        ):
            raise ValueError("Stage-one audit is inconsistent with retained trajectory")
        kept.append(row)
    if len({r["task_id"] for r in source}) != len(source):
        raise ValueError("Duplicate source task")
    val_labels = select_tasks(
        [labels[r["task_id"]] for r in kept],
        round(len(kept) * 0.1),
        seed=42,
        namespace=VERSION + ":validation",
    )
    val_ids = {r["task_id"] for r in val_labels}
    train = sorted(
        [r for r in kept if r["task_id"] not in val_ids],
        key=lambda r: stable_rank(42, VERSION + ":train", labels[r["task_id"]]["group_id"]),
    )
    by_kept_id = {r["task_id"]: r for r in kept}
    validation = [by_kept_id[r["task_id"]] for r in val_labels]
    if {labels[r["task_id"]]["group_id"] for r in train} & {
        labels[r["task_id"]]["group_id"] for r in validation
    }:
        raise ValueError("Instruction group crosses SFT train/validation")
    return kept, rejected, train, validation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--swanlab-mode", choices=["online", "offline"], default="online")
    args = p.parse_args()
    manifest_path = ROOT / "data/splits/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("sft_final", {}).get("version") == VERSION:
        print(json.dumps(validate_current_sft(ROOT), ensure_ascii=False))
        return
    source_path = ROOT / "outputs/sft-unified-collection-64k/accepted.jsonl"
    audit_dir = ROOT / "outputs/sft-quality-audit"
    audit = json.loads((audit_dir / "summary.json").read_text())
    if sha256_file(source_path) != audit["config"]["source_sha256"]:
        raise ValueError("Source changed after quality audit")
    for name, expected in audit["config"]["code_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError("Audit preprocessing code changed: " + name)
    source = read_rows(source_path)
    metrics = read_rows(audit_dir / "trajectory_metrics.jsonl")
    labels = {r["task_id"]: r for r in read_rows(ROOT / "data/splits/labels.jsonl")}
    candidates = {r["task_id"] for r in read_rows(ROOT / "data/sft/candidate_tasks.jsonl")}
    if len(source) != 1000 or any(t["task_id"] not in candidates for t in source):
        raise ValueError("Expected the completed 1,000-task SFT collection")
    kept, rejected, train, validation = prepare_rows(source, metrics, labels)
    if (len(kept), len(rejected), len(train), len(validation)) != (988, 12, 889, 99):
        raise ValueError("Stage-one-v1 counts differ from the approved audit")
    report_dir = ROOT / "outputs/sft-stage1-v1"
    report_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = report_dir / "previous"
    names = [
        "data/sft/train.jsonl",
        "data/sft/validation.jsonl",
        "data/sft/metadata.json",
        "data/splits/manifest.json",
        "data/splits/registry.jsonl",
    ]
    for name in names:
        src, dst = ROOT / name, backup_dir / name
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    import swanlab

    swanlab.init(
        project="shopping-grpo-longhorizon",
        experiment_name=f"{datetime.now(timezone.utc):%Y-%m-%d}-COLLECT-第一版筛选与划分",
        group="01-数据清洗与采集",
        job_type="data",
        description="执行第一阶段 SFT 长度/合法性筛选并按 seed=42 分层划分正式 889 训练 / 99 验证集。",
        tags=["DATA", "SFT", "Split"],
        config={
            "policy": VERSION,
            "stage2_applied": False,
            "seed": 42,
            "validation_ratio": 0.1,
            "source_sha256": sha256_file(source_path),
        },
        mode=args.swanlab_mode,
        logdir=str(report_dir / "swanlab"),
    )
    try:
        outputs = {
            "data/sft/trajectories.jsonl": kept,
            "data/sft/train.jsonl": train,
            "data/sft/validation.jsonl": validation,
            "data/sft/rejected_trajectories.jsonl": rejected,
            "data/sft/final_train_tasks.jsonl": [{"task_id": r["task_id"]} for r in train],
            "data/sft/final_validation_tasks.jsonl": [
                {"task_id": r["task_id"]} for r in validation
            ],
        }
        for name, rows in outputs.items():
            write_jsonl_atomic(ROOT / name, rows, force=True)
        quality = {
            "version": VERSION,
            "completed": True,
            "stage2_applied": False,
            "max_length": 24576,
            "student_model": audit["config"]["student_model"],
            "student_revision": audit["config"]["student_revision"],
            "source": {
                "path": str(source_path.relative_to(ROOT)),
                "sha256": sha256_file(source_path),
                "rows": len(source),
            },
            "audit_sha256": {
                name: sha256_file(audit_dir / name)
                for name in ["summary.json", "trajectory_metrics.jsonl"]
            },
            "excluded": rejected,
            "retained": [
                {
                    "task_id": r["task_id"],
                    "student_tokens": r["student_tokens"],
                    "assistant_loss_tokens": r["assistant_loss_tokens"],
                }
                for r in metrics
                if not r["stage1_issues"]
            ],
        }
        write_json_atomic(ROOT / "data/sft/quality_report.json", quality, force=True)
        metadata = {
            "schema_version": VERSION,
            "environment": "shopsimulator-environment-v2.1",
            "reward": REWARD_VERSION,
            "observation_version": "shopping-observation-v2",
            "tool_version": "shopping-tools-v2",
            "quality_screening": {
                "version": VERSION,
                "completed": True,
                "stage2_applied": False,
                "path": "data/sft/quality_report.json",
                "sha256": sha256_file(ROOT / "data/sft/quality_report.json"),
            },
            "split": {
                "finalized": True,
                "seed": 42,
                "validation_ratio": 0.1,
                "method": "whole_instruction_group_joint_stratification_natural_margins",
                "rounding": "round(total * 0.1)",
            },
            "source": quality["source"],
        }
        for role, rows in [("trajectories", kept), ("train", train), ("validation", validation)]:
            filename = "data/sft/" + role + ".jsonl"
            metadata[role] = {
                "path": filename,
                "rows": len(rows),
                "sha256": sha256_file(ROOT / filename),
                "distribution": distribution([labels[r["task_id"]] for r in rows]),
            }
        write_json_atomic(ROOT / "data/sft/metadata.json", metadata, force=True)
        final = {"version": VERSION, "total": len(kept), "metadata": "data/sft/metadata.json"}
        for role, rows in [("train", train), ("validation", validation)]:
            name = "data/sft/final_" + role + "_tasks.jsonl"
            final[role] = {
                "path": name,
                "sha256": sha256_file(ROOT / name),
                **distribution([labels[r["task_id"]] for r in rows]),
            }
        manifest["sft_final"] = final
        manifest["sft_demonstrations"].update(
            {
                "split_status": "finalized_after_stage1_v1",
                "quality_status": "stage1_complete_stage2_not_applied",
                "retained_trajectories": len(kept),
                "train_tasks": len(train),
                "validation_tasks": len(validation),
            }
        )
        train_ids, val_ids = {r["task_id"] for r in train}, {r["task_id"] for r in validation}
        rejected_ids = {r["task_id"] for r in rejected}
        registry = read_rows(ROOT / "data/splits/registry.jsonl")
        for row in registry:
            if row["task_id"] in train_ids | val_ids | rejected_ids:
                row["sft_final_role"] = (
                    "train"
                    if row["task_id"] in train_ids
                    else "validation"
                    if row["task_id"] in val_ids
                    else "excluded_stage1"
                )
                row["sft_quality_policy"] = VERSION
        write_jsonl_atomic(ROOT / "data/splits/registry.jsonl", registry, force=True)
        changed = [
            *outputs,
            "data/sft/metadata.json",
            "data/sft/quality_report.json",
            "data/splits/registry.jsonl",
        ]
        for name in changed:
            manifest["files"][name] = sha256_file(ROOT / name)
        write_json_atomic(manifest_path, manifest, force=True)
        verified = validate_current_sft(ROOT)
        swanlab.log(
            {
                "data/source_trajectories": len(source),
                "data/rejected_stage1": len(rejected),
                "data/retained": len(kept),
                "data/train": len(train),
                "data/validation": len(validation),
            }
        )
        print(
            json.dumps(
                {
                    "quality_version": VERSION,
                    **verified,
                    "retained": len(kept),
                    "rejected": len(rejected),
                },
                ensure_ascii=False,
            )
        )
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
