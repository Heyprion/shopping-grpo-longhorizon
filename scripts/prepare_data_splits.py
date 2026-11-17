#!/usr/bin/env python3
"""Materialize the fixed task pools, initial selections, and unused-task ledger."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from collections import defaultdict
from pathlib import Path

from shopping_grpo.collection.stratification import (
    EVALUATION_ASSET,
    EVALUATION_CONTRACT,
    LABEL_VERSION,
    LEVELS,
    SPLIT_VERSION,
    TRAIN_RATIOS,
    distribution,
    label_task,
    partition_pools,
    proportional_counts,
    select_tasks,
)
from shopping_grpo.collection.task_cleaning import fingerprint
from shopping_grpo.environment.manifest import sha256_file
from shopping_grpo.environment.reward import REWARD_VERSION
from shopping_grpo.evaluation.artifacts import write_json_atomic, write_jsonl_atomic

ROOT = Path(__file__).resolve().parents[1]
INITIAL = {
    "sft_train": (800, "data/sft/train_tasks.jsonl"),
    "sft_validation": (200, "data/sft/validation_tasks.jsonl"),
    "rl_train": (1000, "data/grpo/train.jsonl"),
    "rl_validation": (50, "data/grpo/validation.jsonl"),
    "evaluation": (200, "data/evaluation/tasks.jsonl"),
}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    write_jsonl_atomic(path, rows, force=True)


def write_json(path, value):
    write_json_atomic(path, value, force=True)


def task_rows(rows):
    return [{"task_id": r["task_id"]} for r in rows]


def parquet_tasks(path, rows, cleaned_by_id, split):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from shopping_grpo.evaluation.rollout import SYSTEM_PROMPT

    records = [
        {
            "data_source": "shopsimulator",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Instruction: " + cleaned_by_id[r["task_id"]]["instruction"],
                },
            ],
            "ability": "shopping",
            "reward_model": {"style": "rule", "ground_truth": None},
            "extra_info": {"split": split, "index": i, "task_id": r["task_id"]},
        }
        for i, r in enumerate(rows)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(records), temporary, compression="zstd")
    temporary.replace(path)


def file_spec(root, name, labels):
    path = root / name
    ids = [r["task_id"] for r in read_rows(path)]
    return {"path": name, "sha256": sha256_file(path), **distribution([labels[i] for i in ids])}


def snapshot_replaced_files(root):
    # Reversible local backup; old assets are never added as another active dataset.
    backup = root / "outputs/data-split-migration/previous"
    names = [
        "data/grpo/train.jsonl",
        "data/grpo/validation.jsonl",
        "data/grpo/train.parquet",
        "data/grpo/validation.parquet",
        "data/grpo/metadata.json",
        "data/evaluation/tasks.jsonl",
        "data/evaluation/metadata.json",
        "src/shopping_grpo/resources/blind_guard.json",
        "src/shopping_grpo/resources/blind_final_task_ids.json",
    ]
    for name in names:
        source, target = root / name, backup / name
        if source.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def build(root, seed=42):
    manifest_path = root / "data/splits/manifest.json"
    if manifest_path.exists():
        verify(root)
        return {"status": "already_materialized", "manifest": "data/splits/manifest.json"}
    cleaned_root = root / "data/cleaned"
    metadata = json.loads((cleaned_root / "metadata.json").read_text())
    for name, spec in metadata["files"].items():
        if sha256_file(cleaned_root / name) != spec["sha256"]:
            raise ValueError(f"cleaned source checksum mismatch: {name}")
    source = root / metadata["source"]["path"]
    if sha256_file(source) != metadata["source"]["sha256"]:
        raise ValueError("raw catalog checksum mismatch")
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        products = json.load(stream)
    cleaned = read_rows(cleaned_root / "tasks.jsonl")
    cleaned_by_id = {r["task_id"]: r for r in cleaned}
    labels = []
    for row in sorted(cleaned, key=lambda r: r["task_id"]):
        product = products[row["task_id"]]
        if product["asin"] != row["asin"] or fingerprint(product) != row["record_hash"]:
            raise ValueError(f"task identity mismatch: {row['task_id']}")
        labels.append(label_task(row, product))
    if len(labels) != len(cleaned_by_id) or len(labels) != 16741:
        raise ValueError("unexpected cleaned task population")
    pools = partition_pools(labels, seed)
    active = {
        name: select_tasks(
            pools[name],
            count,
            seed=seed,
            namespace="initial:" + name,
            ratios=TRAIN_RATIOS[name.split("_")[0]] if name in ("sft_train", "rl_train") else None,
        )
        for name, (count, _) in INITIAL.items()
    }
    # Finish the entire allocation in memory before replacing any canonical list.
    snapshot_replaced_files(root)
    write_rows(root / "data/splits/labels.jsonl", labels)
    labels_by_id = {r["task_id"]: r for r in labels}
    paths = {"data/splits/labels.jsonl"}
    pool_specs, initial_specs, reserve_specs, registry = {}, {}, {}, []
    for name, rows in pools.items():
        path = f"data/splits/pools/{name}.jsonl"
        write_rows(root / path, task_rows(rows))
        paths.add(path)
        pool_specs[name] = file_spec(root, path, labels_by_id)
        selected = {r["task_id"] for r in active.get(name, [])}
        remaining = [r for r in rows if r["task_id"] not in selected]
        if name != "quarantine":
            reserve_path = f"data/splits/reserves/{name}.jsonl"
            write_rows(root / reserve_path, task_rows(remaining))
            paths.add(reserve_path)
            reserve_specs[name] = file_spec(root, reserve_path, labels_by_id)
        for order, row in enumerate(rows):
            registry.append(
                {
                    "task_id": row["task_id"],
                    "pool": name,
                    "difficulty": row["difficulty"],
                    "domain": row["domain"],
                    "group_id": row["group_id"],
                    "reserve_order": order,
                    "selection_status": "excluded"
                    if name == "quarantine"
                    else "selected"
                    if row["task_id"] in selected
                    else "available",
                    "selection_batch": "initial" if row["task_id"] in selected else None,
                    "execution_status": "not_started",
                }
            )
    write_rows(root / "data/splits/registry.jsonl", sorted(registry, key=lambda r: r["task_id"]))
    paths.add("data/splits/registry.jsonl")
    for name, (_, path) in INITIAL.items():
        write_rows(root / path, task_rows(active[name]))
        paths.add(path)
        initial_specs[name] = file_spec(root, path, labels_by_id)
    for name in ("rl_train", "rl_validation"):
        filename = "train" if name == "rl_train" else "validation"
        path = f"data/grpo/{filename}.parquet"
        parquet_tasks(root / path, active[name], cleaned_by_id, filename)
        paths.add(path)
    write_json(
        root / "data/grpo/metadata.json",
        {
            "schema_version": "shopping-grpo-dataset-v2",
            "environment": "shopsimulator-environment-v2.1",
            "reward": REWARD_VERSION,
            "split_version": SPLIT_VERSION,
            "seed": seed,
            "manifest": "data/splits/manifest.json",
            "train": {
                **initial_specs["rl_train"],
                "parquet": "data/grpo/train.parquet",
                "parquet_sha256": sha256_file(root / "data/grpo/train.parquet"),
            },
            "validation": {
                **initial_specs["rl_validation"],
                "parquet": "data/grpo/validation.parquet",
                "parquet_sha256": sha256_file(root / "data/grpo/validation.parquet"),
            },
        },
    )
    paths.add("data/grpo/metadata.json")
    evaluation = {
        "schema_version": "shopping-evaluation-dataset-v4",
        "name": "Strict-Stratified-200",
        "asset": EVALUATION_ASSET,
        "contract": EVALUATION_CONTRACT,
        "environment": "shopsimulator-environment-v2.1",
        "environment_version": "shopsimulator-environment-v2.1",
        "reward": REWARD_VERSION,
        "reward_version": REWARD_VERSION,
        "evaluated": False,
        **initial_specs["evaluation"],
        "training_overlap": 0,
        "source_tag": "eval",
        "pool_tasks": 944,
        "seed": seed,
        "split_version": SPLIT_VERSION,
        "manifest": "data/splits/manifest.json",
    }
    write_json(root / "data/evaluation/metadata.json", evaluation)
    paths.add("data/evaluation/metadata.json")
    resources = "src/shopping_grpo/resources/"
    final_ids = [r["task_id"] for r in active["evaluation"]]
    protected = sorted(r["task_id"] for name in ("evaluation", "quarantine") for r in pools[name])
    write_json(
        root / (resources + "blind_final_task_ids.json"),
        {"schema_version": "shopping-blind-task-ids-v1", "task_ids": final_ids},
    )
    write_json(
        root / (resources + "held_out_task_ids.json"),
        {"schema_version": "shopping-held-out-task-ids-v1", "task_ids": protected},
    )
    write_json(
        root / (resources + "blind_guard.json"),
        {
            "schema_version": "shopping-blind-asset-guard-v1",
            "manifest_version": 1,
            "asset": EVALUATION_ASSET,
            "split_role": "blind_final_test",
            "task_sha256": evaluation["sha256"],
            "task_count": 200,
            "metadata_sha256": sha256_file(root / "data/evaluation/metadata.json"),
            "required_metadata": {
                k: evaluation[k]
                for k in ("asset", "contract", "environment_version", "reward_version", "evaluated")
            },
            "task_ids_resource": "blind_final_task_ids.json",
            "protected_task_ids_resource": "held_out_task_ids.json",
            "protected_task_count": len(protected),
        },
    )
    paths.update(
        resources + p
        for p in ("blind_final_task_ids.json", "held_out_task_ids.json", "blind_guard.json")
    )
    metadata.update(
        split_status="partitioned",
        split_manifest="data/splits/manifest.json",
        label_version=LABEL_VERSION,
    )
    write_json(cleaned_root / "metadata.json", metadata)
    paths.add("data/cleaned/metadata.json")
    manifest = {
        "schema_version": SPLIT_VERSION,
        "status": "materialized",
        "seed": seed,
        "label_version": LABEL_VERSION,
        "reward_version": REWARD_VERSION,
        "source": {
            "archive": metadata["source"],
            "cleaned_tasks_sha256": metadata["files"]["tasks.jsonl"]["sha256"],
            "budget_sha256": metadata["files"]["task_budgets.jsonl"]["sha256"],
        },
        "allocation": "Exact difficulty margins; Hamilton natural category margins; IPF associations; capacity-constrained min-cost integer rounding; whole-instruction groups.",
        "sampling_unit": "unique_task",
        "selection_does_not_mean_trained": True,
        "sft_demonstrations": {
            "status": "pending_collection",
            "train_tasks": INITIAL["sft_train"][1],
            "validation_tasks": INITIAL["sft_validation"][1],
        },
        "label_distribution": distribution(labels),
        "pools": pool_specs,
        "initial": initial_specs,
        "reserves": reserve_specs,
        "training_ratios": {k: dict(zip(LEVELS, v)) for k, v in TRAIN_RATIOS.items()},
        "rounds": [],
        "files": {name: sha256_file(root / name) for name in sorted(paths)},
        "code_hashes": {
            name: sha256_file(ROOT / name)
            for name in (
                "scripts/prepare_data_splits.py",
                "src/shopping_grpo/collection/stratification.py",
            )
        },
    }
    write_json(manifest_path, manifest)
    return verify(root)


def verify(root):
    manifest = json.loads((root / "data/splits/manifest.json").read_text())
    if manifest["schema_version"] != SPLIT_VERSION:
        raise ValueError("unsupported split manifest")
    for name, expected in manifest["files"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"split artifact checksum mismatch: {name}")
    labels = {r["task_id"]: r for r in read_rows(root / "data/splits/labels.jsonl")}
    assigned, group_pools = {}, defaultdict(set)
    for name, spec in manifest["pools"].items():
        ids = [r["task_id"] for r in read_rows(root / spec["path"])]
        if len(ids) != spec["tasks"] or len(ids) != len(set(ids)):
            raise ValueError(f"pool count mismatch: {name}")
        for task_id in ids:
            if task_id in assigned:
                raise ValueError("task occurs in multiple pools")
            assigned[task_id] = name
            row = labels[task_id]
            if name != "quarantine":
                group_pools[row["group_id"]].add(name)
            if (name == "evaluation") != (row["source_tag"] == "eval"):
                raise ValueError("source tag/pool mismatch")
    if set(assigned) != set(labels) or any(len(v) != 1 for v in group_pools.values()):
        raise ValueError("pool coverage or repeated-instruction isolation failure")
    chosen = set()
    for name, spec in manifest["initial"].items():
        rows = [labels[r["task_id"]] for r in read_rows(root / spec["path"])]
        if len(rows) != INITIAL[name][0] or any(assigned[r["task_id"]] != name for r in rows):
            raise ValueError("initial selection count/pool mismatch")
        ids = {r["task_id"] for r in rows}
        if len(ids) != len(rows) or ids & chosen:
            raise ValueError("initial task overlap")
        chosen.update(ids)
        actual = distribution(rows)
        pool = manifest["pools"][name]
        if actual["domain"] != proportional_counts(pool["domain"], len(rows)):
            raise ValueError("category margin is not naturally proportional")
        expected = proportional_counts(
            dict(zip(LEVELS, TRAIN_RATIOS[name.split("_")[0]]))
            if name in ("sft_train", "rl_train")
            else pool["difficulty"],
            len(rows),
        )
        if actual["difficulty"] != expected:
            raise ValueError("difficulty margin mismatch")
    for batch in manifest["rounds"]:
        ids = {r["task_id"] for r in read_rows(root / batch["path"])}
        if len(ids) != batch["tasks"] or ids & chosen:
            raise ValueError("extension reused a previously selected task")
        if any(assigned[i] != batch["pool"] for i in ids):
            raise ValueError("extension crosses pool boundary")
        chosen.update(ids)
    registry = read_rows(root / "data/splits/registry.jsonl")
    if len(registry) != len(labels) or len({r["task_id"] for r in registry}) != len(labels):
        raise ValueError("registry population mismatch")
    for row in registry:
        if row["pool"] != assigned[row["task_id"]]:
            raise ValueError("registry pool mismatch")
        if (row["selection_status"] == "selected") != (row["task_id"] in chosen):
            raise ValueError("registry selection mismatch")
    for name, spec in manifest["reserves"].items():
        ids = {r["task_id"] for r in read_rows(root / spec["path"])}
        expected = {i for i, pool in assigned.items() if pool == name} - chosen
        if ids != expected or len(ids) != spec["tasks"]:
            raise ValueError("reserve ledger mismatch")
    import pyarrow.parquet as pq

    for name, filename in (("rl_train", "train"), ("rl_validation", "validation")):
        parquet_ids = [
            r["extra_info"]["task_id"]
            for r in pq.read_table(root / f"data/grpo/{filename}.parquet").to_pylist()
        ]
        if parquet_ids != [
            r["task_id"] for r in read_rows(root / manifest["initial"][name]["path"])
        ]:
            raise ValueError("RL JSONL/Parquet task mismatch")
    result = {
        "status": "verified",
        "tasks": len(labels),
        "label_distribution": manifest["label_distribution"],
        "initial": {name: spec["tasks"] for name, spec in manifest["initial"].items()},
        "reserves": {name: spec["tasks"] for name, spec in manifest["reserves"].items()},
        "rounds": len(manifest["rounds"]),
    }
    if (
        manifest.get("sft_demonstrations", {}).get("split_status")
        in ("deferred_until_after_quality_screening", "finalized_after_stage1_v1")
    ):
        plan = json.loads((root / manifest["sft_collection_plan"]["path"]).read_text())
        candidate_ids = [r["task_id"] for r in read_rows(root / plan["candidate_pool"]["path"])]
        expected = {i for i, pool in assigned.items() if pool in ("sft_train", "sft_validation")}
        if (
            len(candidate_ids) != 5000
            or len(set(candidate_ids)) != 5000
            or set(candidate_ids) != expected
        ):
            raise ValueError("unified SFT pool identity mismatch")
        population = distribution([labels[i] for i in candidate_ids])
        if plan["target_accepted_unique_trajectories"] != 1000 or "splits" in plan:
            raise ValueError("SFT collection must target one unsplit 1000-trajectory set")
        for field in ("difficulty", "domain", "joint"):
            quotas = plan["accepted_" + field + "_quotas"]
            if sum(quotas.values()) != 1000 or any(
                n < 0 or n > population[field].get(k, 0) for k, n in quotas.items()
            ):
                raise ValueError("invalid unified SFT quotas")
        if plan["accepted_difficulty_quotas"] != {"simple": 300, "medium": 500, "hard": 200}:
            raise ValueError("unified SFT difficulty quotas mismatch")
        if plan["accepted_domain_quotas"] != proportional_counts(population["domain"], 1000):
            raise ValueError("unified SFT category margins must follow the merged pool")
        for domain, total in plan["accepted_domain_quotas"].items():
            if (
                sum(
                    v for k, v in plan["accepted_joint_quotas"].items() if k.split("|")[0] == domain
                )
                != total
            ):
                raise ValueError("unified SFT joint/domain quotas disagree")
        for level, total in plan["accepted_difficulty_quotas"].items():
            if (
                sum(v for k, v in plan["accepted_joint_quotas"].items() if k.split("|")[1] == level)
                != total
            ):
                raise ValueError("unified SFT joint/difficulty quotas disagree")
        result["initial"] = {k: v for k, v in result["initial"].items() if not k.startswith("sft_")}
        result["reserves"] = {
            k: v for k, v in result["reserves"].items() if not k.startswith("sft_")
        }
        result["sft_collection"] = {
            "candidate_tasks": 5000,
            "accepted_target": 1000,
            "difficulty_quotas": plan["accepted_difficulty_quotas"],
            "train_validation_split": "after_later_quality_screening",
        }
        if manifest.get("sft_final"):
            from shopping_grpo.collection.data_contract import validate_current_sft
            result["sft_final"] = validate_current_sft(root)
            result["sft_collection"]["train_validation_split"] = "finalized_after_stage1_v1"
    return result


def extend(root, stage, count, ratios=None, dry_run=False):
    verify(root)
    manifest_path = root / "data/splits/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        stage == "sft"
        and manifest.get("sft_collection_plan")
    ):
        raise ValueError(
            "SFT 已改为统一 5,000 题候选池采集，不能按旧训练/验证分块补题；请按 collection_plan.json 执行统一采集。"
        )
    pool = stage + "_train"
    labels = {r["task_id"]: r for r in read_rows(root / "data/splits/labels.jsonl")}
    reserve_path = root / manifest["reserves"][pool]["path"]
    available = [labels[r["task_id"]] for r in read_rows(reserve_path)]
    number = 1 + sum(batch["pool"] == pool for batch in manifest["rounds"])
    name = f"{stage}_{number:03d}"
    selected = select_tasks(
        available,
        count,
        seed=manifest["seed"],
        namespace=name,
        ratios=ratios or TRAIN_RATIOS[stage],
    )
    selected_ids = {r["task_id"] for r in selected}
    record = {
        "batch": name,
        "pool": pool,
        "path": f"data/splits/rounds/{name}.jsonl",
        "ratios": dict(zip(LEVELS, ratios or TRAIN_RATIOS[stage])),
        **distribution(selected),
    }
    if dry_run:
        return {"dry_run": True, **record, "task_ids": [r["task_id"] for r in selected]}
    write_rows(root / record["path"], task_rows(selected))
    paths = {record["path"], "data/splits/registry.jsonl", manifest["reserves"][pool]["path"]}
    if stage == "rl":
        cleaned = {r["task_id"]: r for r in read_rows(root / "data/cleaned/tasks.jsonl")}
        record["parquet"] = f"data/splits/rounds/{name}.parquet"
        parquet_tasks(root / record["parquet"], selected, cleaned, name)
        paths.add(record["parquet"])
    write_rows(reserve_path, task_rows([r for r in available if r["task_id"] not in selected_ids]))
    registry = read_rows(root / "data/splits/registry.jsonl")
    for row in registry:
        if row["task_id"] in selected_ids:
            if row["selection_status"] != "available":
                raise ValueError("extension attempted to reuse a selected task")
            row.update(selection_status="selected", selection_batch=name)
    write_rows(root / "data/splits/registry.jsonl", registry)
    record["sha256"] = sha256_file(root / record["path"])
    manifest["rounds"].append(record)
    manifest["reserves"][pool] = file_spec(root, manifest["reserves"][pool]["path"], labels)
    manifest["files"].update({path: sha256_file(root / path) for path in paths})
    write_json(manifest_path, manifest)
    verify(root)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify", "extend"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=("sft", "rl"), default="rl")
    parser.add_argument("--count", type=int)
    parser.add_argument("--ratios", type=int, nargs=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.action == "extend" and (not args.count or args.count < 1):
        parser.error("extend requires a positive --count")
    if args.ratios is not None and (min(args.ratios) < 0 or sum(args.ratios) <= 0):
        parser.error("ratios must be nonnegative with a positive sum")
    result = (
        build(args.root)
        if args.action == "build"
        else verify(args.root)
        if args.action == "verify"
        else extend(args.root, args.stage, args.count, args.ratios, args.dry_run)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
