import json

import pytest

from shopping_grpo.collection.data_contract import validate_current_sft
from shopping_grpo.environment.manifest import sha256_file
from shopping_grpo.environment.reward import REWARD_VERSION
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS


@pytest.fixture
def final_root(tmp_path):
    root = tmp_path
    (root / "data/sft").mkdir(parents=True)
    (root / "data/splits").mkdir()

    def write(name, rows):
        (root / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    rows = [
        {
            "task_id": i,
            "trajectory_id": str(i),
            "messages": [
                {"role": "system", "content": "shop"},
                {"role": "user", "content": str(i)},
            ],
            "tools": SHOP_TOOL_SCHEMAS,
        }
        for i in range(10)
    ]
    write("data/sft/trajectories.jsonl", rows)
    write(
        "data/splits/labels.jsonl",
        [{"task_id": i, "source_tag": "train", "group_id": str(i)} for i in range(10)],
    )
    write("data/sft/candidate_tasks.jsonl", [{"task_id": i} for i in range(10)])
    report = {
        "retained": [
            {"task_id": i, "student_tokens": 100, "assistant_loss_tokens": 10} for i in range(10)
        ]
    }
    (root / "data/sft/quality_report.json").write_text(json.dumps(report))
    metadata = {
        "reward": REWARD_VERSION,
        "quality_screening": {
            "version": "shopping-sft-stage1-v1",
            "completed": True,
            "stage2_applied": False,
            "path": "data/sft/quality_report.json",
            "sha256": sha256_file(root / "data/sft/quality_report.json"),
        },
        "split": {"finalized": True, "seed": 42},
        "trajectories": {
            "path": "data/sft/trajectories.jsonl",
            "sha256": sha256_file(root / "data/sft/trajectories.jsonl"),
        },
    }
    final = {"total": 10}
    for role, part in [("train", rows[:9]), ("validation", rows[9:])]:
        task_name = f"data/sft/final_{role}_tasks.jsonl"
        data_name = f"data/sft/{role}.jsonl"
        write(task_name, [{"task_id": r["task_id"]} for r in part])
        write(data_name, part)
        final[role] = {
            "path": task_name,
            "sha256": sha256_file(root / task_name),
            "tasks": len(part),
        }
        metadata[role] = {"rows": len(part), "sha256": sha256_file(root / data_name)}
    (root / "data/sft/metadata.json").write_text(json.dumps(metadata))
    (root / "data/splits/manifest.json").write_text(
        json.dumps(
            {
                "sft_final": final,
                "files": {"data/sft/metadata.json": sha256_file(root / "data/sft/metadata.json")},
            }
        )
    )
    return root


def test_final_contract_accepts_the_frozen_union(final_root):
    assert validate_current_sft(final_root) == {"status": "verified", "train": 9, "validation": 1}


def test_final_contract_blocks_instruction_group_overlap(final_root):
    p = final_root / "data/splits/labels.jsonl"
    labels = [json.loads(l) for l in p.read_text().splitlines()]
    labels[-1]["group_id"] = labels[0]["group_id"]
    p.write_text("".join(json.dumps(r) + "\n" for r in labels))
    with pytest.raises(ValueError, match="重复指令组"):
        validate_current_sft(final_root)


def test_final_contract_blocks_changed_conversations(final_root):
    p = final_root / "data/sft/train.jsonl"
    p.write_text(p.read_text().replace("shop", "changed"))
    with pytest.raises(ValueError, match="校验和"):
        validate_current_sft(final_root)
