import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/collect_sft_unified.py"
spec = importlib.util.spec_from_file_location("sft_unified", SCRIPT)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


@pytest.fixture
def collection_setup(tmp_path, monkeypatch):
    root = tmp_path
    for folder in ["data/sft", "data/splits", "data/cleaned"]:
        (root / folder).mkdir(parents=True)
    candidates = root / "data/sft/candidate_tasks.jsonl"
    candidates.write_text("".join(json.dumps({"task_id": i}) + "\n" for i in range(5000)))
    (root / "data/splits/labels.jsonl").write_text(
        "".join(
            json.dumps({"task_id": i, "source_tag": "train", "domain": "A", "difficulty": "simple"})
            + "\n"
            for i in range(5000)
        )
    )
    (root / "data/cleaned/task_budgets.jsonl").write_text("{}\n")
    plan = {
        "sampling_policy": "first_success_max_3",
        "attempts_per_task": 3,
        "stop_task_after_first_success": True,
        "candidate_pool": {
            "path": "data/sft/candidate_tasks.jsonl",
            "sha256": hashlib.sha256(candidates.read_bytes()).hexdigest(),
        },
        "accepted_joint_quotas": {"A|simple": 2},
    }
    (root / "data/sft/collection_plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(collector, "ROOT", root)
    monkeypatch.setattr(collector, "OpenAIChatClient", lambda **kwargs: None)
    monkeypatch.setattr(collector, "memory_fraction", lambda: 0)
    monkeypatch.setitem(
        sys.modules,
        "swanlab",
        SimpleNamespace(init=lambda **k: None, log=lambda *a, **k: None, finish=lambda: None),
    )
    monkeypatch.setenv("SHOPPING_TASK_BUDGETS", str(root / "data/cleaned/task_budgets.jsonl"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    output = root / "out"
    monkeypatch.setattr(sys, "argv", ["collect", "--output-dir", str(output), "--workers", "2"])
    return output


def test_joint_quota_replacement_earliest_success_and_resume(collection_setup, monkeypatch):
    output = collection_setup
    calls = []

    def fake_client(**kwargs):
        assert kwargs["max_tokens"] == 65536
        assert kwargs["reasoning_effort"] == "high"
        assert kwargs["timeout"] == 600

    monkeypatch.setattr(collector, "OpenAIChatClient", fake_client)

    def fake_collect(task, client, base_url, max_steps, attempt_index):
        task_id = task["task_id"]
        calls.append((task_id, attempt_index))
        return {
            "task_id": task_id,
            "attempt_index": attempt_index,
            "done": True,
            "final_reward": int(task_id != 0 and attempt_index in (1, 2)),
            "error": None,
            "release_error": None,
        }

    monkeypatch.setattr(collector, "collect_for_task", fake_collect)
    assert collector.main() == 0
    accepted = [json.loads(s) for s in (output / "accepted.jsonl").read_text().splitlines()]
    assert {t["task_id"] for t in accepted} == {1, 2}
    assert all(t["attempt_index"] == 1 for t in accepted)
    assert len(calls) == 7
    assert sorted(calls) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)]
    statuses = [json.loads(s) for s in (output / "task_status.jsonl").read_text().splitlines()]
    assert statuses[0]["status"] == "exhausted"
    assert statuses[2]["replacement_for"] == 0
    assert not (output / "train.jsonl").exists()
    assert not (output / "validation.jsonl").exists()
    assert collector.main() == 0
    assert len(calls) == 7


def test_reward_only():
    assert collector.valid_reward(
        {"done": True, "final_reward": 1, "blocked_tool_calls": [1], "terminal_result": {}}
    )
    assert not collector.valid_reward({"done": False, "final_reward": 1})
    assert not collector.valid_reward({"done": True, "final_reward": 0.9})


def test_truncation_pauses_without_consuming_attempt_and_can_resume(collection_setup, monkeypatch):
    output = collection_setup
    monkeypatch.setattr(sys, "argv", ["collect", "--output-dir", str(output), "--workers", "1"])
    calls = []
    fail = True

    def fake_collect(task, **kwargs):
        attempt = kwargs["attempt_index"]
        calls.append((task["task_id"], attempt))
        return {
            "task_id": task["task_id"],
            "attempt_index": attempt,
            "done": not fail,
            "final_reward": 0 if fail else 1,
            "error": {"type": "ModelOutputTruncatedError", "message": "length"} if fail else None,
            "release_error": None,
            "model_completions": [
                {
                    "finish_reason": "length" if fail else "tool_calls",
                    "truncated": fail,
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 65536,
                        "completion_tokens_details": {"reasoning_tokens": 65520},
                    },
                }
            ],
        }

    monkeypatch.setattr(collector, "collect_for_task", fake_collect)
    assert collector.main() == 2
    summary = json.loads((output / "summary.json").read_text())
    assert summary["attempts"] == 0
    assert summary["accepted"] == 0
    assert calls == [(0, 0)]
    error = json.loads((output / "infrastructure_errors.jsonl").read_text())
    assert error["trajectory"]["model_completions"][0]["finish_reason"] == "length"
    assert collector.completion_usage(error["trajectory"]) == {
        "responses": 1,
        "truncated_responses": 1,
        "prompt_tokens": 100,
        "completion_tokens": 65536,
        "reasoning_tokens": 65520,
    }
    fail = False
    assert collector.main() == 0
    assert calls[1] == (0, 0)
    assert json.loads((output / "summary.json").read_text())["attempts"] == 2


def test_different_output_budget_cannot_mix_with_existing_collection(collection_setup, monkeypatch):
    output = collection_setup

    def fake_collect(task, **kwargs):
        return {
            "task_id": task["task_id"],
            "attempt_index": kwargs["attempt_index"],
            "done": True,
            "final_reward": 1,
        }

    monkeypatch.setattr(collector, "collect_for_task", fake_collect)
    assert collector.main() == 0
    monkeypatch.setattr(
        sys, "argv", ["collect", "--output-dir", str(output), "--max-tokens", "2048"]
    )
    with pytest.raises(SystemExit, match="configuration differs"):
        collector.main()
    assert json.loads((output / "summary.json").read_text())["attempts"] == 2


@pytest.mark.parametrize("success_at", [0, 1, 2])
def test_first_success_is_immediately_accepted_and_stops_sampling(
    collection_setup, monkeypatch, capsys, success_at
):
    output = collection_setup
    calls = []

    def fake_collect(task, **kwargs):
        calls.append((task["task_id"], kwargs["attempt_index"]))
        return {
            "task_id": task["task_id"],
            "attempt_index": kwargs["attempt_index"],
            "done": True,
            "final_reward": int(kwargs["attempt_index"] == success_at),
        }

    monkeypatch.setattr(collector, "collect_for_task", fake_collect)
    assert collector.main() == 0
    first = next(line for line in capsys.readouterr().out.splitlines() if "reward=1" in line)
    assert "reward=1" in first
    assert "successful_attempts=1" in first
    assert "successful_tasks=1" in first
    assert "accepted=1" in first
    assert sorted(calls) == [
        (task_id, attempt) for task_id in (0, 1) for attempt in range(success_at + 1)
    ]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["successful_attempts"] == 2
    assert summary["attempts"] == 2 * (success_at + 1)
    assert summary["successful_tasks"] == summary["accepted"] == 2
    assert collector.main() == 0
    resumed = json.loads((output / "summary.json").read_text())
    assert resumed["successful_attempts"] == 2
    assert resumed["successful_tasks"] == 2


def test_saved_successes_are_adopted_without_new_api_calls(collection_setup, monkeypatch):
    output = collection_setup

    def fake_collect(task, **kwargs):
        return {
            "task_id": task["task_id"],
            "attempt_index": kwargs["attempt_index"],
            "done": True,
            "final_reward": 1,
        }

    monkeypatch.setattr(collector, "collect_for_task", fake_collect)
    assert collector.main() == 0
    with sqlite3.connect(output / "collection.sqlite3") as db:
        previous = json.loads(db.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0])
        previous.pop("sampling_policy")
        db.execute("UPDATE meta SET value=? WHERE key='config'", (json.dumps(previous),))
        db.execute("UPDATE tasks SET status='active'")
        extra = {"task_id": 0, "attempt_index": 1, "done": True, "final_reward": 1}
        db.execute("INSERT INTO attempts VALUES (0, 1, ?)", (json.dumps(extra),))
    plan_path = collector.ROOT / "data/sft/collection_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["previous_plan_sha256"] = previous["plan_sha256"]
    plan_path.write_text(json.dumps(plan))

    def unexpected_call(*args, **kwargs):
        pytest.fail("A recorded successful task was sampled again")

    monkeypatch.setattr(collector, "collect_for_task", unexpected_call)
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    monkeypatch.setattr(sys, "argv", ["collect", "--output-dir", str(output), "--reconcile-only"])
    assert collector.main() == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["accepted"] == 2
    assert summary["attempts"] == 3
    accepted = [json.loads(line) for line in (output / "accepted.jsonl").read_text().splitlines()]
    assert all(t["attempt_index"] == 0 for t in accepted)
    assert (output / "collection.before-first-success.sqlite3").exists()
    with sqlite3.connect(output / "collection.sqlite3") as db:
        assert dict(db.execute("SELECT task_id,status FROM tasks")) == {
            0: "accepted",
            1: "accepted",
        }
        assert (
            json.loads(
                db.execute(
                    "SELECT value FROM meta WHERE key='config_before_first_success'"
                ).fetchone()[0]
            )
            == previous
        )
