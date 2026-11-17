import pytest

from scripts import prepare_sft_dataset as prepare


def fixture_rows():
    source = []
    metrics = []
    labels = {}
    for i in range(30):
        source.append(
            {
                "task_id": i,
                "trajectory_id": str(i),
                "messages": [
                    {"role": "system", "content": "shop"},
                    {"role": "user", "content": str(i)},
                ],
                "steps": [],
                "done": True,
            }
        )
        metrics.append(
            {
                "task_id": i,
                "trajectory_id": str(i),
                "stage1_issues": [],
                "student_tokens": 100,
                "assistant_loss_tokens": 10,
                "guard_blocks": 100,
                "repeated_state_action_steps": [1, 2, 3],
            }
        )
        labels[i] = {
            "task_id": i,
            "group_id": str(i),
            "domain": "A",
            "difficulty": ("simple", "medium", "hard")[i % 3],
        }
    metrics[0]["stage1_issues"] = ["student_context_over_limit"]
    metrics[1]["stage1_issues"] = ["action_not_in_structured_observation"]
    return source, metrics, labels


def test_only_stage_one_affects_membership_and_split_is_reproducible(monkeypatch):
    monkeypatch.setattr(prepare, "structure_issues", lambda *a: [])
    source, metrics, labels = fixture_rows()
    kept, rejected, train, val = prepare.prepare_rows(source, metrics, labels)
    assert len(kept) == 28 and len(rejected) == 2 and len(train) == 25 and len(val) == 3
    assert {r["task_id"] for r in rejected} == {0, 1}
    assert {r["task_id"] for r in train}.isdisjoint(r["task_id"] for r in val)
    assert prepare.prepare_rows(source, metrics, labels) == (kept, rejected, train, val)


def test_audit_cannot_be_reused_for_another_trajectory(monkeypatch):
    monkeypatch.setattr(prepare, "structure_issues", lambda *a: [])
    source, metrics, labels = fixture_rows()
    metrics[3]["trajectory_id"] = "different"
    with pytest.raises(ValueError, match="identity"):
        prepare.prepare_rows(source, metrics, labels)
