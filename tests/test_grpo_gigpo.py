from types import SimpleNamespace

import torch

from shopping_grpo.training.grpo.gigpo import (
    anchor_credits, apply_gigpo_advantages, retained_turns,
)


def test_credit_requires_action_return_contrast_across_trajectories_of_the_same_task():
    agent_turns = [
        [("shared", "a", 0, 1), ("shared", "b", 3, 4)],
        [("shared", "b", 0, 1)],
        [("same-action", "a", 0, 1)],
        [("same-action", "a", 0, 1)],
        [("shared", "c", 0, 1)],
    ]
    uids = ["task", "task", "other", "other", "unrelated"]
    scores = [1, 0, 1, 0, 1]
    credits, shared, supported = anchor_credits(uids, agent_turns, scores, 1e-8)
    assert (shared, supported) == (2, 1)
    assert credits == {(0, 0, 1): 0.5, (1, 0, 1): -0.5}
    assert anchor_credits(uids, agent_turns, [0.5] * 5, 1e-8) == ({}, 2, 0)


def test_gigpo_preserves_terminal_reward_and_maps_credit_only_to_retained_assistant_tokens():
    turns = [("dropped", "a", 0, 3), ("shared", "b", 5, 7)]
    assert retained_turns(turns, removed_tokens=4, response_length=2) == [("shared", "b", 1, 2)]
    scores = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    batch = SimpleNamespace(
        batch={
            "token_level_rewards": scores,
            "response_mask": torch.tensor([[1, 0, 1], [1, 0, 1]]),
            "advantages": torch.tensor([[0.5, 0.0, 0.5], [-0.5, 0.0, -0.5]]),
        },
        non_tensor_batch={
            "uid": ["group", "group"],
            "shopping_gigpo": [
                {"turns": [("shared", "buy", 2, 3)]},
                {"turns": [("shared", "leave", 2, 3)]},
            ],
        },
    )
    metrics = apply_gigpo_advantages(batch, tolerance=1e-8)
    assert batch.batch["advantages"].tolist() == [[0.5, 0.0, 0.75], [-0.5, 0.0, -0.75]]
    assert batch.batch["returns"] is batch.batch["advantages"]
    assert batch.batch["token_level_rewards"] is scores
    assert metrics["gigpo/credited_tokens"] == 2
    assert metrics["gigpo/nonzero_grpo_tokens"] == metrics["gigpo/nonzero_mixed_tokens"] == 4
