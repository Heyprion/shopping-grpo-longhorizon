"""Policy selection must distinguish strict gold success from partial utility."""

import pytest

from shopping_grpo.training.grpo.validation import strict_success


@pytest.mark.parametrize("change,expected", [
    ({}, True), ({"done": False}, False),
    ({"reward_type": "partial_alternative_purchase"}, False),
    ({"reward_valid": False}, False), ({"reward_unverifiable": True}, False),
    ({"infrastructure_invalid": True}, False), ({"reward_version": "unknown"}, False),
])
def test_strict_validation_never_confuses_reward_with_gold_success(change, expected):
    info = {"done": True, "reward_type": "gold_purchase", "reward_valid": True,
            "reward_version": "shopsimulator-reward-v3", "reward": {"purchase_success": 1.0}}
    assert strict_success({**info, **change}) is expected
