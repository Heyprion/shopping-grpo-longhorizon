"""The current public reward contract, shared by collection and training."""

from __future__ import annotations

import math
from collections.abc import Mapping

REWARD_VERSION = "shopsimulator-reward-strict-v1"
REWARD_FORMULA = "r_type*r_att*r_option*r_price"
REWARD_COMPONENT_KEYS = ("r_type", "r_att", "r_option", "r_price")


class RewardContractError(ValueError):
    """A terminal reward does not satisfy the active four-factor contract."""


def validate_reward_components(value):
    if not isinstance(value, Mapping):
        raise RewardContractError("reward_components must be an object")
    missing = set(REWARD_COMPONENT_KEYS) - set(value)
    if missing:
        raise ValueError("reward_components missing: " + ", ".join(sorted(missing)))
    result = {}
    for key in REWARD_COMPONENT_KEYS:
        if isinstance(value[key], bool):
            raise RewardContractError(f"reward component {key} must be numeric")
        try:
            number = float(value[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"reward component {key} must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"reward component {key} must be finite")
        if not 0 <= number <= 1:
            raise ValueError(f"reward component {key} must be in [0, 1]")
        result[key] = number
    return result


def strict_reward(components):
    return math.prod(validate_reward_components(components).values())


def full_strict_reward(detail):
    """Complete verified reward, independent of the target-ASIN outcome label."""
    if not isinstance(detail, Mapping) or detail.get("reward_version") != REWARD_VERSION:
        return False
    try:
        return (
            detail.get("reward_valid") is True
            and math.isclose(strict_reward(detail.get("reward_components")), 1.0, abs_tol=1e-8)
            and float(detail.get("terminal_utility", 0)) == 1.0
        )
    except (TypeError, ValueError):
        return False
