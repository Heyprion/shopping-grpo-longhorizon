"""每条 veRL trajectory 的轻量运行状态；不保存 ShopSimulator 隐藏 goal。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from contextvars import ContextVar

from shopping_grpo.environment.reward import (
    REWARD_VERSION,
    RewardContractError,
    strict_reward,
    validate_reward_components,
)

current_environment: ContextVar = ContextVar("shopsimulator_environment", default=None)
current_runtime_state: ContextVar = ContextVar("shopsimulator_runtime_state", default=None)
REWARD_TYPES = {
    "gold_purchase",
    "valid_alternative_purchase",
    "partial_alternative_purchase",
    "graceful_stop",
    "early_abstain",
    "wrong_purchase",
    "repeat_loop",
    "max_steps",
    "reward_unverifiable",
}


def make_runtime_state(task_id: int, max_steps: int) -> dict:
    """创建只含公共运行诊断的状态，reward 仅在环境正常终局后写入。"""
    return {
        "task_id": int(task_id),
        "max_steps": int(max_steps),
        "steps": [],
        "done": False,
        "terminate": False,
        "termination_reason": None,
        "consecutive_guard_rejections": 0,
        "action_attempt_count": 0,
        "repeat_action_count": 0,
        "recent_action_signatures": [],
        "terminal_result": {},
        "final_reward": 0.0,
        "reward_version": None,
        "reward_type": None,
        "reward_valid": True,
        "reward_unverifiable": False,
        "reward_detail": None,
        "reward_components": None,
        "infrastructure_invalid": False,
        "error": None,
        "context_compactions": 0,
        "context_tokens_removed": 0,
        "context_max_input_tokens": 0,
        "observation_projection_count": 0,
        "observation_truncated_count": 0,
        "observation_raw_tokens": 0,
        "observation_visible_tokens": 0,
        "observation_max_raw_tokens": 0,
        "observation_max_visible_tokens": 0,
        "observation_visible_asin_count": 0,
        "observation_visible_button_count": 0,
        "observation_any_truncated": False,
        "latest_observation_truncated": False,
        "observation_footer_failures": 0,
        "guard_rejection_count": 0,
        "guard_rejection_reason_counts": {},
        "guard_rejection_after_truncation_count": 0,
        "action_attempt_after_truncation_count": 0,
    }


def record_observation_projection(state: dict, meta: dict) -> None:
    """Aggregate public projection diagnostics without retaining hidden environment state."""
    raw_tokens = int(meta["raw_tokens"])
    visible_tokens = int(meta["visible_tokens"])
    state["observation_projection_count"] += 1
    state["observation_truncated_count"] += int(bool(meta["truncated"]))
    state["observation_raw_tokens"] += raw_tokens
    state["observation_visible_tokens"] += visible_tokens
    state["observation_max_raw_tokens"] = max(state["observation_max_raw_tokens"], raw_tokens)
    state["observation_max_visible_tokens"] = max(
        state["observation_max_visible_tokens"], visible_tokens
    )
    state["observation_visible_asin_count"] += int(meta["visible_asin_count"])
    state["observation_visible_button_count"] += int(meta["visible_button_count"])
    state["observation_any_truncated"] = state["observation_any_truncated"] or bool(
        meta["truncated"]
    )
    state["latest_observation_truncated"] = bool(meta["truncated"])
    state["observation_footer_failures"] += int(not bool(meta["critical_footer_preserved"]))


def record_action_attempt(state: dict, tool_name: str, parameters: dict, observation: str) -> None:
    """记录环境动作尝试，并统计最近三次中的重复签名。"""
    if tool_name == "think":
        return
    canonical_parameters = json.dumps(
        parameters,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    observation_fingerprint = hashlib.sha256(str(observation).encode("utf-8")).hexdigest()
    signature = (str(tool_name), canonical_parameters, observation_fingerprint)
    recent = state["recent_action_signatures"]
    state["action_attempt_count"] += 1
    if signature in recent:
        state["repeat_action_count"] += 1
    recent.append(signature)
    del recent[:-3]


def validate_reward(raw_detail: object) -> dict:
    """Validate the four factors and their product before exposing public data."""
    if not isinstance(raw_detail, Mapping):
        raise RewardContractError("reward_detail must be an object")
    if raw_detail.get("reward_version") != REWARD_VERSION:
        raise ValueError("reward_detail has an unsupported reward_version")
    kind = raw_detail.get("reward_type")
    if kind not in REWARD_TYPES:
        raise ValueError(f"unknown strict reward_type: {kind!r}")
    if raw_detail.get("termination_reason") != kind:
        raise ValueError("termination_reason must equal reward_type")
    valid = raw_detail.get("reward_valid")
    if type(valid) is not bool or (kind == "reward_unverifiable") != (not valid):
        raise ValueError("only reward_unverifiable may set reward_valid=false")
    components = validate_reward_components(raw_detail.get("reward_components"))
    score = strict_reward(components)
    try:
        utility = float(raw_detail["terminal_utility"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("terminal_utility must be numeric") from exc
    if not math.isfinite(utility) or not math.isclose(utility, score, abs_tol=1e-8):
        raise ValueError("terminal_utility must equal the strict component product")
    try:
        reported_score = float(raw_detail.get("strict_score", utility))
    except (TypeError, ValueError) as exc:
        raise ValueError("strict_score must be numeric") from exc
    if not math.isfinite(reported_score) or not math.isclose(reported_score, score, abs_tol=1e-8):
        raise ValueError("strict_score must equal the component product")
    success = kind in {"gold_purchase", "valid_alternative_purchase"}
    if success != math.isclose(score, 1.0, abs_tol=1e-8):
        raise ValueError("purchase success must agree with all four reward components")
    if raw_detail.get("purchase_success") is not success:
        raise ValueError("purchase_success is inconsistent with reward_type")
    if raw_detail.get("sampling_invalid") is not (not valid):
        raise ValueError("sampling_invalid must equal not reward_valid")
    if (
        kind not in {"gold_purchase", "valid_alternative_purchase", "partial_alternative_purchase"}
        and score != 0
    ):
        raise ValueError("non-purchase and failed outcomes must have zero strict reward")
    if kind == "partial_alternative_purchase" and not 0 < score < 1:
        raise ValueError("partial purchase must have a fractional strict reward")
    return {
        "reward_version": REWARD_VERSION,
        "reward_type": kind,
        "reward_valid": valid,
        "termination_reason": kind,
        "target_asin_match": bool(raw_detail.get("target_asin_match")),
        "reward_components": components,
        "strict_score": score,
        "terminal_utility": score,
        "purchase_success": success,
        "sampling_invalid": not valid,
        "match_score": components["r_att"] * components["r_option"],
        "evidence_coverage": float(raw_detail.get("evidence_coverage", 0.0)),
        "dimension_scores": {
            "category": components["r_type"],
            "attributes": components["r_att"],
            "options": components["r_option"],
            "price": components["r_price"],
        },
    }


def _normal_terminal(state: dict) -> bool:
    terminal = state.get("terminal_result") or {}
    return (
        state.get("done") is True and terminal.get("done") is True and terminal.get("over") is True
    )


def reward_breakdown(state: dict) -> dict:
    """The only learning signal is the four-factor product; no shaping terms."""
    normal = _normal_terminal(state)
    invalid = bool(state.get("infrastructure_invalid"))
    unverifiable = not bool(state.get("reward_valid", True))
    components = dict.fromkeys(("r_type", "r_att", "r_option", "r_price"), 0.0)
    native = 0.0
    if normal:
        try:
            if state.get("reward_version") != REWARD_VERSION:
                raise ValueError("unexpected reward version")
            components = validate_reward_components(
                (state.get("reward_detail") or {}).get("reward_components")
            )
            native = float(state["final_reward"])
            if not math.isfinite(native) or not math.isclose(
                native, strict_reward(components), abs_tol=1e-8
            ):
                raise ValueError("native reward differs from the strict product")
        except (KeyError, TypeError, ValueError):
            invalid = True
            components = dict.fromkeys(components, 0.0)
            native = 0.0
    score = strict_reward(components) if normal and not invalid and not unverifiable else 0.0
    full = float(score == 1 and state.get("reward_type") == "gold_purchase")
    return {
        **components,
        "full": full,
        "strict": score,
        "native": native,
        "semantic": score,
        "total": score,
        "terminal_utility": score,
        "match_score": components["r_att"] * components["r_option"],
        "evidence_coverage": float(
            (state.get("reward_detail") or {}).get("evidence_coverage", 0.0)
        ),
        "purchase_success": float(score == 1),
        "efficiency": 0.0,
        "penalty_overlong": 0.0,
        "penalty_unfinished": 0.0,
        "penalty_repeat": 0.0,
        "repeat_action_rate": int(state.get("repeat_action_count", 0))
        / max(int(state.get("action_attempt_count", 0)), 1),
        "sampling_invalid": bool(invalid or unverifiable),
        "infrastructure_invalid": invalid,
        "reward_unverifiable": unverifiable,
    }


def apply_reward_length_shaping(
    reward, state, *, enabled, soft_threshold=20, penalty_per_step=0.01, max_penalty=0.15
):
    """Strict reward rejects optional shaping instead of silently changing Eq. (4)."""
    if enabled:
        raise ValueError("strict reward forbids length shaping")
    return dict(reward)


def terminal_reward(state: dict, mode: str = "strict") -> float:
    if mode != "strict":
        raise ValueError(f"unknown shopping reward mode: {mode!r}; expected strict")
    return float(reward_breakdown(state)["total"])


def task_id_from_kwargs(kwargs: dict) -> int:
    """从 veRL parquet 的 extra_info 读取当前任务，缺失时立即失败。"""
    extra_info = kwargs.get("extra_info")
    if hasattr(extra_info, "item"):
        extra_info = extra_info.item()
    if not isinstance(extra_info, dict) or "task_id" not in extra_info:
        raise ValueError("veRL sample extra_info is missing task_id")
    return int(extra_info["task_id"])
