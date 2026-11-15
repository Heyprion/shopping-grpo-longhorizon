"""ShopSimulator paper Eq. (4): category × attributes × options × price.

Matching follows Appendix C.3 (soft category and fuzzy attribute/option
coverage). Chinese title nouns use jieba POS tagging. Invalid evidence is
explicitly marked, and every non-purchase terminal has zero utility.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache

from rapidfuzz.fuzz import token_set_ratio

from web_agent_site.engine.comparators import FAIL, PASS, UNVERIFIABLE, comparison, normalize_text
from web_agent_site.engine.reward_features import normalize_option_text
from web_agent_site.engine.variant_price import (
    candidate_options_for_evaluation,
    resolve_variant_price,
)

REWARD_VERSION = "shopsimulator-reward-strict-v1"
REWARD_FORMULA = "r_type*r_att*r_option*r_price"
FUZZY_THRESHOLD = 85
DEFAULT_REWARDS = {
    "gold_purchase": 1.0,
    "valid_alternative_purchase": 1.0,
    "partial_alternative_purchase": 0.0,  # Actual purchase utility is the product.
    "wrong_purchase": 0.0,
    "graceful_stop": 0.0,
    "early_abstain": 0.0,
    "max_steps": 0.0,
    "repeat_loop": 0.0,
    "reward_unverifiable": 0.0,
}


@dataclass(frozen=True)
class RewardResult:
    reward: float
    reward_type: str
    reward_valid: bool
    termination_reason: str
    target_asin_match: bool
    reward_components: dict
    evidence: dict

    def to_dict(self):
        components = self.reward_components
        component_evidence = self.evidence.get("components", {})
        coverage = sum(x.get("status") != UNVERIFIABLE for x in component_evidence.values())
        return {
            "reward": self.reward,
            "reward_version": REWARD_VERSION,
            "reward_formula": REWARD_FORMULA,
            "reward_type": self.reward_type,
            "reward_valid": self.reward_valid,
            "termination_reason": self.termination_reason,
            "target_asin_match": self.target_asin_match,
            "reward_components": components,
            "strict_score": self.reward,
            "terminal_utility": self.reward,
            "purchase_success": self.reward_type in {"gold_purchase", "valid_alternative_purchase"},
            "sampling_invalid": not self.reward_valid,
            "match_score": components["r_att"] * components["r_option"],
            "evidence_coverage": coverage / 4 if component_evidence else 0.0,
            "dimension_scores": {
                "category": components["r_type"],
                "attributes": components["r_att"],
                "options": components["r_option"],
                "price": components["r_price"],
            },
            "evidence": self.evidence,
        }


def _validate_rewards(rewards):
    for key, value in (rewards or {}).items():
        if key in DEFAULT_REWARDS and float(value) != DEFAULT_REWARDS[key]:
            raise ValueError("strict reward does not permit terminal bonuses or penalties")


def _result(score, kind, *, valid=True, asin_match=False, components=None, evidence=None):
    return RewardResult(
        float(score),
        kind,
        valid,
        kind,
        asin_match,
        components or dict.fromkeys(("r_type", "r_att", "r_option", "r_price"), 0.0),
        evidence or {},
    )


@lru_cache(maxsize=32768)
def _title_keywords(title):
    import jieba
    import jieba.posseg

    jieba.setLogLevel(logging.ERROR)
    return frozenset(
        normalize_text(token.word)
        for token in jieba.posseg.cut(title)
        if token.flag.startswith("n") or token.flag == "eng"
    )


def _type_component(product, goal):
    category = str(product.get("category") or "")
    target_category = str(goal.get("category") or "")
    title = str(product.get("title") or product.get("Title") or "")
    target_title = str(goal.get("name") or goal.get("title") or "")
    query = normalize_text(product.get("query"))
    target_query = normalize_text(goal.get("query"))
    # The frozen catalog has empty query fields: empty == empty is not evidence.
    query_match = bool(query and target_query and query == target_query)
    actual_nodes = {normalize_text(x) for x in category.split("›") if x.strip()}
    target_nodes = {normalize_text(x) for x in target_category.split("›") if x.strip()}
    category_match = len(actual_nodes & target_nodes) >= 2
    required = _title_keywords(target_title) if target_title else frozenset()
    actual = _title_keywords(title) if title else frozenset()
    overlap = len(required & actual) / len(required) if required else 0.0
    if query_match or category_match or overlap > 0.2:
        score, status = 1.0, PASS
    elif not required or not title:
        score, status = 0.0, UNVERIFIABLE
    else:
        score = 0.0 if overlap == 0 else 0.1 if overlap < 0.1 else 0.5
        status = PASS if score == 1 else FAIL
    evidence = comparison(
        status,
        comparator="paper_category_v1",
        required=target_category,
        actual=category,
        source_field="query|category|title",
        evidence={
            "query_match": query_match,
            "category_match": category_match,
            "title_overlap": overlap,
        },
    )
    return score, evidence


def _attribute_component(product, goal):
    required = goal.get("attributes", goal.get("expected_core_functions"))
    actual = product.get("attribute", product.get("Attributes"))
    if not isinstance(required, list) or not isinstance(actual, list):
        return 0.0, comparison(
            UNVERIFIABLE,
            comparator="paper_attributes_v1",
            required=required,
            actual=actual,
            source_field="attribute|title|description",
        )
    matched = []
    fallback = normalize_text(
        " ".join(
            str(product.get(key) or "")
            for key in ("title", "Title", "full_description", "Description")
        )
    )
    for value in required:
        text = normalize_text(value)
        if text and (
            any(token_set_ratio(text, normalize_text(x)) > FUZZY_THRESHOLD for x in actual)
            or text in fallback
        ):
            matched.append(value)
    score = len(matched) / len(required) if required else 1.0
    return score, comparison(
        PASS if score == 1 else FAIL,
        comparator="paper_attributes_v1",
        required=required,
        actual=actual,
        source_field="attribute|title|description",
        evidence={"matched": matched, "threshold": FUZZY_THRESHOLD},
    )


def _option_component(product, goal, selected):
    required = goal.get("goal_options")
    if required is None:
        required = [x["value"] for x in (goal.get("required_options_by_key") or {}).values()]
        required += [x["value"] for x in goal.get("unresolved_option_requirements", [])]
    if not isinstance(required, list) or not isinstance(selected, dict):
        return 0.0, comparison(
            UNVERIFIABLE,
            comparator="paper_options_v1",
            required=required,
            actual=selected,
            source_field="selected_options",
        )
    actual_axes = {
        normalize_option_text(k): v for k, v in (product.get("customization_options") or {}).items()
    }
    selected_axes = {normalize_option_text(k): v for k, v in selected.items()}
    valid_selection = set(actual_axes) == set(selected_axes) and all(
        any(
            normalize_option_text(entry.get("value")) == normalize_option_text(value)
            for entry in actual_axes[axis] or []
        )
        for axis, value in selected_axes.items()
    )
    if not valid_selection:
        return 0.0, comparison(
            UNVERIFIABLE,
            comparator="paper_options_v1",
            required=required,
            actual=selected,
            source_field="selected_options",
        )
    matched = [
        value
        for value in required
        if any(
            token_set_ratio(normalize_option_text(value), normalize_option_text(actual))
            > FUZZY_THRESHOLD
            for actual in selected.values()
        )
    ]
    score = len(matched) / len(required) if required else 1.0
    return score, comparison(
        PASS if score == 1 else FAIL,
        comparator="paper_options_v1",
        required=required,
        actual=selected,
        source_field="selected_options",
        evidence={"matched": matched, "threshold": FUZZY_THRESHOLD},
    )


def _price_component(resolution, goal):
    if "task_budget" in goal:
        from web_agent_site.engine.task_budget import budget_reasons

        try:
            if resolution.get("status") != PASS:
                raise ValueError("variant_price_unverifiable")
            reasons = budget_reasons(goal["task_budget"], resolution.get("price"))
            status = FAIL if reasons else PASS
        except (TypeError, ValueError, AttributeError) as exc:
            status, reasons = UNVERIFIABLE, [str(exc)]
        return float(status == PASS), comparison(
            status,
            comparator="canonical_task_budget_v1",
            required=goal["task_budget"],
            actual=resolution.get("price"),
            source_field="variant_price",
            evidence={"resolution": resolution, "reasons": reasons},
        )
    upper = goal.get("price_upper")
    try:
        actual = float(resolution["price"])
        limit = None if upper is None else float(upper)
        valid = (
            resolution.get("status") == PASS
            and math.isfinite(actual)
            and actual >= 0
            and (limit is None or math.isfinite(limit) and limit > 0)
        )
    except (TypeError, ValueError, KeyError):
        actual, limit, valid = None, upper, False
    status = UNVERIFIABLE if not valid else PASS if limit is None or actual <= limit else FAIL
    return float(status == PASS), comparison(
        status,
        comparator="paper_price_v1" if limit is not None else "budget_not_declared_v1",
        required=limit,
        actual=actual,
        source_field="variant_price",
        evidence=resolution,
    )


def evaluate_purchase(
    product, goal, *, selected_options, price_resolution=None, price=None, rewards=None
):
    _validate_rewards(rewards)
    if price_resolution is None:
        if price is None:
            price_resolution = resolve_variant_price(product, selected_options)
        else:
            price_resolution = {"status": PASS, "price": price, "method": "explicit_test_price"}
    pairs = {
        "r_type": _type_component(product, goal),
        "r_att": _attribute_component(product, goal),
        "r_option": _option_component(product, goal, selected_options),
        "r_price": _price_component(price_resolution, goal),
    }
    components = {key: float(value[0]) for key, value in pairs.items()}
    valid = all(value[1]["status"] != UNVERIFIABLE for value in pairs.values())
    score = math.prod(components.values())
    asin_match = str(product.get("asin")) == str(goal.get("asin"))
    if not valid:
        kind = "reward_unverifiable"
    elif math.isclose(score, 1.0, abs_tol=1e-8):
        kind = "gold_purchase" if asin_match else "valid_alternative_purchase"
    else:
        kind = "partial_alternative_purchase" if score > 0 else "wrong_purchase"
    return _result(
        score,
        kind,
        valid=valid,
        asin_match=asin_match,
        components=components,
        evidence={
            "components": {k: v[1] for k, v in pairs.items()},
            "price_resolution": price_resolution,
        },
    )


def evaluate_candidate_eligibility(product, goal):
    selected, resolution = candidate_options_for_evaluation(
        product, goal.get("required_options_by_key")
    )
    price = resolve_variant_price(product, selected)
    result = evaluate_purchase(product, goal, selected_options=selected, price_resolution=price)
    known = result.reward_valid and math.isclose(result.reward, 1.0, abs_tol=1e-8)
    components = result.evidence["components"]
    return {
        "status": PASS if known else FAIL,
        "known_acceptable": known,
        "known_valid": known,
        "selected_options": selected,
        "option_resolution": resolution,
        "price_resolution": price,
        "hard_gates": {"category": components["r_type"], "budget": components["r_price"]},
        "reward_components": result.reward_components,
        "match_score": result.reward_components["r_att"] * result.reward_components["r_option"],
        "evidence_coverage": result.to_dict()["evidence_coverage"],
    }


def evaluate_abstain(
    *,
    effective_result_sets,
    opened_candidates,
    known_acceptable_candidates=None,
    known_valid_candidates=None,
    rewards=None,
):
    _validate_rewards(rewards)
    count = (
        known_acceptable_candidates
        if known_acceptable_candidates is not None
        else known_valid_candidates or 0
    )
    eligible = effective_result_sets >= 2 and opened_candidates >= 2 and count == 0
    return _result(
        0.0, "graceful_stop" if eligible else "early_abstain", evidence={"eligible": eligible}
    )


def fixed_termination(reason, rewards=None):
    _validate_rewards(rewards)
    if reason not in {"repeat_loop", "max_steps"}:
        raise ValueError(f"unsupported fixed termination reason: {reason}")
    return _result(0.0, reason)
