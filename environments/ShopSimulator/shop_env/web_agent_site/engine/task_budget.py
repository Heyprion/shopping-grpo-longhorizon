"""Canonical model-extracted budgets shared by screening and Strict Reward."""

import hashlib
import json
from decimal import Decimal, InvalidOperation
from functools import lru_cache

VERSION = "shopping-task-budget-v1"


def query_hash(query):
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def budget_contract(query, prices, state):
    return {
        "version": VERSION,
        "query_sha256": query_hash(query),
        "state": state,
        "currency": "CNY",
        "around_upper_multiplier": "1.1",
        "prices": prices,
    }


def budget_limits(contract):
    """Validate the frozen interpretation; never interpret natural language."""
    if (
        contract.get("version") != VERSION
        or contract.get("currency") != "CNY"
        or contract.get("around_upper_multiplier") != "1.1"
    ):
        raise ValueError("invalid_budget_contract")
    prices = contract.get("prices")
    state = contract.get("state")
    if not isinstance(prices, list) or state not in {"parsed", "not_mentioned"}:
        raise ValueError("budget_unresolved")
    if (state == "parsed") != bool(prices):
        raise ValueError("budget_state_inconsistent")
    bounds = []
    for price in prices:
        if price.get("scope") != "purchase":
            raise ValueError("price_semantics_unsupported")
        op = price.get("op")
        if op not in {"lte", "lt", "gte", "gt", "between", "around"}:
            raise ValueError("price_semantics_unsupported")
        try:
            low, high = (
                None if price.get(key) is None else Decimal(str(price[key]))
                for key in ("low", "high")
            )
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError("invalid_price_number") from exc
        if any(v is not None and (not v.is_finite() or v < 0) for v in (low, high)):
            raise ValueError("invalid_price_number")
        if op in {"lte", "lt"} and (high is None or low is not None):
            raise ValueError("invalid_upper_budget")
        if op in {"gte", "gt"} and (low is None or high is not None):
            raise ValueError("invalid_lower_budget")
        if op in {"between", "around"} and (
            low is None
            or high is None
            or low > high
            or (op == "between" and low == high)
            or (op == "around" and low != high)
        ):
            raise ValueError("invalid_price_interval")
        if op == "around":
            low, high = None, high * Decimal("1.1")
        bounds.append((op, low, high))
    return bounds


def budget_upper(contract):
    limits = [high for _, _, high in budget_limits(contract) if high is not None]
    return float(min(limits)) if limits else None


def budget_reasons(contract, actual):
    bounds = budget_limits(contract)
    try:
        actual = Decimal(str(actual))
    except InvalidOperation as exc:
        raise ValueError("variant_price_unverifiable") from exc
    if not actual.is_finite() or actual < 0:
        raise ValueError("variant_price_unverifiable")
    reasons = []
    for op, low, high in bounds:
        if high is not None and (actual > high or op == "lt" and actual == high):
            reasons.append("gold_over_budget")
        if low is not None and (actual < low or op == "gt" and actual == low):
            reasons.append("gold_below_price_minimum")
    return sorted(set(reasons))


@lru_cache(maxsize=8)
def load_budget_index(path):
    index = {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            contract = row["budget"]
            budget_limits(contract)
            key = (row["asin"], contract["query_sha256"])
            if key in index and index[key] != contract:
                raise ValueError("conflicting_budget_annotations")
            index[key] = contract
    return index
