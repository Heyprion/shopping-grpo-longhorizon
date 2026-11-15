"""Fail-closed, two-stage task screening; never changes the frozen catalog.

Language interpretation belongs to the extraction/verification model. This
module validates its evidence and owns numeric price and runtime checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from decimal import Decimal

POLICY_VERSION = "shopping-task-cleaning-v2"


class CleaningContractError(ValueError):
    """A model response cannot satisfy the evidence contract."""


def fingerprint(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def normalized(value):
    return "".join(unicodedata.normalize("NFKC", str(value)).casefold().split()).replace("/", "|")


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def string_list(value, *, nonempty=False):
    return (
        isinstance(value, list)
        and (bool(value) or not nonempty)
        and all(isinstance(x, str) and bool(x.strip()) for x in value)
    )


def check_structure(product):
    """Check the raw record, before any runtime fallback/normalization."""
    reasons = []
    if not isinstance(product, dict):
        return ["product_not_object"]
    for key in ("asin", "title", "category", "shop_name"):
        if not isinstance(product.get(key), str) or not product[key].strip():
            reasons.append(f"invalid_{key}")
    asin = product.get("asin")
    if isinstance(asin, str) and (asin == "nan" or len(asin) > 20):
        reasons.append("invalid_product_id")
    if not isinstance(product.get("full_description"), str):
        reasons.append("invalid_full_description")
    for key in ("attribute", "images"):
        if not string_list(product.get(key), nonempty=True):
            reasons.append(f"invalid_{key}")
    pricing = product.get("pricing")
    if (
        not isinstance(pricing, list)
        or len(pricing) not in (1, 2)
        or not all(finite_number(x) for x in pricing)
        or (len(pricing) == 2 and pricing[0] > pricing[1])
    ):
        reasons.append("invalid_pricing")
    axes = product.get("customization_options")
    option_values = {}
    if not isinstance(axes, dict):
        reasons.append("invalid_customization_options")
    else:
        seen_axes = set()
        for axis, entries in axes.items():
            if not isinstance(axis, str) or not axis.strip() or normalized(axis) in seen_axes:
                reasons.append("invalid_or_duplicate_option_axis")
            seen_axes.add(normalized(axis))
            if not isinstance(entries, list) or not entries:
                reasons.append("invalid_option_entries")
                continue
            seen = set()
            for entry in entries:
                if not isinstance(entry, dict):
                    reasons.append("option_not_object")
                    continue
                value = entry.get("value")
                if not isinstance(value, str) or not value.strip():
                    reasons.append("invalid_option_value")
                    continue
                key = normalized(value)
                if key in seen:
                    reasons.append("duplicate_option_value")
                seen.add(key)
                option_values.setdefault(key, []).append(axis)
                if not finite_number(entry.get("price")):
                    reasons.append("invalid_option_price")
                if entry.get("is_available") is not True:
                    reasons.append("option_availability_unverifiable")
    instructions = product.get("instructions")
    if not isinstance(instructions, list) or len(instructions) != 1:
        return sorted(set(reasons + ["expected_one_instruction"]))
    instruction = instructions[0]
    if not isinstance(instruction, dict):
        return sorted(set(reasons + ["instruction_not_object"]))
    if instruction.get("asin") != asin:
        reasons.append("instruction_product_id_mismatch")
    if (
        not isinstance(instruction.get("instruction"), str)
        or not instruction["instruction"].strip()
    ):
        reasons.append("invalid_instruction")
    if not string_list(instruction.get("attributes"), nonempty=True):
        reasons.append("invalid_instruction_attributes")
    gold = instruction.get("instruction_options")
    if not string_list(gold):
        reasons.append("invalid_gold_options")
    else:
        occupied = set()
        for value in gold:
            matches = option_values.get(normalized(value), [])
            if len(matches) != 1:
                reasons.append("gold_option_missing_or_ambiguous")
            elif matches[0] in occupied:
                reasons.append("multiple_gold_values_on_one_axis")
            else:
                occupied.add(matches[0])
    return sorted(set(reasons))


def validate_extraction(result, query):
    """Reject omitted schema/evidence and unsupported or ambiguous extraction."""
    if not isinstance(result, dict) or set(result) != {
        "requirements",
        "prices",
        "uncertain",
        "budget_state",
    }:
        raise ValueError("invalid_extraction_schema")
    if result["uncertain"] is not False:
        raise ValueError("extraction_uncertain")
    requirements = result["requirements"]
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("missing_requirements")
    categories = 0
    for requirement in requirements:
        if not isinstance(requirement, dict) or set(requirement) != {
            "kind",
            "text",
            "quote",
            "hard",
        }:
            raise ValueError("invalid_requirement_schema")
        if requirement["kind"] not in {
            "category",
            "brand",
            "model",
            "function",
            "spec",
            "quantity",
            "preference",
        } or not isinstance(requirement["hard"], bool):
            raise ValueError("invalid_requirement_type")
        if not isinstance(requirement["text"], str) or not requirement["text"].strip():
            raise ValueError("empty_requirement")
        if not isinstance(requirement["quote"], str) or not requirement["quote"]:
            raise ValueError("missing_query_evidence")
        if requirement["quote"] not in query:
            raise ValueError("fabricated_query_evidence")
        categories += requirement["kind"] == "category" and requirement["hard"]
    if not categories:
        raise ValueError("missing_required_category")
    if not isinstance(result["prices"], list):
        raise CleaningContractError("invalid_prices")
    if result["budget_state"] not in {"parsed", "not_mentioned", "unresolved"}:
        raise CleaningContractError("invalid_budget_state")
    if result["budget_state"] == "unresolved":
        raise CleaningContractError("budget_unresolved")
    if (result["budget_state"] == "parsed") != bool(result["prices"]):
        raise CleaningContractError("budget_state_inconsistent")
    for price in result["prices"]:
        if not isinstance(price, dict) or set(price) != {"quote", "op", "low", "high", "scope"}:
            raise ValueError("invalid_price_schema")
        if not isinstance(price["quote"], str) or not price["quote"] or price["quote"] not in query:
            raise ValueError("fabricated_price_evidence")
        if price["op"] not in {"lte", "lt", "gte", "gt", "between", "around", "unknown"}:
            raise ValueError("invalid_price_operator")
        if price["scope"] not in {"purchase", "unit", "component", "unknown"}:
            raise ValueError("invalid_price_scope")
        for key in ("low", "high"):
            if price[key] is not None and not finite_number(price[key]):
                raise ValueError("invalid_price_number")
        op = price["op"]
        if op in {"lte", "lt"} and (price["high"] is None or price["low"] is not None):
            raise ValueError("invalid_upper_budget")
        if op in {"gte", "gt"} and (price["low"] is None or price["high"] is not None):
            raise ValueError("invalid_lower_budget")
        if op in {"between", "around"} and (
            price["low"] is None or price["high"] is None or price["low"] > price["high"]
        ):
            raise ValueError("invalid_price_interval")
        if op == "between" and price["low"] == price["high"]:
            raise ValueError("degenerate_price_interval")
        if op == "around" and price["low"] != price["high"]:
            raise ValueError("approximate_price_requires_center")
    return result


def query_clauses(query):
    """Number evidence spans by punctuation, without extracting any semantics."""
    clauses = []
    start = 0
    for index, char in enumerate(query):
        if char in "，。；,;！？!?\n":
            if query[start:index].strip():
                clauses.append(query[start:index].strip())
            start = index + 1
    if query[start:].strip():
        clauses.append(query[start:].strip())
    return clauses


def ground_extraction(result, query):
    """Code supplies verbatim evidence; the model only selects span IDs."""
    if not isinstance(result, dict) or set(result) != {
        "requirements",
        "prices",
        "uncertain",
        "budget_state",
    }:
        raise ValueError("invalid_extraction_schema")
    clauses = query_clauses(query)
    grounded = {
        "requirements": [],
        "prices": [],
        "uncertain": result["uncertain"],
        "budget_state": result["budget_state"],
    }
    for key in ("requirements", "prices"):
        if not isinstance(result[key], list):
            raise CleaningContractError("invalid_extraction_items")
        for item in result[key]:
            if not isinstance(item, dict):
                raise CleaningContractError("invalid_extraction_item")
            item = dict(item)
            source = item.pop("source", None)
            if type(source) is not int or not 0 <= source < len(clauses):
                raise ValueError("invalid_query_evidence_id")
            item["quote"] = clauses[source]
            if key == "requirements" and item.get("kind") == "price":
                if not result["prices"]:
                    raise ValueError("price_missing_from_prices")
                continue
            grounded[key].append(item)
    return validate_extraction(grounded, query)


def product_evidence(facts):
    return [{"id": i, "source": key, "text": value} for i, (key, value) in enumerate(facts.items())]


def ground_verification(result, requirements, facts, selected_options):
    if not isinstance(result, dict) or set(result) != {"checks", "option_conflict", "axis_sizes"}:
        raise ValueError("invalid_verification_schema")
    if not isinstance(result["checks"], list) or type(result["option_conflict"]) is not bool:
        raise ValueError("invalid_verification_items")
    evidence = product_evidence(facts)
    sizes = result["axis_sizes"]
    if not isinstance(sizes, dict) or set(sizes) != set(selected_options):
        raise ValueError("incomplete_axis_size_checks")
    if any(
        value is not None and (not isinstance(value, str) or not value.strip())
        for value in sizes.values()
    ):
        raise ValueError("invalid_axis_size")
    aliases = {"xxs": "2xs", "xxl": "2xl", "xxxl": "3xl", "xxxxl": "4xl"}
    size_conflict = (
        len({aliases.get(normalized(v), normalized(v)) for v in sizes.values() if v is not None})
        > 1
    )
    grounded = {"checks": [], "option_conflicts": []}
    for check in result["checks"]:
        if not isinstance(check, dict) or set(check) != {"index", "status", "evidence"}:
            raise ValueError("invalid_check_schema")
        index = check["evidence"]
        if type(index) is not int or not -1 <= index < len(evidence):
            raise ValueError("invalid_product_evidence_id")
        source = evidence[index] if index >= 0 else {"source": "category", "text": ""}
        grounded["checks"].append(
            {
                "index": check["index"],
                "status": check["status"],
                "source": source["source"],
                "quote": source["text"],
            }
        )
    if result["option_conflict"] or size_conflict:
        grounded["option_conflicts"] = [facts["selected_options"]]
    grounded["axis_sizes"] = sizes
    reasons = validate_verification(
        {k: v for k, v in grounded.items() if k != "axis_sizes"}, requirements, facts
    )
    return grounded, reasons


def check_prices(prices, actual, runtime_upper):
    """Exact money comparisons. Around uses the existing v2.1 +10% policy."""
    reasons = []
    upper_limits = []
    if not finite_number(actual):
        return ["variant_price_unverifiable"]
    actual = Decimal(str(actual))
    for price in prices:
        if price["scope"] != "purchase" or price["op"] == "unknown":
            reasons.append("price_semantics_unsupported")
            continue
        low = None if price["low"] is None else Decimal(str(price["low"]))
        high = None if price["high"] is None else Decimal(str(price["high"]))
        op = price["op"]
        if op == "around":
            high *= Decimal("1.1")
            low = None
        if high is not None:
            upper_limits.append(high)
            if actual > high or (op == "lt" and actual == high):
                reasons.append("gold_over_budget")
        if low is not None and (actual < low or (op == "gt" and actual == low)):
            reasons.append("gold_below_price_minimum")
    if upper_limits:
        if runtime_upper is None:
            reasons.append("runtime_budget_missing")
        elif not finite_number(runtime_upper) or abs(
            Decimal(str(runtime_upper)) - min(upper_limits)
        ) > Decimal("0.000001"):
            reasons.append("runtime_budget_mismatch")
    elif runtime_upper is not None:
        reasons.append("runtime_budget_not_in_extraction")
    return sorted(set(reasons))


def evidence_view(product, selected_options):
    """Expose the selected variant, never the other variants as positive evidence."""
    return {
        "category": product["category"],
        "title": product["title"],
        "shop_name": product["shop_name"],
        "attributes": "\n".join(product["attribute"]),
        "description": product["full_description"],
        "selected_options": "\n".join(f"{k}: {v}" for k, v in selected_options.items()),
    }


def validate_verification(result, requirements, facts):
    """Require one evidence-backed verdict for every independently extracted need."""
    if not isinstance(result, dict) or set(result) != {"checks", "option_conflicts"}:
        raise ValueError("invalid_verification_schema")
    checks = result["checks"]
    if not isinstance(checks, list) or len(checks) != len(requirements):
        raise ValueError("incomplete_requirement_checks")
    seen = set()
    reasons = []
    for check in checks:
        if not isinstance(check, dict) or set(check) != {"index", "status", "source", "quote"}:
            raise ValueError("invalid_check_schema")
        index = check["index"]
        if type(index) is not int or not 0 <= index < len(requirements) or index in seen:
            raise ValueError("invalid_check_index")
        seen.add(index)
        if check["status"] not in {"pass", "fail", "unknown"}:
            raise ValueError("invalid_check_status")
        source, quote = check["source"], check["quote"]
        if source not in facts or not isinstance(quote, str):
            raise ValueError("invalid_product_evidence")
        if quote and quote not in facts[source]:
            raise ValueError("fabricated_product_evidence")
        if check["status"] == "pass" and not quote:
            raise ValueError("missing_product_evidence")
        if requirements[index]["hard"] and check["status"] != "pass":
            reasons.append(f"requirement_{check['status']}")
    conflicts = result["option_conflicts"]
    if not isinstance(conflicts, list) or not all(
        isinstance(x, str) and x and x in facts["selected_options"] for x in conflicts
    ):
        raise ValueError("invalid_option_conflict_evidence")
    if conflicts:
        reasons.append("gold_options_internally_conflict")
    return sorted(set(reasons))
