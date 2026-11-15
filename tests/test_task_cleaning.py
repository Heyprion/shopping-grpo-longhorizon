"""Evidence, price-boundary, isolation and fail-closed screening tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from scripts.clean_tasks import JSONModel, choose_sample, controlled_cases, process
from shopping_grpo.collection.task_cleaning import (
    check_prices,
    check_structure,
    evidence_view,
    ground_extraction,
    ground_verification,
    validate_extraction,
    validate_verification,
)


def extraction(query="保温杯", *, hard=True):
    return {
        "requirements": [{"kind": "category", "text": "保温杯", "quote": query, "hard": hard}],
        "prices": [],
        "uncertain": False,
        "budget_state": "not_mentioned",
    }


def price(op="lte", low=None, high=100, scope="purchase"):
    return {"op": op, "low": low, "high": high, "scope": scope, "quote": "预算"}


class TaskCleaningTests(unittest.TestCase):
    def setUp(self):
        self.product = deepcopy(controlled_cases()[0]["product"])

    def test_raw_checks_do_not_accept_runtime_price_fallbacks(self):
        self.assertEqual(check_structure(self.product), [])
        self.product["pricing"] = []
        self.product["customization_options"]["规格"][0]["price"] = float("nan")
        self.assertIn("invalid_pricing", check_structure(self.product))
        self.assertIn("invalid_option_price", check_structure(self.product))

    def test_gold_reference_and_same_axis_conflicts(self):
        self.product["instructions"][0]["instruction_options"] = ["不存在"]
        self.assertIn("gold_option_missing_or_ambiguous", check_structure(self.product))
        self.product["customization_options"]["规格"].append(
            {"value": "红色", "price": 80, "is_available": True}
        )
        self.product["instructions"][0]["instruction_options"] = ["黑色500ml单只装", "红色"]
        self.assertIn("multiple_gold_values_on_one_axis", check_structure(self.product))

    def test_extractor_must_cite_query_and_include_category(self):
        result = extraction()
        self.assertEqual(validate_extraction(result, "买个保温杯"), result)
        with self.assertRaisesRegex(ValueError, "fabricated_query_evidence"):
            validate_extraction(result, "买个水壶")
        result["requirements"][0]["kind"] = "spec"
        with self.assertRaisesRegex(ValueError, "missing_required_category"):
            validate_extraction(result, "买个保温杯")

    def test_unsupported_and_uncertain_extractions_cannot_pass(self):
        result = extraction()
        result["uncertain"] = True
        with self.assertRaisesRegex(ValueError, "extraction_uncertain"):
            validate_extraction(result, "保温杯")
        self.assertIn("price_semantics_unsupported", check_prices([price(scope="unit")], 80, 100))
        self.assertIn("price_semantics_unsupported", check_prices([price(op="unknown")], 80, 100))

    def test_budget_boundaries_lower_bounds_and_runtime_disagreement(self):
        self.assertEqual(check_prices([price()], 100, 100), [])
        self.assertIn("gold_over_budget", check_prices([price(op="lt")], 100, 100))
        self.assertIn("gold_over_budget", check_prices([price()], 100.01, 100))
        self.assertEqual(check_prices([price(op="around", low=100)], 110, 110), [])
        self.assertIn(
            "gold_below_price_minimum", check_prices([price(op="gt", low=90, high=None)], 90, None)
        )
        self.assertIn("runtime_budget_missing", check_prices([price()], 80, None))
        self.assertIn("runtime_budget_mismatch", check_prices([price()], 80, 200))
        self.assertIn("runtime_budget_not_in_extraction", check_prices([], 80, 100))
        self.assertIn(
            "runtime_budget_not_in_extraction",
            check_prices([price(op="gte", low=30, high=None)], 30, 30),
        )

    def test_verifier_cannot_omit_requirements_or_invent_product_evidence(self):
        requirements = extraction()["requirements"]
        facts = evidence_view(self.product, {"规格": "黑色500ml单只装"})
        result = {"checks": [], "option_conflicts": []}
        with self.assertRaisesRegex(ValueError, "incomplete_requirement_checks"):
            validate_verification(result, requirements, facts)
        result["checks"] = [{"index": 0, "status": "pass", "source": "title", "quote": "红色"}]
        with self.assertRaisesRegex(ValueError, "fabricated_product_evidence"):
            validate_verification(result, requirements, facts)
        result["checks"][0]["quote"] = "保温杯"
        self.assertEqual(validate_verification(result, requirements, facts), [])
        result["checks"][0].update(status="unknown", quote="")
        self.assertIn("requirement_unknown", validate_verification(result, requirements, facts))

    def test_structure_failure_never_calls_model(self):
        class NoModel:
            def call(self, *args):
                raise AssertionError("invalid structure must not spend inference")

        row = next(r for r in controlled_cases() if r["case_id"] == "missing_instruction")
        result, calls = process(row, NoModel())
        self.assertFalse(result["passed"])
        self.assertFalse(result["stage1_pass"])
        self.assertEqual(calls, [])

    def test_query_only_extraction_and_single_selected_variant_evidence(self):
        seen = []

        class FakeModel:
            def call(self, stage, value):
                seen.append((stage, value))
                if stage == "extract":
                    result = extraction()
                    result["requirements"][0].pop("quote")
                    result["requirements"][0]["source"] = 0
                    p = price()
                    p.pop("quote")
                    result["prices"] = [{**p, "source": 1}]
                    result["budget_state"] = "parsed"
                else:
                    result = {
                        "checks": [{"index": 0, "status": "pass", "evidence": 1}],
                        "option_conflict": False,
                        "axis_sizes": {axis: None for axis in value["selected_options"]},
                    }
                return {"key": stage, "result": result}

        row = {"case_id": "unit", "group": "control", "product": self.product}
        result, _ = process(row, FakeModel())
        self.assertTrue(result["passed"], result["reasons"])
        self.assertEqual(set(seen[0][1]), {"clauses"})
        self.assertNotIn("instruction", seen[1][1])
        self.assertNotIn("customization_options", seen[1][1])

    def test_evidence_ids_are_validated_and_resolved_by_code(self):
        raw = {
            "requirements": [{"kind": "category", "text": "保温杯", "source": 0, "hard": True}],
            "prices": [],
            "uncertain": False,
            "budget_state": "not_mentioned",
        }
        grounded = ground_extraction(raw, "买一只保温杯。")
        self.assertEqual(grounded["requirements"][0]["quote"], "买一只保温杯")
        raw["requirements"][0]["source"] = 99
        with self.assertRaisesRegex(ValueError, "invalid_query_evidence_id"):
            ground_extraction(raw, "买一只保温杯。")
        facts = evidence_view(self.product, {"规格": "黑色500ml单只装"})
        verdict = {
            "checks": [{"index": 0, "status": "pass", "evidence": -1}],
            "option_conflict": False,
            "axis_sizes": {"规格": None},
        }
        with self.assertRaisesRegex(ValueError, "missing_product_evidence"):
            ground_verification(
                verdict, grounded["requirements"], facts, {"规格": "黑色500ml单只装"}
            )
        verdict["checks"][0]["evidence"] = 1
        result, reasons = ground_verification(
            verdict, grounded["requirements"], facts, {"规格": "黑色500ml单只装"}
        )
        self.assertEqual(reasons, [])
        self.assertEqual(result["checks"][0]["quote"], self.product["title"])

    def test_cross_axis_size_conflict_is_checked_in_code_even_when_model_says_pass(self):
        selected = {"颜色分类": "S-2只装", "尺码": "L"}
        facts = evidence_view(self.product, selected)
        requirements = extraction()["requirements"]
        verdict = {
            "checks": [{"index": 0, "status": "pass", "evidence": 1}],
            "option_conflict": False,
            "axis_sizes": {"颜色分类": "S", "尺码": "L"},
        }
        _, reasons = ground_verification(verdict, requirements, facts, selected)
        self.assertIn("gold_options_internally_conflict", reasons)
        verdict["axis_sizes"]["颜色分类"] = "L"
        _, reasons = ground_verification(verdict, requirements, facts, selected)
        self.assertEqual(reasons, [])
        selected = {"套餐": "XXL-2件装", "尺码": "2XL"}
        facts = evidence_view(self.product, selected)
        verdict["axis_sizes"] = {"套餐": "XXL", "尺码": "2XL"}
        _, reasons = ground_verification(verdict, requirements, facts, selected)
        self.assertEqual(reasons, [])

    def test_sampling_protects_held_out_query_duplicates_and_preserves_ids(self):
        products = [deepcopy(self.product) for _ in range(4)]
        products[2]["instructions"][0]["instruction"] = "other"
        products[3]["instructions"][0]["instruction"] = "different"
        rows, pool = choose_sample(products, {0}, 2, 42)
        self.assertEqual(pool, [2, 3])
        self.assertEqual([r["task_id"] for r in rows], [2, 3])
        self.assertIs(rows[0]["product"], products[2])

    def test_truncated_model_output_is_retried_once_and_never_accepted(self):
        args = Namespace(
            model="test", revision="test", max_tokens=10, endpoint="http://localhost/v1", timeout=1
        )
        response = {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}
        with tempfile.TemporaryDirectory() as directory:
            model = JSONModel(args, Path(directory))
            with patch("scripts.clean_tasks.urlopen") as request:
                request.return_value.__enter__.return_value.read.return_value = json.dumps(
                    response
                ).encode()
                result = model.call("extract", {"instruction": "保温杯"})
            self.assertEqual(request.call_count, 2)
            self.assertEqual(result["error"], "model_call_failed")

    def test_canonical_budget_distinguishes_unmentioned_and_unresolved(self):
        from web_agent_site.engine.reward import evaluate_purchase
        from web_agent_site.engine.task_budget import budget_contract, budget_reasons

        from scripts.clean_tasks import runtime_facts

        query = "价格不超过一百元"
        canonical = budget_contract(query, [price()], "parsed")
        self.assertEqual(budget_reasons(canonical, 100), [])
        self.assertEqual(budget_reasons(canonical, 101), ["gold_over_budget"])
        self.assertEqual(
            runtime_facts(self.product, canonical)["runtime_reward_type"], "gold_purchase"
        )
        self.assertEqual(budget_reasons(budget_contract("保温杯", [], "not_mentioned"), 900), [])
        with self.assertRaisesRegex(ValueError, "budget_unresolved"):
            budget_reasons(budget_contract(query, [], "unresolved"), 80)
        result = evaluate_purchase(
            self.product,
            {"asin": "control", "task_budget": budget_contract(query, [], "unresolved")},
            selected_options={},
        )
        self.assertFalse(result.reward_valid)
        result = extraction()
        result["budget_state"] = "parsed"
        with self.assertRaisesRegex(ValueError, "budget_state_inconsistent"):
            validate_extraction(result, "保温杯")

    def test_environment_joins_canonical_budget_by_product_and_instruction(self):
        from web_agent_site.engine.goal import get_existed_goals
        from web_agent_site.engine.task_budget import budget_contract

        product = deepcopy(self.product)
        product.update(user_persona={}, reason_key=None, query="保温杯")
        product["instructions"][0]["instruction_simple"] = "保温杯"
        query = product["instructions"][0]["instruction"]
        other = deepcopy(product)
        other["asin"] = "other"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budgets.jsonl"
            path.write_text(
                json.dumps(
                    {"asin": "control", "budget": budget_contract(query, [price()], "parsed")}
                )
            )
            with patch.dict("os.environ", {"SHOPPING_TASK_BUDGETS": str(path)}):
                goals = get_existed_goals([product, other], {})
                self.assertEqual(len(goals), 2)
                self.assertEqual(goals[0]["price_upper"], 100)
                self.assertEqual(goals[1]["task_budget"]["state"], "unresolved")
                product["instructions"][0]["instruction"] += "改买两只"
                changed = get_existed_goals([product], {})
                self.assertEqual(changed[0]["task_budget"]["state"], "unresolved")

    def test_runtime_canonical_price_obeys_all_boundaries(self):
        from web_agent_site.engine.reward import _price_component
        from web_agent_site.engine.task_budget import budget_contract

        cases = [
            (price(op="lt"), 100, 0),
            (price(op="lte"), 100, 1),
            (price(op="gt", low=80, high=None), 80, 0),
            (price(op="between", low=80), 80, 1),
            (price(op="around", low=100), 110, 1),
            (price(op="around", low=100), 110.01, 0),
        ]
        for constraint, actual, expected in cases:
            with self.subTest(constraint=constraint, actual=actual):
                score, evidence = _price_component(
                    {"status": "pass", "price": actual},
                    {"task_budget": budget_contract("预算", [constraint], "parsed")},
                )
                self.assertEqual(score, expected)
                self.assertEqual(evidence["status"], "pass" if expected else "fail")


if __name__ == "__main__":
    unittest.main()
