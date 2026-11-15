"""Paper Eq. (4), its soft components, and invalid-evidence boundaries."""

import math
import unittest
from unittest.mock import patch

from web_agent_site.engine.reward import (
    REWARD_VERSION,
    evaluate_abstain,
    evaluate_candidate_eligibility,
    evaluate_purchase,
    fixed_termination,
)


def product(asin="111111111111"):
    return {
        "asin": asin,
        "title": "保温防水杯",
        "category": "家居›杯具›保温杯",
        "attribute": ["保温", "防水"],
        "pricing": [80],
        "customization_options": {
            "颜色": [{"value": "白色", "price": 80}, {"value": "黑色", "price": 80}],
            "规格": [{"value": "大杯", "price": 80}, {"value": "小杯", "price": 80}],
        },
    }


def goal():
    return {
        "asin": "111111111111",
        "name": "保温防水杯",
        "category": "家居›杯具›保温杯",
        "query": "",
        "price_upper": 100,
        "attributes": ["保温", "防水"],
        "goal_options": ["白色", "大杯"],
        "required_options_by_key": {"color": {"value": "白色"}, "specification": {"value": "大杯"}},
    }


class StrictRewardTest(unittest.TestCase):
    def test_full_gold_and_full_alternative_both_receive_one(self):
        for asin, kind in [
            ("111111111111", "gold_purchase"),
            ("222222222222", "valid_alternative_purchase"),
        ]:
            result = evaluate_purchase(
                product(asin), goal(), selected_options={"颜色": "白色", "规格": "大杯"}
            )
            self.assertEqual(result.reward, 1.0)
            self.assertEqual(result.reward_type, kind)
            self.assertEqual(result.to_dict()["reward_version"], REWARD_VERSION)
            self.assertEqual(result.reward, math.prod(result.reward_components.values()))

    def test_fractional_attributes_and_options_are_multiplied(self):
        candidate = product()
        candidate.update(attribute=["保温"], title="保温杯")
        result = evaluate_purchase(
            candidate, goal(), selected_options={"颜色": "黑色", "规格": "大杯"}
        )
        self.assertEqual(
            result.reward_components, {"r_type": 1, "r_att": 0.5, "r_option": 0.5, "r_price": 1}
        )
        self.assertEqual(result.reward, 0.25)
        self.assertEqual(result.reward_type, "partial_alternative_purchase")

    def test_any_zero_factor_zeroes_the_reward(self):
        task = goal()
        task["price_upper"] = 79
        result = evaluate_purchase(
            product(), task, selected_options={"颜色": "白色", "规格": "大杯"}
        )
        self.assertEqual(result.reward_components["r_price"], 0)
        self.assertEqual(result.reward, 0)
        self.assertTrue(result.reward_valid)

    def test_category_matches_are_soft_and_empty_query_is_not_a_match(self):
        candidate = product()
        candidate.update(category="数码›电脑›笔记本", title="other", query="")
        task = goal()
        task["name"] = "target"
        with patch(
            "web_agent_site.engine.reward._title_keywords",
            side_effect=[{"a", "b", "c", "d", "e", "f"}, {"a"}],
        ):
            result = evaluate_purchase(
                candidate, task, selected_options={"颜色": "白色", "规格": "大杯"}
            )
        self.assertEqual(result.reward_components["r_type"], 0.5)
        self.assertEqual(result.reward, 0.5)

    def test_no_budget_does_not_invent_a_limit_but_unknown_price_is_invalid(self):
        task = goal()
        task["price_upper"] = None
        result = evaluate_purchase(
            product(), task, selected_options={"颜色": "白色", "规格": "大杯"}, price=9999
        )
        self.assertEqual(result.reward_components["r_price"], 1)
        invalid = evaluate_purchase(
            product(),
            task,
            selected_options={"颜色": "白色", "规格": "大杯"},
            price_resolution={"status": "unverifiable", "price": None},
        )
        self.assertFalse(invalid.reward_valid)
        self.assertEqual(invalid.reward, 0)

    def test_empty_requirements_use_vacuous_full_coverage(self):
        task = goal()
        task.update(attributes=[], goal_options=[])
        result = evaluate_purchase(
            product(), task, selected_options={"颜色": "白色", "规格": "大杯"}
        )
        self.assertEqual(result.reward_components["r_att"], 1)
        self.assertEqual(result.reward_components["r_option"], 1)

    def test_incomplete_variant_is_not_a_verified_purchase(self):
        result = evaluate_purchase(product(), goal(), selected_options={"颜色": "白色"})
        self.assertEqual(result.reward, 0)
        self.assertFalse(result.reward_valid)

    def test_termination_reasons_remain_distinct_but_all_have_zero_utility(self):
        for reason in ("max_steps", "repeat_loop"):
            self.assertEqual(fixed_termination(reason).reward, 0)
        early = evaluate_abstain(effective_result_sets=0, opened_candidates=0)
        graceful = evaluate_abstain(
            effective_result_sets=2, opened_candidates=2, known_acceptable_candidates=0
        )
        self.assertEqual((early.reward, graceful.reward), (0, 0))
        self.assertEqual(
            (early.reward_type, graceful.reward_type), ("early_abstain", "graceful_stop")
        )

    def test_candidate_acceptability_requires_all_four_factors(self):
        self.assertTrue(evaluate_candidate_eligibility(product(), goal())["known_acceptable"])
        candidate = product()
        candidate.update(attribute=["保温"], title="保温杯")
        self.assertFalse(evaluate_candidate_eligibility(candidate, goal())["known_acceptable"])

    def test_terminal_reward_overrides_cannot_reintroduce_shaping(self):
        with self.assertRaisesRegex(ValueError, "bonuses or penalties"):
            fixed_termination("max_steps", {"max_steps": -0.5})


if __name__ == "__main__":
    unittest.main()
