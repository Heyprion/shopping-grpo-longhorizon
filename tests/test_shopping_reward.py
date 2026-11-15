"""The GRPO adapter must optimize exactly the public strict product."""

import unittest

from shopping_grpo.environment.reward import REWARD_VERSION
from shopping_grpo.training.grpo.adapter.runtime import (
    make_runtime_state,
    record_action_attempt,
    reward_breakdown,
    terminal_reward,
    validate_reward,
    validate_reward_components,
)


def terminal_state(components=None, kind="gold_purchase"):
    components = components or {"r_type": 1.0, "r_att": 1.0, "r_option": 1.0, "r_price": 1.0}
    state = make_runtime_state(1, 35)
    score = (
        components["r_type"] * components["r_att"] * components["r_option"] * components["r_price"]
    )
    state.update(
        done=True,
        terminal_result={"done": True, "over": True},
        final_reward=score,
        reward_version=REWARD_VERSION,
        reward_type=kind,
        reward_valid=True,
        reward_detail={"reward_components": components, "evidence_coverage": 1.0},
    )
    return state


class ShoppingRewardTest(unittest.TestCase):
    def test_full_reward_has_no_efficiency_bonus(self):
        state = terminal_state()
        state["steps"] = [{}] * 8
        result = reward_breakdown(state)
        self.assertEqual(result["total"], 1.0)
        self.assertEqual(result["strict"], 1.0)
        self.assertEqual(result["efficiency"], 0.0)
        self.assertEqual(terminal_reward(state), 1.0)

    def test_fractional_product_is_the_actual_learning_signal(self):
        state = terminal_state(
            {"r_type": 1.0, "r_att": 0.5, "r_option": 0.5, "r_price": 1.0},
            "partial_alternative_purchase",
        )
        result = reward_breakdown(state)
        self.assertEqual(result["strict"], 0.25)
        self.assertEqual(result["total"], 0.25)
        self.assertEqual(result["full"], 0.0)
        self.assertFalse(result["sampling_invalid"])

    def test_partial_native_mismatch_is_rejected(self):
        state = terminal_state(
            {"r_type": 1.0, "r_att": 1.0, "r_option": 0.5, "r_price": 1.0},
            "partial_alternative_purchase",
        )
        state["final_reward"] = 0.6
        result = reward_breakdown(state)
        self.assertTrue(result["infrastructure_invalid"])
        self.assertEqual(result["total"], 0.0)

    def test_unfinished_episode_has_zero_reward_without_artificial_penalty(self):
        state = make_runtime_state(1, 35)
        state["termination_reason"] = "assistant_finished_without_environment_done"
        result = reward_breakdown(state)
        self.assertEqual(result["total"], 0.0)
        self.assertEqual(result["penalty_unfinished"], 0.0)

    def test_old_reward_version_is_not_relabelled_as_strict(self):
        state = terminal_state()
        state["reward_version"] = "shopsimulator-reward-v3"
        self.assertTrue(reward_breakdown(state)["sampling_invalid"])

    def test_components_must_be_complete_finite_and_bounded(self):
        for values in [
            {"r_type": 1},
            {"r_type": 1, "r_att": 1, "r_option": float("nan"), "r_price": 1},
            {"r_type": 1, "r_att": 1, "r_option": 1.1, "r_price": 1},
        ]:
            with self.assertRaises(ValueError):
                validate_reward_components(values)

    def test_public_terminal_product_is_validated(self):
        detail = {
            "reward_version": REWARD_VERSION,
            "reward_type": "partial_alternative_purchase",
            "reward_valid": True,
            "termination_reason": "partial_alternative_purchase",
            "reward_components": {"r_type": 1, "r_att": 1, "r_option": 0.5, "r_price": 1},
            "terminal_utility": 0.5,
            "purchase_success": False,
            "sampling_invalid": False,
        }
        self.assertEqual(validate_reward(detail)["strict_score"], 0.5)
        detail["terminal_utility"] = 0.6
        with self.assertRaisesRegex(ValueError, "component product"):
            validate_reward(detail)

    def test_repeat_attempt_diagnostics_do_not_change_reward(self):
        state = make_runtime_state(1, 35)
        record_action_attempt(state, "search_products", {"query": "mug"}, "page")
        record_action_attempt(state, "search_products", {"query": "mug"}, "page")
        self.assertEqual(state["repeat_action_count"], 1)
        self.assertEqual(reward_breakdown(state)["total"], 0.0)


if __name__ == "__main__":
    unittest.main()
