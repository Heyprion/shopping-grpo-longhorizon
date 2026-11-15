import unittest

from scripts.check_grpo_runtime import ppo_gradient_accumulation_steps
from shopping_grpo.training.grpo.adapter.runtime import apply_reward_length_shaping
from shopping_grpo.training.grpo.dynamic_sampling import (
    aggregate_shopping_metrics,
    extract_shopping_group_signals,
)


class GrpoAblationTest(unittest.TestCase):
    def test_ppo_mini_and_micro_batches_define_gradient_accumulation(self):
        self.assertEqual(ppo_gradient_accumulation_steps(4, 2), 2)
        with self.assertRaisesRegex(ValueError, "divisible"):
            ppo_gradient_accumulation_steps(3, 2)

    def test_disabled_length_shaping_is_a_no_op(self):
        reward = {"total": 0.8, "terminal_utility": 0.8, "penalty_overlong": 0.0, "sampling_invalid": False}
        state = {"steps": [{}] * 25, "max_steps": 35, "termination_reason": "gold_purchase"}

        shaped = apply_reward_length_shaping(reward, state, enabled=False)

        self.assertEqual(shaped, reward)
        self.assertIsNot(shaped, reward)

    def test_strict_reward_refuses_length_penalties(self):
        with self.assertRaisesRegex(ValueError, "forbids length shaping"):
            apply_reward_length_shaping({"total": .8}, {"steps": [{}] * 30}, enabled=True)

    def test_max_steps_is_a_valid_zero_reward_without_shaping(self):
        info = {"overlong": False, "infrastructure_invalid": False, "reward_unverifiable": False,
                "reward": {"terminal_utility": 0., "purchase_success": False, "sampling_invalid": False}}
        utilities, _, invalid, reasons = extract_shopping_group_signals([info])
        self.assertEqual(utilities, [0.])
        self.assertEqual(invalid, [False])
        self.assertEqual(reasons, [()])

    def test_metrics_keep_overlong_repeat_loop_and_max_step_rates(self):
        base_reward = {
            "full": 0.0, "strict": 0.0, "native": 0.0, "semantic": 0.0,
            "total": 0.0, "terminal_utility": 0.0, "efficiency": 0.0,
            "penalty_overlong": 0.0, "penalty_unfinished": 0.0, "penalty_repeat": 0.0,
            "repeat_action_rate": 1.0, "purchase_success": False, "sampling_invalid": True,
            "r_type": 0.0, "r_att": 0.0, "r_option": 0.0, "r_price": 0.0,
        }
        metrics = aggregate_shopping_metrics([
            {
                "steps": 35,
                "done": True,
                "overlong": True,
                "termination_reason": "max_steps",
                "reward_type": "repeat_loop",
                "infrastructure_invalid": False,
                "reward": base_reward,
            }
        ])

        self.assertEqual(metrics["trajectory/overlong_rate"], 1.0)
        self.assertEqual(metrics["trajectory/repeat_loop_rate"], 1.0)
        self.assertEqual(metrics["trajectory/max_steps_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
