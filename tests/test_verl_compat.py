"""veRL 不应为了纯 padding 操作强制依赖 FlashAttention。"""

import sys
import unittest
from types import ModuleType
from unittest.mock import patch


class VerlCompatTest(unittest.TestCase):
    def test_installs_verl_builtin_padding_functions(self):
        attention = ModuleType("verl.utils.attention_utils")
        fallback = ModuleType("verl.utils.npu_flash_attn_utils")
        expected = tuple(object() for _ in range(4))
        (
            fallback.index_first_axis,
            fallback.pad_input,
            fallback.rearrange,
            fallback.unpad_input,
        ) = expected
        utils = ModuleType("verl.utils")
        utils.attention_utils = attention
        utils.npu_flash_attn_utils = fallback
        verl = ModuleType("verl")
        verl.utils = utils
        trainer = ModuleType("verl.trainer")
        ppo = ModuleType("verl.trainer.ppo")
        ray_trainer = ModuleType("verl.trainer.ppo.ray_trainer")

        class RayPPOTrainer:
            def fit(self):
                return self.total_training_steps

            def _update_actor(self, batch):
                return batch

        ray_trainer.RayPPOTrainer = RayPPOTrainer

        with patch.dict(
            sys.modules,
            {
                "verl": verl,
                "verl.utils": utils,
                "verl.utils.attention_utils": attention,
                "verl.utils.npu_flash_attn_utils": fallback,
                "verl.trainer": trainer,
                "verl.trainer.ppo": ppo,
                "verl.trainer.ppo.ray_trainer": ray_trainer,
            },
        ):
            from shopping_grpo.training.grpo.compat import install_torch_padding_fallback

            with patch("shopping_grpo.training.grpo.compat._seed_runtime") as seeder, \
                patch("shopping_grpo.training.grpo.compat._install_rollout_wakeup_cache_cleanup"), \
                patch("shopping_grpo.training.grpo.validation.install_validation_metrics"), \
                patch("shopping_grpo.training.grpo.fused_output.install_fused_output_layout"):
                install_torch_padding_fallback()
                seeder.assert_called_once_with(42)

        self.assertEqual(attention._get_attention_functions(), expected)
        self.assertTrue(RayPPOTrainer._update_actor._shopping_trace)
        self.assertTrue(RayPPOTrainer.fit._shopping_pilot)

    def test_releases_cache_before_colocated_rollout_wakes_up(self):
        import asyncio
        from types import SimpleNamespace

        from shopping_grpo.training.grpo.compat import _install_rollout_wakeup_cache_cleanup

        calls = []

        class Worker:
            async def update_weights(self, global_steps=None, mode="auto"):
                calls.append(("wake_up", global_steps, mode))
                return "synced"

        Worker.update_weights._dispatch_metadata = object()
        metadata = Worker.update_weights._dispatch_metadata
        workers = ModuleType("verl.workers.engine_workers")
        workers.ActorRolloutRefWorker = Worker
        memory = ModuleType("verl.utils.memory_utils")
        memory.aggressive_empty_cache = lambda **kwargs: calls.append(("cleanup", kwargs))
        with patch.dict(sys.modules, {
            "verl.workers.engine_workers": workers,
            "verl.utils.memory_utils": memory,
        }):
            _install_rollout_wakeup_cache_cleanup()
            installed = Worker.update_weights
            _install_rollout_wakeup_cache_cleanup()
        self.assertIs(Worker.update_weights, installed)
        self.assertIs(Worker.update_weights._dispatch_metadata, metadata)
        worker = Worker()
        worker.config = SimpleNamespace(rollout=SimpleNamespace(
            checkpoint_engine=SimpleNamespace(backend="naive"), free_cache_engine=True))
        result = asyncio.run(worker.update_weights(global_steps=6))
        self.assertEqual(result, "synced")
        self.assertEqual(calls, [("cleanup", {"force_sync": True}), ("wake_up", 6, "auto")])
        calls.clear()
        asyncio.run(worker.update_weights(global_steps=7, mode="nccl"))
        self.assertEqual(calls, [("wake_up", 7, "nccl")])
        calls.clear()
        worker.config.rollout.free_cache_engine = False
        asyncio.run(worker.update_weights(global_steps=8))
        self.assertEqual(calls, [("wake_up", 8, "auto")])

    def test_pilot_bounds_updates_without_changing_optimizer_schedule(self):
        from types import SimpleNamespace

        from shopping_grpo.training.grpo.compat import _install_bounded_pilot

        class RayPPOTrainer:
            def fit(self):
                assert self.config.actor_rollout_ref.actor.optim.total_training_steps == 500
                return self.total_training_steps

        module = ModuleType("verl.trainer.ppo.ray_trainer")
        module.RayPPOTrainer = RayPPOTrainer
        with patch.dict(sys.modules, {"verl.trainer.ppo.ray_trainer": module}):
            _install_bounded_pilot()
        class Config(dict):
            actor_rollout_ref = SimpleNamespace(actor=SimpleNamespace(
                optim=SimpleNamespace(total_training_steps=500)))
        trainer = RayPPOTrainer()
        trainer.config = Config(shopping_pilot={"max_steps": 3})
        trainer.total_training_steps = 500
        self.assertEqual(trainer.fit(), 3)
        self.assertEqual(trainer.total_training_steps, 500)

    def test_resource_pilot_skips_validation_and_restores_it(self):
        from types import SimpleNamespace

        from shopping_grpo.training.grpo.compat import _install_bounded_pilot

        class RayPPOTrainer:
            def _validate(self):
                raise AssertionError("Resource-only pilot must not run validation")

            def fit(self):
                return self._validate()

        module = ModuleType("verl.trainer.ppo.ray_trainer")
        module.RayPPOTrainer = RayPPOTrainer
        with patch.dict(sys.modules, {"verl.trainer.ppo.ray_trainer": module}):
            _install_bounded_pilot()

        class Config(dict):
            actor_rollout_ref = SimpleNamespace(actor=SimpleNamespace(
                optim=SimpleNamespace(total_training_steps=500)))

        trainer = RayPPOTrainer()
        original = trainer._validate
        trainer.config = Config(shopping_pilot={"max_steps":1,"skip_validation":True})
        trainer.total_training_steps = 500
        self.assertEqual(trainer.fit(), {})
        self.assertEqual(trainer._validate, original)
        self.assertEqual(trainer.total_training_steps, 500)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
