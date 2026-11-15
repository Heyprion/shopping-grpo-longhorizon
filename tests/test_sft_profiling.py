"""Protect data identity, assistant-token measurements and bounded pilot semantics."""

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.train_lora_sft import _load_examples_cached
from shopping_grpo.training.sft.profiling import make_profile_callback


class CacheIdentityTest(unittest.TestCase):
    def test_cache_reuses_identical_contract_and_invalidates_changed_data_or_length(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data.jsonl"
            source.write_text('{"task_id":1}\n')
            args = argparse.Namespace(model="model", revision="fixed", max_length=24576,
                                      preprocessing_cache=root / "cache")
            tokenizer = [0, 1, 2]
            template = argparse.Namespace(chat_template="template")
            with patch("scripts.train_lora_sft.load_supervised_examples",
                       return_value=([{"task_id": 1}], {"kept": 1})) as loader:
                _load_examples_cached(args, source, tokenizer, template, None)
                _load_examples_cached(args, source, tokenizer, template, None)
                self.assertEqual(loader.call_count, 1)
                source.write_text('{"task_id":2}\n')
                _load_examples_cached(args, source, tokenizer, template, None)
                self.assertEqual(loader.call_count, 2)
                args.max_length = 32768
                _load_examples_cached(args, source, tokenizer, template, None)
                self.assertEqual(loader.call_count, 3)
                _load_examples_cached(args, source, tokenizer, template, {2})
                self.assertEqual(loader.call_count, 4)


class PilotControlTest(unittest.TestCase):
    def test_step_pause_keeps_original_scheduler_budget(self):
        torch = MagicMock()
        torch.cuda.mem_get_info.return_value = (1, 2)
        torch.tensor.return_value.reshape.return_value.tolist.return_value = [
            [0.5, 1, 2, 3, 10, 2, 10, 1],
        ]
        trainer = argparse.Namespace(
            sft_counts=[10, 2, 10, 1], sft_profile={"steps": []},
            accelerator=argparse.Namespace(gather=lambda value: value),
        )
        cli = argparse.Namespace(stop_after_steps=20)
        callback = make_profile_callback(object, torch, cli)
        callback.profile_trainer = trainer
        callback.started = 0
        args = argparse.Namespace(device="cpu")
        state = argparse.Namespace(global_step=19, epoch=0.19, is_world_process_zero=False,
                                   max_steps=300)
        control = argparse.Namespace(should_training_stop=False)
        callback.on_step_end(args, state, control)
        self.assertFalse(control.should_training_stop)
        state.global_step = 20
        callback.on_step_end(args, state, control)
        self.assertTrue(control.should_training_stop)
        self.assertEqual(state.max_steps, 300)

    def test_epoch_pause_stops_only_at_requested_boundary_and_preserves_save_eval(self):
        callback = make_profile_callback(object, None, argparse.Namespace(stop_after_epoch=1))
        control = argparse.Namespace(should_training_stop=False, should_save=True, should_evaluate=True)
        callback.on_epoch_end(None, argparse.Namespace(epoch=0.9), control)
        self.assertFalse(control.should_training_stop)
        callback.on_epoch_end(None, argparse.Namespace(epoch=1), control)
        self.assertTrue(control.should_training_stop)
        self.assertTrue(control.should_save)
        self.assertTrue(control.should_evaluate)

    def test_nonfinite_gradient_aborts_instead_of_silently_reporting_success(self):
        callback = make_profile_callback(object, None, argparse.Namespace(stop_after_epoch=None))
        with self.assertRaises(FloatingPointError):
            callback.on_log(None, None, None, logs={"grad_norm": float("nan")})


if __name__ == "__main__":
    unittest.main()
