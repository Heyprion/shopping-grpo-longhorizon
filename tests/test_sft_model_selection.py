import copy

import pytest

from scripts.evaluate_sft_checkpoint import validate_training_summary
from scripts.train_lora_sft import _best_model_settings
from shopping_grpo.training.sft.profiling import unique_validation_samples


def test_profile_selects_lowest_token_weighted_validation_nll():
    assert _best_model_settings(True, True, False) == {
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_assistant_nll",
        "greater_is_better": False,
    }
    assert _best_model_settings(True, True, True) == {"load_best_model_at_end": False}
    assert _best_model_settings(False, True, False) == {"load_best_model_at_end": False}


def test_distributed_eval_padding_is_removed_before_token_weighting():
    a = {"input_sha256": "a", "supervised_tokens": 10, "loss": 1.0, "input_tokens": 100}
    b = {"input_sha256": "b", "supervised_tokens": 90, "loss": 3.0, "input_tokens": 1000}
    rows = unique_validation_samples([a, b, copy.deepcopy(b)])
    assert len(rows) == 2
    assert (
        sum(r["loss"] * r["supervised_tokens"] for r in rows)
        / sum(r["supervised_tokens"] for r in rows)
        == 2.8
    )


def test_final_test_requires_full_training_and_validation_best_adapter():
    summary = {
        "model_selection": {
            "metric": "eval_assistant_nll",
            "best_metric": 0.2,
            "best_checkpoint": "checkpoint-112",
            "final_adapter_is_best": True,
        },
        "arguments": {"epochs": 3, "max_steps": -1},
        "train_data": {"dropped": 0},
        "validation_data": {"dropped": 0},
    }
    assert validate_training_summary(summary)["best_checkpoint"] == "checkpoint-112"
    summary["model_selection"]["final_adapter_is_best"] = False
    with pytest.raises(ValueError, match="selected"):
        validate_training_summary(summary)
    summary["model_selection"]["final_adapter_is_best"] = True
    summary["arguments"]["stop_after_steps"] = 2
    with pytest.raises(ValueError, match="three-epoch"):
        validate_training_summary(summary)
