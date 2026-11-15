"""Recomputation must preserve dropout gradients and adapter state-dict names."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from shopping_grpo.training.sft.checkpointing import (
    checkpoint_decoder_layers,
    load_fsdp2_lora_checkpoint,
)


class Qwen3_5DecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.2))

    def forward(self, value):
        return self.mlp(value).tanh()


def test_partial_checkpoint_preserves_dropout_gradients_and_state_names():
    torch.manual_seed(42)
    model = torch.nn.Sequential(*(Qwen3_5DecoderLayer() for _ in range(4)))
    measured = copy.deepcopy(model)
    metadata = checkpoint_decoder_layers(measured, 2)
    assert metadata["checkpointed_layer_indices"] == [0, 2]
    assert metadata["decoder_layers"] == 4
    assert list(model.state_dict()) == list(measured.state_dict())
    value = torch.randn(2, 4)
    for candidate in (model, measured):
        torch.manual_seed(42)
        candidate(value).square().sum().backward()
    for expected, actual in zip(model.parameters(), measured.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-7, rtol=1e-6)


def test_partial_checkpoint_rejects_invalid_stride_or_unknown_architecture():
    with pytest.raises(ValueError, match="at least 2"):
        checkpoint_decoder_layers(torch.nn.Linear(4, 4), 1)
    with pytest.raises(ValueError, match="Qwen3_5DecoderLayer"):
        checkpoint_decoder_layers(torch.nn.Linear(4, 4), 2)


def test_quarter_uncheckpointed_selection_is_consistent_across_groups():
    model = torch.nn.Sequential(*(Qwen3_5DecoderLayer() for _ in range(8)))
    metadata = checkpoint_decoder_layers(model, 4)
    assert metadata["checkpointed_layer_indices"] == [0, 1, 2, 4, 5, 6]


@pytest.mark.parametrize("main_process", [True, False])
def test_resume_loads_gathered_adapter_and_broadcasts_to_every_rank(tmp_path, main_process):
    from safetensors.torch import save_file

    tensors = {"layer.lora_A.weight": torch.ones(2, 4),
               "layer.lora_B.weight": torch.ones(4, 2)}
    save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    model = SimpleNamespace(
        active_adapter="default",
        named_parameters=lambda: [
            ("layer._checkpoint_wrapped_module.lora_A.default.weight", torch.nn.Parameter(tensors["layer.lora_A.weight"])),
            ("layer._checkpoint_wrapped_module.lora_B.default.weight", torch.nn.Parameter(tensors["layer.lora_B.weight"])),
        ],
    )
    with patch("torch.distributed.checkpoint.state_dict.set_model_state_dict",
               return_value=SimpleNamespace(missing_keys=["frozen.weight"], unexpected_keys=[])) as setter:
        assert load_fsdp2_lora_checkpoint(
            model, tmp_path, SimpleNamespace(is_main_process=main_process))
    state = setter.call_args.args[1]
    options = setter.call_args.kwargs["options"]
    assert options.full_state_dict and options.broadcast_from_rank0 and not options.strict
    if main_process:
        assert set(state) == {"layer.lora_A.default.weight", "layer.lora_B.default.weight"}
        torch.testing.assert_close(state["layer.lora_A.default.weight"], tensors["layer.lora_A.weight"])
    else:
        assert state == {}
