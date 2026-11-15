"""Changing recomputation must preserve the forward and dropout gradients."""

import copy
from functools import partial

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_layers import GradientCheckpointingLayer

from shopping_grpo.training.grpo.checkpointing import set_decoder_checkpointing


class Qwen3_5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,4)
        self.dropout = nn.Dropout(0.2)
        self._gradient_checkpointing_func = partial(
            checkpoint,use_reentrant=False,preserve_rng_state=True)

    def forward(self, inputs):
        return inputs+self.dropout(self.linear(inputs))*0.1


def test_partial_recomputation_preserves_dropout_outputs_and_gradients():
    torch.manual_seed(42)
    full = nn.Sequential(*(Qwen3_5DecoderLayer() for _ in range(24)))
    partial_model = copy.deepcopy(full)
    keys = list(partial_model.state_dict())
    set_decoder_checkpointing(full,24)
    info = set_decoder_checkpointing(partial_model,20)
    assert info['layers_without_recomputation']==[5,11,17,23]
    assert list(partial_model.state_dict())==keys
    inputs = torch.randn(2,4)
    torch.manual_seed(42)
    first = full(inputs.clone().requires_grad_(True))
    first.square().sum().backward()
    torch.manual_seed(42)
    second = partial_model(inputs.clone().requires_grad_(True))
    second.square().sum().backward()
    torch.testing.assert_close(first,second)
    for p,q in zip(full.parameters(),partial_model.parameters(),strict=True):
        torch.testing.assert_close(p.grad,q.grad)


def test_recomputation_budget_rejects_incompatible_layers():
    with pytest.raises(ValueError,match='24 layers'):
        set_decoder_checkpointing(nn.Linear(4,4),20)
