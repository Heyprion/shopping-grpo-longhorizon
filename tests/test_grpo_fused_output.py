"""Fused outputs must retain sample boundaries and drop only padding gradients."""

from types import SimpleNamespace

import torch

from shopping_grpo.training.grpo.fused_output import dense_token_outputs_to_nested


def test_fused_padded_to_nested_preserves_token_order_and_gradients():
    ids = torch.nested.nested_tensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
                                   layout=torch.jagged)
    probs = torch.arange(6, dtype=torch.float32, requires_grad=True)
    entropies = (10 + torch.arange(6, dtype=torch.float32)).requires_grad_()
    output = dense_token_outputs_to_nested(
        SimpleNamespace(log_probs=probs, entropy=entropies), ids, True, torch)
    torch.testing.assert_close(output["log_probs"].values(), torch.tensor([0., 1., 2., 3., 4.]))
    torch.testing.assert_close(output["entropy"].values(), torch.tensor([10., 11., 12., 13., 14.]))
    (output["log_probs"].values().sum() + output["entropy"].values().sum()).backward()
    torch.testing.assert_close(probs.grad, torch.tensor([1., 1., 1., 1., 1., 0.]))
    torch.testing.assert_close(entropies.grad, torch.tensor([1., 1., 1., 1., 1., 0.]))
