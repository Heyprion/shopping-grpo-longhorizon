"""Check optional actor kernel selection, state preservation and convolution parity."""

import pytest
import torch
from torch.nn import functional as F

from shopping_grpo.training.grpo.kernel_optimizations import apply_actor_fusions, fla_convolution


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA kernel parity test")
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_fla_convolution_matches_padded_torch_values_and_gradients(dtype):
    torch.manual_seed(42)
    x=torch.randn(2,6144,33,device='cuda',dtype=dtype)
    x[1,:,17:]=0
    weight=torch.randn(6144,4,device='cuda',dtype=dtype)*0.1
    a=x.clone().requires_grad_()
    b=x.clone().requires_grad_()
    wa=weight.clone().requires_grad_()
    wb=weight.clone().requires_grad_()
    tf32=torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32=False
    try:
        expected=F.silu(F.conv1d(a,wa.unsqueeze(1),padding=3,groups=6144)[...,:33])
        actual=fla_convolution(b,wb,activation='silu')
        gradient=torch.randn_like(actual)
        ga=torch.autograd.grad(expected,(a,wa),gradient)
        gb=torch.autograd.grad(actual,(b,wb),gradient)
    finally:
        torch.backends.cudnn.allow_tf32=tf32
    atol,rtol=(2e-6,2e-5) if dtype==torch.float32 else (0.004,0.02)
    torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
    for ref,result in zip(ga,gb,strict=True):
        relative=(result.float()-ref.float()).norm()/ref.float().norm()
        assert relative < (1e-5 if dtype==torch.float32 else 0.01)


def test_packed_sequence_indices_are_rejected():
    with pytest.raises(ValueError,match='independent sequences'):
        fla_convolution(None,None,seq_idx=torch.tensor([0,1]))


def test_norm_and_convolution_patch_only_selected_instances():
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from transformers.models.qwen3_5 import modeling_qwen3_5

    class Qwen3_5MLP(torch.nn.Linear):
        pass

    class Qwen3_5GatedDeltaNet(torch.nn.Module):
        pass

    norm_class = modeling_qwen3_5.Qwen3_5RMSNorm
    norm_forward = norm_class.forward
    model = torch.nn.ModuleDict({
        "norm": norm_class(4, eps=1e-6),
        "mlp": Qwen3_5MLP(4, 4),
        "delta": Qwen3_5GatedDeltaNet(),
    })
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    mlp_forward = model["mlp"].forward.__func__

    counts = apply_actor_fusions(model, ["fla_conv", "liger_norm"])

    assert counts == {"fla_conv": 1, "liger_norm": 1}
    assert model["norm"].forward.__func__ is LigerRMSNorm.forward
    assert model["norm"].offset == 1.0
    assert model["norm"].casting_mode == "gemma"
    assert model["norm"].in_place is False
    assert model["norm"].variance_epsilon == 1e-6
    assert model["delta"].causal_conv1d_fn is fla_convolution
    assert model["mlp"].forward.__func__ is mlp_forward
    assert modeling_qwen3_5.Qwen3_5RMSNorm is norm_class
    assert norm_class.forward is norm_forward
    assert model.state_dict().keys() == weights.keys()
    for key, value in model.state_dict().items():
        assert torch.equal(value, weights[key])


def test_actor_kernel_hook_keeps_forward_only_engines_unchanged(monkeypatch):
    import sys
    from types import SimpleNamespace

    from shopping_grpo.training.grpo import kernel_optimizations

    model = object()
    calls = []

    class Engine:
        def __init__(self, forward_only=False, remove_padding=False):
            self.engine_config = SimpleNamespace(forward_only=forward_only)
            self.model_config = SimpleNamespace(use_remove_padding=remove_padding)

        def _build_module(self):
            return model

    monkeypatch.setitem(
        sys.modules, "verl.workers.engine.fsdp.transformer_impl",
        SimpleNamespace(FSDPEngine=Engine),
    )
    monkeypatch.setattr(
        kernel_optimizations, "apply_actor_fusions",
        lambda module, names: calls.append((module, names)) or {name: 1 for name in names},
    )
    monkeypatch.setenv("SHOPPING_GRPO_FAST_KERNELS", "fla_conv,liger_norm")
    kernel_optimizations.install_actor_fusions()
    installed = Engine._build_module
    kernel_optimizations.install_actor_fusions()
    assert Engine._build_module is installed
    assert Engine(forward_only=True)._build_module() is model
    assert calls == []
    assert Engine()._build_module() is model
    assert calls == [(model, ["fla_conv", "liger_norm"])]
    with pytest.raises(ValueError, match="use_remove_padding=false"):
        Engine(remove_padding=True)._build_module()
    monkeypatch.setenv("SHOPPING_GRPO_FAST_KERNELS", "liger_swiglu")
    with pytest.raises(ValueError, match="only fla_conv and liger_norm"):
        kernel_optimizations.install_actor_fusions()
