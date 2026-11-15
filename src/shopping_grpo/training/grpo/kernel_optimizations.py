"""Opt-in actor fusions; keep the PPO head and padded sequence contract."""

from functools import lru_cache
from types import MethodType


@lru_cache(maxsize=1)
def native_swiglu():
    import torch
    from torch.nn import functional as F

    return torch.compile(lambda gate, up: F.silu(gate)*up, fullgraph=True, dynamic=True,
                         options={"emulate_precision_casts":True})


def native_mlp_forward(self, inputs):
    return self.down_proj(native_swiglu()(self.gate_proj(inputs),self.up_proj(inputs)))


def fla_convolution(x, weight, bias=None, activation=None, seq_idx=None, **kwargs):
    from fla.modules.causal_conv1d import causal_conv1d
    from torch.nn import functional as F

    if seq_idx is not None:
        raise ValueError("FLA actor convolution requires padded, independent sequences")
    if activation not in (None,"silu","swish"):
        raise ValueError("Unsupported actor convolution activation")
    # Retain the BF16 convolution-output rounding before the separate activation.
    output,_=causal_conv1d(x.transpose(1,2),weight=weight,bias=bias,activation=None,backend="triton")
    output=output.transpose(1,2)
    return F.silu(output) if activation else output


def apply_actor_fusions(module, names):
    names=set(names)
    allowed={"liger_norm","liger_swiglu","native_swiglu","fla_conv"}
    if not names <= allowed or {"liger_swiglu","native_swiglu"} <= names:
        raise ValueError(f"Unsupported actor fusion selection: {names}")
    counts={name:0 for name in names}
    for child in module.modules():
        classes={cls.__name__ for cls in type(child).__mro__}
        if "Qwen3_5RMSNorm" in classes and "liger_norm" in names:
            from liger_kernel.transformers.monkey_patch import _patch_rms_norm_module

            # Match Liger's Qwen3.5 adapter on this actor instance only; avoid
            # replacing HF classes globally or enabling its MLP / loss patches.
            _patch_rms_norm_module(child, offset=1.0, casting_mode="gemma", in_place=False)
            counts["liger_norm"]+=1
        if "Qwen3_5MLP" in classes:
            if "liger_swiglu" in names:
                from liger_kernel.transformers.swiglu import LigerQwen3MoeSwiGLUMLP

                child.forward=MethodType(LigerQwen3MoeSwiGLUMLP.forward,child)
                counts["liger_swiglu"]+=1
            elif "native_swiglu" in names:
                child.forward=MethodType(native_mlp_forward,child)
                counts["native_swiglu"]+=1
        if "Qwen3_5GatedDeltaNet" in classes and "fla_conv" in names:
            child.causal_conv1d_fn=fla_convolution
            counts["fla_conv"]+=1
    if any(value==0 for value in counts.values()):
        raise ValueError(f"Actor fusion found no matching module: {counts}")
    return counts


def install_actor_fusions():
    import functools
    import os

    names=[name.strip() for name in os.environ.get("SHOPPING_GRPO_FAST_KERNELS","").split(",") if name.strip()]
    if not names:
        return
    if not set(names) <= {"fla_conv", "liger_norm"}:
        raise ValueError("Normal training supports only fla_conv and liger_norm actor fusions")
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    if getattr(FSDPEngine._build_module,"_shopping_actor_fusions",False):
        return
    original=FSDPEngine._build_module

    @functools.wraps(original)
    def build(self,*args,**kwargs):
        module=original(self,*args,**kwargs)
        if not self.engine_config.forward_only:
            if self.model_config.use_remove_padding:
                raise ValueError("Actor fusion tests require use_remove_padding=false")
            print(f"SHOPPING_GRPO_ACTOR_FUSIONS {apply_actor_fusions(module,names)}",flush=True)
        return module

    build._shopping_actor_fusions=True
    FSDPEngine._build_module=build
