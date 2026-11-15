"""Opt-in decoder recomputation budget for the single GRPO workflow."""


def set_decoder_checkpointing(module, checkpointed_layers):
    layers = [m for m in module.modules() if any(
        cls.__name__ == "Qwen3_5DecoderLayer" for cls in type(m).__mro__)]
    if len(layers) != 24 or not 0 <= checkpointed_layers <= 24:
        raise ValueError("GRPO decoder checkpointing requires 24 layers and a budget in [0, 24]")
    disabled_count = 24-checkpointed_layers
    disabled = {(j+1)*24//disabled_count-1 for j in range(disabled_count)}
    for index, layer in enumerate(layers):
        layer.gradient_checkpointing = index not in disabled
    return {"decoder_layers":24,"checkpointed_layers":checkpointed_layers,
            "layers_without_recomputation":sorted(disabled)}


def install_selective_checkpointing():
    import functools
    import os

    setting = os.environ.get("SHOPPING_GRPO_CHECKPOINTED_LAYERS")
    if setting is None:
        return
    count = int(setting)
    if not 0 <= count <= 24:
        raise ValueError("SHOPPING_GRPO_CHECKPOINTED_LAYERS must be in [0, 24]")

    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    if getattr(FSDPEngine._build_module, "_shopping_checkpoint_budget", False):
        return
    original = FSDPEngine._build_module

    @functools.wraps(original)
    def build(self, *args, **kwargs):
        module = original(self,*args,**kwargs)
        if not self.engine_config.forward_only:
            if not self.model_config.enable_gradient_checkpointing:
                raise ValueError("Selective recomputation requires enable_gradient_checkpointing=true")
            info = set_decoder_checkpointing(module,count)
            print(f"SHOPPING_GRPO_CHECKPOINT_BUDGET {info}",flush=True)
        return module

    build._shopping_checkpoint_budget = True
    FSDPEngine._build_module = build
