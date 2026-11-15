"""Selective decoder-layer recomputation for the one-stage FSDP2 recipe."""

from pathlib import Path


def checkpoint_decoder_layers(model, stride):
    """Checkpoint all but the last decoder in each group of N, before FSDP wrapping.

    This uses the same child boundaries as Accelerate's full FSDP2 checkpointing.
    Preserve RNG state so LoRA dropout uses its forward mask during recomputation.
    The wrapper preserves ordinary state-dict names for adapter saving.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    if stride < 2:
        raise ValueError("Selective checkpoint stride must be at least 2")
    layers = [module for module in model.modules()
              if module.__class__.__name__ == "Qwen3_5DecoderLayer"]
    if not layers:
        raise ValueError("Selective checkpointing requires Qwen3_5DecoderLayer")
    selected = []
    for index, layer in enumerate(layers):
        if (index + 1) % stride != 0:
            for name, child in list(layer.named_children()):
                layer.register_module(name, checkpoint_wrapper(child, preserve_rng_state=True))
            selected.append(index)
    return {"decoder_layers": len(layers), "checkpointed_layer_indices": selected,
            "preserve_rng_state": True}


def load_fsdp2_lora_checkpoint(model, checkpoint, accelerator):
    """Restore full adapter tensors to both ranks using PyTorch's FSDP2 loader.

    Accelerate's adapter-only FULL_STATE_DICT path saves raw DTensors from rank 0
    and bypasses its broadcast options on load. Trainer also saves a gathered
    adapter.safetensors; use those complete tensors with the public state loader.
    Optimizer, scheduler and RNG restoration remain Trainer's responsibility.
    """
    from safetensors.torch import load_file
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    path = Path(checkpoint) / "adapter_model.safetensors"
    if not path.exists():
        return False
    adapter = model.active_adapter
    if not isinstance(adapter, str):
        raise TypeError("FSDP2 LoRA resume requires one active adapter")
    state = load_file(str(path), device="cpu") if accelerator.is_main_process else {}
    if accelerator.is_main_process:
        state = {key.replace(".lora_A.", f".lora_A.{adapter}.")
                 .replace(".lora_B.", f".lora_B.{adapter}."): value
                 for key, value in state.items()}
        expected = {name.replace("._checkpoint_wrapped_module", "")
                    for name, param in model.named_parameters() if param.requires_grad}
        if state.keys() != expected:
            raise ValueError("Saved adapter keys do not match trainable FSDP2 parameters")
    result = set_model_state_dict(
        model, state,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True,
                                 broadcast_from_rank0=True, strict=False),
    )
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise ValueError("Incomplete FSDP2 LoRA restoration")
    return True
