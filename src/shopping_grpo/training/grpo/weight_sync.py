"""Keep merged LoRA values alive until the rollout consumes the weight iterator."""

from functools import wraps


def snapshot_state_dict(state):
    """Detach a snapshot from state_dict's shared parameter storage, including DTensors."""
    return {name: value.detach().clone() for name, value in state.items()}


def install_lora_sync_snapshot():
    """Fix the pinned veRL 0.8 merge path without changing actor parameters or sync keys.

    veRL restores the actor's base weights on leaving merged_lora_context, then
    consumes a lazy iterator over state_dict values. Those values share storage
    with the restored actor, so the rollout otherwise receives the base alone.
    Clone the sharded state while it still contains the adapter contribution.
    """
    import torch
    from torch.distributed.tensor import DTensor
    from verl.workers.engine.fsdp import transformer_impl as impl

    original = impl.FSDPEngine.get_per_tensor_param
    if getattr(original, "_shopping_lora_sync_snapshot", False):
        return

    @wraps(original)
    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if (
            not self.model_config.lora.get("merge", False)
            or not hasattr(peft_model, "peft_config")
            or self._qat_enabled
        ):
            return original(
                self, layered_summon=layered_summon, base_sync_done=base_sync_done, **kwargs
            )

        if not self._uses_fsdp2_cpu_offload_policy:
            impl.load_fsdp_model_to_gpu(self.module)
        with impl.merged_lora_context(self.module, backup_adapters=True):
            params = impl.normalize_peft_param_name(self.module.state_dict())
            params = snapshot_state_dict(params)
        params = impl.convert_weight_keys(params, peft_model)
        if self._is_offload_param:
            impl.offload_fsdp_model_to_cpu(self.module)
        device = impl.get_device_id()
        per_tensor_param = (
            (
                name,
                param.to(device, non_blocking=True).full_tensor().to(
                    torch.bfloat16, non_blocking=True
                )
                if isinstance(param, DTensor)
                else param,
            )
            for name, param in params.items()
        )
        return per_tensor_param, None

    get_per_tensor_param._shopping_lora_sync_snapshot = True
    impl.FSDPEngine.get_per_tensor_param = get_per_tensor_param
