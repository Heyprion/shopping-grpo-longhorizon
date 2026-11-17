"""A rollout snapshot must retain LoRA deltas after the actor base is restored."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from shopping_grpo.training.grpo.weight_sync import snapshot_state_dict


def test_snapshot_preserves_adapter_delta_after_base_restore():
    layer = torch.nn.Linear(4, 4, bias=False)
    original = layer.weight.detach().clone()
    delta = torch.full_like(original, 0.25)
    with torch.no_grad():
        layer.weight.add_(delta)
        live_state = layer.state_dict()
        snapshot = snapshot_state_dict(live_state)
        layer.weight.copy_(original)

    torch.testing.assert_close(live_state["weight"], original)
    torch.testing.assert_close(snapshot["weight"], original + delta)
    assert not snapshot["weight"].requires_grad


def test_sync_snapshot_installation_is_idempotent():
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    from shopping_grpo.training.grpo.weight_sync import install_lora_sync_snapshot

    install_lora_sync_snapshot()
    patched = FSDPEngine.get_per_tensor_param
    install_lora_sync_snapshot()
    assert FSDPEngine.get_per_tensor_param is patched
    assert patched._shopping_lora_sync_snapshot


def test_merged_sync_iterator_retains_delta_and_restores_actor(monkeypatch):
    from verl.workers.engine.fsdp import transformer_impl as impl

    from shopping_grpo.training.grpo.weight_sync import install_lora_sync_snapshot

    module = torch.nn.Linear(4, 4, bias=False)
    module.peft_config = {"default": object()}
    baseline = module.weight.detach().clone()

    @contextmanager
    def merge(actor, backup_adapters):
        assert actor is module and backup_adapters
        with torch.no_grad():
            actor.weight.add_(0.25)
        try:
            yield
        finally:
            with torch.no_grad():
                actor.weight.copy_(baseline)

    monkeypatch.setattr(impl, "merged_lora_context", merge)
    monkeypatch.setattr(impl, "normalize_peft_param_name", lambda state: state)
    monkeypatch.setattr(impl, "convert_weight_keys", lambda state, model: state)
    monkeypatch.setattr(impl, "get_device_id", lambda: "cpu")
    engine = SimpleNamespace(
        module=module,
        model_config=SimpleNamespace(lora={"merge": True}),
        _uses_fsdp2_cpu_offload_policy=True,
        _is_offload_param=False,
        _qat_enabled=False,
    )
    install_lora_sync_snapshot()
    iterator, peft_config = impl.FSDPEngine.get_per_tensor_param(engine)
    torch.testing.assert_close(module.weight, baseline)
    assert peft_config is None
    sent = dict(iterator)
    torch.testing.assert_close(sent["weight"], baseline + 0.25)
