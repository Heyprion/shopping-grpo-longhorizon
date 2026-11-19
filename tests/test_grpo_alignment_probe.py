"""The diagnostic measures an update without replacing its behavior probabilities."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from omegaconf import OmegaConf


def test_alignment_probe_measures_masked_policy_improvement_once(tmp_path, monkeypatch):
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from shopping_grpo.training.grpo import alignment_probe, compat

    monkeypatch.setattr(compat, "install_torch_padding_fallback", lambda: None)
    path = tmp_path / "alignment.jsonl"
    monkeypatch.setenv("SHOPPING_GRPO_ALIGNMENT_PROBE", str(path))
    original = Mock(return_value=SimpleNamespace(meta_info={"metrics": {}}))
    original._shopping_alignment_probe = False
    monkeypatch.setattr(RayPPOTrainer, "_update_actor", original)
    alignment_probe.install()
    installed = RayPPOTrainer._update_actor
    alignment_probe.install()
    assert RayPPOTrainer._update_actor is installed
    behavior = torch.full((2, 3), -1.0)
    before = behavior.clone()
    after = torch.tensor([[-0.9, -0.9, -20.0], [-1.1, -1.1, 0.0]])
    batch = SimpleNamespace(batch={
        "old_log_probs": behavior,
        "response_mask": torch.tensor([[1, 1, 0], [1, 1, 0]]),
        "advantages": torch.tensor([[1, 1, 0], [-1, -1, 0]]),
    }, meta_info={}, non_tensor_batch={"shopping": [
        {"task_id": 1, "reward": {"total": 1.0}, "steps": 1},
        {"task_id": 1, "reward": {"total": 0.0}, "steps": 1},
    ]})
    trainer = SimpleNamespace(global_steps=51,
        config=OmegaConf.create({"actor_rollout_ref": {
            "actor": {"clip_ratio_low": 0.2, "clip_ratio_high": 0.2},
            "rollout": {"temperature": 1.0, "multi_turn": {"enable": True}},
        }}),
        _compute_old_log_prob=Mock(side_effect=[
            (SimpleNamespace(batch={"old_log_probs": before}), 0),
            (SimpleNamespace(batch={"old_log_probs": after}), 0),
        ]))
    result = installed(trainer, batch)
    row = json.loads(path.read_text())
    assert row["clipped_policy_loss_after"] < row["clipped_policy_loss_before"]
    assert row["advantage_weighted_delta_logp"] > 0
    assert row["positive_mean_delta_logp"] > 0 > row["negative_mean_delta_logp"]
    assert torch.equal(behavior, torch.full((2, 3), -1.0))
    assert result.meta_info["metrics"]["alignment/ppo_loss_improvement"] > 0
    original.assert_called_once_with(trainer, batch)


def test_counterfactual_restores_parameters_optimizer_and_scheduler(monkeypatch):
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
    from shopping_grpo.training.grpo import alignment_probe

    original = Mock()
    original._shopping_counterfactual_restore = False
    monkeypatch.setattr(FSDPEngine, "load_checkpoint", original)
    alignment_probe.install_counterfactual_restore()
    load = FSDPEngine.load_checkpoint
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    engine = SimpleNamespace(module=model, optimizer=optimizer, lr_scheduler=scheduler)

    def step():
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()

    step()
    load(engine, "shopping-probe:snapshot-pre")
    expected_pre = alignment_probe.snapshot_engine(engine)
    step()
    expected_post = alignment_probe.snapshot_engine(engine)
    load(engine, "shopping-probe:counterfactual-beta0")
    assert optimizer.param_groups[0]["betas"][0] == 0.0
    assert all(torch.equal(p, expected_pre["parameters"][n]) for n, p in model.named_parameters())
    step()
    load(engine, "shopping-probe:restore-post")
    assert optimizer.param_groups[0]["betas"][0] == 0.9
    assert all(torch.equal(p, expected_post["parameters"][n]) for n, p in model.named_parameters())
    assert scheduler.state_dict() == expected_post["scheduler"]
    for index, state in optimizer.state_dict()["state"].items():
        assert all(torch.equal(value, expected_post["optimizer"]["state"][index][key])
                   for key, value in state.items())
    assert not hasattr(engine, "_shopping_probe_pre")
    load(engine, "/normal/checkpoint")
    original.assert_called_once_with(engine, "/normal/checkpoint")
