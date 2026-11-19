"""Opt-in measurement of the actual PPO objective before and after an update."""


def clone_state(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_state(v) for v in value)
    return value


def reshard_engine(engine):
    from torch.distributed.fsdp import FSDPModule
    # Inference with reshard_after_forward=false exposes plain, full parameters.
    # Snapshot and restore the canonical sharded parameters held by the optimizer.
    for module in reversed(list(engine.module.modules())):
        if isinstance(module, FSDPModule):
            module.reshard()


def snapshot_engine(engine):
    import torch
    reshard_engine(engine)
    device = next(engine.module.parameters()).device
    return {
        "parameters": {name: p.detach().clone() for name, p in engine.module.named_parameters() if p.requires_grad},
        "optimizer": clone_state(engine.optimizer.state_dict()),
        "scheduler": clone_state(engine.lr_scheduler.state_dict()),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_engine(engine, state):
    import torch
    reshard_engine(engine)
    with torch.no_grad():
        for name, p in engine.module.named_parameters():
            if name in state["parameters"]:
                p.copy_(state["parameters"][name])
                actual = p.to_local() if hasattr(p, "to_local") else p
                target = state["parameters"][name]
                target = target.to_local() if hasattr(target, "to_local") else target
                if not torch.equal(actual, target):
                    raise RuntimeError("Counterfactual parameter restore mismatch: " + name)
    engine.optimizer.load_state_dict(clone_state(state["optimizer"]))
    engine.lr_scheduler.load_state_dict(clone_state(state["scheduler"]))
    engine.optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(state["cpu_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"], next(engine.module.parameters()).device)


def install_counterfactual_restore():
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    original = FSDPEngine.load_checkpoint
    if getattr(original, "_shopping_counterfactual_restore", False):
        return

    def load(self, local_path, *args, **kwargs):
        if local_path == "shopping-probe:snapshot-pre":
            self._shopping_probe_pre = snapshot_engine(self)
        elif local_path == "shopping-probe:counterfactual-beta0":
            self._shopping_probe_post = snapshot_engine(self)
            restore_engine(self, self._shopping_probe_pre)
            for group in self.optimizer.param_groups:
                group["betas"] = (0.0, group["betas"][1])
        elif local_path == "shopping-probe:restore-post":
            restore_engine(self, self._shopping_probe_post)
            del self._shopping_probe_pre, self._shopping_probe_post
        else:
            return original(self, local_path, *args, **kwargs)

    load._shopping_counterfactual_restore = True
    FSDPEngine.load_checkpoint = load


def install():
    import json
    import os
    import torch
    from pathlib import Path
    from shopping_grpo.training.grpo.compat import install_torch_padding_fallback

    install_torch_padding_fallback()
    path = os.environ.get("SHOPPING_GRPO_ALIGNMENT_PROBE")
    if not path:
        return
    counterfactual = os.environ.get("SHOPPING_GRPO_ALIGNMENT_COUNTERFACTUAL") == "1"
    if counterfactual:
        install_counterfactual_restore()
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    original = RayPPOTrainer._update_actor
    if getattr(original, "_shopping_alignment_probe", False):
        return

    def update(self, batch):
        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        if counterfactual:
            self.actor_rollout_wg.load_checkpoint("shopping-probe:snapshot-pre")
        before, _ = self._compute_old_log_prob(batch)
        pre = before.batch["old_log_probs"].detach().float().clone()
        result = original(self, batch)
        after, _ = self._compute_old_log_prob(batch)
        post = after.batch["old_log_probs"].detach().float()
        mask = batch.batch["response_mask"].float().cpu()
        advantages = batch.batch["advantages"].float().cpu()
        behavior = batch.batch["old_log_probs"].float().cpu()
        config = self.config.actor_rollout_ref.actor
        low = float(config.clip_ratio_low)
        high = float(config.clip_ratio_high)
        dual = float(config.get("clip_ratio_c", 3.0))

        def objective(logp):
            ratio = (logp - behavior).clamp(-20, 20).exp()
            loss = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(1-low, 1+high))
            loss = torch.where(advantages < 0, torch.minimum(loss, -advantages * dual), loss)
            return float((loss * mask).sum() / mask.sum())

        delta = post - pre
        positive = mask * (advantages > 0)
        negative = mask * (advantages < 0)
        record = {
            "global_step": int(self.global_steps),
            "clipped_policy_loss_before": objective(pre),
            "clipped_policy_loss_after": objective(post),
            "advantage_weighted_delta_logp": float((advantages * delta * mask).sum() / mask.sum()),
            "positive_mean_delta_logp": float((delta * positive).sum() / positive.sum()),
            "negative_mean_delta_logp": float((delta * negative).sum() / negative.sum()),
            "trajectories": [],
        }
        if counterfactual:
            self.actor_rollout_wg.load_checkpoint("shopping-probe:counterfactual-beta0")
            original(self, batch)
            alternative, _ = self._compute_old_log_prob(batch)
            alternative = alternative.batch["old_log_probs"].detach().float()
            alternative_delta = alternative - pre
            record.update({
                "counterfactual_beta1": 0.0,
                "counterfactual_policy_loss_after": objective(alternative),
                "counterfactual_weighted_delta_logp": float((advantages * alternative_delta * mask).sum() / mask.sum()),
                "counterfactual_positive_mean_delta_logp": float((alternative_delta * positive).sum() / positive.sum()),
                "counterfactual_negative_mean_delta_logp": float((alternative_delta * negative).sum() / negative.sum()),
                "retained_branch": "standard_adam_beta1_0.9",
            })
            self.actor_rollout_wg.load_checkpoint("shopping-probe:restore-post")
        for i, info in enumerate(batch.non_tensor_batch["shopping"]):
            count = mask[i].sum()
            record["trajectories"].append({
                "task_id": int(info["task_id"]), "reward": float(info["reward"]["total"]),
                "tokens": int(count), "steps": int(info["steps"]),
                "advantage": float((advantages[i] * mask[i]).sum() / count),
                "mean_delta_logp": float((delta[i] * mask[i]).sum() / count),
            })
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        result.meta_info["metrics"].update({
            "alignment/weighted_delta_logp": record["advantage_weighted_delta_logp"],
            "alignment/ppo_loss_improvement": record["clipped_policy_loss_before"] - record["clipped_policy_loss_after"],
        })
        return result

    update._shopping_alignment_probe = True
    RayPPOTrainer._update_actor = update
