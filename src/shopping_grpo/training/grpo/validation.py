"""Strict Reward-v3 validation metrics without changing the optimization reward."""


def strict_success(info):
    return bool(
        info.get("done") is True
        and info.get("reward_type") == "gold_purchase"
        and info.get("reward_version") == "shopsimulator-reward-v3"
        and info.get("reward_valid") is True
        and info.get("reward", {}).get("purchase_success") == 1.0
        and not info.get("infrastructure_invalid")
        and not info.get("reward_unverifiable")
    )


def install_validation_metrics():
    import os
    from contextvars import ContextVar

    from verl.trainer.ppo import ray_trainer

    from shopping_grpo.training.grpo.dynamic_sampling import append_training_diagnostic

    if getattr(ray_trainer.extract_reward, "_shopping_validation", False):
        return
    original_extract = ray_trainer.extract_reward
    original_validate = ray_trainer.RayPPOTrainer._validate
    current_step = ContextVar("shopping_validation_step", default=None)

    def validate(self, *args, **kwargs):
        token = current_step.set(self.global_steps)
        try:
            return original_validate(self, *args, **kwargs)
        finally:
            current_step.reset(token)

    def extract(batch):
        score, extra = original_extract(batch)
        infos = batch.non_tensor_batch.get("shopping")
        step = current_step.get()
        if step is not None and infos is not None:
            infos = list(infos)
            extra = {**extra, "acc": [float(strict_success(info)) for info in infos]}
            append_training_diagnostic(
                os.environ.get("SHOPPING_GRPO_VALIDATION_DIAGNOSTICS_PATH"),
                "validation_batch", step, rollouts=infos,
            )
        return score, extra

    extract._shopping_validation = True
    ray_trainer.extract_reward = extract
    ray_trainer.RayPPOTrainer._validate = validate
