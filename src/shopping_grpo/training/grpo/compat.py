"""veRL 0.8 的窄范围运行时兼容。"""


def _install_rollout_wakeup_cache_cleanup():
    """Release actor allocator caches before colocated vLLM allocates weights.

    veRL 0.8 cleans these caches only after the weight transfer. With resident
    FSDP2 parameters, a long actor update can leave enough unused cached memory
    to make vLLM's earlier CUDA allocator wake-up fail.
    """
    from functools import wraps

    from verl.utils.memory_utils import aggressive_empty_cache
    from verl.workers.engine_workers import ActorRolloutRefWorker

    original = ActorRolloutRefWorker.update_weights
    if getattr(original, "_shopping_wakeup_cache_cleanup", False):
        return

    @wraps(original)
    async def update_weights(self, global_steps=None, mode="auto"):
        effective_mode = (
            self.config.rollout.checkpoint_engine.backend if mode == "auto" else mode
        )
        if effective_mode == "naive" and self.config.rollout.free_cache_engine:
            aggressive_empty_cache(force_sync=True)
        return await original(self, global_steps=global_steps, mode=mode)

    update_weights._shopping_wakeup_cache_cleanup = True
    ActorRolloutRefWorker.update_weights = update_weights


def _install_bounded_pilot():
    """Bound search updates after workers have built the full-budget scheduler."""
    import json

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    if getattr(RayPPOTrainer.fit, "_shopping_pilot", False):
        return
    original_fit = RayPPOTrainer.fit

    def fit(self, *args, **kwargs):
        pilot = self.config.get("shopping_pilot", {})
        limit = int(pilot.get("max_steps", 0))
        if not limit:
            return original_fit(self, *args, **kwargs)
        planned = self.total_training_steps
        if not 0 < limit <= planned:
            raise ValueError("shopping pilot steps must be within the planned training budget")
        # init_workers has already constructed optimizer/scheduler from config.
        # Change only fit's stopping condition, not the optimizer's 500-step plan.
        self.total_training_steps = limit
        skip_validation = bool(pilot.get("skip_validation", False))
        validate = getattr(self, "_validate", None)
        if skip_validation:
            self._validate = lambda *args, **kwargs: {}
        print("SHOPPING_GRPO_PILOT " + json.dumps({"stop_at_step": limit,
              "planned_training_steps": planned,
              "optimizer_planned_steps": self.config.actor_rollout_ref.actor.optim.total_training_steps}))
        try:
            return original_fit(self, *args, **kwargs)
        finally:
            self.total_training_steps = planned
            if skip_validation:
                self._validate = validate

    fit._shopping_pilot = True
    RayPPOTrainer.fit = fit


def _install_trace_actor_update():
    """Score private gold targets before the existing veRL actor update."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    if getattr(RayPPOTrainer._update_actor, "_shopping_trace", False):
        return

    from shopping_grpo.training.grpo.trace import (
        apply_trace_advantages,
        build_trace_score_batch,
    )

    original_update_actor = RayPPOTrainer._update_actor

    def update_actor_with_trace(self, batch):
        config = self.config.get("shopping_trace", {})
        enabled = bool(config.get("enable", False))
        if not enabled:
            batch.non_tensor_batch.pop("trace_target", None)
            output = original_update_actor(self, batch)
            tensors = getattr(batch, "batch", {})
            if "old_log_probs" in tensors and "response_mask" in tensors:
                mask = tensors["response_mask"]
                count = mask.sum()
                if count.item() > 0:
                    # Monte Carlo surprisal is a cheap diagnostic, not full-vocab entropy.
                    output.meta_info["metrics"]["actor/sampled_token_surprisal"] = float(
                        (-(tensors["old_log_probs"] * mask).sum() / count).item()
                    )
            return output
        if not self.ref_in_actor:
            raise RuntimeError("shopping TRACE requires LoRA so the frozen base model is available")

        score_batch, state_counts = build_trace_score_batch(
            batch,
            self.tokenizer,
            max_sequence_length=int(config["max_sequence_length"]),
        )
        reference = self._compute_ref_log_prob(score_batch)
        target_mask = score_batch.batch["response_mask"]
        mean_log_probs = (
            reference.batch["ref_log_prob"] * target_mask
        ).sum(dim=-1) / target_mask.sum(dim=-1)
        batch.non_tensor_batch.pop("trace_target", None)
        trace_metrics = apply_trace_advantages(
            batch,
            mean_log_probs,
            state_counts,
            epsilon=float(config["epsilon"]),
            horizon=int(config["horizon"]),
            discount=float(config["discount"]),
            terminal_weight=float(config["terminal_weight"]),
            outcome_weight=float(config["outcome_weight"]),
            turn_weight=float(config["turn_weight"]),
        )
        trace_metrics["trace/target_log_prob_mean"] = float(mean_log_probs.mean())
        output = original_update_actor(self, batch)
        output.meta_info["metrics"].update(trace_metrics)
        return output

    update_actor_with_trace._shopping_trace = True
    RayPPOTrainer._update_actor = update_actor_with_trace


def _seed_runtime(seed):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def install_torch_padding_fallback():
    """安装项目所需的 veRL 窄范围 runtime hooks。"""
    import os

    from verl.utils import attention_utils
    from verl.utils import npu_flash_attn_utils as fallback

    from shopping_grpo.training.grpo.checkpointing import install_selective_checkpointing
    from shopping_grpo.training.grpo.fused_output import install_fused_output_layout
    from shopping_grpo.training.grpo.kernel_optimizations import install_actor_fusions
    from shopping_grpo.training.grpo.validation import install_validation_metrics

    functions = (
        fallback.index_first_axis,
        fallback.pad_input,
        fallback.rearrange,
        fallback.unpad_input,
    )
    # ponytail: veRL 0.8 在 CUDA 上硬导入 FA2；上游提供 torch fallback 后删除此 hook。
    attention_utils._get_attention_functions = lambda: functions
    _install_trace_actor_update()
    _install_bounded_pilot()
    install_validation_metrics()
    install_fused_output_layout()
    install_selective_checkpointing()
    install_actor_fusions()
    _install_rollout_wakeup_cache_cleanup()
    seed = int(os.environ.get("SHOPPING_GRPO_SEED", "42"))
    _seed_runtime(seed)
