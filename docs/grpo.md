# GRPO with veRL

## Purpose

SFT teaches the action format and a strong initial policy. GRPO then samples
fresh trajectories in ShopSimulator and optimizes the terminal Reward v3 signal.
The goal is to improve constraint satisfaction and termination behavior without
requiring a learned reward model.

## Integration boundary

veRL is installed from the pinned `verl==0.8.0` package. This repository does
not vendor the veRL source tree. Project-owned integration code lives in:

```text
src/shopping_grpo/training/grpo/
  adapter/              AgentLoop and ShopSimulator tools
  compat.py             narrow runtime compatibility hook
  dynamic_sampling.py   bounded non-zero-reward sampling
  fused_output.py       per-sequence layout for the fused output head
  validation.py         strict Reward-v3 validation diagnostics
```

`scripts/setup.sh` applies one SHA-256-checked patch needed to connect the
bounded dynamic sampler to veRL 0.8.0. Setup fails rather than patching an
unknown veRL version.

## Inputs

- Initial policy: `outputs/models/sft-merged`
- Train set: `data/grpo/train.parquet` (1,000 tasks)
- Validation set: `data/grpo/validation.parquet` (50 tasks)
- Environment: ShopSimulator Environment v2.1
- Reward: Reward v3

Hashes are recorded in [`data/grpo/metadata.json`](../data/grpo/metadata.json).

## Run

Inspect the resolved command first:

```bash
bash scripts/grpo.sh --dry-run
```

Train:

```bash
bash scripts/grpo.sh
```

Important defaults:

| Setting | Value |
|---|---|
| Algorithm | GRPO |
| Training backend | FSDP2, one node / two GPUs |
| FSDP shard group | 2 GPUs, reshard after forward |
| Rollout tensor parallel size | 1 (two data-parallel replicas) |
| Rollouts per prompt | 4 |
| Rollout temperature / top-p | 0.7 / 0.9 |
| Train / validation batch | 2 / 2 |
| Policy learning rate | `1e-6` |
| LoRA rank / alpha | 16 / 32 |
| Maximum model length | 24,576 |
| Maximum training steps | 500 |
| Save / validation frequency | 50 / 50 |
| KL reward / KL loss | disabled / disabled |
| Policy entropy measurement | enabled (logging only) |

Dynamic sampling can generate at most three batches to find a useful update and
permits at most ten consecutive skipped updates. These bounds prevent an
all-equal reward batch from causing an unbounded resampling loop.

Each run also appends `training_diagnostics.jsonl` under its output directory.
`generation_batch` records contain every generated rollout, its public tool
sequence, terminal result, reward breakdown, Guard rejection reasons and group
keep/drop decision. `optimizer_step` records preserve the scalar veRL metrics,
including entropy, PPO KL, clip fractions, response lengths and effective-group
rates. `skipped_update` records make zero-signal attempts visible even though
they do not advance the optimizer step.

The canonical configuration is [`configs/grpo.yaml`](../configs/grpo.yaml).
An optional resource configuration keeps the same objective, rank 16 / alpha
32, learning rate `1e-6`, four rollouts per prompt, and eight trajectories per
optimizer update. It uses microbatch 2 per GPU, resident actor parameters and
optimizer state, no resharding after forward, and the Triton fused
log-probability/entropy head.
Gradient checkpointing remains enabled. The fused head avoids materializing
the full sequence-by-vocabulary tensor; it still measures full-vocabulary
entropy. The project hook preserves each padded sequence's token boundaries
in veRL 0.8.0's output layout. Padding removal remains disabled.

An additional optional resource setting uses actor microbatch 4 per GPU and
`SHOPPING_GRPO_CHECKPOINTED_LAYERS=18`. With two tasks and four rollouts per task,
the effective batch remains eight and accumulation becomes one. The environment
setting retains HF recomputation on 18 of 24 decoder layers, preserving RNG for
dropout; it is applied before FSDP wrapping. Unset it to keep full recomputation.
It changes memory and computation, not the reward or learning-rate schedule.
Short trajectories can still use less memory than a longest-context stress run.

`SHOPPING_GRPO_FAST_KERNELS=fla_conv` optionally uses the installed FLA Triton
causal convolution in the actor. It retains the separate BF16 convolution output
and SiLU rounding boundary, requires `use_remove_padding=false`, and leaves the
PPO head, RMSNorm, rollout implementation and learning parameters unchanged.
Unset the variable to use the original PyTorch convolution. General
`model.use_liger` is independent of this narrower option.

`SHOPPING_GRPO_FAST_KERNELS=fla_conv,liger_norm` additionally enables Liger
RMSNorm on the actor's Qwen3.5 normalization modules. It uses the installed
Qwen3.5 adapter's `offset=1.0`, `casting_mode="gemma"`, `in_place=false`
settings without changing HF classes globally. Keep `model.use_liger=false`
to select Norm separately from Liger MLP and loss fusion. Pass the same kernel
selection as a quoted string to Ray's `runtime_env.env_vars` so worker
processes receive it. Fixed-input Norm-only measurements reduced actor compute
time by about 11%, with gradient relative L2 differences of 1.37–1.85%; these
are numerical differences, not measured changes in success rate. Combined
kernel speed and long-run training quality have not been measured.

This configuration validates all 50 tasks in one batch. veRL pads that to 56 concurrent
rollouts with eight agent workers, so start the environment with 64 slots:

```bash
SHOPSIM_ENV_SLOTS=64 bash scripts/start_environment.sh
```

Use the supported launcher and explicitly select the intended merged SFT:

```bash
.venv/bin/python scripts/train_grpo.py \
  --model outputs/models/sft-balanced-best-merged \
  --output outputs/models/grpo-balanced \
  --logger swanlab --experiment-name grpo-balanced-seed42 -- \
  data.seed=42 data.val_batch_size=50 data.dataloader_num_workers=0 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.seed=42 \
  actor_rollout_ref.model.use_fused_kernels=true \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=false
```

The launcher uses the existing SwanLab login. Each validation also appends
`validation_diagnostics.jsonl`; `val-core/shopsimulator/acc/mean@1` requires
a complete, valid `gold_purchase` terminal result. Validation metrics do not
change the training reward. Resource short tests cannot establish final
500-update quality or the maximum safe memory use of every long trajectory.

Both the actor strategy and its engine strategy select `fsdp2`; the reference
configuration follows the same two-GPU shard group. Run the prerequisite check
before training:

```bash
.venv/bin/python scripts/check_fsdp2_runtime.py --distributed
```

GRPO still requires a completed SFT merged model. For curriculum SFT, pass
`--model outputs/models/sft-curriculum/stage-c/merged` to `scripts/grpo.sh`.
The default path remains the output of `scripts/sft.sh`.

The environment manifest includes the repository's `d99a0ac` correction removing
an undefined item-page argument; the runtime hash reflects that committed fix.
Advanced overrides may be appended after `--`:

```bash
bash scripts/grpo.sh -- \
  trainer.total_training_steps=20 \
  trainer.save_freq=10
```

## Export

veRL checkpoints are not directly served by the evaluation launcher. Export the
selected actor:

```bash
bash scripts/export_grpo.sh \
  outputs/models/grpo/global_step_100/actor \
  outputs/models/grpo-merged
```

The reported comparison uses step 100. Select checkpoints using validation
metrics rather than assuming that the final training step is best.
