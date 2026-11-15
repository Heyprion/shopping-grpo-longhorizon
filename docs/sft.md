# One-stage LoRA SFT

The current recipe trains `Qwen/Qwen3.5-2B` directly on
`data/sft/train.jsonl` and scores `data/sft/validation.jsonl` after each epoch.
Only assistant tokens contribute to the loss. User and tool-observation tokens
are masked. Training uses a single stage and seed **42**.

The source split contains 800 training and 200 validation trajectories. At the
24,576-token limit, 799 training trajectories and all 200 validation trajectories
are retained. Overlength trajectories are dropped in full. The SFT splits and
`data/evaluation/tasks.jsonl` have disjoint task IDs.

## Start

Run these commands from the repository root:

```bash
# Print the training command without loading a model or starting training.
bash scripts/sft.sh --dry-run

# Start the selected one-stage, two-GPU recipe.
bash scripts/sft.sh
```

The launcher checks FSDP2 prerequisites and verifies that Liger and the FLA
linear-attention kernel are available before training. It saves a LoRA adapter
and training checkpoints. Model merging and Final-200 evaluation are separate
actions.

The prepared environment was tested with `liger-kernel==0.8.5`,
`flash-linear-attention==0.6.0`, `fla-core==0.6.0`, PyTorch 2.11.0,
Accelerate 1.14.0 and PEFT 0.20.0. The model revision is pinned to
`15852e8c16360a2fea060d615a32b45270f8a8fc` for the default Qwen model.

## Selected parameters

| Setting | Value |
|---|---|
| Parallelism | One node, two GPUs, FSDP2 |
| Precision / attention | BF16 / SDPA, with FLA linear attention |
| Liger fused loss | Enabled for training and loss-only validation |
| Epochs / seed | 3 / 42 |
| Learning rate / warmup | `1e-4` / `0.03` |
| Scheduler / gradient clipping | Linear / norm 1.0 |
| Per-device train / validation batch | 1 / 1 |
| Gradient accumulation | 4 |
| Effective batch | `2 GPUs × 1 × 4 = 8` |
| Maximum sequence length | 24,576 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Activation checkpointing | Enabled through FSDP2 |
| Loss normalization | Global supervised assistant-token count |
| Checkpoint policy | Save each epoch, retain three |

[`configs/fsdp2.yaml`](../configs/fsdp2.yaml) enables transformer-layer
sharding, resharding after forward and FSDP activation checkpointing. The
Trainer's separate gradient-checkpointing path stays disabled. Both the outer
model cache and the nested Qwen text cache are disabled during training.

## Balanced preset for two 96 GiB GPUs

```bash
bash scripts/sft.sh --balanced --dry-run
bash scripts/sft.sh --balanced
```

This preset keeps seed 42, learning rate `1e-4`, warmup `0.03`, the 24,576-token
limit, and effective batch 8 (`2 GPUs × microbatch 1 × accumulation 4`). It uses
LoRA rank 32, alpha 64 and dropout 0.05, with BF16, Liger and FLA still enabled.

[`configs/fsdp2-balanced.yaml`](../configs/fsdp2-balanced.yaml) retains unsharded
parameters after forward, avoiding a second all-gather before backward. The
trainer checkpoints alternate text decoder layers (12 of 24), preserving RNG
state for LoRA dropout during recomputation. Full-layer Accelerate checkpointing
is disabled so the two checkpointing paths do not overlap.
`PYTORCH_ALLOC_CONF=expandable_segments:True` limits allocator fragmentation from
variable sequence lengths. The dry-run command includes this environment setting.

The default output is `outputs/models/sft-lora-balanced`, and the SwanLab run name
is `sft-balanced-seed42`. The original preset remains available without
`--balanced`; both presets support the same output and monitoring environment
overrides. Inference services should release the GPUs before training.

The LoRA comparison uses ranks 8/16/32 and alpha/rank 1/2, with matching training
samples and the original 300-update learning-rate schedule. Initial screening
uses 20 updates per candidate; finalists are checked through one epoch. This is
a bounded recipe comparison, not a completed three-epoch convergence study or a
new shopping success-rate evaluation.

## Outputs and monitoring

```text
outputs/models/sft-lora/
  console.log
  checkpoint-100/
  checkpoint-200/
  checkpoint-300/
  adapter_config.json
  adapter_model.safetensors
  train_summary.json
  step_profile.jsonl
  validation_epoch_*.json
  swanlab/
```

Checkpoint step numbers above correspond to the current data and batch sizes.
SwanLab records configuration, loss, gradients, validation NLL, per-GPU peak
memory and token throughput in project `shopping-grpo-sft`.

Optional environment settings:

| Variable | Default |
|---|---|
| `SFT_PYTHON` | Repository `.venv/bin/python` |
| `SFT_ADAPTER_DIR` | `outputs/models/sft-lora` |
| `SFT_SWANLAB_PROJECT` | `shopping-grpo-sft` |
| `SFT_SWANLAB_RUN_NAME` | `sft-single-stage-seed42` |

To resume an interrupted run, use a checkpoint containing the adapter,
optimizer, scheduler and RNG states:

```bash
bash scripts/sft.sh \
  --resume-from-checkpoint outputs/models/sft-lora/checkpoint-100
```

For the balanced preset, include `--balanced` when resuming. FSDP2 restores the
gathered adapter through PyTorch's distributed state-dict loader so both ranks
receive the correct weights; Trainer restores optimizer, scheduler and RNG state.

The training launcher also accepts the existing `--fsdp2` flag; the selected
recipe already uses FSDP2 by default. Final-200 remains reserved for evaluation
after the recipe is frozen. Validation NLL measures supervised prediction;
shopping rollout success requires a separate evaluation.
