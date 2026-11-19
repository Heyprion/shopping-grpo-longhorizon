# GRPO with veRL

SFT provides the initial policy; GRPO optimizes fresh ShopSimulator trajectories.
The runtime is Environment v2.1, observation/tool schema v2, and Strict Reward
`r_type * r_att * r_option * r_price`. The data boundaries and selected recipe
in [AGENTS.md](../../AGENTS.md) and [the data plan](数据清洗与任务池划分方案.md) take precedence over
bare YAML defaults.

## Train and evaluate

Run from the repository root in the training terminal. Inspect the complete
resolved configuration first:

```bash
bash scripts/grpo.sh --dry-run
```

Start the full workflow:

```bash
bash scripts/grpo.sh
```

The launcher performs these steps automatically:

1. Verify the frozen split artifacts, final SFT ownership, RL Parquet files and
   held-out isolation using `prepare_data_splits.py verify`. Verify that the
   initial merged model comes from the validation-best completed SFT adapter.
2. Start its own ShopSimulator service with 64 slots on port 15705. Every
   environment process and Ray worker receives the canonical
   `data/cleaned/task_budgets.jsonl` path.
3. Run configuration C (temperature/top-p 1.0/1.0, top-k -1) and the selected two-GPU recipe from
   `outputs/models/sft-stage1-v1-best-merged`, with a new rank-16/alpha-32 LoRA.
   The low-level trainer checks installed runtime versions before loading weights.
4. After 500 consecutive effective updates, select the saved checkpoint with
   the highest `val-core/shopsimulator/acc/mean@1` on the fixed 50 validation tasks;
   ties select the earlier step. Initial step-0 SFT validation is a reference,
   not a saved RL candidate. All ten scheduled checkpoints are retained.
5. Export that actor's FSDP shards. veRL exports frozen base weights and an
   RL `lora_adapter` separately, so merge the adapter into the exported base
   before starting the model server on port 18005.
6. Run the fixed 200-task evaluation once, then generate the HTML report and
   `final_report.json`. Test results do not select checkpoints or tune parameters.

The launcher stops if training fails, updates or validation records are incomplete,
checkpoints are missing, or data change during training. It closes the services
it started on completion, failure or interruption. It does not stop other jobs.
The dry run performs read-only validation and configuration composition; it does
not start services, Ray, training, model export or evaluation.

Existing nonempty output directories are rejected. To run a separate experiment,
choose a new name; the merged model, log and evaluation names follow it:

```bash
bash scripts/grpo.sh --output outputs/models/grpo-stage1-v2-config-c-cp20-02
```

Use `--environment-port` and `--llm-port` if the default ports are occupied.
`CUDA_VISIBLE_DEVICES` defaults to `0,1` and must expose two GPUs. The final
model server uses the first of these GPUs. Launch in a persistent terminal
session if the remote terminal may disconnect.

## Selected recipe

| Setting | Value |
|---|---|
| Data | Frozen 1,000-task train pool / 50 validation; fixed 200-task final test |
| Candidate task sampling | Simple/medium/hard 1:3:6; preserve all nine category marginals |
| Seed | 42, including vLLM and Ray worker settings |
| Training | One node, two GPUs, FSDP2, BF16 mixed precision / SDPA |
| LoRA | Fresh rank 16 / alpha 32, all-linear |
| Updates / LR / warmup | 500 effective updates / `1e-6` / 15 steps |
| Task batch / PPO mini batch | 2 / 2 |
| Group / per-GPU micro batch / accumulation | 4 / 4 / 1; 8 effective trajectories |
| Actor parameters / optimizer offload | Both disabled |
| Reshard after forward | Disabled |
| Recomputed decoder layers | 20 of 24, preserving RNG |
| Actor fast kernels | FLA convolution, Liger Norm, Triton fused PPO output head |
| General Liger / padding removal / dynamic batch | Disabled |
| Rollout | Async vLLM, TP1 × 2 replicas |
| Rollout memory / max sequences | 0.45 / 8 |
| Train sampling | Configuration C: temperature 1.0 / top-p 1.0 / top-k -1 |
| Maximum sequence | 24,576 tokens |
| GRPO advantage | Group mean subtraction, no standard deviation normalization |
| PPO clip | 0.2 |
| KL reward / KL loss / entropy reward | Disabled; entropy diagnostics retained |
| Validation | Before training, then every 50 updates; batch 50 |
| Save | Every 50 updates, retain all candidates |
| Logging | SwanLab online plus console / JSONL diagnostics |

`SHOPPING_GRPO_CHECKPOINTED_LAYERS="20"` and
`SHOPPING_GRPO_FAST_KERNELS="fla_conv,liger_norm"` are set both in the launcher
environment and as strings in Ray's `runtime_env.env_vars`. Allocator overrides
are unset to match the selected RL command. The complete combination's end-to-end
speed and long-run quality have not yet been measured.

For each update, dynamic sampling accumulates valid groups with unequal rewards
and distinct instructions while the rollout policy stays fixed. It samples at
most 30 generation batches total. If it still cannot assemble the required two
groups, it records `sampling_exhausted` and stops before an optimizer update.

## Outputs

| Artifact | Default path |
|---|---|
| Checkpoints and training/validation diagnostics | `outputs/models/grpo-stage1-v2-config-c-cp20/` |
| Validation selection and all candidate scores | `outputs/models/grpo-stage1-v2-config-c-cp20/model_selection.json` |
| Exported base and RL adapter | `outputs/models/grpo-stage1-v2-config-c-cp20/best-export/` |
| Standalone best RL model | `outputs/models/grpo-stage1-v2-config-c-cp20-best-merged/` |
| Configuration, provenance, training/service/export/evaluation logs | `outputs/logs/grpo-stage1-v2-config-c-cp20/` |
| Final test trajectories, summary, final report and HTML | `outputs/evaluation/grpo-stage1-v2-config-c-cp20/` |

Training and final testing both log to the unified SwanLab project
`shopping-grpo-longhorizon`, separated by `group` (`06-GRPO正式训练` /
`05-GRPO工程摸底与排障` vs `07-GRPO评测`) and `job_type` (`train` vs `eval`),
with concise Chinese descriptions (see [SwanLab 实验记录与命名规范](SwanLab实验记录与命名规范.md)).
Names use the actual start date and stage, for example
`YYYY-MM-DD-RL-配置C显存调整` and `YYYY-MM-DD-EVAL-配置C显存调整最佳模型`.
Strict success requires a complete, valid `gold_purchase` terminal result.

## Integration

The supported dependency is pinned `verl==0.8.0`; the repository does not vendor
veRL. Project integration lives in `src/shopping_grpo/training/grpo/`. Setup
applies the SHA-256-checked bounded sampling patch and refuses unknown versions.
`training_diagnostics.jsonl` records generated groups, skipped attempts and
optimizer metrics; `validation_diagnostics.jsonl` preserves validation rollouts.

`scripts/run_grpo_pipeline.py` orchestrates the workflow above through the existing
`scripts/train_grpo.py`, `scripts/export_grpo.sh`, `scripts/merge_lora_adapter.py`
and evaluation tools. Advanced Hydra overrides belong to the low-level
`train_grpo.py` entry point; it only trains and does not manage services or final
model evaluation. `configs/grpo.yaml` remains the base configuration; the pipeline
applies the explicitly selected overrides and saves the resolved configuration.

## Frozen SFT sampling study before RL

Prepare and inspect the bounded study without running models:

```bash
bash scripts/probe_rl_sampling.sh --dry-run
```

Run it in a separate terminal:

```bash
bash scripts/probe_rl_sampling.sh
```

The default total wall-clock budget is 90 minutes, including data preflight,
service/model startup and cleanup. Set `--budget-minutes 60` for a shorter bound;
values above 120 are rejected. Completed results survive a timeout or interruption.
The study exits as soon as its planned observations finish; it does not fill the
remaining time with extra trials.

The study freezes `sft-stage1-v1-best-merged` and uses native veRL standalone
vLLM TP1 × two replicas plus `ShoppingToolAgentLoop`. It never constructs an actor,
optimizer, new LoRA or model merger. Reward, observations, tools, token budgets
and prompts match the RL path. All environment processes use canonical task
budgets. The study leaves the training limit of 30 generation batches per update in place.

The fixed seed-42 plan uses only the current 1,000 RL training tasks, never RL
validation or any held-out test tasks. The task manifests and registry are read
without allocating or replacing data. The panels are deterministic, have no
shared instruction groups, and cover all nine domains:

| Phase | Tasks | Sampling | Maximum trajectories |
|---|---|---|---:|
| Shared warmup, excluded from analysis | 2 | Group 4 | 8 |
| Screen | 48: 16 simple / 16 medium / 16 hard | Same tasks under A/B/C, group 4 | 576 |
| Fresh training-task confirmation | 24: 8 / 8 / 8 | Locked candidate vs A; group 4 | 192 |
| Conditional fallback | Same 24 confirmation tasks | Only selected setting, native group 8 | 192 |

A/B/C are temperature/top-p `0.7/0.9`, `1.0/0.9`, `1.0/1.0`; top-k stays `-1`.
Sample-specific streams derive from master seed 42 and task/sample/turn index; the
same indices match across settings. Settings rotate within matched task pairs
so initial warmup and run order do not systematically favor an arm. Every task
is sampled in full; successful trajectories do not stop subsequent samples.

Difficulty mixtures `20/50/30`, `10/40/50`, and `10/30/60` are estimated by
reweighting the balanced panel, without extra model calls. They represent task
sampling probabilities and do not replace the frozen 1,000-task list. Selection
ranks valid reward-varying groups per sampling minute, with an invalid-group ceiling of
`max(10%, baseline+5 percentage points)`; gains below 5% retain the unchanged
recipe. These are predeclared diagnostic rules, not claims of optimal RL quality.
The candidate is locked before confirmation on fresh training tasks. Group 8 is
only tested when the confirmed effective-group rate is below 50% or its sampling
throughput is below baseline, and enough time remains according to measured
sampling cost. It is a fallback, not another full parameter sweep.

Raw Strict Reward differences determine learning signal, including fractional
scores. Strict gold success is reported separately. Whole-dataset success near
50% cannot replace per-task success histograms or group reward variance.
Timeouts and exceptions are recorded as invalid, never relabelled as valid
zero-reward trajectories.

The maximum is 968 trajectories, including warmup. Outputs are under
`outputs/pre-rl-sampling/<UTC timestamp>/`: `plan.json`, `panels.json`,
`resolved_config.json`, `console.log`, `environment.log`, `trajectories.jsonl`,
`groups.jsonl`, `summary.json`, `report.md` and `recommendation.json`. SwanLab
reuses `shopping-grpo` with an actual-date `RL-SFT采样诊断` run name. Partial
matched panels remain visibly incomplete, and small or overlapping confidence
intervals are labelled inconclusive. Recommendations never start training or
modify the frozen data/configuration.

Native group-8 sampling timing is reported on the same confirmation task IDs.
It excludes actor update and weight synchronization. At two tasks per update,
group 8 means 16 trajectories and accumulation 2 at microbatch 4 per GPU; a
sampling-rate improvement alone does not establish a full-training speedup.

For the synchronization-fixed 50-update experiment, use two tasks per update
and the fixed 30-task SFT-failure evaluation:

```bash
bash scripts/grpo.sh --pilot-steps 50
```

This uses configuration C and the best SFT source, preserving the 500-step
optimizer schedule and 15-step warmup while stopping at 50 effective updates.
Training and PPO mini-batches stay at two tasks, group4, micro4 per GPU and
accumulation1: 400 optimized trajectories. Batch4 experiments are no longer used.

The evaluation is fixed in `data/grpo/pilot_eval30.jsonl` and `.parquet`.
All 30 tasks failed in the saved current SFT test result
(`outputs/evaluation/sft-stage1-v1/summary.json`): zero strict successes.
Seed42 selects from those failures with simple/medium/hard quotas 3/18/9,
matching the complete 200-task test distribution 21/119/60 after rounding.
Selection metadata is in `data/grpo/pilot_eval30_metadata.json`. The user has
authorized this test-failure subset for short-experiment feedback; see
`docs/data-plan.md`. It is only evaluated, never added to gradient training.

The pipeline automatically evaluates these same 30 tasks before training and
after step50, saves the checkpoint, and writes success counts, solved task IDs,
and gains/losses against the current run's actual baseline to
`outputs/models/grpo-stage1-v3-syncfix-pilot50-eval30/validation_report.json`.
It also records the saved SFT reference of 0/30. Logs, `resolved_config.json`
and `run_contract.json` are under
`outputs/logs/grpo-stage1-v3-syncfix-pilot50-eval30/`. SwanLab remains enabled.
Use `--output` for another run. The original 50-task validation and complete
200-task test files remain available for the full pipeline.

For the learning-rate comparison on the same 30-task subset:

```bash
bash scripts/grpo.sh --pilot-steps 50 --learning-rate 3e-6
```

This starts from the same best SFT source with batch2, group4, micro4,
accumulation1, 15-step warmup and 50 effective updates (400 optimized
trajectories). Only the peak learning rate changes to 3e-6. Baseline and final
validation, the checkpoint and SwanLab logging remain automatic. The report is
`outputs/models/grpo-stage1-v3-syncfix-pilot50-eval30-lr3e-6/validation_report.json`;
logs use the same name under `outputs/logs/`. It does not resume either prior
pilot. Short experiments do not export/merge a model or run the complete
200-task test. Append `--dry-run` to inspect preparation without training.

The pilot also supports an explicit first-moment control while retaining the
second moment, two-task batches and the fixed 30-task evaluation:

```bash
bash scripts/grpo.sh --pilot-steps 50 --learning-rate 3e-5 --adam-beta1 0 \
  --output outputs/models/grpo-pilot50-lr3e5-beta0-repeat
```

This is the current candidate for a fresh repeat from SFT, using Adam betas
`[0, 0.999]` and LR `3e-5`. One completed 50-update run improved the fixed
30-task diagnostic score from 2 to 7, with five gains and no losses; LR `3e-6`
with the same betas gave 1 to 1. Two-task batches, group4, micro4, accumulation1,
warmup15 and the original 1,000-task source remain unchanged. This is a short
diagnostic result, not a complete 200-task test gain or a default change to the
500-update production recipe. Check reproducibility and retention on the
original 50-task validation before extending the training budget.

To run the current candidate for 100 effective updates from the same best SFT:

```bash
bash scripts/grpo.sh --pilot-steps 100 --learning-rate 3e-5 --adam-beta1 0
```

This is a fresh 100-update run, using batch2/group4/micro4/accumulation1,
warmup15 and the same 500-update scheduler plan as the 50-update comparison.
It optimizes 800 trajectories and automatically evaluates the fixed 30-task
subset at steps0,50,100, saving both step50 and step100 checkpoints. SwanLab
logging remains enabled. The output is
`outputs/models/grpo-stage1-v3-syncfix-pilot100-eval30-lr3e-5-beta0/`;
logs and resolved configuration use the same name under `outputs/logs/`.
`validation_report.json` reports all three snapshots, final-step gains/losses,
and the final step100 checkpoint. Its `model_selection` field separately
records the best checkpoint, which can be step50. The step100 result is never
replaced by the best score. As with the 50-update pilot, this mode does not
export/merge a model or run the complete 200-task evaluation.

To evaluate the saved step100 checkpoint on the formal 200-task benchmark:

```bash
TMPDIR="$PWD/outputs" bash scripts/grpo.sh --evaluate-checkpoint \
  outputs/models/grpo-stage1-v3-syncfix-pilot100-eval30-lr3e-5-beta0/global_step_100/actor
```

The launcher exports and merges the specified actor in temporary storage, then
runs the formal evaluator and prints the Strict success percentage and count.
It saves the summary, HTML report and logs under `outputs/evaluation/` and removes
its temporary model copies after the run. Use `--output` to name a new evaluation
directory for a repeat.

To resume the best-recipe run at step100:

```bash
bash scripts/grpo.sh \
  --resume-from outputs/models/grpo-stage1-v3-syncfix-pilot100-eval30-lr3e-5-beta0/global_step_100 \
  --stop-at-step 200 \
  --output outputs/models/grpo-step200-lr3e-5-beta0-continuous
```

The runner restores the actor, optimizer, scheduler and data loader state from
step100. Accepted groups accumulate continuously for up to 30 generation
batches per update, using the same policy. The runner checks for all 100 new effective updates
before exporting step200 and evaluating the fixed 200 tasks. The terminal
prints the Strict success rate and the verified Reward=1 count. Results go to
`outputs/evaluation/grpo-step200-lr3e-5-beta0-continuous/`.

The configuration-C task sampler is installed through the existing runtime hook
at the trainer's data-loader boundary. It replaces only the training sampler,
including the sampler passed by the native TaskRunner. The source 1,000 task IDs
remain frozen. Each 1,000-draw candidate epoch has 100/300/600
simple/medium/hard draws and the exact original nine-category counts, fitted
with joint category/difficulty quotas. Within each cell, task IDs cycle through
seeded shuffled permutations; instruction duplicates cannot share a candidate
batch. Both sampler and iterator states are saved by StatefulDataLoader, and
restore rejects mismatched source membership or weights. Candidate epoch plans
are logged in `task_sampling_diagnostics.jsonl`; actual generated/used tasks
remain visible in the training diagnostics. Reward filtering can change the
mixture of tasks that ultimately reach the optimizer. No split, label, registry,
validation or test file is rewritten.
