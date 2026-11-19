# Shopping GRPO

<div align="center">

**English** · [简体中文](README.md)

<br />

Reproducible post-training and evaluation for long-horizon shopping agents

<br />

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![LoRA SFT](https://img.shields.io/badge/Post--training-LoRA%20SFT-7B61FF)](docs/sft.md)
[![veRL](https://img.shields.io/badge/veRL-0.8.0-0E8A16)](https://github.com/verl-project/verl)
[![ShopSimulator](https://img.shields.io/badge/Environment-ShopSimulator%20v2.1-4C78A8)](https://arxiv.org/pdf/2601.18225)
[![Benchmark](https://img.shields.io/badge/Benchmark-Final--200--Clean-F59E0B)](docs/evaluation-dataset.md)

<br />

Teacher rollouts and LoRA SFT → online GRPO with veRL → auditable comparison on
a frozen benchmark

</div>

![Shopping GRPO project overview](docs/images/project-overview-pipeline.png)

## What is ShopSimulator?

[ShopSimulator](https://arxiv.org/pdf/2601.18225) is a large-scale Chinese
shopping environment for evaluating long-horizon LLM agents. A task describes
what a user wants—including category, budget, brand, model, functions and
product options—but the agent must discover the right item through interaction.

In this project the agent can search products, open candidates, inspect details,
select variants, buy, or stop when no acceptable item can be verified. Success
therefore requires more than producing a plausible answer: the agent must gather
evidence, obey constraints, choose the correct variant and terminate correctly.

The frozen Environment v2.1 source and product archive are embedded under
[`environments/ShopSimulator/`](environments/ShopSimulator/), so the tutorial
does not depend on a separately running third-party repository.

![ShopSimulator overview](docs/images/shopsimulator-overview.png)

## The four stages

| Stage | What happens | Entry point | Details |
|---|---|---|---|
| Baseline | Evaluate the untouched base model | `bash scripts/baseline.sh` | [Evaluation](docs/evaluation.md) |
| SFT | Learn tool use from accepted teacher trajectories | `bash scripts/sft.sh` | [SFT](项目文档/规范与方案/SFT第一版训练指南.md) |
| GRPO | Optimize terminal Strict Reward with online rollouts | `bash scripts/grpo.sh` | [GRPO](项目文档/规范与方案/GRPO训练与配置C指南.md) |
| Evaluation | Run the curated Final-200 Clean protocol | `bash scripts/evaluate.sh NAME` | [Evaluation](docs/evaluation.md) |

The checked-in SFT data was produced by a separate collection stage documented
in [Data collection](项目文档/规范与方案/SFT统一采集方案.md). The custom constraint-aware
reward is specified in [Strict Reward](项目文档/规范与方案/Strict_Reward奖励规范.md).

```mermaid
flowchart LR
    A[Teacher rollouts] --> B[Strict Reward acceptance]
    B --> C[Action-only SFT data]
    C --> D[LoRA SFT]
    D --> E[Online GRPO with veRL]
    F[Frozen ShopSimulator v2.1] --> E
    G[Final-200 Clean held-out tasks] --> H[Shared evaluation pipeline]
    I[Baseline] --> H
    D --> H
    E --> H
```

### How the SFT data was collected

The current collection used `deepseek-v4-flash` as a teacher in ShopSimulator
Environment v2.1. It produced 2,498 raw trajectories, of which 1,026 passed the
strict acceptance filter. This frozen revision uses 1,000 trajectories split
into 800 training and 200 validation rows. SFT, GRPO and Final-200 Clean task IDs are
pairwise disjoint. Dataset hashes and the audit are in
[Data collection](项目文档/规范与方案/SFT统一采集方案.md).

The resumable collection entry point is:

```bash
python scripts/collect_sft_data.py \
  --tasks data/grpo/train.jsonl \
  --output-dir outputs/sft-collection \
  --target-accepted 1000 \
  --workers 4
```

### How GRPO is trained

GRPO starts from the merged SFT model. veRL generates four online trajectories
per prompt in ShopSimulator. The paper's Strict Reward multiplies category,
attribute, option and price scores. Non-purchase outcomes receive zero; fully
matching alternatives and gold items both receive one. See
[Strict Reward](项目文档/规范与方案/Strict_Reward奖励规范.md) for the contract and migration. No additional
LLM-as-a-Judge reward model is used for training.

The repository pins `verl==0.8.0` instead of copying its source. It keeps only
the project-specific AgentLoop, tool adapter, runtime compatibility code and a
small SHA-256-checked patch. See the [GRPO guide](项目文档/规范与方案/GRPO训练与配置C指南.md) for details.

### How evaluation works

Formal evaluation combines deterministic checks with two LLM-as-Judge roles.
DeepSeek V4 Flash curates a frozen requirement Rubric from code-generated
category, brand, model, function, option and price candidates. It may select and
deduplicate candidates, but cannot invent fields or expected values. The same
Rubric is shared by Baseline, SFT and GRPO.

After each Actor completes a rollout, code normalizes events and computes
Reward, legality, repetition, context and infrastructure checks. Valid
trajectories then go to DeepSeek V4 Pro with the original Query, frozen Rubric,
Actor-visible trajectory, neutral terminal flags and allowlisted behavioral
metrics. Reward values, hidden Gold fields, raw observations, success labels and
other models' results are excluded.

```mermaid
flowchart TD
    A[Benchmark task ID] --> B[Private TaskFacts]
    B --> C[Code-generated candidates]
    C --> D[V4 Flash frozen Rubric]
    A --> E[Actor rollout]
    E --> F[Normalization and hard checks]
    F -->|valid| G[Judge-safe payload]
    D --> G
    G --> H[V4 Pro requirement and five-dimension judgment]
    F -->|infrastructure invalid| I[not_judged]
    H --> J[Four-panel aggregation]
    I --> J
    J --> K[Paired Baseline / SFT / GRPO comparison]
```

V4 Pro scores Search Strategy, Candidate Utilization, Evidence Verification,
Decision Quality and Termination Efficiency independently on a 0/1/2 scale. It
also assesses each Rubric and assigns errors from a frozen taxonomy. The final
report keeps Reward/terminal, requirement Rubric, trajectory quality and
deterministic behavior as four separate panels—there is no composite score.
Failed, missing and not-judged tasks remain in the Final-200 Clean denominator. The
[evaluation guide](docs/evaluation.md) contains the complete prompts, one
worked benchmark example, input-isolation rules and aggregation contract. An
historical static view is available in the [Final-200 Benchmark Dashboard](docs/evaluation-dashboard.html); the current task set is documented in [Final-200 Clean](项目文档/规范与方案/Strict-Stratified-200测试集说明.md).

> **Reserved figure — Training and evaluation pipeline.** A full-width diagram
> showing teacher data collection, LoRA SFT, online GRPO rollouts and the shared
> held-out evaluation path, with artifacts produced at each boundary.

## Results

A contributor reproduction is now available on the current Final-200 Clean
benchmark. See the [evaluation update log](docs/evaluation-updates.md) for the
full protocol, failure distribution and artifact hashes:

| Model | Strict success | Purchase success | Done rate | Mean reward |
|---|---:|---:|---:|---:|
| Qwen3.8-27B (BF16 weights, thinking disabled) | 73.0% | 73.0% | 99.5% | 0.6354 |

The following are archived Final-200 results; new comparisons use Final-200 Clean:

| Model | Strict success | Purchase success | Mean reward |
|---|---:|---:|---:|
| Qwen3.5-2B baseline | 0.0% | 0.0% | -0.1105 |
| LoRA SFT | 60.5% | 60.5% | 0.4729 |
| GRPO step 100 | 62.0% | 62.5% | 0.5158 |

The complete compact summaries and reproduction settings are in
[`experiments/`](experiments/). These are reported results, not a promise that
different hardware or dependency versions will produce bit-identical training.

## Training hardware and time

All training used a single NVIDIA RTX 6000 with 96 GB of GPU memory.

### LoRA SFT training (448 training examples, 3 epochs)

| Stage | Time | Peak GPU memory |
|---|---:|---:|
| One epoch (56 steps) | ~62 min | 89 GiB |
| Full 3-epoch training | ~3 h | 89 GiB |

### GRPO training (veRL 0.8, 8 environment workers)

| Step range | Per-step time | Cumulative time |
|---|---:|---:|
| steps 0–24 | ~140 s/step, including Ray startup | ~56 min |
| stable steps 20–30 | ~73–120 s/step | ~2 min/step in the steady state |
| 100 steps (reported checkpoint) | ~110 s/step on average | ~3–4 h |
| Full 500 steps | ~100 s/step | ~14 h |

### Other stages

| Stage | Estimated time |
|---|---:|
| Teacher collection (2,498 raw trajectories) | Depends on endpoint concurrency and rate limits |
| Final-200 Clean evaluation (Base) | ~20 min |
| Final-200 Clean evaluation (SFT/GRPO) | ~40–60 min |
| LLM Judge scoring for 200 trajectories | ~30–60 min |

## Requirements

- Linux with an NVIDIA GPU and a compatible CUDA driver;
- [`uv`](https://docs.astral.sh/uv/);
- about 25 GB of free disk for environments, weights and generated artifacts;
- approximately 48 GB GPU memory for the provided SFT recipe;
- one 96 GB GPU for the provided GRPO recipe.

The main environment uses Python 3.12. ShopSimulator is isolated on Python 3.10.
`uv` creates both environments. veRL is **installed as the pinned
`verl==0.8.0` dependency**; its source is not copied into this repository. Only
the Shopping Agent adapter and a small version-checked patch live here.

## Quick start

Run every command from the repository root.

### 1. Install

```bash
bash scripts/setup.sh
```

This installs the pinned SFT and GRPO dependencies, creates the isolated
ShopSimulator environment, verifies and expands the product archive, builds the
search index and applies the version-checked veRL patch.

### 2. Start ShopSimulator

Keep this terminal running:

```bash
bash scripts/start_environment.sh
```

The service listens on `http://127.0.0.1:5700`.

### 3. Evaluate the baseline

Start the base model server in a second terminal:

```bash
bash scripts/serve_model.sh Qwen/Qwen3.5-2B
```

Evaluate it in a third terminal:

```bash
bash scripts/baseline.sh
```

Stop the model server before training so it releases the GPU.

### 4. Train and evaluate SFT

```bash
bash scripts/sft.sh
```

This runs two-GPU FSDP2 with seed 42, learning rate `1e-4`, effective batch 8
and three epochs, with SwanLab monitoring. The adapter is saved to
`outputs/models/sft-lora`. Merge and evaluate separately after training:

```bash
.venv/bin/python scripts/merge_lora_adapter.py \
  --base-model Qwen/Qwen3.5-2B \
  --adapter outputs/models/sft-lora \
  --output outputs/models/sft-merged --bf16
bash scripts/serve_model.sh outputs/models/sft-merged
bash scripts/evaluate.sh sft
```

Stop the model server again before GRPO.

### 5. Train GRPO

First verify the frozen data and best SFT provenance and inspect the resolved configuration:

```bash
bash scripts/grpo.sh --dry-run
```

Then train and automatically evaluate:

```bash
bash scripts/grpo.sh
```

The launcher starts a 64-slot environment and runs configuration C (temperature/top-p
1.0/1.0, top-k -1, candidate difficulty sampling 1:3:6) with the selected two-GPU recipe
for 500 effective updates from `outputs/models/sft-stage1-v1-best-merged`.
It selects the highest strict success rate on the 50-task validation set
(earliest checkpoint on ties), merges the RL adapter, then tests that model
once on the fixed 200 tasks. See [GRPO](项目文档/规范与方案/GRPO训练与配置C指南.md).
The actor recomputes 20 of 24 decoder layers and keeps a per-GPU micro batch of 4.
Checkpoints and the merged best model are saved as `outputs/models/grpo-stage1-v2-config-c-cp20`
and `grpo-stage1-v2-config-c-cp20-best-merged`; logs and reports are in
`outputs/logs/grpo-stage1-v2-config-c-cp20/` and `outputs/evaluation/grpo-stage1-v2-config-c-cp20/`.

Generated checkpoints, rollouts and logs are written under `outputs/`, which is
ignored by Git.

## Strict Reward

The current objective is the four-factor product in
[ShopSimulator Eq. (4)](https://arxiv.org/html/2601.18225v1):

```text
R_strict = r_type × r_att × r_option × r_price
```

Attribute and option coverage may be fractional. Full gold and alternative
purchases both receive one; non-purchase terminals receive zero. There are no
brand/model weights, identity discounts, negative utilities or length penalties.
Budget extraction and semantic annotation checks remain separate requirements.
See the [Strict Reward change record](项目文档/规范与方案/Strict_Reward奖励规范.md).

## Repository map

```text
configs/                         current GRPO, AgentLoop and tool configuration
data/
  sft/                           800 train + 200 validation trajectories
  grpo/                          ready-to-train JSONL and veRL Parquet
  evaluation/                    curated Final-200 Clean held-out set
docs/                            pre-Nov 2026 upstream reference docs
项目文档/                        post-Nov 2026 project specs, reports, and diagnostics (Chinese)
environments/ShopSimulator/      embedded environment and product archive
experiments/
  baseline/                      baseline config and result summary
  sft/                           SFT config and result summary
  grpo/                          GRPO config and result summary
scripts/                         thin user-facing tutorial entry points
src/shopping_grpo/
  collection/                    Teacher acceptance and SFT data construction
  environment/                   HTTP client, tools, actions and observations
  training/sft/                  SFT dataset masking and collation
  training/grpo/                 veRL adapter, compatibility and sampling logic
  evaluation/                    hard checks, Rubrics, trajectory Judge and metrics
tests/                           focused unit, launcher and packaging checks
```

The project keeps focused checks for the CPU smoke path, offline trajectory
evaluation, GRPO launcher arguments, non-editable wheel installation, Reward
aggregation and the frozen environment manifest. Cleanup does not mean deleting
tests that protect the public workflow.

## Configuration

Most users only need these environment variables:

| Variable | Default |
|---|---|
| `BASE_MODEL` | `Qwen/Qwen3.5-2B` |
| `SHOPSIM_BASE_URL` | `http://127.0.0.1:5700` |
| `LLM_BASE_URL` | `http://127.0.0.1:8000/v1` |
| `SERVED_MODEL_NAME` | `shopping-agent` |
| `SFT_ADAPTER_DIR` | `outputs/models/sft-lora` |
| `SFT_PYTHON` | `.venv/bin/python` |
| `SFT_SWANLAB_PROJECT` | `shopping-grpo-longhorizon` |
| `SFT_SWANLAB_RUN_NAME` | `YYYY-MM-DD-SFT-第一版正式训练` |

Advanced Hydra overrides use the lower-level training-only entry point:

```bash
.venv/bin/python scripts/train_grpo.py --logger swanlab -- \
  trainer.total_training_steps=20 \
  trainer.save_freq=10
```

All stages log to the unified SwanLab project `shopping-grpo-longhorizon`, using
`group` to organize stages, `job_type` (`train` / `eval` / `data` / `probe`) to
separate training from inference/evaluation, date-prefixed run names, and concise
Chinese descriptions (see [SwanLab tracking specification](项目文档/规范与方案/SwanLab实验记录与命名规范.md)).

## Documentation

- **[Project documentation index (post-Nov 2026 specs & reports)](项目文档/README.md)**
- [Data plan and task pool splits](项目文档/规范与方案/数据清洗与任务池划分方案.md)
- [Data collection and dataset provenance](项目文档/规范与方案/SFT统一采集方案.md)
- [LoRA SFT](项目文档/规范与方案/SFT第一版训练指南.md)
- [SFT Stage-1 v1 Training Report](项目文档/正式报告/SFT正式第一版训练报告.md)
- [GRPO with veRL](项目文档/规范与方案/GRPO训练与配置C指南.md)
- [SwanLab tracking and naming specification](项目文档/规范与方案/SwanLab实验记录与命名规范.md)
- [Held-out evaluation](docs/evaluation.md)
- [Strict-Stratified-200 evaluation dataset](项目文档/规范与方案/Strict-Stratified-200测试集说明.md)
- [Strict Reward design and migration](项目文档/规范与方案/Strict_Reward奖励规范.md)
- [Auditable experiment results](experiments/comparison.md)

## Future improvement plan

To further improve planning, tool use, and constraint satisfaction on long-horizon shopping tasks, the project will pursue the following directions:

- **Scale up the model**: Train a larger `Qwen3.5-9B` model on top of the current experiments and evaluate how model scale affects long-horizon stability and end-task success rate.
- **Upgrade the Teacher models and trajectory collection**: Move beyond a single Teacher model by using a mixture of `Qwen3.8 Flash`, `GLM-5.3-Flash`, and `DeepSeek-V4-Flash-0731` to collect trajectories, improving the coverage, behavioral diversity, and quality ceiling of the training data.
- **Expand the independent validation set**: Increase the number and variety of validation tasks to cover more product categories, constraint combinations, and interaction paths, strengthening the statistical reliability and generalization assessment of the results.
- **Introduce finer-grained credit assignment**: Address sparse terminal rewards and the difficulty of distinguishing individual action contributions in long-horizon interactions by providing more effective training signals for important intermediate steps.

## References and acknowledgements

This tutorial builds on the
[ShopSimulator paper](https://arxiv.org/pdf/2601.18225) and source project,
[veRL](https://github.com/verl-project/verl), and
[Qwen](https://github.com/QwenLM/Qwen3).

The evaluation protocol and Benchmark construction were also informed by
[VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications](https://arxiv.org/pdf/2509.26490)
and
[EComAgentBench: Benchmarking Shopping Agents on Long-Horizon Tasks with Distributed Hidden Intent](https://arxiv.org/pdf/2606.17698).

The repository organization and tutorial presentation were informed by
[qiqihezh/agentic-grpo-longhorizon](https://github.com/qiqihezh/agentic-grpo-longhorizon).
Thanks to the [OpenCode Go plan](https://dev.opencode.ai/go) for supporting the
development workflow.

### Contributors

<a href="https://github.com/Guochangwei917">
  <img src="https://github.com/Guochangwei917.png?size=80" width="80px;" alt="Guochangwei917"/>
</a>
