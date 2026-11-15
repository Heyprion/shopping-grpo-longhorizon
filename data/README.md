# Data

The current task selections are fixed and audited. Read [the data plan](../docs/data-plan.md) and [the split index](splits/README.md) before launching training.

| Purpose | Fixed task list | Tasks | Remaining training pool |
|---|---|---:|---:|
| SFT training | `sft/train_tasks.jsonl` | 800 | 3,700 |
| SFT validation | `sft/validation_tasks.jsonl` | 200 | — |
| RL training | `grpo/train.jsonl` and `train.parquet` | 1,000 | 9,494 |
| RL validation | `grpo/validation.jsonl` and `validation.parquet` | 50 | — |
| Test | `evaluation/tasks.jsonl` | 200 | — |

All selections cover the nine original product domains. Training difficulty mixtures are 30/50/20 for SFT and 20/50/30 for RL; category margins follow the natural source-pool proportions. Validation and test difficulty margins also follow their natural source pools.

The 16,741 passing source tasks remain in [`cleaned/`](cleaned/README.md). Their per-task labels, fixed candidate pools, unused tasks and selection batches are in `splits/`. The complete 944-task original-eval pool and its three cross-tag instruction duplicates are held out from training. All active phase pools are disjoint.

SFT **task identities are ready**; new demonstrations have not been collected. The old `sft/train.jsonl`, `sft/validation.jsonl` and their metadata retain their original Reward-v3 provenance and cannot train this selection. The SFT launcher checks the reward version and exact fixed task membership before training. See [`sft/README.md`](sft/README.md).

Expand training using `scripts/prepare_data_splits.py extend`, which selects only previously unselected tasks from the same training pool and records a new batch. It preserves initial, validation and test lists. Nothing in this preparation starts training or evaluation.
