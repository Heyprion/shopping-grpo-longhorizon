# Data

The current data decisions are in [the data plan](../docs/data-plan.md). Cleaning, complexity labels, SFT-versus-RL/test ownership, and the RL/test selections are fixed. **SFT stage-one-v1 is finalized: 988 formatted trajectories, with 889 training and 99 validation examples (seed 42).** Only the first-stage context/legality filters are applied; stage two is deferred.

| Purpose | Current source/list | Current target |
|---|---|---:|
| Unified SFT collection | `sft/candidate_tasks.jsonl`: 5,000 candidates | 1,000 accepted trajectories |
| RL training | `grpo/train.jsonl` / `train.parquet` | 1,000 tasks |
| RL validation | `grpo/validation.jsonl` / `validation.parquet` | 50 tasks |
| Test | `evaluation/tasks.jsonl` | 200 tasks |

SFT uses at most three rejection-sampling attempts per candidate and stops at the first success, accepts solely actual terminal Strict Reward=1, and retains one trajectory per task. Its final accepted mixture is simple/medium/hard=300/500/200 with natural category margins across nine domains. See [`sft/README.md`](sft/README.md) and [`collection_plan.json`](sft/collection_plan.json).

The old 4,500/500 SFT partitions and 800/200 task files remain provenance records only. Both SFT source chunks are now merged for collection; they are not current training/validation assignments. The current final memberships are in `manifest.sft_final` and `sft/final_*_tasks.jsonl`. Current conversation files have been replaced with this new Strict Reward dataset; replaced files are backed up under `outputs/sft-stage1-v1/previous/`.

The 16,741 passing tasks are in [`cleaned/`](cleaned/README.md), with labels and provenance in [`splits/`](splits/README.md). The entire original-eval pool and its cross-tag instruction duplicates remain held out. RL has 9,494 additional unselected training tasks, accessible through `prepare_data_splits.py extend --stage rl`; its validation and test lists remain unchanged.

This update does not launch collection, quality screening, splitting, training or evaluation.
