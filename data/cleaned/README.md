# Cleaned task pool

The authoritative cleaning history, stratification and split plan, and
provisional SFT/RL quantities and difficulty mixtures are in
[the current data plan](../../docs/data-plan.md). The passing pool is fixed;
formal labels and all phase selections are materialized in
[`../splits/`](../splits/README.md).

This is the cleaned source task pool. All 23,421 original tasks were
screened, including the 200 previous evaluation tasks. There is no manual-review
queue. The original catalog is unchanged; the later stratification has replaced
the RL and test selections and fixed the new SFT task lists.

| Result | Tasks |
|---|---:|
| Passed both stages | 16,741 |
| Rejected by structure/reference checks | 625 |
| Rejected by feasibility/semantic checks | 6,055 |
| Total rejected | 6,680 |

The small model is Qwen3.5-4B, revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, with thinking disabled. Screening
uses `shopping-task-cleaning-v2` and Strict Reward. The full run took 42.6 minutes
on one RTX PRO 6000 Blackwell GPU with 128 concurrent requests. Inference was
local; GPU rental cost depends on the account tariff.

| File | Contents |
|---|---|
| `tasks.jsonl` | Passing tasks with original ID, ASIN, instruction, extracted requirements, selected options, gold price and canonical budget |
| `task_ids.jsonl` | Passing original task IDs |
| `task_budgets.jsonl` | Frozen budget annotations consumed by the environment |
| `rejected_tasks.jsonl` | Every rejected original task and its reason codes |
| `metadata.json` | Source, model, policy and file checksums, audit links and duplicate groups |

Every passing task was independently replayed through the static Strict Reward
oracle and produced a valid `gold_purchase` with all four factors equal to one.
All 26 automatic controls passed; the two known negative tasks were rejected.
The previous evaluation set contributed 160 passing and 40 rejected tasks.
Existing SFT and GRPO task IDs are also fully covered by this cleaning.

Rejection includes insufficient evidence, unsupported price semantics, uncertain
extraction and persistent model-generation failures. The automatic controls are
regression checks; they do not measure natural-task precision or recall. Static
feasibility does not run a shopping trajectory or establish search/step-limit
reachability. Full decisions and raw model responses remain under
`outputs/task-cleaning/full/`.

For an environment using this pool, set the budget annotation path from the
repository root before launching its process:

```bash
export SHOPPING_TASK_BUDGETS="$PWD/data/cleaned/task_budgets.jsonl"
```

The environment joins annotations by ASIN and exact instruction hash. Missing
annotations are unverifiable. The same decimal budget code is used in cleaning
and scoring; old regular-expression results are diagnostic only for this pool.

Train/validation/test task selections are fixed in `data/splits/manifest.json`.
SFT demonstration collection is a separate later step. Four pairs of passing
tasks have identical instructions; three cross-tag train tasks are quarantined
and the remaining train pair is assigned atomically to one candidate pool.
