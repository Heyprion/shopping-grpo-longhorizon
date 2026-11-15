# Documentation

Follow the guides in workflow order:

1. [Current data plan](data-plan.md) is the authority for completed full-corpus
   cleaning, materialized stratification and pools, fixed initial task lists,
   and provisional SFT/RL sizes and difficulty mixtures. The unused-task ledger
   supports later expansion without reusing selected tasks.
2. [Task cleaning](task-cleaning.md) explains the two-stage implementation and
   canonical budgets shared with Strict Reward.
3. [Data collection](data-collection.md) explains how the checked-in SFT data
   was produced and audited.
4. [SFT](sft.md) trains the first useful shopping agent.
5. [GRPO](grpo.md) improves that model with online environment reward.
6. [Evaluation](evaluation.md) compares baseline, SFT and GRPO fairly.
7. [Strict-Stratified-200 evaluation list](evaluation-dataset.md) documents the
   fixed test selected from the original eval-tag pool.

[Strict Reward](reward-strict.md) is the detailed specification shared by collection,
GRPO and evaluation.
