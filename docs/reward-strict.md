# Strict Reward

## Decision and implementation record

On 2026-11-15, at the user's request, the active reward changed from the local
Reward-v3 weighted utility to the multiplicative Strict Reward in
[ShopSimulator, Eq. (4) and Appendix C.3](https://arxiv.org/html/2601.18225v1).
The version is `shopsimulator-reward-strict-v1`. Environment v2.1, observation v2
and tool schema v2 retain their interfaces.

Full-corpus cleaning now uses `shopping-task-budget-v1`: the small model extracts
the user's budget once, and both screening and `r_price` use the same decimal
comparison code. The environment reads the frozen annotations through
`SHOPPING_TASK_BUDGETS`. This fixes divergence from the old language parser when
using the cleaned pool; changing the four-factor formula alone did not fix it.
The user explicitly included previous evaluation tasks in this cleaning and
has now repartitioned the resulting pool. See [the current data plan](data-plan.md)
and [task cleaning](task-cleaning.md).

```text
R_strict = r_type × r_att × r_option × r_price ∈ [0, 1]
```

| Factor | Calculation |
|---|---|
| `r_type` | Soft category coefficient using nonempty source-query equality, two shared category nodes, or title-keyword overlap above 0.2. Otherwise 0.5, reduced to 0.1 below 0.1 overlap, and 0 for zero overlap. |
| `r_att` | Fraction of required attributes matched by fuzzy comparison, with title/description fallback. |
| `r_option` | Fraction of required options matched against the actually selected values by fuzzy comparison. |
| `r_price` | 1 when the resolved variant price is within the declared upper limit, otherwise 0. An explicitly absent limit is unrestricted. |

An empty attribute/option requirement set has coverage 1. Fuzzy matching uses
`token_set_ratio > 85`. Chinese title keywords use jieba noun/English-token
segmentation; the paper does not prescribe a tokenizer, so this is not a claim
of bit-identical reproduction of the upstream spaCy implementation. Empty
query strings do not count as category evidence.

Partial coverage yields fractional reward: `1 × 0.5 × 0.5 × 1 = 0.25`.
Strict Reward is not the binary full-success metric. Gold identity does not
multiply or discount reward: a fully matching alternative also receives 1.
The existing `gold_purchase`/alternative labels remain diagnostic; the project's
gold-success selection metric still requires a complete, valid gold terminal.

All non-purchase terminals, including abstention, repeat loops and max steps,
receive 0. No efficiency bonus, length penalty, brand/model weighting or
alternative-ASIN discount is added. Invalid evidence receives 0 with
`reward_valid=false`, preserving the infrastructure/sampling validity boundary.
Current variant availability and deterministic-price checks are retained.

## Integration

- The environment's `engine/reward.py` computes and exports all four components
  and their exact product.
- Environment configuration and the public reward version reject old contracts.
- The GRPO adapter validates finite components in `[0,1]`, checks that the
  environment utility equals their product, and optimizes only that product.
  Enabling length shaping is rejected.
- SFT collection and gold-success validation require the new contract and full
  components. New evaluation summaries report the continuous strict score.
- Existing collected trajectories, model checkpoints, frozen dataset metadata
  and blind-final curation records retain their original provenance. They were
  not recollected, retrained, merged or reevaluated by this change. The fixed
  evaluation task list and its leakage guard are unchanged.

Implementation checks cover full/partial/zero products, equivalent reward for
full alternatives, soft category coefficients, invalid evidence, zero-valued
non-purchase terminals, protocol mismatch and training-side product validation.

## Consequences for data cleaning

Changing aggregation does not repair a missing or incorrectly extracted budget.
The runtime still derives `price_upper` with the existing parser. When that
parser returns `None` for a stated budget, it is indistinguishable from an
unstated budget to `r_price`. A four-factor product can therefore still ignore
the real user limit.

Similarly, fuzzy matching can confuse nearby model/specification strings or
negations. Reproducing the paper's multiplicative objective does not make the
gold annotation or underlying string comparisons semantically correct.

The two-stage cleaning scheme remains necessary: raw integrity checks, followed
by independent model extraction and semantic verification with exact price
arithmetic. Its next change should freeze extracted requirements and an
explicit-state budget as canonical task data, then make the runtime read the
same contract. Until that work is done, retain budget-parser mismatch rejection
and reject unresolved prices. Full SKU-combination price gaps also remain.
