"""GiGPO step credit using undiscounted ShopSimulator terminal returns."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from statistics import fmean


STEP_WEIGHT = 0.5


def observation_anchor(raw: str, visible: str) -> str:
    return sha256(json.dumps([raw, visible], ensure_ascii=False).encode()).hexdigest()


def retained_turns(turns, removed_tokens, response_length):
    """Translate absolute turn offsets after prefix compaction and output truncation."""
    return [
        (anchor, action, start - removed_tokens, min(end - removed_tokens, response_length))
        for anchor, action, start, end in turns
        if removed_tokens <= start < removed_tokens + response_length
    ]


def anchor_credits(uids, rollouts, scores, tolerance):
    """Return credits keyed by rollout row and assistant-token span."""
    anchors = defaultdict(list)
    for row, (uid, turns, score) in enumerate(zip(uids, rollouts, scores, strict=True)):
        seen = set()
        for state, action, start, end in turns:
            if state not in seen:
                anchors[uid, state].append((row, action, start, end, score))
                seen.add(state)

    shared = supported = 0
    credits = {}
    for members in anchors.values():
        if len(members) < 2:
            continue
        shared += 1
        by_action = defaultdict(list)
        for _, action, _, _, score in members:
            by_action[action].append(score)
        averages = [fmean(values) for values in by_action.values()]
        if len(averages) < 2 or max(averages) - min(averages) <= tolerance:
            continue
        supported += 1
        center = fmean(member[4] for member in members)
        for row, _, start, end, score in members:
            if abs(score - center) > tolerance:
                credits[row, start, end] = score - center
    return credits, shared, supported


def apply_gigpo_advantages(batch, *, tolerance):
    """Add bounded step credit to GRPO advantages on assistant tokens."""
    data = batch.non_tensor_batch
    turns = [item["turns"] for item in data["shopping_gigpo"]]
    scores = batch.batch["token_level_rewards"].sum(dim=-1).tolist()
    credits, shared, supported = anchor_credits(data["uid"], turns, scores, tolerance)
    mask = batch.batch["response_mask"].bool()
    advantages = batch.batch["advantages"]
    mixed = advantages.clone()
    token_count = 0
    for (row, start, end), credit in credits.items():
        if not (0 <= start < end <= mask.shape[1]) or not bool(mask[row, start:end].all()):
            raise ValueError("GiGPO assistant span is not aligned with the response mask")
        mixed[row, start:end] += STEP_WEIGHT * credit
        token_count += end - start
    batch.batch["advantages"] = mixed
    batch.batch["returns"] = mixed
    return {
        "gigpo/shared_anchors": shared,
        "gigpo/supported_anchors": supported,
        "gigpo/recorded_turns": sum(map(len, turns)),
        "gigpo/credited_turns": len(credits),
        "gigpo/credited_tokens": token_count,
        "gigpo/assistant_tokens": int(mask.sum()),
        "gigpo/nonzero_grpo_tokens": int((mask & (advantages.abs() > tolerance)).sum()),
        "gigpo/nonzero_mixed_tokens": int((mask & (mixed.abs() > tolerance)).sum()),
    }
