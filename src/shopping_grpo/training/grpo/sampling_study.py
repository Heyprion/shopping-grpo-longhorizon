"""Bounded, frozen-SFT sampling study: selection, accounting and paired reports."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from shopping_grpo.collection.stratification import LEVELS, select_tasks
from shopping_grpo.training.grpo.dynamic_sampling import (
    extract_shopping_group_signals,
    select_reward_varying_groups,
)
from shopping_grpo.training.grpo.validation import strict_success

SETTINGS = {
    "A": {"temperature": 0.7, "top_p": 0.9, "top_k": -1},
    "B": {"temperature": 1.0, "top_p": 0.9, "top_k": -1},
    "C": {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
}
MIXTURES = {
    "20-50-30": [0.2, 0.5, 0.3],
    "10-40-50": [0.1, 0.4, 0.5],
    "10-30-60": [0.1, 0.3, 0.6],
}
PLAN = {
    "version": "shopping-pre-rl-sampling-v1",
    "seed": 42,
    "model_updates": 0,
    "screen_tasks_per_difficulty": 16,
    "confirmation_tasks_per_difficulty": 8,
    "screen_group": 4,
    "settings": SETTINGS,
    "mixtures_simple_medium_hard": MIXTURES,
    "mixture_semantics": "task sampling probabilities, not replacements of the frozen 1000-task list",
    "request_seed_derivation": "SHA256(master42, task_id, sample_index, generation_turn)",
    "priority": ["difficulty_mixture", "temperature_top_p", "conditional_group8"],
    "training_resample_batches_unchanged": 3,
    "warmup_trajectories": 8,
    "maximum_screen_trajectories": 576,
    "maximum_confirmation_trajectories": 192,
    "maximum_conditional_group8_trajectories": 192,
    "maximum_total_trajectories": 968,
    "group8_trigger": "confirmed effective-group rate < 0.5 or throughput not above baseline",
    "selection": "effective groups per sampling minute; invalid-group rate <= max(0.10, baseline+0.05)",
    "confirmation": "new train tasks; lock screen-selected setting and mixture before confirmation",
    "runtime": "native veRL standalone vLLM TP1 x2 and ShoppingToolAgentLoop; no actor/optimizer",
    "timing_scope": "sampling only, excludes actor update and weight synchronization",
}


def read_rows(path, allow_partial=False):
    lines = Path(path).read_text().splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if allow_partial and index == len(lines) - 1 and not line.endswith("\n"):
                break
            raise
    return rows


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sample_seed(task_id, sample_index, generation_turn=0):
    # One master seed, no seed search. Different samples never reuse the same
    # request stream; matching indices across settings enable paired comparisons.
    value = hashlib.sha256(f"42:{task_id}:{sample_index}:{generation_turn}".encode()).digest()
    return int.from_bytes(value[:4], "big") % (2**31 - 1)


def choose_tasks(root):
    root = Path(root)
    train_ids = {r["task_id"] for r in read_rows(root / "data/grpo/train.jsonl")}
    labels = [r for r in read_rows(root / "data/splits/labels.jsonl") if r["task_id"] in train_ids]
    # Keep one representative per instruction group, never inflate pilot counts
    # by selecting instruction duplicates from the same training split.
    unique = {}
    for row in sorted(labels, key=lambda r: r["task_id"]):
        unique.setdefault(row["group_id"], row)
    population = list(unique.values())
    selected = select_tasks(
        population, 72, seed=42, namespace="pre-rl-study-panel", ratios=[1, 1, 1]
    )
    screen = select_tasks(selected, 48, seed=42, namespace="pre-rl-study-screen", ratios=[1, 1, 1])
    used = {r["task_id"] for r in screen}
    confirmation = [r for r in selected if r["task_id"] not in used]
    selected_ids = {r["task_id"] for r in selected}
    warmup = sorted(
        (r for r in population if r["task_id"] not in selected_ids),
        key=lambda r: sample_seed(r["task_id"], 0),
    )[:2]
    for rows, expected in ((screen, 16), (confirmation, 8)):
        if any(sum(r["difficulty"] == level for r in rows) != expected for level in LEVELS):
            raise ValueError("study panel difficulty allocation is not balanced")
    return {"screen": screen, "confirmation": confirmation, "warmup": warmup}


def task_pairs(rows):
    """Pair equal-difficulty tasks so batch wall time can be attributed to a stratum."""
    by_level = {
        level: sorted(
            (r for r in rows if r["difficulty"] == level),
            key=lambda r: sample_seed(r["task_id"], 0),
        )
        for level in LEVELS
    }
    for offset in range(0, len(rows) // 3, 2):
        for level in LEVELS:
            yield by_level[level][offset : offset + 2]


def describe_group(phase, setting, task, infos, elapsed_seconds, token_counts):
    """Use the exact training filter, including partial strict rewards and invalid groups."""
    utility, success, invalid, reasons = extract_shopping_group_signals(infos)
    _, stats = select_reward_varying_groups(
        [str(task["task_id"])] * len(infos),
        utility,
        terminal_utilities=utility,
        purchase_success=success,
        sampling_invalid=invalid,
        sampling_invalid_reasons=reasons,
    )
    group = stats["groups"][0]
    strict_count = sum(strict_success(info) for info in infos)
    return {
        "phase": phase,
        "setting": setting,
        "task_id": task["task_id"],
        "difficulty": task["difficulty"],
        "domain": task["domain"],
        "group_size": len(infos),
        "effective": group["kept"],
        "invalid": group["sampling_invalid"],
        "invalid_reasons": list(group["sampling_invalid_reasons"]),
        "drop_reason": group["drop_reason"],
        "rewards": utility,
        "strict_success_count": strict_count,
        "strict_success_rate": strict_count / len(infos),
        "all_reward_one": all(value == 1 for value in utility) and not group["sampling_invalid"],
        "all_reward_zero": all(value == 0 for value in utility) and not group["sampling_invalid"],
        "exactly_half_strict_success": strict_count * 2 == len(infos),
        "reward_variance": sum((v - sum(utility) / len(utility)) ** 2 for v in utility)
        / len(utility),
        "sampling_seconds": elapsed_seconds,
        "generated_tokens": sum(token_counts),
        "distinct_action_sequences": len(
            {json.dumps(i.get("actions", []), sort_keys=True) for i in infos}
        ),
    }


def wilson(successes, count):
    if not count:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize_cell(rows):
    n = len(rows)
    keys = (
        "effective",
        "invalid",
        "all_reward_one",
        "all_reward_zero",
        "strict_success_rate",
        "exactly_half_strict_success",
        "sampling_seconds",
        "generated_tokens",
        "reward_variance",
    )
    means = {key: sum(r[key] for r in rows) / n for key in keys}
    histogram = defaultdict(int)
    for row in rows:
        histogram[str(row["strict_success_count"]) + "/" + str(row["group_size"])] += 1
    return {
        "tasks": n,
        **means,
        "success_count_histogram": dict(histogram),
        "effective_rate_wilson95": wilson(sum(r["effective"] for r in rows), n),
    }


def candidates(groups, phase="screen", settings=None):
    settings = list(settings or SETTINGS)
    by_setting = {
        setting: [r for r in groups if r["phase"] == phase and r["setting"] == setting]
        for setting in settings
    }
    # An interrupted panel cannot advantage an arm that happened to finish more tasks.
    common = set.intersection(*({r["task_id"] for r in rows} for rows in by_setting.values()))
    cells = {
        setting: {
            level: [r for r in rows if r["task_id"] in common and r["difficulty"] == level]
            for level in LEVELS
        }
        for setting, rows in by_setting.items()
    }
    if any(not rows for strata in cells.values() for rows in strata.values()):
        return [], {}
    summaries = {
        setting: {level: summarize_cell(rows) for level, rows in strata.items()}
        for setting, strata in cells.items()
    }
    baseline_invalid = (
        sum(
            MIXTURES["20-50-30"][i] * summaries["A"][level]["invalid"]
            for i, level in enumerate(LEVELS)
        )
        if "A" in summaries
        else 0.0
    )
    results = []
    for setting in settings:
        for mixture, weights in MIXTURES.items():
            summary = {
                key: sum(
                    weights[i] * summaries[setting][level][key] for i, level in enumerate(LEVELS)
                )
                for key in (
                    "effective",
                    "invalid",
                    "strict_success_rate",
                    "sampling_seconds",
                    "generated_tokens",
                    "reward_variance",
                    "all_reward_one",
                    "all_reward_zero",
                )
            }
            summary.update(
                setting=setting,
                mixture=mixture,
                group_size=4,
                tasks=len(common),
                tasks_per_difficulty={level: len(cells[setting][level]) for level in LEVELS},
                effective_groups_per_minute=60
                * summary["effective"]
                / max(summary["sampling_seconds"], 1e-9),
                quality_eligible=summary["invalid"] <= max(0.10, baseline_invalid + 0.05),
            )
            results.append(summary)
    return sorted(
        results,
        key=lambda r: (r["quality_eligible"], r["effective_groups_per_minute"]),
        reverse=True,
    ), summaries


def select_candidate(groups):
    ranked, _ = candidates(groups)
    if not ranked or any(n < 8 for n in ranked[0]["tasks_per_difficulty"].values()):
        return None
    # Treat the unchanged recipe as a tie when the estimated throughput gain is <5%.
    baseline = next(r for r in ranked if r["setting"] == "A" and r["mixture"] == "20-50-30")
    best = next((r for r in ranked if r["quality_eligible"]), baseline)
    if (
        baseline["quality_eligible"]
        and best["effective_groups_per_minute"] <= 1.05 * baseline["effective_groups_per_minute"]
    ):
        best = baseline
    return best


def weighted_candidate(groups, phase, selection):
    ranked, _ = candidates(
        groups, phase=phase, settings=list(dict.fromkeys(["A", selection["setting"]]))
    )
    return next(
        (
            r
            for r in ranked
            if r["setting"] == selection["setting"] and r["mixture"] == selection["mixture"]
        ),
        None,
    )


def group8_needed(groups, selection):
    confirmed = weighted_candidate(groups, "confirmation", selection)
    ranked, _ = candidates(
        groups, phase="confirmation", settings=list(dict.fromkeys(["A", selection["setting"]]))
    )
    baseline = next((r for r in ranked if r["setting"] == "A" and r["mixture"] == "20-50-30"), None)
    if confirmed is None or baseline is None or confirmed["tasks"] < 24:
        return False
    return (
        confirmed["effective"] < 0.5
        or confirmed["effective_groups_per_minute"] < baseline["effective_groups_per_minute"]
    )


def paired_confirmation(groups, selection, repeats=400):
    """Task-stratified paired bootstrap: ratio changes are evaluated on fresh train tasks."""
    chosen = selection["setting"]
    lookup = {(r["setting"], r["task_id"]): r for r in groups if r["phase"] == "confirmation"}
    ids = {
        level: [
            task
            for arm, task in lookup
            if arm == "A" and (chosen, task) in lookup and lookup[arm, task]["difficulty"] == level
        ]
        for level in LEVELS
    }
    if any(not values for values in ids.values()):
        return None
    rng = random.Random(42)
    deltas = []
    for _ in range(repeats):
        values = {}
        for arm, mixture in (("baseline", "20-50-30"), ("candidate", selection["mixture"])):
            values[arm] = {"effective": 0.0, "seconds": 0.0}
        for index, level in enumerate(LEVELS):
            sampled = rng.choices(ids[level], k=len(ids[level]))
            for name, arm, mix in (
                ("baseline", "A", "20-50-30"),
                ("candidate", chosen, selection["mixture"]),
            ):
                weight = MIXTURES[mix][index]
                values[name]["effective"] += (
                    weight * sum(lookup[arm, i]["effective"] for i in sampled) / len(sampled)
                )
                values[name]["seconds"] += (
                    weight * sum(lookup[arm, i]["sampling_seconds"] for i in sampled) / len(sampled)
                )
        rates = {name: 60 * v["effective"] / max(v["seconds"], 1e-9) for name, v in values.items()}
        deltas.append(rates["candidate"] - rates["baseline"])
    deltas.sort()
    return {
        "tasks": sum(map(len, ids.values())),
        "paired_bootstrap_repeats": repeats,
        "effective_groups_per_minute_delta_95": [
            deltas[int(0.025 * repeats)],
            deltas[int(0.975 * repeats)],
        ],
    }


def build_report(output, status):
    output = Path(output)
    groups = (
        read_rows(output / "groups.jsonl", allow_partial=True)
        if (output / "groups.jsonl").exists()
        else []
    )
    ranked, strata = candidates(groups)
    selection = (
        json.loads((output / "selection.json").read_text())
        if (output / "selection.json").exists()
        else None
    )
    confirmation = weighted_candidate(groups, "confirmation", selection) if selection else None
    comparison = paired_confirmation(groups, selection) if selection else None
    plan = json.loads((output / "plan.json").read_text())
    screen_complete = all(
        sum(r["phase"] == "screen" and r["setting"] == setting for r in groups) == 48
        for setting in SETTINGS
    )
    confirm_complete = confirmation is not None and confirmation["tasks"] == 24
    evidence = "incomplete"
    if screen_complete and confirm_complete:
        evidence = "exploratory_confirmation"
        interval = comparison["effective_groups_per_minute_delta_95"]
        if interval[0] > 0 and confirmation["quality_eligible"]:
            evidence = "sampling_throughput_improvement_supported"
        elif selection["setting"] == "A" and selection["mixture"] == "20-50-30":
            evidence = "baseline_retained"
        else:
            evidence = "inconclusive_small_sample"
    fallback = [r for r in groups if r["phase"] == "group8"]
    group8 = None
    if fallback and selection:
        # Only compare the same task IDs with native group=4 confirmation runs.
        chosen = selection["setting"]
        common = {r["task_id"] for r in fallback} & {
            r["task_id"] for r in groups if r["phase"] == "confirmation" and r["setting"] == chosen
        }
        cells = {
            g: {
                level: [
                    r
                    for r in groups
                    if r["task_id"] in common
                    and r["setting"] == chosen
                    and r["phase"] == ("group8" if g == 8 else "confirmation")
                    and r["difficulty"] == level
                ]
                for level in LEVELS
            }
            for g in (4, 8)
        }
        if all(rows for strata_rows in cells.values() for rows in strata_rows.values()):
            group8 = {}
            for g, strata_rows in cells.items():
                summary = {level: summarize_cell(rows) for level, rows in strata_rows.items()}
                weights = MIXTURES[selection["mixture"]]
                weighted = {
                    k: sum(weights[i] * summary[level][k] for i, level in enumerate(LEVELS))
                    for k in ("effective", "sampling_seconds", "invalid", "generated_tokens")
                }
                group8[str(g)] = {
                    "strata": summary,
                    **weighted,
                    "effective_groups_per_minute": 60
                    * weighted["effective"]
                    / max(weighted["sampling_seconds"], 1e-9),
                }
    report = {
        "status": status,
        "evidence": evidence,
        "screen_complete": screen_complete,
        "confirmation_complete": confirm_complete,
        "completed_groups": len(groups),
        "completed_trajectories": sum(r["group_size"] for r in groups),
        "plan": plan,
        "screen_by_difficulty": strata,
        "screen_candidates": ranked,
        "screen_selection": selection,
        "fresh_train_confirmation": confirmation,
        "paired_confirmation": comparison,
        "native_group4_vs_group8": group8,
        "training_resample_batches": 3,
        "training_started": False,
        "limitations": [
            "small diagnostic panel, not evidence of RL learning gains",
            "sampling times exclude actor update and weight sync",
            "group8 doubles actor trajectories at two tasks/update; micro4 implies accumulation2",
            "no training data allocation or hyperparameters are modified automatically",
        ],
    }
    write_json(output / "summary.json", report)
    write_json(
        output / "recommendation.json",
        {
            "evidence": evidence,
            "candidate": selection,
            "confirmation": confirmation,
            "settings": SETTINGS.get(selection["setting"]) if selection else None,
            "mixture": MIXTURES.get(selection["mixture"]) if selection else None,
            "group": 4,
            "resample_batches": 3,
            "group8_measured_separately": group8 is not None,
            "group8_candidate_requires_actor_timing": group8,
            "automatic_training": False,
        },
    )
    lines = [
        "# SFT → RL 采样诊断",
        "",
        f"状态：{status}；证据：{evidence}",
        f"完成 {len(groups)} 个任务组 / {report['completed_trajectories']} 条轨迹（另有预热记录）。",
        "",
        "模型全程冻结；只使用当前 RL 训练池，不使用验证或测试题。",
        "",
        "| 设置 | 难度比 simple/medium/hard | 有效组率 | 有效组/分钟 | 严格成功率 | 无效组率 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['setting']} | {row['mixture']} | {row['effective']:.1%} | {row['effective_groups_per_minute']:.2f} | {row['strict_success_rate']:.1%} | {row['invalid']:.1%} |"
        )
    lines += [
        "",
        "| 设置 | 难度 | 任务数 | 有效组率 | 严格成功率 | 全满分组率 | 无效组率 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for setting, levels in strata.items():
        for level, values in levels.items():
            lines.append(
                f"| {setting} | {level} | {values['tasks']} | {values['effective']:.1%} | {values['strict_success_rate']:.1%} | {values['all_reward_one']:.1%} | {values['invalid']:.1%} |"
            )
    if selection:
        lines += [
            "",
            f"预先锁定的初筛候选：设置 {selection['setting']}，难度比 {selection['mixture']}，group=4。",
        ]
    if confirmation:
        lines += [
            f"新训练题确认：有效组率 {confirmation['effective']:.1%}，有效组/分钟 {confirmation['effective_groups_per_minute']:.2f}。"
        ]
    if comparison:
        lines += [
            "相对原配方的有效组/分钟差值，按任务分层配对 bootstrap 95% 区间："
            + str(comparison["effective_groups_per_minute_delta_95"])
            + "。"
        ]
    if group8:
        lines += [
            "",
            "| 相同确认题 | 有效组率 | 有效组/分钟 | 每组采样秒数 |",
            "|---|---:|---:|---:|",
        ]
        for size, row in group8.items():
            lines.append(
                f"| group={size} | {row['effective']:.1%} | {row['effective_groups_per_minute']:.2f} | {row['sampling_seconds']:.2f} |"
            )
    lines += [
        "",
        "完整分难度指标、成功数直方图、置信区间和新训练题确认见 summary.json。",
        "group=8 若运行，包含真实原生采样计时；不代表完整训练耗时。",
        "总成功率接近 50% 不等于每道题都有学习信号；有效组按原始 Strict Reward 差异判定。",
        "这些结果只给采样建议，不自动启动 RL，也不修改固定名单。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
