#!/usr/bin/env python3
"""Audit collected SFT trajectories without filtering or splitting the source data."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from shopping_grpo.collection.sft import build_sft_row
from shopping_grpo.environment.actions import action_reject_reason
from shopping_grpo.environment.observation import render_structured_observation
from shopping_grpo.environment.reward import REWARD_VERSION, full_strict_reward
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS, tool_call_to_action
from shopping_grpo.training.sft.dataset import (
    IGNORE_INDEX,
    build_supervised_example,
    normalize_messages_for_chat_template,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in SHOP_TOOL_SCHEMAS}


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def structure_issues(t, student):
    issues = []
    required = [
        "trajectory_id",
        "task_id",
        "attempt_index",
        "messages",
        "steps",
        "initial_result",
        "terminal_result",
        "status",
        "done",
        "final_reward",
    ]
    issues.extend("missing:" + k for k in required if k not in t)
    if t.get("status") != "done" or t.get("done") is not True or t.get("final_reward") != 1:
        issues.append("not_completed_reward_one")
    if t.get("error") or t.get("release_error"):
        issues.append("runtime_error")
    terminal = t.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    if (
        detail.get("reward_version") != REWARD_VERSION
        or detail.get("reward_valid") is not True
        or not full_strict_reward(detail)
        or terminal.get("done") is not True
        or terminal.get("over") is not True
    ):
        issues.append("invalid_terminal_strict_reward")
    if not t.get("steps") or t["steps"][-1].get("tool_name") != "buy_now":
        issues.append("missing_terminal_purchase")
    initial = t.get("initial_result") or {}
    latest = render_structured_observation(initial["observation_state"])
    state = initial["observation_state"]
    for step in t["steps"]:
        action = step["env_action"]
        if action and action.startswith("click["):
            target = action[6:-1].casefold()
            if target not in {str(value).casefold() for value in state.get("actions", [])}:
                issues.append("action_not_in_structured_observation")
        state = step["result"].get("observation_state") or {}
    steps = {s["tool_call"]["id"]: s for s in t["steps"]}
    pending = None
    seen = set()
    matched = set()
    if student["tools"] != SHOP_TOOL_SCHEMAS:
        issues.append("tool_schema_mismatch")
    if not student["messages"] or student["messages"][0].get("role") != "system":
        issues.append("missing_system")
    if sum(m.get("role") == "user" for m in student["messages"]) != 1:
        issues.append("user_instruction_count")
    for m in student["messages"]:
        role = m.get("role")
        if role == "assistant":
            if pending is not None:
                issues.append("assistant_before_tool_reply")
            calls = m.get("tool_calls") or []
            if len(calls) != 1:
                issues.append("not_one_tool_call_per_assistant")
                continue
            call = calls[0]
            call_id = call.get("id")
            if not call_id or call_id in seen:
                issues.append("invalid_or_duplicate_call_id")
            seen.add(call_id)
            function = call.get("function") or {}
            name = function.get("name")
            args = json.loads(function.get("arguments") or "{}")
            schema = SCHEMAS.get(name)
            if schema is None or not isinstance(args, dict):
                issues.append("unknown_tool_or_invalid_arguments")
                continue
            if (
                set(args) - set(schema["properties"])
                or set(schema["required"]) - set(args)
                or any(not isinstance(v, str) for v in args.values())
                or any(
                    "enum" in schema["properties"][k] and v not in schema["properties"][k]["enum"]
                    for k, v in args.items()
                    if k in schema["properties"]
                )
            ):
                issues.append("tool_arguments_violate_schema")
            if action_reject_reason(name, args, latest):
                issues.append("illegal_action_in_student_context")
            step = steps.get(call_id)
            if (
                step is None
                or step["tool_name"] != name
                or step["env_action"] != tool_call_to_action(name, args)
            ):
                issues.append("step_message_action_mismatch")
            else:
                matched.add(call_id)
            pending = call_id
        elif role == "tool":
            call_id = m.get("tool_call_id")
            if call_id != pending or call_id not in steps:
                issues.append("unmatched_tool_reply")
            else:
                step = steps[call_id]
                expected = "购买已完成。" if step.get("done") else step["observation"]
                if m.get("content") != expected:
                    issues.append("tool_observation_mismatch")
                latest = m.get("content") or ""
            pending = None
        elif role not in ("system", "user"):
            issues.append("invalid_message_role")
        if "reasoning_content" in m or m.get("runtime_action_guard"):
            issues.append("teacher_audit_field_in_student")
    if pending is not None:
        issues.append("missing_final_tool_reply")
    if matched != set(steps):
        issues.append("executed_step_missing_from_student")
    if any(e.get("truncated") for e in t.get("model_completions", [])):
        issues.append("truncated_model_response")
    return sorted(set(issues))


def behavioral_metrics(t):
    repeated_state_actions = []
    unchanged_actions = []
    repeated_queries = []
    queries = set()
    opens = []
    last = t["initial_result"]["observation_state"]
    seen = set()
    pairs = []
    redundant_info_roundtrips = []
    repeat_types = Counter()
    option_revisions = []
    for i, s in enumerate(t["steps"]):
        name, args = s["tool_name"], s["parameters"]
        state_key = fingerprint(last)
        pair = (state_key, name, fingerprint(args))
        if pair in seen:
            repeated_state_actions.append(i)
            repeat_types[name] += 1
        if (
            i >= 2
            and pair == pairs[i - 2]
            and name in {"view_description", "view_features", "view_reviews", "view_attributes"}
            and t["steps"][i - 1]["tool_name"] == "prev_page"
        ):
            redundant_info_roundtrips.append(i)
        pairs.append(pair)
        seen.add(pair)
        current = s["result"].get("observation_state") or {}
        if not s.get("done") and fingerprint(current) == state_key:
            unchanged_actions.append(i)
        if name == "search_products":
            query = " ".join(args["query"].casefold().split())
            if query in queries:
                repeated_queries.append(i)
            queries.add(query)
        if name == "open_product":
            opens.append(args["asin"])
        if name == "select_option":
            before = last.get("selected_options") or {}
            after = current.get("selected_options") or {}
            for axis, value in after.items():
                if axis in before and value != before[axis]:
                    option_revisions.append(
                        {"step": i, "axis": axis, "from": before[axis], "to": value}
                    )
        last = current
    tool_counts = Counter(s["tool_name"] for s in t["steps"])
    return {
        "steps": len(t["steps"]),
        "tool_counts": dict(tool_counts),
        "guard_blocks": len(t.get("blocked_tool_calls", [])),
        "guard_reasons": dict(Counter(b["reason"] for b in t.get("blocked_tool_calls", []))),
        "serial_tool_truncations": len(t.get("tool_call_truncations", [])),
        "repeated_state_action_steps": repeated_state_actions,
        "repeated_state_action_types": dict(repeat_types),
        "redundant_info_roundtrip_steps": redundant_info_roundtrips,
        "unchanged_action_steps": unchanged_actions,
        "repeated_query_steps": repeated_queries,
        "reopened_products": len(opens) - len(set(opens)),
        "products_opened": len(set(opens)),
        "option_revisions": option_revisions,
        "assistant_content_messages": sum(
            m.get("role") == "assistant" and bool(m.get("content")) for m in t["messages"]
        ),
        "viewed_info_subpage": bool(
            sum(tool_counts[n] for n in ["view_description", "view_features", "view_attributes"])
        ),
        "terminal_type": t["terminal_result"]["reward_detail"].get("reward_type"),
    }


def distribution(values):
    ordered = sorted(values)
    return {
        "min": min(ordered),
        "median": statistics.median(ordered),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "p99": ordered[int(0.99 * (len(ordered) - 1))],
        "max": max(ordered),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "outputs/sft-unified-collection-64k/accepted.jsonl"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sft-quality-audit")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--swanlab-mode", choices=["online", "offline"], default="online")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    import swanlab
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    args.output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    labels = {
        r["task_id"]: r
        for r in map(json.loads, (ROOT / "data/splits/labels.jsonl").read_text().splitlines())
    }
    candidates = {
        r["task_id"]
        for r in map(json.loads, (ROOT / "data/sft/candidate_tasks.jsonl").read_text().splitlines())
    }
    protected = {r["group_id"] for r in labels.values() if r["source_tag"] == "eval"}
    config = {
        "purpose": "audit_only_no_filtering_or_split",
        "source_sha256": source_hash,
        "student_model": MODEL,
        "student_revision": REVISION,
        "max_length": 24576,
        "formatting": "collection.sft.build_sft_row + processor chat template + build_supervised_example",
        "code_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "scripts/audit_sft_trajectories.py",
                "src/shopping_grpo/collection/sft.py",
                "src/shopping_grpo/training/sft/dataset.py",
            ]
        },
    }
    swanlab.init(
        project="shopping-grpo-sft-collection",
        experiment_name=args.run_name
        or f"{datetime.now(timezone.utc):%Y-%m-%d}-COLLECT-教师轨迹质量审计",
        config=config,
        mode=args.swanlab_mode,
        logdir=str(args.output / "swanlab"),
    )
    started = time.monotonic()

    def audit(t):
        result = {
            "task_id": t["task_id"],
            "trajectory_id": t["trajectory_id"],
            "difficulty": labels[t["task_id"]]["difficulty"],
            "domain": labels[t["task_id"]]["domain"],
            "instruction_group": labels[t["task_id"]]["group_id"],
            **behavioral_metrics(t),
        }
        try:
            student = build_sft_row(t)
            issues = structure_issues(t, student)
            if (
                t["task_id"] not in candidates
                or labels[t["task_id"]]["source_tag"] != "train"
                or labels[t["task_id"]]["group_id"] in protected
            ):
                issues.append("source_boundary_violation")
            normalized = normalize_messages_for_chat_template(student["messages"])
            text = processor.apply_chat_template(
                normalized, tools=student["tools"], tokenize=False, add_generation_prompt=False
            )
            n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            result["student_tokens"] = n
            raw_normalized = normalize_messages_for_chat_template(t["messages"])
            raw_text = processor.apply_chat_template(
                raw_normalized, tools=SHOP_TOOL_SCHEMAS, tokenize=False, add_generation_prompt=False
            )
            result["raw_teacher_tokens"] = len(
                tokenizer(raw_text, add_special_tokens=False)["input_ids"]
            )
            if n > 24576:
                issues.append("student_context_over_limit")
            example = build_supervised_example(
                student["messages"],
                student["tools"],
                tokenizer,
                max_length=24576,
                chat_template=processor,
            )
            result["assistant_loss_tokens"] = (
                sum(v != IGNORE_INDEX for v in example["labels"]) if example else None
            )
            if example is None and n <= 24576:
                issues.append("training_template_or_label_failure")
            result["stage1_issues"] = sorted(set(issues))
        except (KeyError, TypeError, ValueError) as e:
            result["stage1_issues"] = ["malformed_record:" + type(e).__name__ + ":" + str(e)]
        return result

    trajectories = [
        json.loads(line) for line in args.source.read_text().splitlines() if line.strip()
    ]
    metrics = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, row in enumerate(pool.map(audit, trajectories), 1):
                metrics.append(row)
                if i % 100 == 0:
                    swanlab.log(
                        {
                            "audit/completed": i,
                            "audit/stage1_flagged": sum(bool(r["stage1_issues"]) for r in metrics),
                            "audit/elapsed_seconds": time.monotonic() - started,
                        }
                    )
                    print("audited", i, flush=True)
        with (args.output / "trajectory_metrics.jsonl").open("w") as f:
            for row in metrics:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        flags = Counter(issue for r in metrics for issue in r["stage1_issues"])
        duplicates = Counter(t["task_id"] for t in trajectories)
        groups = Counter(r["instruction_group"] for r in metrics)
        summary = {
            "config": config,
            "trajectories": len(metrics),
            "unique_tasks": len(duplicates),
            "duplicate_task_ids": [k for k, v in duplicates.items() if v > 1],
            "duplicate_instruction_groups": sum(v - 1 for v in groups.values() if v > 1),
            "stage1_flag_counts": dict(flags),
            "stage1_flagged": sum(bool(r["stage1_issues"]) for r in metrics),
            "stage1_preview_kept": sum(not r["stage1_issues"] for r in metrics),
            "stage1_preview_difficulty": dict(
                Counter(r["difficulty"] for r in metrics if not r["stage1_issues"])
            ),
            "raw_only_overlength": sum(
                r.get("raw_teacher_tokens", 0) > 24576 >= r.get("student_tokens", 24577)
                for r in metrics
            ),
            "student_tokens": distribution(
                [r["student_tokens"] for r in metrics if "student_tokens" in r]
            ),
            "raw_teacher_tokens": distribution(
                [r["raw_teacher_tokens"] for r in metrics if "raw_teacher_tokens" in r]
            ),
            "steps": distribution([r["steps"] for r in metrics]),
            "diagnostic_trajectory_counts": {
                name: sum(bool(r[name]) for r in metrics)
                for name in [
                    "guard_blocks",
                    "serial_tool_truncations",
                    "repeated_state_action_steps",
                    "redundant_info_roundtrip_steps",
                    "unchanged_action_steps",
                    "repeated_query_steps",
                    "reopened_products",
                    "option_revisions",
                    "assistant_content_messages",
                ]
            },
            "no_info_subpage": sum(not r["viewed_info_subpage"] for r in metrics),
            "skill_coverage": {
                "multiple_searches": sum(
                    r["tool_counts"].get("search_products", 0) > 1 for r in metrics
                ),
                "multiple_products": sum(r["products_opened"] > 1 for r in metrics),
                "pagination": sum(r["tool_counts"].get("next_page", 0) > 0 for r in metrics),
            },
            "terminal_types": dict(Counter(r["terminal_type"] for r in metrics)),
            "elapsed_seconds": time.monotonic() - started,
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        swanlab.log(
            {
                "audit/stage1_flagged": summary["stage1_flagged"],
                "audit/trajectories": len(metrics),
                "audit/max_student_tokens": summary["student_tokens"]["max"],
                **{
                    "diagnostic/" + k: v for k, v in summary["diagnostic_trajectory_counts"].items()
                },
            }
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        swanlab.finish()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != source_hash:
        raise SystemExit("Source changed during audit")


if __name__ == "__main__":
    main()
