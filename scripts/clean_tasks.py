#!/usr/bin/env python3
"""Screen a sample or the complete catalog without renumbering source tasks."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from shopping_grpo.collection.task_cleaning import (
    POLICY_VERSION,
    check_structure,
    evidence_view,
    fingerprint,
    ground_extraction,
    ground_verification,
    product_evidence,
    query_clauses,
)
from shopping_grpo.collection.task_cleaning_prompts import (
    EXTRACT_PROMPT,
    EXTRACT_SCHEMA,
    VERIFY_PROMPT,
    VERIFY_SCHEMA,
)
from shopping_grpo.environment.manifest import sha256_file
from shopping_grpo.environment.reward import REWARD_VERSION

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "environments/ShopSimulator/shop_env"))

from web_agent_site.engine.constraints import explicit_budget_from_instruction
from web_agent_site.engine.reward import evaluate_purchase
from web_agent_site.engine.reward_features import compile_reward_features
from web_agent_site.engine.task_budget import budget_contract, budget_reasons, budget_upper
from web_agent_site.engine.variant_price import (
    candidate_options_for_evaluation,
    resolve_variant_price,
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in rows)
    )


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


class JSONModel:
    def __init__(self, args, output):
        self.args = args
        self.path = output / "model_calls.jsonl"
        self.cache = {}
        if self.path.exists():
            for row in read_rows(self.path):
                if "result" in row:
                    self.cache[row["key"]] = row

    def call(self, stage, value):
        prompt, schema = (
            (EXTRACT_PROMPT, EXTRACT_SCHEMA)
            if stage == "extract"
            else (VERIFY_PROMPT, VERIFY_SCHEMA)
        )
        if stage == "verify":
            schema = deepcopy(schema)
            schema["properties"]["checks"].update(
                minItems=len(value["requirements"]), maxItems=len(value["requirements"])
            )
            schema["properties"]["axis_sizes"] = {
                "type": "object",
                "properties": {
                    axis: {"type": ["string", "null"]} for axis in value["selected_options"]
                },
                "required": list(value["selected_options"]),
                "additionalProperties": False,
            }
        key = fingerprint(
            {
                "policy": POLICY_VERSION,
                "stage": stage,
                "input": value,
                "prompt": prompt,
                "schema": schema,
                "model": self.args.model,
                "revision": self.args.revision,
                "max_tokens": self.args.max_tokens,
            }
        )
        if key in self.cache:
            return {**self.cache[key], "cached": True}
        payload = {
            "model": self.args.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(value, ensure_ascii=False)},
            ],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": self.args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": stage,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        started = time.perf_counter()
        usage = Counter()
        errors = []
        error_kind = "model"
        for attempt in range(2):
            try:
                request = Request(
                    self.args.endpoint.rstrip("/") + "/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "local"),
                    },
                )
                with urlopen(request, timeout=self.args.timeout) as stream:
                    response = json.load(stream)
                for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    usage[name] += int(response.get("usage", {}).get(name, 0))
                choice = response["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError("generation_not_complete")
                result = json.loads(choice["message"]["content"])
                return {
                    "key": key,
                    "stage": stage,
                    "input": value,
                    "result": result,
                    "latency_seconds": time.perf_counter() - started,
                    "attempts": attempt + 1,
                    "usage": dict(usage),
                    "cached": False,
                    "provider_model": response.get("model"),
                    "errors": errors,
                }
            except HTTPError as exc:
                # A broken endpoint/schema is an infrastructure error, not bad task data.
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise RuntimeError(f"model endpoint HTTP {exc.code}") from exc
                errors.append(f"HTTP {exc.code}")
                error_kind = "transport"
            except (URLError, TimeoutError, ValueError, KeyError, IndexError) as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                error_kind = (
                    "model" if isinstance(exc, (ValueError, KeyError, IndexError)) else "transport"
                )
        return {
            "key": key,
            "stage": stage,
            "input": value,
            "error": "model_call_failed",
            "error_kind": error_kind,
            "latency_seconds": time.perf_counter() - started,
            "attempts": 2,
            "usage": dict(usage),
            "cached": False,
            "errors": errors,
        }


def runtime_facts(product, budget=None):
    instruction = product["instructions"][0]
    features = compile_reward_features(instruction, product)
    selected, selection = candidate_options_for_evaluation(
        product, features["required_options_by_key"]
    )
    price = resolve_variant_price(product, selected)
    upper = explicit_budget_from_instruction(instruction["instruction"])
    prepared = {
        **product,
        "Title": product["title"],
        "Description": product["full_description"],
        "Attributes": product["attribute"],
        "BulletPoints": [],
    }
    goal = {
        **features,
        "asin": product["asin"],
        "price_upper": upper,
        "name": product["title"],
        "query": product.get("query", ""),
        "attributes": instruction["attributes"],
        "goal_options": instruction["instruction_options"],
    }
    if budget is not None:
        goal["task_budget"] = budget
        goal["price_upper"] = budget_upper(budget)
    terminal = evaluate_purchase(prepared, goal, selected_options=selected, price_resolution=price)
    return {
        "selected_options": selected,
        "selection": selection,
        "price": price,
        "runtime_upper": goal["price_upper"],
        "legacy_parser_upper": upper,
        "runtime_reward_type": terminal.reward_type,
        "runtime_reward_valid": terminal.reward_valid,
        "runtime_reward_version": REWARD_VERSION,
        "runtime_reward_components": terminal.reward_components,
        "unresolved_options": features["unresolved_option_requirements"],
    }


def process(row, model):
    started = time.perf_counter()
    product = row["product"]
    output = {k: v for k, v in row.items() if k != "product"}
    output["record_hash"] = fingerprint(product)
    reasons = check_structure(product)
    output["stage1_pass"] = not reasons
    output["stage1_seconds"] = time.perf_counter() - started
    calls = []
    if not reasons:
        runtime = runtime_facts(product)
        output["runtime"] = runtime
        if runtime["selection"]["status"] != "pass" or runtime["unresolved_options"]:
            reasons.append("runtime_options_unverifiable")
        if runtime["price"]["status"] != "pass":
            reasons.append("variant_price_unverifiable")
        if not reasons:
            query = product["instructions"][0]["instruction"]
            extracted = model.call(
                "extract",
                {
                    "clauses": [
                        {"id": i, "text": text} for i, text in enumerate(query_clauses(query))
                    ]
                },
            )
            calls.append(extracted)
            output["extraction_key"] = extracted["key"]
            if "error" in extracted:
                reasons.append(extracted["error"])
            else:
                try:
                    extraction = ground_extraction(extracted["result"], query)
                except ValueError as exc:
                    reasons.append(str(exc))
                else:
                    output["extraction"] = extraction
                    budget = budget_contract(
                        query, extraction["prices"], extraction["budget_state"]
                    )
                    try:
                        reasons.extend(budget_reasons(budget, runtime["price"]["price"]))
                        runtime = runtime_facts(product, budget)
                        output["runtime"] = runtime
                        output["budget"] = budget
                    except ValueError as exc:
                        reasons.append(str(exc))
                    if (
                        runtime["runtime_reward_type"] != "gold_purchase"
                        or not runtime["runtime_reward_valid"]
                    ):
                        reasons.append("runtime_gold_not_reachable")
                    if not reasons:
                        option_axes = {
                            str(i): axis for i, axis in enumerate(runtime["selected_options"])
                        }
                        option_values = {
                            key: runtime["selected_options"][axis]
                            for key, axis in option_axes.items()
                        }
                        # Raw axes such as "颜色分类" often actually contain size/bundle
                        # facts. Hide misleading axis names from semantic extraction.
                        facts = evidence_view(product, option_values)
                        output["option_axis_ids"] = option_axes
                        verified = model.call(
                            "verify",
                            {
                                "requirements": [
                                    {"id": i, **r} for i, r in enumerate(extraction["requirements"])
                                ],
                                "evidence": product_evidence(facts),
                                "selected_options": option_values,
                            },
                        )
                        calls.append(verified)
                        output["verification_key"] = verified["key"]
                        if "error" in verified:
                            reasons.append(verified["error"])
                        else:
                            output["verification"] = verified["result"]
                            try:
                                grounded, verification_reasons = ground_verification(
                                    verified["result"],
                                    extraction["requirements"],
                                    facts,
                                    option_values,
                                )
                                output["verification"] = grounded
                                reasons.extend(verification_reasons)
                            except ValueError as exc:
                                reasons.append(str(exc))
    output["passed"] = not reasons
    output["reasons"] = sorted(set(reasons))
    return output, calls


def controlled_cases():
    """Independent positive/negative controls, not extra training tasks."""
    base = {
        "asin": "control",
        "title": "黑色500ml保温杯",
        "category": "家居›杯具›保温杯",
        "shop_name": "测试店",
        "full_description": "黑色500ml保温杯，单只装。",
        "attribute": ["保温", "单只装"],
        "images": ["unused"],
        "pricing": [80],
        "customization_options": {
            "规格": [
                {
                    "value": "黑色500ml单只装",
                    "price": 80,
                    "is_available": True,
                }
            ]
        },
        "instructions": [
            {
                "asin": "control",
                "instruction": "买一只黑色500毫升保温杯，价格不超过100元。",
                "attributes": ["保温"],
                "instruction_options": ["黑色500ml单只装"],
            }
        ],
    }
    rows = []

    def add(name, expected, query=None, price=None):
        product = deepcopy(base)
        if query is not None:
            product["instructions"][0]["instruction"] = query
        if price is not None:
            product["pricing"] = [price]
            product["customization_options"]["规格"][0]["price"] = price
        rows.append(
            {"case_id": name, "group": "control", "expected_pass": expected, "product": product}
        )

    add("valid", True)
    add("wrong_color", False, "买一只红色500毫升保温杯，价格不超过100元。")
    add("wrong_capacity", False, "买一只黑色750毫升保温杯，价格不超过100元。")
    add("wrong_quantity", False, "买两只黑色500毫升保温杯，价格不超过100元。")
    add("over_budget", False, price=120)
    add("budget_equal", True, price=100)
    add("strict_budget_equal", False, "买一只黑色500毫升保温杯，价格低于100元。", 100)
    add("below_price_minimum", False, "买一只黑色500毫升保温杯，价格高于90元。")
    add("unit_equivalence", True, "买一只黑色0.5升保温杯，价格不超过100元。")
    add("unspecified_color", True, "买一只500毫升保温杯，价格不超过100元。")
    add("optional_color", True, "买一只500毫升保温杯，最好红色，价格不超过100元。")
    add("chinese_budget_canonical", True, "买一只黑色500毫升保温杯，价格不超过一百元。")
    add("component_price", False, "买一只黑色500毫升保温杯，杯盖配件价格不超过10元。")
    add("missing_price", False)
    rows[-1]["product"]["customization_options"]["规格"][0]["price"] = None
    add("missing_instruction", False)
    rows[-1]["product"]["instructions"][0]["instruction"] = ""
    add("title_lists_other_capacity", False, "买一只黑色750毫升保温杯，价格不超过100元。")
    rows[-1]["product"]["title"] = "黑色500ml/750ml保温杯"
    add("negation", False, "买一只黑色500毫升保温杯，不能带蓝牙，价格不超过100元。")
    rows[-1]["product"]["full_description"] += "内置蓝牙。"
    rows[-1]["product"]["attribute"].append("内置蓝牙")

    def domain_case(name, expected, title, category, attrs, query, options):
        product = deepcopy(base)
        product.update(title=title, category=category, full_description=title, attribute=attrs)
        product["customization_options"] = {
            axis: [{"value": value, "price": 80, "is_available": True}]
            for axis, value in options.items()
        }
        product["instructions"][0].update(
            instruction=query, attributes=attrs, instruction_options=list(options.values())
        )
        rows.append(
            {"case_id": name, "group": "control", "expected_pass": expected, "product": product}
        )

    for size, expected in [("7号GB55", True), ("8号GB60", False)]:
        domain_case(
            "needle_" + size,
            expected,
            "DBX1工业平车防热机针",
            "工业›缝纫›机针",
            ["工业平车", "防热"],
            "买7号DBX1工业平车防热机针，价格不超过100元。",
            {"规格": "DBX1防热" + size},
        )
    for size, expected in [("M", True), ("L", False)]:
        domain_case(
            "shirt_" + size,
            expected,
            "纯棉衬衫，蓝色，尺码M/L",
            "服饰›上衣›衬衫",
            ["纯棉"],
            "买蓝色M码纯棉衬衫，预算不超过100元。",
            {"颜色分类": "蓝色", "尺码": size},
        )
    for variant, expected in [("RTX3080 10G", True), ("RTX3070 8G", False)]:
        domain_case(
            "gpu_" + variant,
            expected,
            "二手显卡RTX3080/RTX3070",
            "电脑›硬件›显卡",
            ["二手"],
            "买二手RTX3080显卡，要10G显存，预算不超过100元。",
            {"型号": variant},
        )
    domain_case(
        "dimensions_equivalence",
        True,
        "纯棉床单150cm*200cm",
        "家居›床品›床单",
        ["纯棉"],
        "买1.5米乘2米的纯棉床单，预算不超过100元。",
        {"尺寸": "150cm*200cm"},
    )
    for size, expected in [("S", True), ("L", False)]:
        domain_case(
            "cross_axis_size_" + size,
            expected,
            "纯棉衬衫两件装",
            "服饰›上衣›衬衫",
            ["纯棉"],
            "买S码纯棉衬衫，两件装，预算不超过100元。",
            {"颜色分类": "S-2件装", "尺码": size},
        )
    return rows


def choose_sample(products, reserved_ids, count, seed):
    # Also protect identical held-out queries, even if their IDs differ.
    reserved_queries = {products[i]["instructions"][0]["instruction"] for i in reserved_ids}
    pool = [
        i
        for i, p in enumerate(products)
        if i not in reserved_ids
        and not (
            isinstance(p, dict)
            and isinstance(p.get("instructions"), list)
            and p["instructions"]
            and isinstance(p["instructions"][0], dict)
            and p["instructions"][0].get("instruction") in reserved_queries
        )
    ]
    if count > len(pool):
        raise ValueError("sample exceeds eligible task pool")
    ids = sorted(random.Random(seed).sample(pool, count))
    return [
        {"task_id": i, "case_id": f"task_{i}", "group": "random", "product": products[i]}
        for i in ids
    ], pool


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--products",
        type=Path,
        default=ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz",
    )
    parser.add_argument("--held-out", type=Path, default=ROOT / "data/evaluation/tasks.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible /v1 URL")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--revision", required=True, help="Actual served weight revision for provenance"
    )
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Clean every source task, including existing evaluation tasks",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--diagnostic-task-ids", type=int, nargs="*", default=[])
    parser.add_argument("--must-reject-task-ids", type=int, nargs="*", default=[])
    parser.add_argument(
        "--swanlab-mode", choices=("disabled", "offline", "online"), default="offline"
    )
    args = parser.parse_args(argv)
    if (
        (not args.all and not 1 <= args.sample_size <= 500)
        or args.workers < 1
        or args.max_tokens < 1
    ):
        parser.error("pilot sample-size must be 1..500; workers/max-tokens must be positive")
    start = time.perf_counter()
    with gzip.open(args.products, "rt", encoding="utf-8") as stream:
        products = json.load(stream)
    # A source reordering/deduplication would invalidate the environment's task IDs.
    asins = [p.get("asin") for p in products if isinstance(p, dict)]
    if len(asins) != len(set(asins)):
        raise ValueError("catalog has duplicate product IDs; task-ID mapping needs repair")
    if any(
        isinstance(p, dict)
        and isinstance(p.get("instructions"), list)
        and len(p["instructions"]) != 1
        for p in products
    ):
        raise ValueError("catalog is not one-record/one-task; do not infer task IDs from rows")
    reserved = {r["task_id"] for r in read_rows(args.held_out)}
    if args.all:
        pool = list(range(len(products)))
        rows = [
            {"task_id": i, "case_id": f"task_{i}", "group": "corpus", "product": p}
            for i, p in enumerate(products)
        ]
    else:
        rows, pool = choose_sample(products, reserved, args.sample_size, args.seed)
    natural_group = "corpus" if args.all else "random"
    for task_id in sorted(set(args.diagnostic_task_ids) | set(args.must_reject_task_ids)):
        if task_id not in pool:
            raise ValueError(f"diagnostic task {task_id} is held out or invalid")
        if task_id not in {row.get("task_id") for row in rows}:
            rows.append(
                {
                    "task_id": task_id,
                    "case_id": f"task_{task_id}",
                    "group": "diagnostic",
                    "product": products[task_id],
                }
            )
    for row in rows:
        if row["task_id"] in args.must_reject_task_ids:
            row["expected_pass"] = False
    rows.extend(controlled_cases())
    with urlopen(
        Request(
            args.endpoint.rstrip("/") + "/models",
            headers={"Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "local")},
        ),
        timeout=args.timeout,
    ) as stream:
        served_models = json.load(stream)
    served = [m for m in served_models["data"] if m["id"] == args.model]
    if len(served) != 1:
        raise ValueError("requested model is not served by the endpoint")
    config = {
        "policy": POLICY_VERSION,
        "reward_version": REWARD_VERSION,
        "scope": "full_corpus_including_previous_evaluation" if args.all else "sample",
        "product_sha256": sha256_file(args.products),
        "held_out_sha256": sha256_file(args.held_out),
        "model": args.model,
        "revision": args.revision,
        "served_model_root": served[0].get("root"),
        "sample_size": len(pool) if args.all else args.sample_size,
        "seed": args.seed,
        "workers": args.workers,
        "max_tokens": args.max_tokens,
        "prompt_hash": fingerprint([EXTRACT_PROMPT, VERIFY_PROMPT, EXTRACT_SCHEMA, VERIFY_SCHEMA]),
        "code_hashes": {
            str(p.relative_to(ROOT)): sha256_file(p)
            for p in [
                Path(__file__),
                ROOT / "src/shopping_grpo/collection/task_cleaning.py",
                ROOT / "environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py",
                ROOT / "environments/ShopSimulator/shop_env/web_agent_site/engine/variant_price.py",
                ROOT
                / "environments/ShopSimulator/shop_env/web_agent_site/engine/reward_features.py",
                ROOT / "environments/ShopSimulator/shop_env/web_agent_site/engine/constraints.py",
                ROOT / "environments/ShopSimulator/shop_env/web_agent_site/engine/task_budget.py",
                ROOT / "environments/ShopSimulator/shop_env/web_agent_site/engine/goal.py",
            ]
        },
        "cases": [{k: v for k, v in r.items() if k != "product"} for r in rows],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("output configuration changed; use a different output directory")
    write_json(manifest, config)
    write_rows(args.output / "sample.jsonl", rows)
    setup_seconds = time.perf_counter() - start
    model = JSONModel(args, args.output)
    # Compile the JSON grammar and warm the GPU outside throughput timing.
    warmup_start = time.perf_counter()
    warmup = model.call(
        "extract",
        {"clauses": [{"id": 0, "text": "买一只黑色保温杯"}, {"id": 1, "text": "价格不超过100元"}]},
    )
    if "error" in warmup:
        raise RuntimeError("model warmup failed; no task results produced")
    warmup_seconds = time.perf_counter() - warmup_start
    write_json(args.output / "warmup.json", warmup)
    tracker = None
    if args.swanlab_mode != "disabled":
        import swanlab

        tracker = swanlab.init(
            project="shopping-grpo-data",
            experiment_name="task-cleaning-full" if args.all else "task-cleaning-pilot",
            config=config,
            mode=args.swanlab_mode,
            logdir=str(args.output / "swanlab"),
        )
    outputs = []
    all_calls = []
    timings = {}
    with (
        model.path.open("a", encoding="utf-8") as log,
        (args.output / "decisions.checkpoint.jsonl").open("w", encoding="utf-8") as checkpoint,
    ):
        for group in ("control", natural_group, "diagnostic"):
            group_rows = [row for row in rows if row["group"] == group]
            group_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process, row, model): row for row in group_rows}
                for index, future in enumerate(as_completed(futures), 1):
                    result, calls = future.result()
                    outputs.append(result)
                    checkpoint.write(json.dumps(result, ensure_ascii=False) + "\n")
                    checkpoint.flush()
                    calls = [
                        {**call, "group": group, "case_id": result["case_id"]} for call in calls
                    ]
                    all_calls.extend(calls)
                    for call in calls:
                        if not call["cached"]:
                            log.write(json.dumps(call, ensure_ascii=False) + "\n")
                    log.flush()
                    if index % 25 == 0 or index == len(group_rows):
                        print(
                            json.dumps(
                                {
                                    "group": group,
                                    "completed": index,
                                    "total": len(group_rows),
                                    "seconds": round(time.perf_counter() - group_start, 2),
                                    "passed": sum(
                                        r["passed"] for r in outputs if r["group"] == group
                                    ),
                                }
                            ),
                            flush=True,
                        )
            timings[group] = time.perf_counter() - group_start
            if group == "control":
                failures = [r for r in outputs if r["passed"] != r["expected_pass"]]
                if failures:
                    write_json(args.output / "control_failures.json", failures)
                    print("Control gate failed; corpus processing was not started.", flush=True)
                    if tracker is not None:
                        tracker.finish()
                    return 2
    outputs.sort(key=lambda x: (x["group"], x["case_id"]))
    write_rows(args.output / "results.jsonl", outputs)
    write_rows(
        args.output / "passed_tasks.jsonl",
        [{"task_id": r["task_id"]} for r in outputs if r["passed"] and "task_id" in r],
    )
    natural_results = [r for r in outputs if "task_id" in r]
    cleaned = [
        {
            "task_id": r["task_id"],
            "asin": products[r["task_id"]]["asin"],
            "instruction": products[r["task_id"]]["instructions"][0]["instruction"],
            "record_hash": r["record_hash"],
            "policy": POLICY_VERSION,
            "reward_version": REWARD_VERSION,
            "requirements": r["extraction"]["requirements"],
            "budget": r["budget"],
            "selected_options": r["runtime"]["selected_options"],
            "gold_price": r["runtime"]["price"]["price"],
        }
        for r in natural_results
        if r["passed"]
    ]
    write_rows(args.output / "cleaned_tasks.jsonl", cleaned)
    write_rows(
        args.output / "task_budgets.jsonl",
        [{"task_id": r["task_id"], "asin": r["asin"], "budget": r["budget"]} for r in cleaned],
    )
    random_results = [r for r in outputs if r["group"] == natural_group]
    control_results = [r for r in outputs if r["group"] == "control"]
    failed_controls = [
        {
            "case_id": r["case_id"],
            "expected": r["expected_pass"],
            "actual": r["passed"],
            "reasons": r["reasons"],
        }
        for r in control_results
        if r["passed"] != r["expected_pass"]
    ]
    failed_diagnostics = [
        {"task_id": r["task_id"], "reasons": r["reasons"]}
        for r in outputs
        if r["group"] != "control" and r.get("expected_pass") is False and r["passed"]
    ]
    fresh_calls = [c for c in all_calls if not c["cached"]]
    usage = Counter()
    for c in fresh_calls:
        usage.update(c["usage"])
    cached = sum(c["cached"] for c in all_calls)
    random_usage = Counter()
    for call in fresh_calls:
        if call["group"] == natural_group:
            random_usage.update(call["usage"])
    failed_model_calls = sum("error" in c for c in all_calls)
    infrastructure_failures = sum(c.get("error_kind") == "transport" for c in all_calls)
    seconds_per_task = timings[natural_group] / len(random_results)
    summary = {
        "policy": POLICY_VERSION,
        "reward_version": REWARD_VERSION,
        "scope": config["scope"],
        "source_tasks": len(products),
        "screened_tasks": len(random_results),
        "included_previous_evaluation_tasks": sum(
            r["task_id"] in reserved for r in natural_results
        ),
        "eligible_tasks": len(pool),
        "reserved_task_ids": len(reserved),
        "random_tasks": len(random_results),
        "stage1_pass": sum(r["stage1_pass"] for r in random_results),
        "stage1_cpu_seconds": sum(r["stage1_seconds"] for r in random_results),
        "passed": sum(r["passed"] for r in random_results),
        "rejected": sum(not r["passed"] for r in random_results),
        "reason_counts": dict(Counter(reason for r in random_results for reason in r["reasons"])),
        "legacy_budget_parser_differences": sum(
            r["runtime"]["legacy_parser_upper"] != r["runtime"]["runtime_upper"]
            for r in random_results
            if "budget" in r
        ),
        "controls": len(control_results),
        "failed_controls": failed_controls,
        "failed_diagnostics": failed_diagnostics,
        "control_false_accepts": sum(
            r["passed"] and not r["expected_pass"] for r in control_results
        ),
        "control_false_rejects": sum(
            not r["passed"] and r["expected_pass"] for r in control_results
        ),
        "fresh_model_calls": len(fresh_calls),
        "failed_model_calls": failed_model_calls,
        "model_infrastructure_failures": infrastructure_failures,
        "cached_model_calls": cached,
        "retry_calls": sum(c["attempts"] > 1 for c in fresh_calls),
        "usage": dict(usage),
        "random_usage": dict(random_usage),
        "projected_eligible_usage": None
        if cached
        else {
            key: round(value * len(pool) / len(random_results))
            for key, value in random_usage.items()
        },
        "setup_seconds": setup_seconds,
        "warmup_seconds": warmup_seconds,
        "group_seconds": timings,
        "seconds_per_random_task": seconds_per_task,
        "projected_eligible_seconds": None if cached else seconds_per_task * len(pool),
        "projected_full_corpus_seconds": None if cached else seconds_per_task * len(products),
        "projection_note": "Warm server, same concurrency and rejection mix; excludes model startup.",
        "pilot_quality_gate_passed": (
            not failed_controls and not failed_diagnostics and not infrastructure_failures
        ),
        "total_seconds": time.perf_counter() - start,
    }
    write_json(args.output / "summary.json", summary)
    if tracker is not None:
        tracker.log(
            {
                "random_passed": summary["passed"],
                "random_rejected": summary["rejected"],
                "control_false_accepts": summary["control_false_accepts"],
                "control_false_rejects": summary["control_false_rejects"],
                "random_seconds": timings[natural_group],
                "projected_eligible_seconds": summary["projected_eligible_seconds"] or 0,
            }
        )
        tracker.finish()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["pilot_quality_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
