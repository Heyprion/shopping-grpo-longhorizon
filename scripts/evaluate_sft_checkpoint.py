#!/usr/bin/env python3
"""Merge the validation-best SFT adapter and run the fixed 200-task evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from shopping_grpo.collection.data_contract import validate_current_sft
from shopping_grpo.environment.manifest import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def validate_training_summary(summary):
    selection = summary.get("model_selection", {})
    if (
        selection.get("metric") != "eval_assistant_nll"
        or not selection.get("final_adapter_is_best")
        or not selection.get("best_checkpoint")
        or not math.isfinite(selection.get("best_metric", math.inf))
    ):
        raise ValueError(
            "SFT adapter must be selected by the lowest validation assistant-token NLL"
        )
    args = summary["arguments"]
    if (
        args["epochs"] != 3
        or args.get("stop_after_steps") is not None
        or args.get("stop_after_epoch") is not None
        or args.get("max_steps", -1) > 0
    ):
        raise ValueError("Automatic final evaluation requires the full three-epoch recipe")
    if summary["train_data"]["dropped"] or summary["validation_data"]["dropped"]:
        raise ValueError("The frozen SFT dataset was not fully used")
    return selection


def wait_service(url, process, timeout=1200):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Service exited before becoming ready; inspect its log")
        try:
            if url.startswith("tcp:"):
                with socket.create_connection(("127.0.0.1", int(url[4:])), timeout=2):
                    return
            else:
                with urlopen(url, timeout=3) as response:
                    if response.status == 200:
                        return
        except (URLError, OSError):
            time.sleep(1)
    raise TimeoutError("Service did not become ready: " + url)


def stop_service(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument(
        "--merged-output", type=Path, default=ROOT / "outputs/models/sft-stage1-v1-best-merged"
    )
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/evaluation/sft-stage1-v1")
    p.add_argument("--environment-port", type=int, default=15704)
    p.add_argument("--llm-port", type=int, default=18004)
    args = p.parse_args()
    os.chdir(ROOT)
    validate_current_sft(ROOT)
    summary_path = args.adapter / "train_summary.json"
    summary = json.loads(summary_path.read_text())
    selection = validate_training_summary(summary)
    for name, expected in summary["data_sha256"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError("Dataset changed after training: " + name)
    adapter_hash = sha256_file(args.adapter / "adapter_model.safetensors")
    contract = {
        "adapter_sha256": adapter_hash,
        "train_summary_sha256": sha256_file(summary_path),
        "test_tasks_sha256": sha256_file(ROOT / "data/evaluation/tasks.jsonl"),
        "model_selection": selection,
        "data_sha256": summary["data_sha256"],
    }
    if len((ROOT / "data/evaluation/tasks.jsonl").read_text().splitlines()) != 200:
        raise ValueError("Expected the fixed 200-task evaluation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "run_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Evaluation output belongs to a different model or dataset")
    elif any(args.output_dir.iterdir()):
        raise ValueError("Evaluation output is nonempty without a matching run contract")
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    merge_manifest = args.merged_output / "merge_manifest.json"
    if merge_manifest.exists():
        if json.loads(merge_manifest.read_text())["source"].get("adapter_sha256") != adapter_hash:
            raise ValueError("Merged model belongs to a different adapter")
    else:
        command = [
            sys.executable,
            "scripts/merge_lora_adapter.py",
            "--base-model",
            summary["arguments"]["model"],
            "--adapter",
            str(args.adapter),
            "--output",
            str(args.merged_output),
            "--bf16",
        ]
        if summary["arguments"].get("revision"):
            command.extend(["--revision", summary["arguments"]["revision"]])
        subprocess.run(command, cwd=ROOT, check=True)
    for port in (args.environment_port, args.llm_port):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    processes = []
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print("Starting the fixed 200-task evaluation; logs:", args.output_dir, flush=True)
    try:
        with ExitStack() as stack:
            env_log = stack.enter_context((args.output_dir / "environment.log").open("a"))
            llm_log = stack.enter_context((args.output_dir / "model-server.log").open("a"))
            env = {
                **os.environ,
                "SHOPPING_TASK_BUDGETS": str(ROOT / "data/cleaned/task_budgets.jsonl"),
                "SHOPSIM_ENV_SLOTS": "8",
                "SHOPSIM_PORT": str(args.environment_port),
            }
            environment = subprocess.Popen(
                ["bash", "scripts/start_environment.sh"],
                cwd=ROOT,
                env=env,
                stdout=env_log,
                stderr=subprocess.STDOUT,
            )
            processes.append(environment)
            serving_env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": os.environ.get("SFT_EVAL_CUDA_VISIBLE_DEVICES", "0"),
                "LLM_PORT": str(args.llm_port),
                "SERVED_MODEL_NAME": "shopping-sft-stage1-v1",
            }
            server = subprocess.Popen(
                ["bash", "scripts/serve_model.sh", str(args.merged_output)],
                cwd=ROOT,
                env=serving_env,
                stdout=llm_log,
                stderr=subprocess.STDOUT,
            )
            processes.append(server)
            wait_service("tcp:" + str(args.environment_port), environment)
            wait_service(f"http://127.0.0.1:{args.llm_port}/v1/models", server)
            subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_shop_benchmark.py",
                    "--benchmark",
                    "data/evaluation/tasks.jsonl",
                    "--output",
                    str(args.output_dir / "trajectories.jsonl"),
                    "--summary",
                    str(args.output_dir / "summary.json"),
                    "--base-url",
                    f"http://127.0.0.1:{args.environment_port}",
                    "--model",
                    "shopping-sft-stage1-v1",
                    "--llm-base-url",
                    f"http://127.0.0.1:{args.llm_port}/v1",
                    "--api-key",
                    "EMPTY",
                    "--swanlab",
                    "--swanlab-project",
                    "shopping-grpo-longhorizon",
                    "--swanlab-group",
                    "04-SFT评测",
                    "--swanlab-job-type",
                    "eval",
                    "--swanlab-description",
                    "第一版 SFT 最佳验证检查点合并模型在固定 200 题 Strict-Stratified-200 上的正式评测（T=0，max_steps=35）。",
                    "--swanlab-tags",
                    "EVAL,SFT,Strict200",
                    "--swanlab-run-name",
                    date + "-EVAL-第一版SFT最佳模型",
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )
            subprocess.run(
                [sys.executable, "scripts/build_eval_report.py", "--run-dir", str(args.output_dir)],
                cwd=ROOT,
                check=True,
            )
    finally:
        for process in reversed(processes):
            stop_service(process)
    evaluation = json.loads((args.output_dir / "summary.json").read_text())
    report = {
        "model_selection": selection,
        "test_results": evaluation,
        "merged_model": str(args.merged_output),
    }
    (args.output_dir / "final_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
