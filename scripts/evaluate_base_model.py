#!/usr/bin/env python3
"""Run the official Qwen3.5-4B baseline with owned two-GPU serving services."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SERVED_MODEL = "qwen3.5-4b"
MODEL_OPTIONS = [
    "--tensor-parallel-size", "2", "--dtype", "bfloat16", "--seed", "42",
    "--generation-config", "vllm", "--language-model-only",
    "--reasoning-parser", "qwen3",
    "--default-chat-template-kwargs", '{"enable_thinking":false}',
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只显示命令，不启动服务或评测")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--model-size", choices=("2b", "4b"), default="4b")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        args.output_dir = ROOT / "outputs/evaluation" / f"qwen3.5-{args.model_size}-strict200-tp2-{stamp}"
    elif not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    return args


def reserve_port(stack):
    probe = stack.enter_context(socket.socket())
    probe.bind(("127.0.0.1", 0))
    return probe.getsockname()[1]


def prepare(args, environment_port, llm_port):
    model = MODEL if args.model_size == "4b" else ROOT / "models/Qwen3.5-2B"
    revision = REVISION if args.model_size == "4b" else "15852e8c16360a2fea060d615a32b45270f8a8fc"
    served_model = f"qwen3.5-{args.model_size}"
    benchmark = ROOT / "data/evaluation/tasks.jsonl"
    metadata = json.loads((ROOT / "data/evaluation/metadata.json").read_text())
    rows = [json.loads(line) for line in benchmark.read_text().splitlines() if line.strip()]
    digest = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    if (
        len(rows) != 200 or len({r["task_id"] for r in rows}) != 200
        or digest != metadata["sha256"]
        or metadata["reward"] != "shopsimulator-reward-strict-v1"
        or metadata["environment"] != "shopsimulator-environment-v2.1"
    ):
        raise ValueError("当前固定 200 题测试集与 Strict Reward 契约不符")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    required = [model / name for name in set(index["weight_map"].values())]
    required += [model / name for name in ("config.json", "tokenizer.json", "chat_template.jinja")]
    required += [ROOT / "data/cleaned/task_budgets.jsonl"]
    if not all(path.is_file() and path.stat().st_size for path in required):
        raise ValueError("本地模型权重、tokenizer、模板或规范任务预算缺失")
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
        "SHOPPING_TASK_BUDGETS": str(ROOT / "data/cleaned/task_budgets.jsonl"),
        "SHOP_ENV_CONFIG": str(ROOT / "environments/ShopSimulator/shop_env/configs/environment.json"),
        "SHOP_MAX_STEPS": "35",
        "SHOPSIM_ENV_SLOTS": "8",
        "SHOPSIM_PORT": str(environment_port),
        "LLM_PORT": str(llm_port),
        "SERVED_MODEL_NAME": served_model,
        "SHOPSIM_BASE_URL": f"http://127.0.0.1:{environment_port}",
        "LLM_BASE_URL": f"http://127.0.0.1:{llm_port}/v1",
        "LLM_API_KEY": "EMPTY",
        "LLM_MAX_NUM_SEQS": "8",
        "LLM_GPU_MEMORY_UTILIZATION": "0.6",
        "EVAL_OUTPUT_DIR": str(args.output_dir),
    }
    commands = [
        ["bash", "scripts/start_environment.sh"],
        ["bash", "scripts/serve_model.sh", str(model), *MODEL_OPTIONS],
        ["bash", "scripts/evaluate.sh", args.output_dir.name],
    ]
    contract = {
        "model": f"Qwen/Qwen3.5-{args.model_size.upper()}", "revision": revision,
        "model_path": str(model), "benchmark_sha256": digest,
        "environment": metadata["environment"], "reward": metadata["reward"],
        "tasks": 200, "tensor_parallel_size": 2, "thinking": False,
        "dtype": "bfloat16", "seed": 42,
        "environment_variables": {k: environment[k] for k in (
            "CUDA_VISIBLE_DEVICES", "SHOPPING_TASK_BUDGETS", "SHOPSIM_PORT", "LLM_PORT",
        )},
        "commands": commands,
    }
    return commands, environment, contract


def wait_service(url, processes, log_path, expected_model=None, timeout=1200):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError(f"服务提前退出，请检查日志：{log_path.parent}")
        try:
            with urlopen(url, timeout=3) as response:
                if expected_model is None:
                    return
                models = json.load(response).get("data", [])
                if any(row.get("id") == expected_model for row in models):
                    return
        except HTTPError as exc:
            # GET=405 checks the POST-only route without leasing any test task.
            if expected_model is None and exc.code == 405:
                return
        except (URLError, OSError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError(f"等待服务就绪超时，请检查日志：{log_path}")


def stop_owned(process):
    """Stop only the process group created by this invocation, including workers."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    # The group leader can exit before a CUDA worker; also reap its group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def main(argv=None):
    args = parse_args(argv)
    with ExitStack() as reservations:
        environment_port = reserve_port(reservations)
        llm_port = reserve_port(reservations)
        commands, environment, contract = prepare(args, environment_port, llm_port)
    if args.dry_run:
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        print("dry-run 完成：未启动服务、加载模型或执行 200 题评测。")
        return

    visible = environment["CUDA_VISIBLE_DEVICES"].split(",")
    if len(visible) < 2:
        raise ValueError("需要两张可见 GPU；CUDA_VISIBLE_DEVICES 至少指定两张卡")
    os.environ["CUDA_VISIBLE_DEVICES"] = environment["CUDA_VISIBLE_DEVICES"]
    # Import CUDA only on execution; preparation/dry-run works without GPUs.
    import torch

    if torch.cuda.device_count() < 2:
        raise RuntimeError("当前没有两张可用 GPU，无法启动双卡评测")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"输出目录非空，请用 --output-dir 指定新目录：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_from:
        previous = json.loads((args.resume_from / "run_contract.json").read_text())
        for key in ("model", "revision", "benchmark_sha256", "environment", "reward",
                    "tasks", "tensor_parallel_size", "thinking", "dtype", "seed"):
            if previous.get(key) != contract.get(key):
                raise ValueError(f"续跑来源契约不一致：{key}")
        shutil.copyfile(args.resume_from / "trajectories.jsonl", args.output_dir / "trajectories.jsonl")
    (args.output_dir / "run_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n"
    )
    processes = []
    previous_handler = signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    print(f"启动双卡 Qwen3.5-{args.model_size.upper()} 评测，日志和结果：{args.output_dir}", flush=True)
    try:
        with ExitStack() as stack:
            for command, filename in zip(commands[:2], ("environment.log", "model-server.log")):
                log = stack.enter_context((args.output_dir / filename).open("w"))
                processes.append(subprocess.Popen(
                    command, cwd=ROOT, env=environment, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True,
                ))
            print("等待购物环境和双卡模型服务就绪……", flush=True)
            wait_service(
                environment["SHOPSIM_BASE_URL"] + "/api/shop_agent", processes,
                args.output_dir / "environment.log",
            )
            wait_service(
                environment["LLM_BASE_URL"] + "/models", processes,
                args.output_dir / "model-server.log", expected_model=environment["SERVED_MODEL_NAME"],
            )
            print("服务已就绪，开始固定 200 题评测。", flush=True)
            evaluation = subprocess.Popen(
                commands[2], cwd=ROOT, env=environment, start_new_session=True,
            )
            processes.append(evaluation)
            returncode = evaluation.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, commands[2])
    finally:
        for process in reversed(processes):
            stop_owned(process)
        signal.signal(signal.SIGTERM, previous_handler)
    summary = json.loads((args.output_dir / "summary.json").read_text())
    if summary.get("expected_tasks") != 200 or summary.get("completed_tasks") != 200:
        raise RuntimeError(f"200 题未完整结束，请检查结果：{args.output_dir}")
    print(f"评测完成，Strict Success：{summary['strict_success_rate']:.2%}", flush=True)
    print(f"结果：{args.output_dir / 'summary.json'}", flush=True)
    print(f"报告：{args.output_dir / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
