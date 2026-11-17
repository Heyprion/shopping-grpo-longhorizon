#!/usr/bin/env python3
"""Time-bounded frozen-SFT sampling study; never trains, merges, or tests held-out tasks."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from shopping_grpo.environment.manifest import sha256_file
from shopping_grpo.training.grpo.sampling_study import (
    PLAN,
    SETTINGS,
    build_report,
    choose_tasks,
    describe_group,
    group8_needed,
    sample_seed,
    select_candidate,
    task_pairs,
    write_json,
)

if __package__:
    from .check_grpo_runtime import compose_runtime_config, validate_environment_contract
    from .evaluate_sft_checkpoint import wait_service
    from .run_grpo_pipeline import validate_sft_source
else:
    from check_grpo_runtime import compose_runtime_config, validate_environment_contract
    from evaluate_sft_checkpoint import wait_service
    from run_grpo_pipeline import validate_sft_source

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "outputs/models/sft-stage1-v1-best-merged"
STUDY_OVERRIDES = [
    "data.seed=42",
    "data.val_batch_size=50",
    "data.dataloader_num_workers=0",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
    "actor_rollout_ref.actor.fsdp_config.param_offload=false",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
    "actor_rollout_ref.actor.fsdp_config.reshard_after_forward=false",
    "actor_rollout_ref.model.use_fused_kernels=true",
    "actor_rollout_ref.model.fused_kernel_options.impl_backend=triton",
    "actor_rollout_ref.model.use_liger=false",
    "actor_rollout_ref.model.lora_rank=0",
    "actor_rollout_ref.model.lora.rank=0",
    "actor_rollout_ref.rollout.nnodes=1",
    "actor_rollout_ref.rollout.n_gpus_per_node=2",
    "actor_rollout_ref.rollout.load_format=auto",
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.seed=42",
    "trainer.logger=[console,swanlab]",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pre-rl-sampling" / stamp)
    parser.add_argument(
        "--budget-minutes",
        type=int,
        default=90,
        help="总墙钟预算，包含服务启动和清理，最多 120 分钟（默认 90）",
    )
    parser.add_argument("--environment-port", type=int, default=15706)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.output = args.output.expanduser().resolve()
    if not 5 <= args.budget_minutes <= 120:
        parser.error("--budget-minutes 必须在 5–120 之间")
    if not 1024 <= args.environment_port <= 65535:
        parser.error("端口必须在 1024–65535 之间")
    return args


def configure_environment(args):
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        os.environ.pop(name, None)
    os.environ.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONHASHSEED": "42",
            "SHOPPING_GRPO_SEED": "42",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "SHOPPING_TASK_BUDGETS": str(ROOT / "data/cleaned/task_budgets.jsonl"),
            "SHOPPING_ENV_MANIFEST": str(ROOT / "data/environment.json"),
            "SHOPPING_ENVIRONMENT_VERSION": "shopsimulator-environment-v2.1",
            "SHOPPING_TOOL_CONFIG": str(ROOT / "configs/tools.json"),
            "SHOPPING_GRPO_ROOT": str(ROOT),
            "GRPO_CONFIG_NAME": "grpo",
            "GRPO_MODEL_PATH": str(MODEL),
            "GRPO_TRAIN_FILE": str(ROOT / "data/grpo/train.parquet"),
            "GRPO_VAL_FILE": str(ROOT / "data/grpo/validation.parquet"),
            "GRPO_OUTPUT_DIR": str(args.output),
            "SHOPSIM_BASE_URL": f"http://127.0.0.1:{args.environment_port}",
            "SHOPSIM_ENV_SLOTS": "64",
            "SHOPSIM_PORT": str(args.environment_port),
            "SHOP_MAX_STEPS": "35",
            "SHOPPING_LENGTH_SHAPING_ENABLE": "false",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "RAY_ADDRESS": "local",
            "SWANLAB_MODE": "online",
            "SWANLAB_LOG_DIR": str(args.output / "swanlab"),
        }
    )
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    if len([s for s in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if s.strip()]) != 2:
        raise ValueError("该诊断使用 TP1 × 2 副本，需要两张可见 GPU")
    if not os.environ.get("SWANLAB_API_KEY"):
        from swanlab.sdk.internal.settings import Settings

        api_key = Settings().api_key
        if hasattr(api_key, "get_secret_value"):
            api_key = api_key.get_secret_value()
        if not api_key:
            raise ValueError("采样诊断要求已登录 SwanLab，或设置 SWANLAB_API_KEY")
        os.environ["SWANLAB_API_KEY"] = str(api_key)


def preflight(args):
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("拒绝覆盖已有实验，请指定新的 --output：" + str(args.output))
    configure_environment(args)
    # Reuse the repository verifier; no cleaning, labelling or allocation is run.
    result = subprocess.run(
        [sys.executable, "scripts/prepare_data_splits.py", "verify"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    print("固定数据与训练/验证/测试隔离核验通过。", flush=True)
    source = validate_sft_source()
    validate_environment_contract()
    compose_runtime_config(STUDY_OVERRIDES)
    panels = choose_tasks(ROOT)
    files = [
        "data/grpo/train.parquet",
        "data/grpo/train.jsonl",
        "data/splits/labels.jsonl",
        "data/splits/manifest.json",
        "data/cleaned/task_budgets.jsonl",
        "data/environment.json",
        "configs/agent_loop.yaml",
        "configs/tools.json",
    ]
    plan = {
        **PLAN,
        "budget_minutes": args.budget_minutes,
        "output": str(args.output),
        "environment_port": args.environment_port,
        "sft_source": source,
        "data_sha256": {p: sha256_file(ROOT / p) for p in files},
        "screen_task_ids": [r["task_id"] for r in panels["screen"]],
        "confirmation_task_ids": [r["task_id"] for r in panels["confirmation"]],
        "warmup_task_ids": [r["task_id"] for r in panels["warmup"]],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "total_runtime_includes_setup_and_cleanup": True,
        "conditional_group8_min_remaining_seconds": 180,
    }
    return plan, panels


def stop_tree(process, descendants=None, grace=8):
    """Only terminate this child's descendants and process group, never other Ray jobs."""
    import psutil

    children = list(descendants or [])
    try:
        children.extend(psutil.Process(process.pid).children(recursive=True))
    except psutil.Error:
        pass
    unique = {}
    for child in children:
        try:
            if child.is_running():
                unique[child.pid, child.create_time()] = child
        except psutil.Error:
            pass
    for child in reversed(list(unique.values())):
        try:
            child.terminate()
        except psutil.Error:
            pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    _, alive = psutil.wait_procs(list(unique.values()), timeout=grace)
    for child in alive:
        try:
            child.kill()
        except psutil.Error:
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def append_row(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


class TrajectoryClient:
    """Pair random draws across settings without reusing a seed across turns."""

    def __init__(self, client, task_id, sample_index):
        self.client, self.task_id, self.sample_index = client, task_id, sample_index
        self.turn = 0

    async def generate(self, request_id, **kwargs):
        params = dict(kwargs["sampling_params"])
        params["seed"] = sample_seed(self.task_id, self.sample_index, self.turn)
        self.turn += 1
        return await self.client.generate(request_id, **{**kwargs, "sampling_params": params})


class NativeSampler:
    """Use native vLLM servers and the production shopping agent, without a trainer."""

    def __init__(self, config, client, model, tools):
        self.config, self.client, self.model, self.tools = config, client, model, tools
        self.agent_config = None

    async def sample(self, row, sample_index, setting):
        import hydra
        from omegaconf import OmegaConf
        from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
        from verl.utils.dataset.rl_dataset import RLHFDataset

        if self.agent_config is None:
            self.agent_config = OmegaConf.load(ROOT / "configs/agent_loop.yaml")[0]
        task_id = row["extra_info"]["task_id"]
        agent = hydra.utils.instantiate(
            self.agent_config,
            trainer_config=DictConfigWrap(self.config),
            server_manager=TrajectoryClient(self.client, task_id, sample_index),
            tokenizer=self.model.tokenizer,
            processor=self.model.processor,
            dataset_cls=RLHFDataset,
            data_config=DictConfigWrap(self.config.data),
            tools=ToolListWrap(self.tools),
        )
        params = {
            **SETTINGS[setting],
            "repetition_penalty": 1.0,
            "logprobs": True,
            "max_tokens": 512,
            "seed": sample_seed(task_id, sample_index),
        }
        result = await agent.run(
            params,
            raw_prompt=copy.deepcopy(row["prompt"]),
            extra_info=copy.deepcopy(row["extra_info"]),
        )
        return result.extra_fields["shopping"], sum(result.response_mask)


async def initialize_sampler(config):
    import ray
    from omegaconf import OmegaConf
    from verl.tools.tool_registry import load_all_tools
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    from verl.utils.config import omega_conf_to_dataclass
    from verl.utils.rollout_trace import RolloutTraceConfig
    from verl.workers.rollout.llm_server import LLMServerManager

    runtime = OmegaConf.to_container(
        OmegaConf.merge(get_ppo_ray_runtime_env(), config.ray_kwargs.ray_init.runtime_env)
    )
    runtime.setdefault("env_vars", {}).update(
        {
            "SHOPPING_TASK_BUDGETS": os.environ["SHOPPING_TASK_BUDGETS"],
            "PYTHONPATH": str(ROOT / "src"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "SHOPPING_GRPO_SEED": "42",
            "PYTHONHASHSEED": "42",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    ray.init(
        address="local",
        num_cpus=32,
        num_gpus=2,
        include_dashboard=False,
        object_store_memory=2147483648,
        runtime_env=runtime,
        _temp_dir=tempfile.mkdtemp(prefix="shopping-sampling-ray-"),
    )
    servers = await LLMServerManager.create(config)
    # Construct only tokenizer/processor metadata locally; no actor or optimizer exists.
    model = omega_conf_to_dataclass(config.actor_rollout_ref.model)
    tools = load_all_tools(str(ROOT / "configs/tools.json"), None)
    RolloutTraceConfig.init("shopping-grpo", "frozen-sft-sampling", backend=None)
    return NativeSampler(config, servers.get_client(), model, tools)


async def study_worker(args):
    import pyarrow.parquet as pq
    import ray
    import swanlab

    configure_environment(args)
    output = args.output
    plan = json.loads((output / "plan.json").read_text())
    panels = json.loads((output / "panels.json").read_text())
    for path, expected in plan["data_sha256"].items():
        if sha256_file(ROOT / path) != expected:
            raise ValueError("实验配置或数据在启动后发生变化：" + path)
    await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "scripts/check_grpo_runtime.py", *STUDY_OVERRIDES],
        cwd=ROOT,
        check=True,
        timeout=90,
    )
    table = {
        r["extra_info"]["task_id"]: r
        for r in pq.read_table(ROOT / "data/grpo/train.parquet").to_pylist()
    }
    deadline = float(os.environ["SHOPPING_STUDY_DEADLINE_UNIX"]) - 45
    name = datetime.now(timezone.utc).strftime("%Y-%m-%d") + "-RL-SFT采样诊断-" + output.name
    swanlab.init(
        project="shopping-grpo",
        experiment_name=name,
        config=plan,
        logdir=str(output / "swanlab"),
        mode="online",
    )
    config = compose_runtime_config(STUDY_OVERRIDES)
    all_groups = []
    completed_trajectories = 0
    status = "failed"
    try:
        sampler = await initialize_sampler(config)

        async def run_pair(phase, setting, pair, group):
            nonlocal completed_trajectories
            if time.time() >= deadline - 20:
                raise TimeoutError("study time budget reached")
            started = time.monotonic()
            jobs = [(task, index) for task in pair for index in range(group)]
            # Exceptions invalidate the group; they are never imputed as valid failures.
            results = await asyncio.wait_for(
                asyncio.gather(
                    *(sampler.sample(table[t["task_id"]], i, setting) for t, i in jobs),
                    return_exceptions=True,
                ),
                timeout=min(180, max(1, deadline - time.time())),
            )
            elapsed = time.monotonic() - started
            infos = defaultdict(list)
            tokens = defaultdict(list)
            for (task, index), result in zip(jobs, results, strict=True):
                if isinstance(result, BaseException):
                    info = {
                        "task_id": task["task_id"],
                        "done": False,
                        "infrastructure_invalid": True,
                        "error": type(result).__name__ + ":" + str(result),
                        "reward_unverifiable": True,
                        "reward_valid": False,
                        "reward": {
                            "terminal_utility": 0.0,
                            "purchase_success": 0.0,
                            "sampling_invalid": True,
                            "infrastructure_invalid": True,
                            "reward_unverifiable": True,
                        },
                    }
                    count = 0
                else:
                    info, count = result
                infos[task["task_id"]].append(info)
                tokens[task["task_id"]].append(count)
                append_row(
                    output / "trajectories.jsonl",
                    {
                        "phase": phase,
                        "setting": setting,
                        "task_id": task["task_id"],
                        "difficulty": task["difficulty"],
                        "sample_index": index,
                        "sample_seed": sample_seed(task["task_id"], index),
                        "generated_tokens": count,
                        "shopping": info,
                    },
                )
            for task in pair:
                record = describe_group(
                    phase,
                    setting,
                    task,
                    infos[task["task_id"]],
                    elapsed / len(pair),
                    tokens[task["task_id"]],
                )
                target = "warmup-groups.jsonl" if phase == "warmup" else "groups.jsonl"
                append_row(output / target, record)
                if phase != "warmup":
                    all_groups.append(record)
            completed_trajectories += len(jobs)
            if all(isinstance(result, BaseException) for result in results):
                raise RuntimeError("整批采样均抛出异常，请查看 trajectories.jsonl 中的 error 字段")
            write_json(
                output / "progress.json",
                {
                    "phase": phase,
                    "setting": setting,
                    "completed_trajectories": completed_trajectories,
                    "remaining_seconds": max(0, deadline - time.time()),
                    "model_updates": 0,
                },
            )
            swanlab.log(
                {
                    "probe/completed_trajectories": completed_trajectories,
                    "probe/pair_seconds": elapsed,
                    "probe/model_updates": 0,
                    f"probe/{phase}/{setting}/pair_effective_rate": sum(
                        all_groups[-len(pair) :][i]["effective"] for i in range(len(pair))
                    )
                    / len(pair)
                    if phase != "warmup"
                    else 0,
                }
            )
            print(
                f"{phase} {setting} group={group}：已完成 {completed_trajectories} 条；本批 {elapsed:.1f}s",
                flush=True,
            )

        await run_pair("warmup", "A", panels["warmup"], 4)
        for number, pair in enumerate(task_pairs(panels["screen"])):
            arms = list(SETTINGS)
            arms = arms[number % 3 :] + arms[: number % 3]
            for setting in arms:
                await run_pair("screen", setting, pair, 4)
        selection = select_candidate(all_groups)
        if selection is None:
            raise RuntimeError("screening produced insufficient matched task coverage")
        write_json(output / "selection.json", selection)
        print("初筛候选：" + json.dumps(selection, ensure_ascii=False), flush=True)
        confirm_arms = list(dict.fromkeys(["A", selection["setting"]]))
        for number, pair in enumerate(task_pairs(panels["confirmation"])):
            arms = confirm_arms if number % 2 == 0 else list(reversed(confirm_arms))
            for setting in arms:
                await run_pair("confirmation", setting, pair, 4)
        needs_group8 = group8_needed(all_groups, selection)
        confirmation_times = [
            r["sampling_seconds"] for r in all_groups if r["phase"] == "confirmation"
        ]
        estimated_extra = 2 * sum(confirmation_times) / max(1, len(confirmation_times)) * 24
        if needs_group8 and deadline - time.time() > max(180, 1.5 * estimated_extra + 60):
            for pair in task_pairs(panels["confirmation"]):
                await run_pair("group8", selection["setting"], pair, 8)
            write_json(
                output / "group8_decision.json", {"run": True, "reason": PLAN["group8_trigger"]}
            )
        else:
            write_json(
                output / "group8_decision.json",
                {
                    "run": False,
                    "reason": "insufficient time"
                    if needs_group8
                    else "mixture/temperature candidate is sufficient",
                    "estimated_extra_sampling_seconds": estimated_extra,
                },
            )
        status = "completed"
    except (TimeoutError, asyncio.TimeoutError):
        status = "time_budget_reached"
    except (KeyboardInterrupt, SystemExit):
        status = "interrupted"
        raise
    finally:
        # Persist the complete report before potentially slow Ray/SwanLab shutdown.
        write_json(output / "worker_status.json", {"status": status})
        report = build_report(output, status)
        metrics = {
            "probe/final_completed_groups": report["completed_groups"],
            "probe/final_completed_trajectories": report["completed_trajectories"],
            "probe/screen_complete": int(report["screen_complete"]),
            "probe/confirmation_complete": int(report["confirmation_complete"]),
        }
        for arm, levels in report["screen_by_difficulty"].items():
            for level, values in levels.items():
                for key in (
                    "tasks",
                    "effective",
                    "invalid",
                    "strict_success_rate",
                    "sampling_seconds",
                    "generated_tokens",
                ):
                    metrics[f"probe/screen/{arm}/{level}/{key}"] = values[key]
        for row in report["screen_candidates"]:
            prefix = f"probe/mixture/{row['setting']}/{row['mixture']}"
            for key in (
                "effective",
                "effective_groups_per_minute",
                "invalid",
                "strict_success_rate",
            ):
                metrics[prefix + "/" + key] = row[key]
        if report["fresh_train_confirmation"]:
            for key in (
                "effective",
                "effective_groups_per_minute",
                "invalid",
                "strict_success_rate",
            ):
                metrics["probe/confirmation/" + key] = report["fresh_train_confirmation"][key]
        if report["native_group4_vs_group8"]:
            for size, values in report["native_group4_vs_group8"].items():
                for key in (
                    "effective",
                    "effective_groups_per_minute",
                    "invalid",
                    "sampling_seconds",
                ):
                    metrics[f"probe/native_group{size}/{key}"] = values[key]
        try:
            swanlab.log(metrics)
        finally:
            try:
                ray.shutdown()
            finally:
                swanlab.finish()


def supervise(args, deadline):
    import psutil

    env = dict(os.environ)
    env["SHOPPING_STUDY_DEADLINE_UNIX"] = str(deadline)
    command = [
        sys.executable,
        "scripts/probe_rl_sampling.py",
        "--worker",
        "--output",
        str(args.output),
        "--budget-minutes",
        str(args.budget_minutes),
        "--environment-port",
        str(args.environment_port),
    ]
    tracked = {}
    status = "failed"
    with (args.output / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        def forward():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)

        reader = threading.Thread(target=forward, daemon=True)
        reader.start()
        try:
            while process.poll() is None:
                try:
                    for child in psutil.Process(process.pid).children(recursive=True):
                        tracked[child.pid, child.create_time()] = child
                except psutil.Error:
                    pass
                if time.time() >= deadline - 30:
                    status = "time_budget_reached"
                    break
                time.sleep(1)
            else:
                worker_status = args.output / "worker_status.json"
                status = (
                    json.loads(worker_status.read_text())["status"]
                    if worker_status.exists()
                    else "failed"
                )
                if process.returncode and status == "completed":
                    status = "failed"
        except KeyboardInterrupt:
            status = "interrupted"
        finally:
            stop_tree(process, list(tracked.values()))
            reader.join(timeout=2)
            process.stdout.close()
    return status


def main(argv=None):
    started = time.time()
    args = parse_args(argv)
    os.chdir(ROOT)
    if args.worker:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        asyncio.run(study_worker(args))
        return
    plan, panels = preflight(args)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("dry-run 通过：未启动服务、Ray、模型或采样。", flush=True)
        return
    deadline = started + args.budget_minutes * 60
    if deadline - time.time() < 120:
        raise TimeoutError("数据预检已耗尽总时间预算，未启动采样")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.environment_port))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "panels.json", panels)
    from omegaconf import OmegaConf

    write_json(
        args.output / "resolved_config.json",
        OmegaConf.to_container(compose_runtime_config(STUDY_OVERRIDES), resolve=True),
    )
    status = "failed"
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        with ExitStack() as stack:
            log = stack.enter_context((args.output / "environment.log").open("w"))
            simulator = subprocess.Popen(
                ["bash", "scripts/start_environment.sh"],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            stack.callback(stop_tree, simulator)
            wait_service(
                "tcp:" + str(args.environment_port),
                simulator,
                timeout=min(300, max(1, deadline - time.time() - 60)),
            )
            status = supervise(args, deadline)
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        report = build_report(args.output, status)
        write_json(
            args.output / "run_status.json",
            {
                "status": status,
                "elapsed_seconds": time.time() - started,
                "budget_seconds": args.budget_minutes * 60,
                "training_started": False,
            },
        )
        print(
            f"采样诊断结束：{status}，{report['completed_trajectories']} 条分析轨迹；报告：{args.output / 'report.md'}",
            flush=True,
        )
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
