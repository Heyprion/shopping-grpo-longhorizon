#!/usr/bin/env python3
"""Run configuration C, with a 50-update validation pilot or the full pipeline."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from shopping_grpo.environment.manifest import sha256_file
from shopping_grpo.training.grpo.task_sampling import StratifiedTaskSampler
from shopping_grpo.training.grpo.validation import strict_success

if __package__:
    from .check_grpo_runtime import compose_runtime_config, validate_environment_contract
    from .evaluate_sft_checkpoint import validate_training_summary, wait_service
    from .train_grpo import DEFAULT_MODEL, build_command
else:
    from check_grpo_runtime import compose_runtime_config, validate_environment_contract
    from evaluate_sft_checkpoint import validate_training_summary, wait_service
    from train_grpo import DEFAULT_MODEL, build_command

ROOT = Path(__file__).resolve().parents[1]
SELECTION_METRIC = "val-core/shopsimulator/acc/mean@1"
TOTAL_STEPS = 500
VALIDATION_STEPS = set(range(50, TOTAL_STEPS + 1, 50))
CHECKPOINTED_LAYERS = 20

# Keep the resource overrides explicit: the bare YAML is not the selected recipe.
OVERRIDES = [
    "data.seed=42",
    "data.train_batch_size=2",
    "data.val_batch_size=50",
    "data.dataloader_num_workers=0",
    "actor_rollout_ref.rollout.temperature=1.0",
    "actor_rollout_ref.rollout.top_p=1.0",
    "actor_rollout_ref.rollout.top_k=-1",
    "actor_rollout_ref.rollout.n=4",
    "shopping_task_sampling.enable=true",
    "shopping_task_sampling.difficulty_weights=[0.1,0.3,0.6]",
    "shopping_task_sampling.seed=42",
    "shopping_task_sampling.preserve_domain_marginal=true",
    "shopping_dynamic_sampling.max_num_gen_batches=3",
    "shopping_dynamic_sampling.max_consecutive_skipped_updates=10",
    "actor_rollout_ref.actor.ppo_mini_batch_size=2",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
    "actor_rollout_ref.actor.optim.lr=1e-6",
    "actor_rollout_ref.actor.optim.lr_warmup_steps=15",
    "actor_rollout_ref.actor.fsdp_config.param_offload=false",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
    "actor_rollout_ref.actor.fsdp_config.reshard_after_forward=false",
    "actor_rollout_ref.model.use_fused_kernels=true",
    "actor_rollout_ref.model.fused_kernel_options.impl_backend=triton",
    "actor_rollout_ref.model.use_liger=false",
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.seed=42",
    f'+ray_kwargs.ray_init.runtime_env.env_vars.SHOPPING_GRPO_CHECKPOINTED_LAYERS="{CHECKPOINTED_LAYERS}"',
    '+ray_kwargs.ray_init.runtime_env.env_vars.SHOPPING_GRPO_FAST_KERNELS="fla_conv,liger_norm"',
    "+ray_kwargs.ray_init.address=local",
    "+ray_kwargs.ray_init.num_cpus=32",
    "+ray_kwargs.ray_init.include_dashboard=false",
    "+ray_kwargs.ray_init.object_store_memory=2147483648",
    "trainer.total_training_steps=500",
    "trainer.save_freq=50",
    "trainer.test_freq=50",
    "trainer.val_before_train=true",
    "trainer.max_actor_ckpt_to_keep=null",
    "trainer.resume_mode=disable",
    "trainer.project_name=shopping-grpo",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require_empty(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"拒绝覆盖非空输出，请用 --output 指定新目录：{path}")


def validate_sft_source():
    adapter = ROOT / "outputs/models/sft-stage1-v1-balanced"
    summary_path = adapter / "train_summary.json"
    summary = read_json(summary_path)
    selection = validate_training_summary(summary)
    for path, expected in summary["data_sha256"].items():
        if sha256_file(ROOT / path) != expected:
            raise ValueError(f"SFT 数据在训练后发生变化：{path}")
    merge = read_json(DEFAULT_MODEL / "merge_manifest.json")
    if merge["source"].get("adapter_sha256") != sha256_file(adapter / "adapter_model.safetensors"):
        raise ValueError("RL 起点不是当前验证最佳 SFT adapter 的合并模型")
    return {
        "model": str(DEFAULT_MODEL),
        "model_selection": selection,
        "train_summary_sha256": sha256_file(summary_path),
        "merge_manifest_sha256": sha256_file(DEFAULT_MODEL / "merge_manifest.json"),
    }


def select_checkpoint(output, total_steps=TOTAL_STEPS):
    """Only full, successful training can proceed to the held-out test."""
    validation_steps = set(range(50, total_steps + 1, 50))
    updates, candidates = [], []
    with (output / "training_diagnostics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") != "optimizer_step":
                continue
            step = row["global_step"]
            updates.append(step)
            if step not in validation_steps:
                continue
            score = float(row["metrics"].get(SELECTION_METRIC, math.nan))
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"step {step} 缺少有效的 Strict Success 验证指标")
            actor = output / f"global_step_{step}" / "actor"
            required = [
                actor / "huggingface/config.json",
                actor / "fsdp_config.json",
                *(actor / f"model_world_size_2_rank_{rank}.pt" for rank in range(2)),
            ]
            if not all(path.is_file() for path in required):
                raise ValueError(f"验证检查点缺失：{actor}")
            fsdp = read_json(actor / "fsdp_config.json")
            if fsdp.get("world_size") != 2 or fsdp.get("FSDP_version") != 2:
                raise ValueError(f"检查点不是双卡 FSDP2：{actor}")
            meta = read_json(actor / "lora_train_meta.json")
            if meta.get("r") != 16 or meta.get("lora_alpha") != 32:
                raise ValueError(f"检查点 LoRA 配方不符：{actor}")
            candidates.append({"step": step, "value": score, "actor": str(actor)})
    if updates != list(range(1, total_steps + 1)):
        raise ValueError(f"未完成连续 {total_steps} 次有效更新，停止后续处理")
    if {r["step"] for r in candidates} != validation_steps:
        raise ValueError("每 50 步的验证记录不完整，停止自动选模")
    # The initial SFT validation is a reference, not an RL checkpoint candidate.
    best = max(candidates, key=lambda r: (r["value"], -r["step"]))
    return {
        "metric": SELECTION_METRIC,
        "greater_is_better": True,
        "tie_break": "earliest_step",
        "best": best,
        "candidates": candidates,
    }


def pilot_validation_report(output, selection, expected_ids, task_batch_size=2, learning_rate=1e-6):
    """Compare the same frozen validation tasks before training and at step 50."""
    batches = {}
    with (output / "validation_diagnostics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") != "validation_batch":
                continue
            step = row["global_step"]
            if step in batches:
                raise ValueError(f"重复的验证记录：step {step}")
            batches[step] = row["rollouts"]
    if set(batches) != {0, 50}:
        raise ValueError("50 步实验必须具有训练前及第 50 步的完整验证")
    summaries, successes = {}, {}
    for step, rows in batches.items():
        ids = [int(row["task_id"]) for row in rows]
        if len(ids) != len(expected_ids) or set(ids) != set(expected_ids):
            raise ValueError(f"step {step} 验证题重复、缺失或偏离固定名单")
        successes[step] = {int(row["task_id"]) for row in rows if strict_success(row)}
        rewards = [float(row["reward"]["total"]) for row in rows]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in rewards):
            raise ValueError("验证记录存在非法 Strict Reward")
        summaries[step] = {
            "tasks": len(rows),
            "strict_successes": len(successes[step]),
            "strict_success_rate": len(successes[step]) / len(rows),
            "mean_strict_reward": sum(rewards) / len(rows),
        }
    if not math.isclose(selection["best"]["value"], summaries[50]["strict_success_rate"], abs_tol=1e-8):
        raise ValueError("第 50 步训练指标与逐题验证记录不一致")
    return {
        "experiment": "config-c-syncfix-pilot50"
        + ("-batch4" if task_batch_size == 4 else "")
        + ("-lr3e-6" if learning_rate == 3e-6 else ""),
        "task_batch_size": task_batch_size,
        "learning_rate": learning_rate,
        "effective_trajectory_batch_size": task_batch_size * 4,
        "optimized_trajectories": 50 * task_batch_size * 4,
        "effective_updates": 50,
        "scheduler_planned_updates": TOTAL_STEPS,
        "evaluation_split": "rl_validation",
        "baseline": summaries[0],
        "step_50": summaries[50],
        "success_rate_delta": summaries[50]["strict_success_rate"] - summaries[0]["strict_success_rate"],
        "gained_task_ids": sorted(successes[50] - successes[0]),
        "lost_task_ids": sorted(successes[0] - successes[50]),
        "checkpoint": selection["best"]["actor"],
        "heldout_test_executed": False,
    }


def training_overrides(args):
    replacements = {
        "data.train_batch_size": args.task_batch_size,
        "actor_rollout_ref.actor.ppo_mini_batch_size": args.task_batch_size,
        "actor_rollout_ref.actor.optim.lr": args.learning_rate,
    }
    overrides = []
    for value in OVERRIDES:
        key = value.split("=", 1)[0]
        overrides.append(f"{key}={replacements[key]}" if key in replacements else value)
    if args.pilot_steps:
        overrides.append(f"+shopping_pilot.max_steps={args.pilot_steps}")
    return overrides


def stop_owned(process):
    """Each child owns a new process group; never stop unrelated Ray/vLLM jobs."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_logged(command, environment, log):
    print("执行：" + shlex.join(map(str, command)), flush=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            process.stdout.close()
            stop_owned(process)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=None
    )
    parser.add_argument(
        "--pilot-steps", type=int, choices=[50],
        help="只执行 50 次有效更新，自动比较训练前/后的固定 50 题验证；不合并或运行测试集",
    )
    parser.add_argument(
        "--task-batch-size", type=int, choices=[2, 4], default=2,
        help="每次更新的任务数；batch4 对照须搭配 --pilot-steps 50",
    )
    parser.add_argument(
        "--learning-rate", type=float, choices=[1e-6, 3e-6], default=1e-6,
        help="pilot 学习率对照；3e-6 须搭配 --pilot-steps 50",
    )
    parser.add_argument("--environment-port", type=int, default=15705)
    parser.add_argument("--llm-port", type=int, default=18005)
    parser.add_argument(
        "--dry-run", action="store_true", help="校验数据并展示配方，不启动服务、训练、合并或评测"
    )
    args = parser.parse_args(argv)
    if args.task_batch_size != 2 and not args.pilot_steps:
        parser.error("batch4 对照请使用 --pilot-steps 50 --task-batch-size 4")
    if args.learning_rate != 1e-6 and not args.pilot_steps:
        parser.error("学习率对照请使用 --pilot-steps 50 --learning-rate 3e-6")
    if args.output is None:
        name = "grpo-stage1-v3-syncfix-pilot50" if args.pilot_steps else "grpo-stage1-v2-config-c-cp20"
        if args.task_batch_size == 4:
            name += "-batch4"
        if args.learning_rate == 3e-6:
            name += "-lr3e-6"
        args.output = ROOT / "outputs/models" / name
    args.output = args.output.expanduser().resolve()
    args.merged = args.output.with_name(args.output.name + "-best-merged")
    args.evaluation = ROOT / "outputs/evaluation" / args.output.name
    args.logs = ROOT / "outputs/logs" / args.output.name
    if args.environment_port == args.llm_port or any(
        not 1024 <= port <= 65535 for port in (args.environment_port, args.llm_port)
    ):
        parser.error("环境端口和模型服务端口必须不同，且在 1024–65535 之间")
    return args


def prepare(args):
    paths_to_check = (args.output, args.logs) if args.pilot_steps else (args.output, args.merged, args.evaluation, args.logs)
    for path in paths_to_check:
        require_empty(path)
    environment = dict(os.environ)
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        environment.pop(name, None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
            "OMP_NUM_THREADS": "8",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONHASHSEED": "42",
            "SHOPPING_GRPO_SEED": "42",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "SHOPPING_TASK_BUDGETS": str(ROOT / "data/cleaned/task_budgets.jsonl"),
            "SHOPPING_GRPO_CHECKPOINTED_LAYERS": str(CHECKPOINTED_LAYERS),
            "SHOPPING_GRPO_FAST_KERNELS": "fla_conv,liger_norm",
            "PYTHONPATH": str(ROOT / "src"),
            "RAY_ADDRESS": "local",
        }
    )
    if len([x for x in environment["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]) != 2:
        raise ValueError("当前配方需要两张可见 GPU，请设置 CUDA_VISIBLE_DEVICES=0,1 或另外两张卡")
    subprocess.run(
        [sys.executable, "scripts/prepare_data_splits.py", "verify"],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    source = validate_sft_source()
    overrides = training_overrides(args)
    experiment = "-RL-配置C同步修复50步" if args.pilot_steps else "-RL-配置C显存调整"
    if args.task_batch_size == 4:
        experiment += "-batch4对照"
    if args.learning_rate == 3e-6:
        experiment += "-lr3e-6对照"
    train_args = argparse.Namespace(
        model=DEFAULT_MODEL,
        train_data=ROOT / "data/grpo/train.parquet",
        val_data=ROOT / "data/grpo/validation.parquet",
        env_url=f"http://127.0.0.1:{args.environment_port}",
        output=args.output,
        logger="swanlab",
        config=ROOT / "configs/grpo.yaml",
        experiment_name=datetime.now(timezone.utc).strftime("%Y-%m-%d") + experiment,
        hydra_overrides=overrides,
    )
    # Compose under exactly the environment that the trainer and Ray will receive.
    original = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(environment)
        resolved_command, environment = build_command(train_args)
        os.environ.update(environment)
        validate_environment_contract()
        config = compose_runtime_config(resolved_command[5:])
        from omegaconf import OmegaConf

        resolved = OmegaConf.to_container(config, resolve=True)
    finally:
        os.environ.clear()
        os.environ.update(original)
    command = [
        sys.executable,
        "scripts/train_grpo.py",
        "--model",
        str(DEFAULT_MODEL),
        "--train-data",
        str(train_args.train_data),
        "--val-data",
        str(train_args.val_data),
        "--env-url",
        train_args.env_url,
        "--output",
        str(args.output),
        "--logger",
        "swanlab",
        "--experiment-name",
        train_args.experiment_name,
        "--",
        *overrides,
    ]
    import pyarrow.parquet as pq

    task_ids = [
        r["task_id"]
        for r in pq.read_table(train_args.train_data, columns=["extra_info"]).to_pydict()[
            "extra_info"
        ]
    ]
    sampler = StratifiedTaskSampler(
        task_ids, resolved["shopping_task_sampling"], batch_size=args.task_batch_size
    )
    sampler.schedule(0)  # Validate exact margins and distinct task pairs without writing a plan.
    paths = [
        "data/grpo/train.parquet",
        "data/grpo/validation.parquet",
        "data/evaluation/tasks.jsonl",
        "data/cleaned/task_budgets.jsonl",
        "data/splits/manifest.json",
        "data/splits/labels.jsonl",
        "data/environment.json",
        "configs/grpo.yaml",
        "configs/agent_loop.yaml",
        "configs/tools.json",
    ]
    contract = {
        "sft_source": source,
        "data_sha256": {p: sha256_file(ROOT / p) for p in paths},
        "recipe": "config-c",
        "task_sampling": sampler.summary(),
        "selection_metric": SELECTION_METRIC,
        "total_effective_updates": args.pilot_steps or TOTAL_STEPS,
        "task_batch_size": args.task_batch_size,
        "learning_rate": args.learning_rate,
        "effective_trajectory_batch_size": args.task_batch_size * 4,
        "gradient_accumulation_steps": args.task_batch_size // 2,
        "scheduler_planned_updates": TOTAL_STEPS,
        "evaluation_split": "rl_validation" if args.pilot_steps else "heldout_test",
        "lora_sync_snapshot_sha256": sha256_file(ROOT / "src/shopping_grpo/training/grpo/weight_sync.py"),
        "train_command": command,
        "merged_model": None if args.pilot_steps else str(args.merged),
        "evaluation_output": str(args.output / "validation_report.json") if args.pilot_steps else str(args.evaluation),
        "log_dir": str(args.logs),
        "environment": {
            k: environment[k]
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "SHOPPING_TASK_BUDGETS",
                "SHOPPING_GRPO_CHECKPOINTED_LAYERS",
                "SHOPPING_GRPO_FAST_KERNELS",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            )
        },
    }
    return command, environment, resolved, contract


def export_best(args, selection, environment):
    exported = args.output / "best-export"
    run_logged(
        ["bash", "scripts/export_grpo.sh", selection["best"]["actor"], str(exported)],
        environment,
        args.logs / "export.log",
    )
    adapter = exported / "lora_adapter"
    config = read_json(adapter / "adapter_config.json")
    if config.get("r") != 16 or config.get("lora_alpha") != 32:
        raise ValueError("导出的 RL adapter 不是 r16/alpha32，停止合并")
    # veRL exports frozen base weights and LoRA separately. Serving only the
    # export root would silently evaluate the pre-RL policy.
    run_logged(
        [
            sys.executable,
            "scripts/merge_lora_adapter.py",
            "--base-model",
            str(exported),
            "--adapter",
            str(adapter),
            "--output",
            str(args.merged),
            "--bf16",
        ],
        environment,
        args.logs / "merge.log",
    )


def start_environment(args, environment, stack):
    env_log = stack.enter_context((args.logs / "environment.log").open("a"))
    env = {**environment, "SHOPSIM_ENV_SLOTS": "64", "SHOPSIM_PORT": str(args.environment_port)}
    simulator = subprocess.Popen(
        ["bash", "scripts/start_environment.sh"],
        cwd=ROOT,
        env=env,
        stdout=env_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stack.callback(stop_owned, simulator)
    wait_service("tcp:" + str(args.environment_port), simulator)
    return env


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    command, environment, resolved, contract = prepare(args)
    print(json.dumps(contract, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("解析后的完整训练配置：", flush=True)
        print(json.dumps(resolved, ensure_ascii=False, indent=2), flush=True)
        print("dry-run 完成：未启动服务、训练、模型合并或 200 题测试。", flush=True)
        return
    ports = (args.environment_port,) if args.pilot_steps else (args.environment_port, args.llm_port)
    for port in ports:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    args.logs.mkdir(parents=True, exist_ok=True)
    if not args.pilot_steps:
        args.evaluation.mkdir(parents=True, exist_ok=True)
    write_json(args.logs / "run_contract.json", contract)
    write_json(args.logs / "resolved_config.json", resolved)
    # SIGTERM follows the same cleanup path as Ctrl-C.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    with ExitStack() as stack:
        env = start_environment(args, environment, stack)
        run_logged(command, environment, args.logs / "train.log")
        if args.pilot_steps:
            selection = select_checkpoint(args.output, total_steps=args.pilot_steps)
            for path, expected in contract["data_sha256"].items():
                if sha256_file(ROOT / path) != expected:
                    raise ValueError("实验期间数据发生变化：" + path)
            expected_ids = [
                json.loads(line)["task_id"]
                for line in (ROOT / "data/grpo/validation.jsonl").read_text().splitlines()
                if line.strip()
            ]
            report = pilot_validation_report(
                args.output, selection, expected_ids, task_batch_size=args.task_batch_size,
                learning_rate=args.learning_rate,
            )
            write_json(args.output / "validation_report.json", report)
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            return
        selection = select_checkpoint(args.output)
        write_json(args.output / "model_selection.json", selection)
        for path, expected in contract["data_sha256"].items():
            if sha256_file(ROOT / path) != expected:
                raise ValueError("训练期间数据发生变化，停止自动测试：" + path)
        export_best(args, selection, environment)
        server_log = stack.enter_context((args.logs / "model-server.log").open("w"))
        serving = {
            **environment,
            "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"].split(",")[0],
            "LLM_PORT": str(args.llm_port),
            "SERVED_MODEL_NAME": "shopping-grpo-config-c",
        }
        server = subprocess.Popen(
            ["bash", "scripts/serve_model.sh", str(args.merged)],
            cwd=ROOT,
            env=serving,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stack.callback(stop_owned, server)
        wait_service(f"http://127.0.0.1:{args.llm_port}/v1/models", server)
        evaluation_env = {
            **env,
            "SHOPSIM_BASE_URL": f"http://127.0.0.1:{args.environment_port}",
            "LLM_BASE_URL": f"http://127.0.0.1:{args.llm_port}/v1",
            "SERVED_MODEL_NAME": serving["SERVED_MODEL_NAME"],
            "LLM_API_KEY": "EMPTY",
            "EVAL_OUTPUT_DIR": str(args.evaluation),
            "EVAL_SWANLAB_PROJECT": "shopping-grpo-evaluation",
        }
        name = datetime.now(timezone.utc).strftime("%Y-%m-%d") + "-EVAL-配置C显存调整最佳模型"
        run_logged(
            ["bash", "scripts/evaluate.sh", name], evaluation_env, args.logs / "evaluation.log"
        )
    result = read_json(args.evaluation / "summary.json")
    if result.get("expected_tasks") != 200 or result.get("completed_tasks") != 200:
        raise ValueError("固定 200 题测试未完整结束，请检查评测日志")
    report = {
        "model_selection": selection,
        "merged_model": str(args.merged),
        "test_results": result,
        "run_contract": contract,
    }
    write_json(args.evaluation / "final_report.json", report)
    print(
        f"RL 训练和评测完成。最佳检查点：step {selection['best']['step']}；报告：{args.evaluation / 'final_report.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
