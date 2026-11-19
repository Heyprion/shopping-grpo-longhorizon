#!/usr/bin/env python3
"""Run GRPO training or evaluate an existing GRPO checkpoint on the fixed test."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from shopping_grpo.environment.manifest import sha256_file
from shopping_grpo.training.grpo.dynamic_sampling import MAX_GENERATION_BATCHES
from shopping_grpo.training.grpo.task_sampling import StratifiedTaskSampler
from shopping_grpo.training.grpo.validation import strict_success

if __package__:
    from .check_grpo_runtime import compose_runtime_config, validate_environment_contract, validate_gigpo
    from .evaluate_sft_checkpoint import validate_training_summary, wait_service
    from .train_grpo import DEFAULT_MODEL, build_command
else:
    from check_grpo_runtime import compose_runtime_config, validate_environment_contract, validate_gigpo
    from evaluate_sft_checkpoint import validate_training_summary, wait_service
    from train_grpo import DEFAULT_MODEL, build_command

ROOT = Path(__file__).resolve().parents[1]
SELECTION_METRIC = "val-core/shopsimulator/acc/mean@1"
TOTAL_STEPS = 500
VALIDATION_STEPS = set(range(50, TOTAL_STEPS + 1, 50))
CHECKPOINTED_LAYERS = 20
PILOT_EVALUATION_SPLIT = "sft_test_failures30"
PILOT_EVALUATION_JSONL = ROOT / "data/grpo/pilot_eval30.jsonl"
PILOT_EVALUATION_PARQUET = ROOT / "data/grpo/pilot_eval30.parquet"
PILOT_EVALUATION_METADATA = ROOT / "data/grpo/pilot_eval30_metadata.json"
SAMPLING_BUDGET_FIELDS = {
    "max_num_gen_batches", "max_consecutive_skipped_updates", "distinct_accepted_instructions",
}


def learning_rate_suffix(value):
    return "" if value == 1e-6 else "-lr" + format(value, ".0e").replace("e-0", "e-")

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
    "trainer.project_name=shopping-grpo-longhorizon",
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


def checkpoint_shards(actor):
    shards = [actor / f"model_world_size_2_rank_{rank}.pt" for rank in range(2)]
    required = [actor / "huggingface/config.json", actor / "fsdp_config.json",
                actor / "lora_train_meta.json", *shards]
    if not all(path.is_file() and path.stat().st_size for path in required):
        raise ValueError(f"检查点缺失或文件为空：{actor}")
    fsdp = read_json(actor / "fsdp_config.json")
    if fsdp.get("world_size") != 2 or fsdp.get("FSDP_version") != 2:
        raise ValueError(f"检查点不是双卡 FSDP2：{actor}")
    meta = read_json(actor / "lora_train_meta.json")
    if meta.get("r") != 16 or meta.get("lora_alpha") != 32:
        raise ValueError(f"检查点 LoRA 配方不符：{actor}")
    return shards


def select_checkpoint(output, total_steps=TOTAL_STEPS, start_step=0):
    """Select the best saved checkpoint after all requested updates."""
    validation_steps = {step for step in range(50, total_steps + 1, 50) if step > start_step}
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
            checkpoint_shards(actor)
            candidates.append({"step": step, "value": score, "actor": str(actor)})
    if updates != list(range(start_step + 1, total_steps + 1)):
        raise ValueError(f"未完成连续 {total_steps - start_step} 次有效更新，停止后续处理")
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


def pilot_validation_report(
    output, selection, expected_ids, task_batch_size=2, learning_rate=1e-6,
    adam_beta1=0.9, total_steps=50, *, start_step=0, group_size=4, experiment=None,
    validation_end_step=None,
):
    """Report every pilot validation, keeping final performance distinct from best."""
    expected_steps = {start_step, *(step for step in range(50, total_steps + 1, 50) if step > start_step)}
    batches = {}
    with (output / "validation_diagnostics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") != "validation_batch":
                continue
            step = row["global_step"]
            if validation_end_step is not None and step > validation_end_step:
                continue
            if step in batches:
                raise ValueError(f"重复的验证记录：step {step}")
            batches[step] = row["rollouts"]
    if set(batches) != expected_steps:
        raise ValueError(f"{total_steps} 步实验必须具有训练前及每 50 步的完整验证")
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
    candidates = {row["step"]: row for row in selection["candidates"]}
    if set(candidates) != expected_steps - {start_step}:
        raise ValueError("实验检查点与验证步数不一致")
    for step, candidate in candidates.items():
        if not math.isclose(candidate["value"], summaries[step]["strict_success_rate"], abs_tol=1e-8):
            raise ValueError(f"第 {step} 步训练指标与逐题验证记录不一致")
    return {
        "experiment": experiment or (
            f"config-c-syncfix-pilot{total_steps}-eval30" + learning_rate_suffix(learning_rate)
            + ("-beta0" if adam_beta1 == 0.0 else "")
        ),
        "task_batch_size": task_batch_size,
        "learning_rate": learning_rate,
        "adam_betas": [adam_beta1, 0.999],
        "effective_trajectory_batch_size": task_batch_size * group_size,
        "optimized_trajectories": (total_steps - start_step) * task_batch_size * group_size,
        "effective_updates": total_steps - start_step,
        "scheduler_planned_updates": TOTAL_STEPS,
        "evaluation_split": PILOT_EVALUATION_SPLIT,
        "baseline": summaries[start_step],
        **{f"step_{step}": summaries[step] for step in sorted(candidates)},
        "final_step": total_steps,
        "final": summaries[total_steps],
        "success_rate_delta": summaries[total_steps]["strict_success_rate"] - summaries[start_step]["strict_success_rate"],
        "gained_task_ids": sorted(successes[total_steps] - successes[start_step]),
        "lost_task_ids": sorted(successes[start_step] - successes[total_steps]),
        "solved_task_ids": sorted(successes[total_steps]),
        "checkpoint": candidates[total_steps]["actor"],
        "model_selection": selection,
        "validation_history": [
            {"global_step": step, **summaries[step],
             "gained_task_ids": sorted(successes[step] - successes[start_step]),
             "lost_task_ids": sorted(successes[start_step] - successes[step])}
            for step in sorted(expected_steps)
        ],
        "full_200_task_test_executed": False,
        "test_subset_executed": True,
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
        overrides = [
            "data.val_batch_size=30" if value.startswith("data.val_batch_size=") else value
            for value in overrides
        ]
        overrides.append(f"+shopping_pilot.max_steps={args.pilot_steps}")
    if args.focus_tasks:
        focus = read_json(args.focus_tasks)["training_task_ids"]
        overrides.append("shopping_task_sampling.focus_task_ids=" + json.dumps(focus))
    if args.adam_beta1 != 0.9:
        overrides.append(f"actor_rollout_ref.actor.optim.betas=[{args.adam_beta1},0.999]")
    return overrides


def continuation_overrides(args, source_contract):
    previous = source_contract["train_command"]
    previous_overrides = previous[previous.index("--") + 1:]
    replaced = {
        "shopping_pilot.max_steps", "trainer.resume_mode", "trainer.resume_from_path",
        "trainer.del_local_ckpt_after_load",
    }
    replaced.update(f"shopping_dynamic_sampling.{key}" for key in SAMPLING_BUDGET_FIELDS)
    if args.continuation_pilot:
        replaced.add("trainer.save_freq")
    if args.continuation_pilot == "group8":
        replaced.add("actor_rollout_ref.rollout.n")
    overrides = [
        value for value in previous_overrides
        if value.split("=", 1)[0].lstrip("+") not in replaced
    ]
    overrides += [
        f"+shopping_pilot.max_steps={args.stop_at_step}",
        "trainer.resume_mode=resume_path",
        f"trainer.resume_from_path={json.dumps(str(args.resume_from))}",
        "trainer.del_local_ckpt_after_load=false",
    ]
    if args.continuation_pilot:
        overrides.append("trainer.save_freq=50")
    if args.continuation_pilot == "group8":
        overrides.append("actor_rollout_ref.rollout.n=8")
    if args.continuation_pilot == "gigpo":
        overrides.append("+shopping_gigpo.enable=true")
    return overrides


def validate_resume_checkpoint(checkpoint):
    actor = checkpoint / "actor"
    checkpoint_shards(actor)
    states = [checkpoint / "data.pt", *(
        actor / f"{kind}_world_size_2_rank_{rank}.pt"
        for kind in ("optim", "extra_state") for rank in range(2)
    )]
    for path in states:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"续训状态缺失或为空：{path}")


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--pilot-steps", type=int, choices=[50, 100],
        help="执行 50 或 100 次有效更新，训练前及每 50 步自动评测固定 30 道 SFT 测试错题",
    )
    mode.add_argument("--evaluate-checkpoint", type=Path,
                      help="用现有 global_step_*/actor 检查点评测固定 200 题；--output 指定评测目录")
    mode.add_argument("--resume-from", type=Path,
                      help="从现有 global_step_* 恢复完整状态，续训后评测固定 200 题")
    parser.add_argument("--stop-at-step", type=int, help="续训结束时的全局步数")
    parser.add_argument("--continuation-pilot", choices=["group8", "gigpo"],
                        help="从同一 RL 检查点续训 50 步并仅做固定 30 题短测")
    parser.add_argument(
        "--task-batch-size", type=int, choices=[2], default=2,
        help="每次更新固定两道任务",
    )
    parser.add_argument(
        "--learning-rate", type=float, choices=[1e-6, 3e-6, 3e-5], default=1e-6,
        help="pilot 学习率对照；非默认值须搭配 --pilot-steps 50 或 100",
    )
    parser.add_argument("--focus-tasks", type=Path, help="pilot复查名单JSON：对training_task_ids中的16道训练题重复采样")
    parser.add_argument("--adam-beta1", type=float, choices=[0.0, 0.9], default=0.9,
                        help="pilot一阶动量对照；0.0关闭Adam一阶动量，保留二阶统计")
    parser.add_argument("--environment-port", type=int, default=15705)
    parser.add_argument("--llm-port", type=int, default=18005)
    parser.add_argument(
        "--dry-run", action="store_true", help="校验数据并展示配方，不启动服务、训练、合并或评测"
    )
    args = parser.parse_args(argv)
    if args.continuation_pilot and not args.resume_from:
        parser.error("--continuation-pilot 须搭配 --resume-from")
    if args.resume_from:
        args.resume_from = args.resume_from.expanduser().resolve()
        prefix, _, step = args.resume_from.name.rpartition("_")
        if prefix != "global_step" or not step.isdecimal():
            parser.error("--resume-from 应指向 global_step_<步数> 目录")
        args.resume_step = int(step)
        if (args.stop_at_step is None
                or not 0 < args.resume_step < args.stop_at_step <= TOTAL_STEPS
                or args.stop_at_step % 50):
            parser.error("--stop-at-step 须大于来源步数，且为 50 的倍数，最多 500 步")
        if args.continuation_pilot and args.stop_at_step - args.resume_step != 50:
            parser.error("--continuation-pilot 须续训 50 步")
        if args.output is None:
            name = f"{args.resume_from.parent.name}-to{args.stop_at_step}"
            if args.continuation_pilot:
                name += "-" + args.continuation_pilot
            args.output = ROOT / "outputs/models" / name
    elif args.stop_at_step is not None:
        parser.error("--stop-at-step 须搭配 --resume-from")
    if args.learning_rate != 1e-6 and not args.pilot_steps:
        parser.error("非默认学习率对照须使用 --pilot-steps 50 或 100")
    if args.focus_tasks and args.pilot_steps != 50:
        parser.error("集中重复采样仅用于 --pilot-steps 50 短实验")
    if args.adam_beta1 != 0.9 and not args.pilot_steps:
        parser.error("一阶动量对照仅用于 --pilot-steps 50 或 100 短实验")
    if args.focus_tasks:
        args.focus_tasks = args.focus_tasks.expanduser().resolve()
    if args.evaluate_checkpoint:
        args.evaluate_checkpoint = args.evaluate_checkpoint.expanduser().resolve()
        if args.output is None:
            step_dir = args.evaluate_checkpoint.parent
            args.output = ROOT / "outputs/evaluation" / f"{step_dir.parent.name}-{step_dir.name}"
    if args.output is None:
        name = f"grpo-stage1-v3-syncfix-pilot{args.pilot_steps}-eval30" if args.pilot_steps else "grpo-stage1-v2-config-c-cp20"
        name += learning_rate_suffix(args.learning_rate)
        if args.focus_tasks:
            name += "-focus16"
        if args.adam_beta1 != 0.9:
            name += "-beta0"
        args.output = ROOT / "outputs/models" / name
    args.output = args.output.expanduser().resolve()
    args.merged = args.output.with_name(args.output.name + "-best-merged")
    args.evaluation = args.output if args.evaluate_checkpoint else ROOT / "outputs/evaluation" / args.output.name
    args.logs = args.evaluation if args.evaluate_checkpoint else ROOT / "outputs/logs" / args.output.name
    if args.environment_port == args.llm_port or any(
        not 1024 <= port <= 65535 for port in (args.environment_port, args.llm_port)
    ):
        parser.error("环境端口和模型服务端口必须不同，且在 1024–65535 之间")
    return args


def prepare(args):
    short_validation = args.pilot_steps or args.continuation_pilot
    paths_to_check = (args.output, args.logs) if short_validation else (args.output, args.merged, args.evaluation, args.logs)
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
    val_data = PILOT_EVALUATION_PARQUET if args.pilot_steps else ROOT / "data/grpo/validation.parquet"
    source_contract = None
    previous_config = None
    experiment = f"-RL-{args.pilot_steps}步训练与30道SFT错题短测" if args.pilot_steps else "-RL-配置C显存调整"
    if args.learning_rate != 1e-6:
        experiment += learning_rate_suffix(args.learning_rate) + "对照"
    if args.focus_tasks:
        experiment += "-16道训练题重复学习"
    if args.adam_beta1 != 0.9:
        experiment += "-无一阶动量对照"
    if args.resume_from:
        validate_resume_checkpoint(args.resume_from)
        environment["SHOP_ENV_CONFIG"] = str(ROOT / "environments/ShopSimulator/shop_env/configs/environment.json")
        environment["SHOP_MAX_STEPS"] = "35"
        source_logs = ROOT / "outputs/logs" / args.resume_from.parent.name
        source_contract = read_json(source_logs / "run_contract.json")
        previous_config = read_json(source_logs / "resolved_config.json")
        if Path(previous_config["trainer"]["default_local_dir"]).resolve() != args.resume_from.parent:
            raise ValueError("续训目录与已保存的训练配置不一致")
        for key in ("train_summary_sha256", "merge_manifest_sha256"):
            if source_contract["sft_source"][key] != source[key]:
                raise ValueError("续训来源的 SFT 初始模型已变化")
        for path, expected in source_contract["data_sha256"].items():
            if path == "configs/grpo.yaml":
                # The resolved training configuration is checked below. The YAML now
                # defines the continuous sampling budget for both fresh and resumed runs.
                continue
            if sha256_file(ROOT / path) != expected:
                raise ValueError(f"续训来源的固定数据或配置已变化：{path}")
        val_data = Path(previous_config["data"]["val_files"])
        if args.continuation_pilot and (
            val_data.resolve() != PILOT_EVALUATION_PARQUET
            or previous_config["data"]["val_batch_size"] != 30
            or previous_config["trainer"]["test_freq"] != 50
            or not previous_config["trainer"]["val_before_train"]
            or previous_config["actor_rollout_ref"]["rollout"]["n"] != 4
        ):
            raise ValueError("续训短测要求来源为 group4，且已配置固定 30 题训练验证")
        overrides = continuation_overrides(args, source_contract)
        experiment = f"-RL-{args.resume_step}至{args.stop_at_step}步续训"
        if args.continuation_pilot:
            experiment += "-" + args.continuation_pilot
    is_probe_pilot = bool(
        (args.pilot_steps and args.pilot_steps < 100)
        or args.focus_tasks
        or args.continuation_pilot
    )
    swanlab_group = "05-GRPO工程摸底与排障" if is_probe_pilot else "06-GRPO正式训练"
    if args.resume_from:
        pilot_note = f"，专项对照={args.continuation_pilot}" if args.continuation_pilot else ""
        swanlab_description = (
            f"GRPO 续训（step {args.resume_step}→{args.stop_at_step}{pilot_note}），"
            f"沿用来源检查点与调度计划，在 Strict Reward 下继续优化。"
        )
    elif args.pilot_steps:
        focus_note = "，focus16 重复采样" if args.focus_tasks else ""
        swanlab_description = (
            f"GRPO {args.pilot_steps} 步实验（batch={args.task_batch_size}，group=4，"
            f"LR={args.learning_rate:g}，beta1={args.adam_beta1:g}{focus_note}），"
            f"配固定 30 道 SFT 错题短测。"
        )
    else:
        swanlab_description = (
            f"GRPO 配置 C 正式训练（500 步，batch={args.task_batch_size}，group=4，"
            f"LR={args.learning_rate:g}，beta1={args.adam_beta1:g}，Strict Reward）。"
        )
    swanlab_tags = "GRPO,LoRA,StrictReward"
    if args.continuation_pilot:
        swanlab_tags += f",{args.continuation_pilot}"
    train_args = argparse.Namespace(
        model=DEFAULT_MODEL,
        train_data=ROOT / "data/grpo/train.parquet",
        val_data=val_data,
        env_url=f"http://127.0.0.1:{args.environment_port}",
        output=args.output,
        logger="swanlab",
        config=ROOT / "configs/grpo.yaml",
        experiment_name=datetime.now(timezone.utc).strftime("%Y-%m-%d") + experiment,
        swanlab_group=swanlab_group,
        swanlab_job_type="train",
        swanlab_description=swanlab_description,
        swanlab_tags=swanlab_tags,
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
        validate_gigpo(config)
        from omegaconf import OmegaConf

        resolved = OmegaConf.to_container(config, resolve=True)
    finally:
        os.environ.clear()
        os.environ.update(original)
    if previous_config:
        # The rollout trace inherits the new run name from trainer metadata.
        previous_config["actor_rollout_ref"]["rollout"]["trace"]["experiment_name"] = (
            resolved["actor_rollout_ref"]["rollout"]["trace"]["experiment_name"]
        )
        for key in SAMPLING_BUDGET_FIELDS:
            previous_config["shopping_dynamic_sampling"].pop(key, None)
        previous_config["shopping_dynamic_sampling"]["max_num_gen_batches"] = MAX_GENERATION_BATCHES
        if args.continuation_pilot == "group8":
            previous_config["actor_rollout_ref"]["rollout"]["n"] = 8
            for role in ("actor", "ref"):
                previous_config["actor_rollout_ref"][role]["rollout_n"] = 8
            previous_config["critic"]["rollout_n"] = 8
        for key in ("data", "actor_rollout_ref", "algorithm", "shopping_task_sampling",
                    "shopping_dynamic_sampling", "shopping_trace", "critic", "reward",
                    "reward_model", "custom_reward_function", "distillation"):
            if resolved[key] != previous_config[key]:
                raise ValueError(f"续训配置与来源不一致：{key}")
        if resolved["trainer"]["total_training_steps"] != previous_config["trainer"]["total_training_steps"]:
            raise ValueError("续训必须沿用原学习率调度计划")
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
        "--swanlab-group",
        train_args.swanlab_group,
        "--swanlab-job-type",
        train_args.swanlab_job_type,
        "--swanlab-description",
        train_args.swanlab_description,
        "--swanlab-tags",
        train_args.swanlab_tags,
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
    sampling = sampler.summary()
    if source_contract and sampling["fingerprint"] != source_contract["task_sampling"]["fingerprint"]:
        raise ValueError("续训任务采样器与原检查点不一致")
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
    if short_validation:
        paths.extend(str(path.relative_to(ROOT)) for path in (
            PILOT_EVALUATION_JSONL, PILOT_EVALUATION_PARQUET, PILOT_EVALUATION_METADATA
        ))
    if args.focus_tasks:
        paths.append(str(args.focus_tasks.relative_to(ROOT)))
    if source_contract:
        paths = list(dict.fromkeys([*paths, *source_contract["data_sha256"]]))
    if args.continuation_pilot:
        paths.extend([
            "patches/verl-0.8.0-shopping-dynamic-sampling.patch",
            "src/shopping_grpo/training/grpo/gigpo.py",
            "src/shopping_grpo/training/grpo/adapter/agent_loop.py",
        ])
    optim = resolved["actor_rollout_ref"]["actor"]["optim"]
    group_size = resolved["actor_rollout_ref"]["rollout"]["n"]
    micro = resolved["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"]
    replicas = resolved["trainer"]["n_gpus_per_node"] * resolved["trainer"]["nnodes"]
    contract = {
        "sft_source": source,
        "data_sha256": {p: sha256_file(ROOT / p) for p in paths},
        "recipe": "config-c",
        "task_sampling": sampling,
        "selection_metric": SELECTION_METRIC,
        "total_effective_updates": args.stop_at_step or args.pilot_steps or TOTAL_STEPS,
        "task_batch_size": args.task_batch_size,
        "learning_rate": optim["lr"],
        "adam_betas": optim["betas"],
        "effective_trajectory_batch_size": args.task_batch_size * group_size,
        "gradient_accumulation_steps": args.task_batch_size * group_size // (micro * replicas),
        "scheduler_planned_updates": resolved["trainer"]["total_training_steps"],
        "evaluation_split": PILOT_EVALUATION_SPLIT if short_validation else "heldout_test",
        "pilot_evaluation": read_json(PILOT_EVALUATION_METADATA) if short_validation else None,
        "lora_sync_snapshot_sha256": sha256_file(ROOT / "src/shopping_grpo/training/grpo/weight_sync.py"),
        "train_command": command,
        "merged_model": None if args.pilot_steps or args.resume_from else str(args.merged),
        "evaluation_output": str(args.output / "validation_report.json") if short_validation else str(args.evaluation),
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
    if args.resume_from:
        contract.update(
            resume_from=str(args.resume_from),
            resume_step=args.resume_step,
            additional_effective_updates=args.stop_at_step - args.resume_step,
            checkpoint_for_test=str(args.output / f"global_step_{args.stop_at_step}/actor"),
            training_validation_split=source_contract.get("training_validation_split", source_contract["evaluation_split"]),
            pilot_evaluation=source_contract.get("pilot_evaluation"),
        )
    if args.continuation_pilot:
        from shopping_grpo.training.grpo.gigpo import STEP_WEIGHT

        contract.update(
            continuation_pilot=args.continuation_pilot,
            rollout_group_size=group_size,
            checkpoint=str(args.output / f"global_step_{args.stop_at_step}/actor"),
            gigpo_step_weight=STEP_WEIGHT if args.continuation_pilot == "gigpo" else 0.0,
        )
        contract.pop("checkpoint_for_test")
    return command, environment, resolved, contract


def export_best(args, selection, environment):
    export_checkpoint(selection["best"]["actor"], args.output / "best-export", args.merged, args.logs, environment)


def export_checkpoint(actor, exported, merged, logs, environment):
    run_logged(
        ["bash", "scripts/export_grpo.sh", str(actor), str(exported)],
        environment,
        logs / "export.log",
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
            str(merged),
            "--bf16",
        ],
        environment,
        logs / "merge.log",
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
    if args.evaluate_checkpoint:
        evaluate_checkpoint_only(args)
        return
    command, environment, resolved, contract = prepare(args)
    print(json.dumps(contract, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("解析后的完整训练配置：", flush=True)
        print(json.dumps(resolved, ensure_ascii=False, indent=2), flush=True)
        print("dry-run 完成：未启动服务、训练、模型合并或 200 题测试。", flush=True)
        return
    short_validation = args.pilot_steps or args.continuation_pilot
    ports = (args.environment_port,) if short_validation else (args.environment_port, args.llm_port)
    for port in ports:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    args.logs.mkdir(parents=True, exist_ok=True)
    if not short_validation:
        args.evaluation.mkdir(parents=True, exist_ok=True)
    write_json(args.logs / "run_contract.json", contract)
    write_json(args.logs / "resolved_config.json", resolved)
    # SIGTERM follows the same cleanup path as Ctrl-C.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if args.resume_from and not short_validation:
        resume_and_test(args, command, environment, contract)
        return
    with ExitStack() as stack:
        env = start_environment(args, environment, stack)
        run_logged(command, environment, args.logs / "train.log")
        if short_validation:
            start = args.resume_step if args.continuation_pilot else 0
            stop = args.stop_at_step or args.pilot_steps
            start_override = {"start_step": start} if args.continuation_pilot else {}
            selection = select_checkpoint(args.output, total_steps=stop, **start_override)
            for path, expected in contract["data_sha256"].items():
                if sha256_file(ROOT / path) != expected:
                    raise ValueError("实验期间数据发生变化：" + path)
            expected_ids = [
                json.loads(line)["task_id"]
                for line in PILOT_EVALUATION_JSONL.read_text().splitlines()
                if line.strip()
            ]
            report_options = {}
            if args.continuation_pilot:
                report_options.update(
                    start_step=start,
                    group_size=contract["rollout_group_size"],
                    experiment=args.continuation_pilot,
                )
            report = pilot_validation_report(
                args.output, selection, expected_ids, task_batch_size=args.task_batch_size,
                learning_rate=contract.get("learning_rate", args.learning_rate),
                adam_beta1=contract.get("adam_betas", [args.adam_beta1])[0],
                total_steps=stop, **report_options,
            )
            if args.continuation_pilot:
                report["resume_from"] = contract["resume_from"]
            else:
                report["reference_sft"] = {
                    "source": str(PILOT_EVALUATION_METADATA.relative_to(ROOT)),
                    "tasks": 30,
                    "strict_successes": 0,
                    "strict_success_rate": 0.0,
                }
            report["task_sampling"] = contract.get("task_sampling")
            write_json(args.output / "validation_report.json", report)
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            return
        selection = select_checkpoint(args.output)
        write_json(args.output / "model_selection.json", selection)
        for path, expected in contract["data_sha256"].items():
            if sha256_file(ROOT / path) != expected:
                raise ValueError("训练期间数据发生变化，停止自动测试：" + path)
        export_best(args, selection, environment)
        evaluate_merged(args, args.merged, environment, env, stack, "配置C显存调整最佳模型")
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


def evaluate_merged(args, merged, environment, env, stack, label):
    server_log = stack.enter_context((args.logs / "model-server.log").open("w"))
    serving = {
        **environment,
        "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"].split(",")[0],
        "LLM_PORT": str(args.llm_port),
        "SERVED_MODEL_NAME": "shopping-grpo-config-c",
    }
    server = subprocess.Popen(
        ["bash", "scripts/serve_model.sh", str(merged)],
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
        "EVAL_SWANLAB_PROJECT": "shopping-grpo-longhorizon",
        "EVAL_SWANLAB_GROUP": "07-GRPO评测",
        "EVAL_SWANLAB_JOB_TYPE": "eval",
        "EVAL_SWANLAB_DESCRIPTION": f"GRPO 检查点（{label}）固定 200 题 Strict-Stratified-200 正式评测（T=0，max_steps=35）。",
        "EVAL_SWANLAB_TAGS": "EVAL,GRPO,Strict200",
    }
    name = datetime.now(timezone.utc).strftime("%Y-%m-%d") + "-EVAL-" + label
    run_logged(
        ["bash", "scripts/evaluate.sh", name], evaluation_env, args.logs / "evaluation.log"
    )


def completed_continuation_actor(args):
    with (args.output / "training_diagnostics.jsonl").open() as handle:
        rows = map(json.loads, handle)
        updates = [row["global_step"] for row in rows if row["event"] == "optimizer_step"]
    if updates != list(range(args.resume_step + 1, args.stop_at_step + 1)):
        raise ValueError(f"续训未完整完成第 {args.resume_step + 1} 至 {args.stop_at_step} 步，停止评测")
    checkpoint = args.output / f"global_step_{args.stop_at_step}"
    validate_resume_checkpoint(checkpoint)
    return checkpoint / "actor"


def resume_and_test(args, command, environment, training_contract):
    scratch_root = ROOT / "outputs"
    with ExitStack() as stack:
        env = start_environment(args, environment, stack)
        run_logged(command, environment, args.logs / "train.log")
        actor = completed_continuation_actor(args)
        for path, expected in training_contract["data_sha256"].items():
            if sha256_file(ROOT / path) != expected:
                raise ValueError(f"续训期间固定数据或配置已变化：{path}")
        contract = checkpoint_test_contract(args, actor, scratch_root)
        write_json(args.evaluation / "run_contract.json", contract)
        run_checkpoint_test(args, actor, scratch_root, environment, env, stack)
    finish_checkpoint_test(args, actor, contract, training_contract=training_contract)


def checkpoint_test_contract(args, actor, scratch_root):
    shards = checkpoint_shards(actor)
    benchmark = ROOT / "data/evaluation/tasks.jsonl"
    metadata = read_json(benchmark.with_name("metadata.json"))
    test_ids = [json.loads(line)["task_id"] for line in benchmark.read_text().splitlines() if line.strip()]
    benchmark_hash = sha256_file(benchmark)
    if len(test_ids) != 200 or len(set(test_ids)) != 200 or benchmark_hash != metadata["sha256"]:
        raise ValueError("固定 200 题评测名单与登记不一致")
    budgets = ROOT / "data/cleaned/task_budgets.jsonl"
    # The veRL export and the separate BF16 merge both need temporary space.
    required_space = sum(path.stat().st_size for path in shards) + 1024**3
    if shutil.disk_usage(scratch_root).free < required_space:
        raise ValueError(f"临时空间不足：{scratch_root}。评测约需 {required_space / 1024**3:.1f} GiB 可用空间")
    return {
        "checkpoint": str(actor.relative_to(ROOT) if actor.is_relative_to(ROOT) else actor),
        "benchmark": str(benchmark.relative_to(ROOT)),
        "benchmark_sha256": benchmark_hash,
        "task_budgets_sha256": sha256_file(budgets),
        "tasks": len(test_ids),
        "scratch_directory": str(scratch_root),
        "output_directory": str(args.evaluation),
    }


def run_checkpoint_test(args, actor, scratch_root, environment, env, stack):
    scratch = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="grpo-eval-", dir=scratch_root)))
    merged = scratch / "merged"
    export_checkpoint(actor, scratch / "export", merged, args.logs, environment)
    evaluate_merged(args, merged, environment, env, stack, f"GRPO-{actor.parent.name}")


def evaluate_checkpoint_only(args):
    actor = args.evaluate_checkpoint
    if args.evaluation.exists():
        raise ValueError(f"评测目录已存在，请用 --output 指定新目录：{args.evaluation}")
    scratch_root = Path(tempfile.gettempdir())
    contract = checkpoint_test_contract(args, actor, scratch_root)
    if args.dry_run:
        print(json.dumps(contract, ensure_ascii=False, indent=2), flush=True)
        return

    for port in (args.environment_port, args.llm_port):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    args.evaluation.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation.mkdir()
    write_json(args.evaluation / "run_contract.json", contract)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "SHOPPING_TASK_BUDGETS": str(ROOT / "data/cleaned/task_budgets.jsonl"),
        "SHOP_ENV_CONFIG": str(ROOT / "environments/ShopSimulator/shop_env/configs/environment.json"),
        "SHOP_MAX_STEPS": "35",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    previous_handler = signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        with ExitStack() as stack:
            env = start_environment(args, environment, stack)
            run_checkpoint_test(args, actor, scratch_root, environment, env, stack)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    finish_checkpoint_test(args, actor, contract)


def finish_checkpoint_test(args, actor, contract, *, training_contract=None):
    summary_path = args.evaluation / "summary.json"
    result = read_json(summary_path)
    if result.get("expected_tasks") != 200 or result.get("completed_tasks") != 200:
        raise ValueError(f"固定 200 题评测未完整结束，请检查 {summary_path}")
    report = {"test_results": result, "run_contract": contract}
    if training_contract is not None:
        report["training"] = training_contract
    write_json(args.evaluation / "final_report.json", report)
    print(
        f"\n{'=' * 64}\n"
        f"  正式 200 题评测  |  {actor.parent.name}\n"
        f"  Strict 成功率     {result['strict_success_rate']:.2%}  ({result['strict_successes']}/200)\n"
        f"  Reward = 1       {result['full_reward_tasks']}/200（已核验）\n"
        f"{'=' * 64}\n"
        f"结果：{summary_path}\n报告：{args.evaluation / 'report.html'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
