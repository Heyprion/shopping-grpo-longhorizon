#!/usr/bin/env python3
"""Run the two 200→250 branches after the active 200→300 continuation exits."""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import json
import os
import select
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from statistics import fmean

from run_grpo_pipeline import (
    PILOT_EVALUATION_JSONL, ROOT, SELECTION_METRIC, pilot_validation_report, read_json,
)

LOGS = ROOT / "outputs/logs"
MODELS = ROOT / "outputs/models"
SOURCE = MODELS / "grpo-step200-lr3e-5-beta0-retry/global_step_200"
CONTROL = "grpo-step300-lr3e-5-beta0"
ARMS = ("group8", "gigpo")
STUDY = ROOT / "outputs/analysis/grpo-group8-gigpo"
START, STOP = 200, 250
IDLE_LIMIT = 20 * 60
RUN_LIMIT = 3 * 60 * 60


def atomic_write(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def publish(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def open_pidfd(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    # The project's Python build predates glibc's wrapper for this Linux syscall.
    if sys.platform != "linux" or os.uname().machine not in {"x86_64", "aarch64"}:
        raise RuntimeError("The experiment supervisor requires Linux pidfd support")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    fd = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def wait_for_exit(fd, logs):
    """Block on the exact process, with an independent inactivity deadline."""
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    hard_deadline = time.monotonic() + RUN_LIMIT
    began = time.time()
    while not poll.poll(0):
        activity = max([began, *(path.stat().st_mtime for path in logs if path.exists())])
        seconds = min(IDLE_LIMIT - (time.time() - activity), hard_deadline - time.monotonic())
        if seconds <= 0:
            raise TimeoutError("进程超过 20 分钟无日志活动，或超过 3 小时运行上限")
        if poll.poll(max(1, int(seconds * 1000))):
            return


def control_pidfd(pid):
    try:
        fd = open_pidfd(pid)
    except ProcessLookupError:
        return None
    try:
        argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        if not argv or argv == [b""]:
            os.close(fd)
            return None
        flags = dict(zip(argv, argv[1:]))
        if (
            not any(part.endswith(b"scripts/run_grpo_pipeline.py") for part in argv)
            or Path(os.fsdecode(flags.get(b"--output", b""))).name != CONTROL
            or flags.get(b"--stop-at-step") != b"300"
            or Path(os.fsdecode(flags.get(b"--resume-from", b""))).name != SOURCE.name
        ):
            raise ValueError(f"PID {pid} 并非指定的 200→300 续训")
        return fd
    except FileNotFoundError:
        os.close(fd)
        return None
    except BaseException:
        os.close(fd)
        raise


def run_logged(command, log, progress_logs=()):
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        fd = open_pidfd(process.pid)
        try:
            wait_for_exit(fd, (log, *progress_logs))
            return process.wait()
        except BaseException as exc:
            process.terminate()
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if isinstance(exc, TimeoutError):
                print(f"{log.name}: {exc}；终止后的进程退出码 {process.returncode}", flush=True)
                return 124
            raise
        finally:
            os.close(fd)


def step_wall_time(model, updates):
    from swanlab.proto.swanlab.record.v1.record_pb2 import Record
    from swanlab.sdk.internal.core_python.store import DataStoreReader

    files = list((model / "swanlab").glob("run*/run*.swanlab"))
    if len(files) != 1:
        return None
    reader = DataStoreReader()
    reader.open(files[0])
    times = {}
    try:
        for data in reader:
            record = Record.FromString(data)
            if record.HasField("scalar"):
                scalar = record.scalar
                if (scalar.key, scalar.step) in (
                    (SELECTION_METRIC, START), ("training/global_step", STOP)
                ):
                    times[scalar.step] = scalar.timestamp.seconds + scalar.timestamp.nanos * 1e-9
    finally:
        reader.close()
    if times.keys() != {START, STOP}:
        return None
    overhead = sum(
        row["metrics"].get(f"timing_s/{key}", 0.0)
        for row in updates for key in ("save_checkpoint", "testing")
    )
    seconds = times[STOP] - times[START] - overhead
    return seconds if seconds > 0 else None


def summarize_run(model, *, group_size):
    with (model / "training_diagnostics.jsonl").open() as handle:
        rows = [
            row for row in map(json.loads, handle) if START < row["global_step"] <= STOP
        ]
    updates = [row for row in rows if row["event"] == "optimizer_step"]
    if [row["global_step"] for row in updates] != list(range(START + 1, STOP + 1)):
        raise ValueError(f"{model.name} 缺少连续的 50 次有效更新")
    batches = [row for row in rows if row["event"] == "generation_batch"]
    groups = [group for row in batches for group in row["groups"]]
    if not groups or any(len(group["terminal_utilities"]) != group_size for group in groups):
        raise ValueError(f"{model.name} 的采样组与预期不符")
    accepted = {group["uid"] for group in groups if group["accepted"]}
    trained_uids = {rollout["uid"] for row in updates for rollout in row["rollouts"]}
    trained = len(trained_uids)
    if trained != 2 * len(updates) or not trained_uids <= accepted:
        raise ValueError(f"{model.name} 的动态采样记录与实际更新数不符")
    valid = [group for group in groups if not group["sampling_invalid"]]
    varying = sum(group["kept"] for group in valid)
    generated = group_size * len(groups)
    seconds = step_wall_time(model, updates)
    generation_seconds = (
        sum(row["generation_seconds"] for row in batches)
        if all("generation_seconds" in row for row in batches) else None
    )
    metrics = {
        "candidate_groups": len(groups),
        "valid_groups": len(valid),
        "reward_varying_groups": varying,
        "trained_groups": trained,
        "reward_varying_group_rate": varying / len(groups),
        "reward_varying_valid_group_rate": varying / len(valid),
        "trained_group_rate": trained / len(groups),
        "generation_batches": len(batches),
        "batches_per_update": len(batches) / len(updates),
        "generated_rollouts": generated,
        "generated_rollouts_per_update": generated / len(updates),
        "candidate_strict_reward_mean": fmean(
            r for group in valid for r in group["terminal_utilities"]
        ),
        "time_excluding_validation_and_save_seconds": seconds,
        "rollouts_per_training_minute": 60 * generated / seconds if seconds else None,
        "trained_groups_per_training_hour": 3600 * trained / seconds if seconds else None,
        "generation_seconds": generation_seconds,
        "rollouts_per_generation_minute": (
            60 * generated / generation_seconds if generation_seconds else None
        ),
    }
    if group_size == 8:
        halves = [group["terminal_utilities"][i:i + 4] for group in valid for i in (0, 4)]
        metrics["paired_group4_reward_varying_rate_on_valid_group8"] = fmean(
            max(scores) - min(scores) > 1e-8 for scores in halves
        )
    credit = {
        key.removeprefix("gigpo/"): sum(row["metrics"][key] for row in updates)
        for key in updates[0]["metrics"] if key.startswith("gigpo/")
    }
    if credit:
        credit["credited_turn_rate"] = credit["credited_turns"] / max(1, credit["recorded_turns"])
        credit["credited_token_rate"] = credit["credited_tokens"] / max(1, credit["assistant_tokens"])
        credit["updates_with_credit"] = sum(
            row["metrics"]["gigpo/credited_turns"] > 0 for row in updates
        )
        metrics["gigpo"] = credit
    return metrics, updates


def summarize_validation(model, updates, group_size):
    if model.name != CONTROL:
        return read_json(model / "validation_report.json")
    checkpoint = {
        "step": STOP, "value": updates[-1]["metrics"][SELECTION_METRIC], "actor": None,
    }
    with PILOT_EVALUATION_JSONL.open() as handle:
        ids = [row["task_id"] for row in map(json.loads, handle)]
    return pilot_validation_report(
        model, {"best": checkpoint, "candidates": [checkpoint]}, ids,
        total_steps=STOP, start_step=START, group_size=group_size,
        learning_rate=3e-5, adam_beta1=0, experiment="group4-grpo",
        validation_end_step=STOP,
    )


def result_for(model, group_size):
    metrics, updates = summarize_run(model, group_size=group_size)
    validation = summarize_validation(model, updates, group_size)
    return {"model_directory": str(model.relative_to(ROOT)), "metrics": metrics, "validation": validation}


def save_arm(results, label, model, group_size):
    try:
        result = result_for(model, group_size)
        if label == "group4 GiGPO" and "gigpo" not in result["metrics"]:
            raise ValueError("GiGPO 训练记录缺少局部优势指标")
        results["arms"][label] = result
    except (OSError, ValueError, KeyError) as exc:
        traceback.print_exc()
        results.setdefault("analysis_errors", {})[label] = f"{type(exc).__name__}: {exc}"


def pct(value):
    return f"{100 * value:.1f}%"


def rate(value):
    return "—" if value is None else f"{value:.1f}"


def collect_control(results, state):
    model = MODELS / CONTROL
    result = result_for(model, 4)
    results["arms"]["group4 GRPO"] = result
    final_report = ROOT / "outputs/evaluation" / CONTROL / "final_report.json"
    if final_report.is_file():
        final = read_json(final_report)
        test = final["test_results"]
        if (test["completed_tasks"], final["training"]["total_effective_updates"]) != (200, 300):
            raise ValueError("原 200→300 续训缺少完整的训练及正式评测结果")
        state["control_completed_by_report"] = True
        results["step300_full_test"] = test
        return
    with (model / "training_diagnostics.jsonl").open() as handle:
        last = None
        for last in map(json.loads, handle):
            pass
    if last is None or last["event"] != "sampling_exhausted" or last["global_step"] <= STOP:
        raise ValueError("原续训未生成正式评测报告，也没有 step 250 后的动态采样耗尽记录")
    results["control_sampling_exhausted"] = last
    state["control_completed_by_report"] = False
    state["control_status"] = "sampling_exhausted_after_comparison_window"


def shutdown_server(command):
    try:
        subprocess.run(command, check=True)
    except OSError as exc:
        if exc.errno != errno.ENOEXEC:
            raise
        # This server exposes shutdown as an executable shell file without a shebang.
        subprocess.run(["bash", *command], check=True)


def write_report(state, results):
    publish(STUDY / "results.json", {"status": state, **results})
    rows = [
        "# Group=8 与 GiGPO 的 50 步对照",
        "",
        "三组均从同一个 step 200 检查点恢复，在 201–250 步做 50 次更新，每次使用 2 道题。",
        "固定 30 题来自 SFT 在原 200 题中的错题。以下成绩用于短测，单次运行和小样本只提供初步信号。",
        "",
        "| 方案 | 候选组有效信号率 | 实际入选率 | 批次/更新 | 轨迹/更新 | 轨迹/训练分钟 | 入选组/训练小时 | Strict 成功 200→250 | 平均 Strict Reward 200→250 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, result in results.get("arms", {}).items():
        m, v = result["metrics"], result["validation"]
        base, final = v["baseline"], v["final"]
        rows.append(
            f"| {key} | {pct(m['reward_varying_group_rate'])} | {pct(m['trained_group_rate'])} "
            f"| {rate(m['batches_per_update'])} | {rate(m['generated_rollouts_per_update'])} "
            f"| {rate(m['rollouts_per_training_minute'])} | {rate(m['trained_groups_per_training_hour'])} "
            f"| {base['strict_successes']}→{final['strict_successes']}/30 "
            f"| {base['mean_strict_reward']:.4f}→{final['mean_strict_reward']:.4f} |"
        )
    rows += [
        "",
        "有效信号率是所有候选任务组中通过校验且终局奖励有差异的比例。训练速率以 SwanLab "
        "记录的 step 200 验证结束至 step 250 更新结束为窗口，扣除保存和验证耗时。",
    ]
    control = results.get("arms", {}).get("group4 GRPO")
    group8 = results.get("arms", {}).get("group8 GRPO")
    gigpo = results.get("arms", {}).get("group4 GiGPO")
    if group8:
        m = group8["metrics"]
        rows += [
            "",
            f"group=8 中完整有效的组，其信号率为 {pct(m['reward_varying_valid_group_rate'])}。"
            f"将这些组各拆成两个 4 轨迹子组后，子组信号率为 "
            f"{pct(m['paired_group4_reward_varying_rate_on_valid_group8'])}。",
        ]
    if gigpo:
        g = gigpo["metrics"]["gigpo"]
        rows += [
            "",
            f"GiGPO 在 {g['updates_with_credit']}/50 次更新中提供局部优势，覆盖 "
            f"{g['credited_turns']}/{g['recorded_turns']} 个记录的环境动作回合和 "
            f"{pct(g['credited_token_rate'])} 的训练 assistant tokens。"
            f"{g['shared_anchors']} 个共享状态中有 {g['supported_anchors']} 个具有动作回报差异。",
            f"非零优势 token 数由 {g['nonzero_grpo_tokens']} 变为 {g['nonzero_mixed_tokens']}。",
            "GiGPO 使用 γ=1 的终局回报，局部优势权重 0.5。初始状态、每条轨迹的重复访问和无动作回报差异的状态"
            "不提供局部优势。同组终局奖励完全相同的候选任务继续被过滤。",
        ]
    if control and group8 and gigpo:
        rows += ["", "相对同起点的 group4 GRPO 对照"]
        a, b = control["metrics"], group8["metrics"]
        signal_delta = 100 * (b["reward_varying_group_rate"] - a["reward_varying_group_rate"])
        rows.append(f"- group=8 的候选组有效信号率变化 {signal_delta:+.1f} 个百分点。")
        if a["trained_groups_per_training_hour"] and b["trained_groups_per_training_hour"]:
            productivity = 100 * (b["trained_groups_per_training_hour"] / a["trained_groups_per_training_hour"] - 1)
            rows.append(f"- group=8 的每训练小时入选组数变化 {productivity:+.1f}%。")
        for name, arm in (("group=8", group8), ("GiGPO", gigpo)):
            base = control["validation"]["final"]
            final = arm["validation"]["final"]
            delta = final["strict_successes"] - base["strict_successes"]
            reward_delta = final["mean_strict_reward"] - base["mean_strict_reward"]
            rows.append(f"- {name} 的 step 250 Strict 成功题数变化 {delta:+d}/30，平均 Strict Reward 变化 {reward_delta:+.4f}。")
    if "step300_full_test" in results:
        test = results["step300_full_test"]
        rows += ["", f"原续训 step 300 的独立 200 题评测为 {test['strict_successes']}/200 Strict 成功。"]
    if "control_sampling_exhausted" in results:
        exhausted = results["control_sampling_exhausted"]
        rows += [
            "",
            f"原续训在 step {exhausted['global_step']} 的 {exhausted['generation_batches']} 次生成后，"
            f"凑到 {exhausted['accepted_prompts']}/{exhausted['target_prompts']} 个有效任务组并退出。"
            "用于比较的 201–250 步及两次 30 题验证完整，step 300 的正式评测未执行。",
        ]
    if state.get("error"):
        rows += ["", f"中止原因为 {state['error']}。详见状态和进程日志。"]
    for key, error in results.get("analysis_errors", {}).items():
        rows += ["", f"{key} 的汇总失败，原始记录仍保留。错误为 {error}。"]
    atomic_write(STUDY / "report.md", "\n".join(rows) + "\n")


def supervise(pid):
    state = {"worker_pid": os.getpid(), "control_pid": pid, "phase": "waiting_for_control", "exit_codes": {}}
    results = {"arms": {}}
    fd = control_pidfd(pid)
    publish(STUDY / "status.json", state)
    try:
        if fd is not None:
            try:
                wait_for_exit(fd, tuple(LOGS.joinpath(CONTROL).glob("*.log")) + (
                    LOGS / CONTROL / "evaluation.log", LOGS / CONTROL / "merge.log",
                    LOGS / CONTROL / "model-server.log",
                ))
            finally:
                os.close(fd)
        collect_control(results, state)
        write_report(state, results)
        state["phase"] = "installing_patch"
        publish(STUDY / "status.json", state)
        code = run_logged(
            [sys.executable, "scripts/apply_verl_dynamic_sampling_patch.py"],
            STUDY / "patch.log",
        )
        state["exit_codes"]["patch"] = code
        if code:
            raise RuntimeError(f"veRL 补丁准备失败，退出码 {code}")
        for arm, label, group_size in zip(ARMS, ("group8 GRPO", "group4 GiGPO"), (8, 4), strict=True):
            state["phase"] = arm
            publish(STUDY / "status.json", state)
            model = MODELS / f"{SOURCE.parent.name}-to{STOP}-{arm}"
            command = [
                "bash", "scripts/grpo.sh", "--resume-from", str(SOURCE),
                "--stop-at-step", str(STOP), "--continuation-pilot", arm, "--output", str(model),
            ]
            code = run_logged(command, STUDY / f"{arm}.log", (LOGS / model.name / "train.log",))
            state["exit_codes"][arm] = code
            publish(STUDY / "status.json", state)
            if code:
                raise RuntimeError(f"{arm} 训练或验证失败，退出码 {code}")
            save_arm(results, label, model, group_size)
            write_report(state, results)
        state.update(phase="finished", exit_code=0 if not results.get("analysis_errors") else 1)
    except BaseException as exc:
        traceback.print_exc()
        state.update(phase="finished", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    finally:
        state["shutdown_command"] = ["/usr/bin/shutdown", "-h", "now"]
        try:
            write_report(state, results)
        finally:
            try:
                publish(STUDY / "status.json", state)
                print(json.dumps(state, ensure_ascii=False), flush=True)
            finally:
                shutdown_server(state["shutdown_command"])
    return state["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-pid", required=True, type=int)
    args = parser.parse_args()
    STUDY.mkdir(parents=True, exist_ok=True)
    with (STUDY / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (STUDY / "status.json").exists():
            raise SystemExit("已有探索实验状态，保留原记录")
        signal.signal(signal.SIGTERM, signal.default_int_handler)
        return supervise(args.wait_for_pid)


if __name__ == "__main__":
    raise SystemExit(main())
