#!/usr/bin/env python3
"""Bounded one-stage SFT search using the existing two-GPU trainer; never merge/evaluate Final-200."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize(summary):
    steps = summary["profile"]["steps"]
    steady = steps[2:] or steps
    evaluations = summary["profile"]["evaluations"]
    memory = max(row["reserved_gib"] for step in steps for row in step["ranks"])
    if evaluations:
        memory = max(memory, max(row[1] for evaluation in evaluations for row in evaluation["memory_gib"]))
    return {
        "tokens_per_second": sum(step["tokens"] for step in steady) / sum(step["seconds"] for step in steady),
        "padding_fraction": 1 - sum(step["tokens"] for step in steady) / sum(step["padded_tokens"] for step in steady),
        "peak_reserved_gib": memory,
        "peak_allocated_gib": max(row["allocated_gib"] for step in steps for row in step["ranks"]),
        "epoch": steps[-1]["epoch"],
        "evaluations": [{"epoch": evaluation["epoch"], **evaluation["metrics"]} for evaluation in evaluations],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/sft-grid"))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--phase", choices=("profiles", "quality", "all"), default="all")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "search_results.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"runs": {}}
    from datetime import datetime, timezone

    import swanlab

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    swanlab.init(project="shopping-grpo-longhorizon", experiment_name=f"{date_str}-PROBE-SFT网格搜索-{args.phase}",
                 group="03-SFT训练与调优", job_type="probe",
                 description=f"SFT 双卡 FSDP2 参数与吞吐网格搜索汇总（phase={args.phase}）。",
                 tags=["PROBE", "SFT", "Grid"],
                 config={"revision": args.revision, "lr_grid": [5e-5, 1e-4, 1.5e-4],
                         "effective_batch_grid": [8, 16], "epochs": 3,
                         "train": "data/sft/train.jsonl", "validation": "data/sft/validation.jsonl"},
                 logdir=str(output / "controller-swanlab"))

    def persist():
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False))

    def run(name, batch, accumulation, *, liger=False, profile=False, mixed=False, lr=1e-4,
            resume=None, pause=False):
        seed = 42
        run_dir = output / name
        summary_path = run_dir / "train_summary.json"
        if summary_path.exists() and (resume is None or json.loads(summary_path.read_text())["arguments"].get("resume_from_checkpoint")):
            summary = json.loads(summary_path.read_text())
            state["runs"][name] = {"status": "ok", "batch": batch, "accumulation": accumulation,
                                   "liger": liger, "lr": lr, "seed": seed, **summarize(summary)}
            persist()
            return state["runs"][name]
        run_dir.mkdir(exist_ok=True)
        command = [sys.executable, "-m", "accelerate.commands.launch", "--config_file", "configs/fsdp2.yaml",
                   "scripts/train_lora_sft.py", "--model", "Qwen/Qwen3.5-2B", "--revision", args.revision,
                   "--train", "data/sft/train.jsonl", "--validation", "data/sft/validation.jsonl",
                   "--output", str(run_dir), "--dtype", "bf16", "--attention-implementation", "sdpa", "--fsdp2", "--profile",
                   "--per-device-train-batch-size", str(batch), "--gradient-accumulation-steps", str(accumulation),
                   "--learning-rate", str(lr), "--epochs", "3", "--seed", str(seed),
                   "--preprocessing-cache", str(output / "cache"), "--logging-steps", "1",
                   "--swanlab", "--swanlab-project", "shopping-grpo-longhorizon",
                   "--swanlab-group", "03-SFT训练与调优", "--swanlab-job-type", "probe",
                   "--swanlab-description", f"SFT 网格搜索子实验：{name}（batch={batch}，accum={accumulation}，LR={lr:g}，liger={liger}）。",
                   "--swanlab-run-name", f"{date_str}-PROBE-SFT-{name}"]
        if liger:
            command.append("--liger-kernel")
        if profile:
            command.extend(["--max-steps", "8", "--no-save"])
            if not mixed:
                command.append("--longest-first")
        if pause:
            command.extend(["--stop-after-epoch", "1"])
        if resume:
            command.extend(["--resume-from-checkpoint", str(resume)])
        (run_dir / ("resume_command.json" if resume else "command.json")).write_text(json.dumps(command, indent=2))
        state["runs"][name] = {"status": "running", "batch": batch, "accumulation": accumulation,
                               "liger": liger, "lr": lr, "seed": seed}
        persist()
        print(f"START {name}", flush=True)
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "8"}
        started = time.time()
        log_path = run_dir / ("resume_console.log" if resume else "console.log")
        with log_path.open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=ROOT)
            state["runs"][name]["pid"] = process.pid
            persist()
            code = process.wait()
        if code:
            content = log_path.read_text()
            status = "oom" if "out of memory" in content.lower() else "failed"
            state["runs"][name].update(status=status, elapsed_seconds=time.time() - started)
            persist()
            swanlab.log({f"runs/{name}/failed": 1})
            print(f"{status.upper()} {name}: {content[-1600:]}", flush=True)
            if status != "oom":
                raise RuntimeError(f"{name} failed; inspect {log_path}")
            return state["runs"][name]
        summary = json.loads(summary_path.read_text())
        state["runs"][name].update(status="ok", **summarize(summary))
        persist()
        swanlab.log({f"runs/{name}/tokens_per_second": state["runs"][name]["tokens_per_second"],
                     f"runs/{name}/peak_reserved_gib": state["runs"][name]["peak_reserved_gib"]})
        print(f"DONE {name}: {json.dumps(state['runs'][name])}", flush=True)
        return state["runs"][name]

    if args.phase in {"all", "profiles"}:
        candidates = []
        for liger in [False, True]:
            for batch in [1, 2, 4]:
                name = f"profile-b{batch}" + ("-liger" if liger else "")
                result = run(name, batch, 4 // batch, liger=liger, profile=True)
                if result["status"] == "ok" and result["peak_reserved_gib"] < 81:
                    candidates.append((name, result))
        # Measure real random-order padding separately from the length-sorted stress test.
        mixed_results = []
        for name, result in sorted(candidates, key=lambda item: -item[1]["tokens_per_second"])[:2]:
            mixed = run(name + "-mixed", result["batch"], result["accumulation"],
                        liger=result["liger"], profile=True, mixed=True)
            if mixed["status"] == "ok":
                mixed_results.append(mixed)
        if not mixed_results:
            raise RuntimeError("No safe resource configuration")
        state["selected_resource"] = max(mixed_results, key=lambda row: row["tokens_per_second"])
        persist()
    if args.phase in {"all", "quality"}:
        selected = state["selected_resource"]
        batch, liger = selected["batch"], selected["liger"]
        quality = []
        for effective_batch in [8, 16]:
            for lr in [5e-5, 1e-4, 1.5e-4]:
                name = f"quality-eb{effective_batch}-lr{lr:g}"
                result = run(name, batch, effective_batch // (2 * batch), liger=liger, lr=lr, pause=True)
                if result["status"] == "ok":
                    quality.append((name, result))
        finalists = sorted(quality, key=lambda item: item[1]["evaluations"][0]["eval_assistant_nll"])[:2]
        for name, result in finalists:
            checkpoints = list((output / name).glob("checkpoint-*"))
            latest = max(checkpoints, key=lambda path: int(path.name.split("-")[-1]))
            # Keep the first-epoch summary as evidence before resuming.
            pilot = output / name / "pilot_summary.json"
            if not pilot.exists():
                pilot.write_text((output / name / "train_summary.json").read_text())
            run(name, batch, result["accumulation"], liger=liger, lr=result["lr"], resume=latest)
        state["finalists"] = [name for name, _ in finalists]
        best = min((state["runs"][name] for name, _ in finalists),
                   key=lambda row: min(e["eval_assistant_nll"] for e in row["evaluations"]))
        state["winner"] = best
        persist()
        state["seed_scope"] = [42]
        state["completed_seed42"] = True
        persist()
    swanlab.finish()


if __name__ == "__main__":
    main()
