#!/usr/bin/env python3
"""Collect the unified SFT pool with durable per-attempt state and joint quotas."""

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from shopping_grpo.evaluation.rollout import OpenAIChatClient, collect_for_task

ROOT = Path(__file__).resolve().parents[1]
SAMPLING_POLICY = "first_success_max_3"


def rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def atomic_jsonl(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def memory_fraction():
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit != "max":
            return int(Path("/sys/fs/cgroup/memory.current").read_text()) / int(limit)
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return 0.0


def valid_reward(t):
    # Acceptance uses the actual terminal reward only; no post-hoc quality rules.
    return t.get("done") is True and t.get("final_reward") == 1


def completion_usage(trajectory):
    totals = Counter()
    for event in trajectory.get("model_completions", []):
        usage = event.get("usage") or {}
        totals["responses"] += 1
        totals["truncated_responses"] += int(bool(event.get("truncated")))
        for name in ("prompt_tokens", "completion_tokens"):
            totals[name] += usage.get(name) or 0
        totals["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        ) or 0
    return totals


def reconcile_state(db, config, plan, pools, quotas, output_dir):
    """Adopt recorded successes without changing raw attempts or model settings."""
    stored = db.execute("SELECT value FROM meta WHERE key='config'").fetchone()
    previous = json.loads(stored[0]) if stored else None
    migrating = previous is not None and previous != config
    if migrating:
        expected = {key: value for key, value in config.items() if key != "sampling_policy"}
        expected["plan_sha256"] = plan.get("previous_plan_sha256")
        if previous != expected or not expected["plan_sha256"]:
            raise SystemExit(
                "Collection configuration differs from existing state; use a new output directory."
            )
    task_states = {}
    inventory = {task_id: stratum for stratum, ids in pools.items() for task_id in ids}
    task_attempts = defaultdict(list)
    for task_id, attempt, raw in db.execute("SELECT task_id,attempt,trajectory FROM attempts"):
        trajectory = json.loads(raw)
        task_attempts[task_id].append((attempt, valid_reward(trajectory)))
    reservations = dict(db.execute("SELECT task_id,stratum FROM tasks"))
    if set(task_attempts) - reservations.keys():
        raise SystemExit("Recorded attempt has no task reservation.")
    counts = Counter()
    for task_id, stratum in reservations.items():
        if inventory.get(task_id) != stratum:
            raise SystemExit("Recorded task is outside the planned SFT stratum.")
        attempts = sorted(task_attempts[task_id])
        if [attempt for attempt, _ in attempts] != list(range(len(attempts))) or len(attempts) > 3:
            raise SystemExit("Recorded attempts must be consecutive and limited to three.")
        status = (
            "accepted"
            if any(success for _, success in attempts)
            else ("exhausted" if len(attempts) == 3 else "active")
        )
        task_states[task_id] = status
        if status in ("accepted", "active"):
            counts[stratum] += 1
    if any(count > quotas.get(stratum, 0) for stratum, count in counts.items()):
        raise SystemExit("Recorded reservations exceed the current joint quotas.")
    if migrating:
        backup_path = output_dir / "collection.before-first-success.sqlite3"
        if backup_path.exists():
            raise SystemExit("First-success migration backup already exists; inspect state first.")
        backup = sqlite3.connect(backup_path)
        try:
            db.backup(backup)
        finally:
            backup.close()
    with db:
        if migrating:
            db.execute("INSERT INTO meta VALUES ('config_before_first_success', ?)", (stored[0],))
        db.execute(
            "INSERT OR REPLACE INTO meta VALUES ('config', ?)",
            (json.dumps(config, sort_keys=True),),
        )
        db.executemany(
            "UPDATE tasks SET status=? WHERE task_id=?",
            ((status, task_id) for task_id, status in task_states.items()),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/sft-unified-collection-64k")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--base-url", default="http://127.0.0.1:5700")
    p.add_argument("--swanlab-mode", choices=["online", "offline"], default="online")
    p.add_argument("--memory-limit", type=float, default=0.90)
    p.add_argument("--max-steps", type=int, default=35)
    p.add_argument("--max-tokens", type=int, default=65536)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument(
        "--reconcile-only",
        action="store_true",
        help="Reconcile saved successes and export progress without API calls.",
    )
    p.add_argument(
        "--probe-attempt-limit",
        type=int,
        help="Stop after this many attempts in a separate experimental output directory.",
    )
    args = p.parse_args()
    if (
        not 1 <= args.workers <= 2500
        or not 0 < args.memory_limit < 1
        or not 1 <= args.max_tokens <= 393216
        or args.timeout < 1
        or args.max_steps < 1
        or (args.probe_attempt_limit is not None and args.probe_attempt_limit < 1)
    ):
        p.error("invalid workers, memory limit, output budget, timeout or step limit")
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key and not args.reconcile_only:
        raise SystemExit("Set DEEPSEEK_API_KEY in your shell.")
    budget = Path(os.environ.get("SHOPPING_TASK_BUDGETS", "")).resolve()
    if budget != (ROOT / "data/cleaned/task_budgets.jsonl").resolve():
        raise SystemExit("SHOPPING_TASK_BUDGETS must point to the canonical budget file.")
    plan_path = ROOT / "data/sft/collection_plan.json"
    plan = json.loads(plan_path.read_text())
    if (
        plan.get("sampling_policy") != SAMPLING_POLICY
        or plan.get("attempts_per_task") != 3
        or plan.get("stop_task_after_first_success") is not True
    ):
        raise SystemExit("Collection plan must stop at first success with at most three attempts.")
    candidate_path = ROOT / plan["candidate_pool"]["path"]
    digest = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    if digest != plan["candidate_pool"]["sha256"]:
        raise SystemExit("Candidate file checksum differs from the plan.")
    candidates = rows(candidate_path)
    labels = {r["task_id"]: r for r in rows(ROOT / "data/splits/labels.jsonl")}
    quotas = plan["accepted_joint_quotas"]
    pools = defaultdict(list)
    for c in candidates:
        label = labels[c["task_id"]]
        if label["source_tag"] != "train":
            raise SystemExit("Candidate is not from train source.")
        pools[label["domain"] + "|" + label["difficulty"]].append(c["task_id"])
    if len({c["task_id"] for c in candidates}) != 5000 or set(pools) != set(quotas):
        raise SystemExit("Candidate pool does not match the plan.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / "collector.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another collector is using this output directory.")
    db = sqlite3.connect(args.output_dir / "collection.sqlite3")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT); CREATE TABLE IF NOT EXISTS tasks (task_id INTEGER PRIMARY KEY, stratum TEXT NOT NULL, status TEXT NOT NULL, replacement_for INTEGER); CREATE TABLE IF NOT EXISTS attempts (task_id INTEGER, attempt INTEGER, trajectory TEXT NOT NULL, PRIMARY KEY(task_id, attempt));"
    )
    config = {
        "candidate_sha256": digest,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "model": "deepseek-flash",
        "thinking": True,
        "reasoning_effort": "high",
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "max_steps": args.max_steps,
        "attempts": 3,
        "sampling_policy": SAMPLING_POLICY,
        "budget_sha256": hashlib.sha256(budget.read_bytes()).hexdigest(),
        "probe_attempt_limit": args.probe_attempt_limit,
    }
    reconcile_state(db, config, plan, pools, quotas, args.output_dir)
    successful_attempts = 0
    successful_task_ids = set()
    for (raw,) in db.execute("SELECT trajectory FROM attempts"):
        trajectory = json.loads(raw)
        if valid_reward(trajectory):
            successful_attempts += 1
            successful_task_ids.add(trajectory["task_id"])

    def snapshot():
        accepted_tasks = db.execute(
            "SELECT task_id,stratum,replacement_for FROM tasks WHERE status='accepted' ORDER BY task_id"
        ).fetchall()

        def accepted_rows():
            for task_id, stratum, replacement in accepted_tasks:
                attempts = db.execute(
                    "SELECT trajectory FROM attempts WHERE task_id=? ORDER BY attempt", (task_id,)
                ).fetchall()
                trajectory = next(
                    json.loads(a[0]) for a in attempts if valid_reward(json.loads(a[0]))
                )
                label = labels[task_id]
                trajectory["collection"] = {
                    "domain": label["domain"],
                    "difficulty": label["difficulty"],
                    "source_tag": label["source_tag"],
                    "stratum": stratum,
                    "replacement_for": replacement,
                }
                yield trajectory

        atomic_jsonl(args.output_dir / "accepted.jsonl", accepted_rows())
        atomic_jsonl(
            args.output_dir / "successful_tasks.jsonl",
            (
                {
                    "task_id": task_id,
                    "domain": labels[task_id]["domain"],
                    "difficulty": labels[task_id]["difficulty"],
                    "source_tag": labels[task_id]["source_tag"],
                    "stratum": stratum,
                    "replacement_for": replacement,
                }
                for task_id, stratum, replacement in accepted_tasks
            ),
        )
        states = {
            task_id: (status, replacement)
            for task_id, status, replacement in db.execute(
                "SELECT task_id,status,replacement_for FROM tasks"
            )
        }
        atomic_jsonl(
            args.output_dir / "task_status.jsonl",
            (
                {
                    "task_id": c["task_id"],
                    "domain": labels[c["task_id"]]["domain"],
                    "difficulty": labels[c["task_id"]]["difficulty"],
                    "status": states.get(c["task_id"], ("unattempted", None))[0],
                    "replacement_for": states.get(c["task_id"], (None, None))[1],
                }
                for c in candidates
            ),
        )
        counts = Counter(stratum for _, stratum, _ in accepted_tasks)
        summary = {
            "accepted": len(accepted_tasks),
            "successful_attempts": successful_attempts,
            "successful_tasks": len(successful_task_ids),
            "quotas": quotas,
            "accepted_by_stratum": dict(counts),
            "shortfalls": {k: v - counts[k] for k, v in quotas.items() if counts[k] < v},
            "attempts": db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
            "exhausted": db.execute(
                "SELECT COUNT(*) FROM tasks WHERE status='exhausted'"
            ).fetchone()[0],
        }
        temp = args.output_dir / "summary.json.tmp"
        temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        os.replace(temp, args.output_dir / "summary.json")
        return summary

    initial_summary = snapshot()
    if args.reconcile_only:
        print(json.dumps(initial_summary, ensure_ascii=False))
        db.close()
        return 0
    import swanlab

    swanlab.init(
        project="shopping-grpo-sft-collection",
        experiment_name=f"{datetime.now(timezone.utc):%Y-%m-%d}-COLLECT-教师轨迹采集",
        config={
            **config,
            "workers": args.workers,
            "memory_limit": args.memory_limit,
            "joint_quotas": quotas,
            "candidate_pool": str(candidate_path),
            "acceptance": "environment terminal reward == 1",
        },
        mode=args.swanlab_mode,
        logdir=str(args.output_dir / "swanlab"),
    )
    swanlab.log(
        {
            f"collection/{key}": initial_summary[key]
            for key in (
                "accepted",
                "attempts",
                "exhausted",
                "successful_attempts",
                "successful_tasks",
            )
        }
    )
    print(
        f"restored accepted={initial_summary['accepted']} attempts={initial_summary['attempts']}; stop each task on first success",
        flush=True,
    )

    def reserve():
        counts = Counter(
            dict(
                db.execute(
                    "SELECT stratum, COUNT(*) FROM tasks WHERE status IN ('accepted','active') GROUP BY stratum"
                )
            )
        )
        for stratum in quotas:
            if counts[stratum] >= quotas[stratum]:
                continue
            used = {x[0] for x in db.execute("SELECT task_id FROM tasks")}
            available = next((task_id for task_id in pools[stratum] if task_id not in used), None)
            if available is None:
                continue
            replacement = db.execute(
                "SELECT task_id FROM tasks WHERE stratum=? AND status='exhausted' AND task_id NOT IN (SELECT replacement_for FROM tasks WHERE replacement_for IS NOT NULL) ORDER BY rowid LIMIT 1",
                (stratum,),
            ).fetchone()
            db.execute(
                "INSERT INTO tasks VALUES (?, ?, 'active', ?)",
                (available, stratum, replacement[0] if replacement else None),
            )
            db.commit()
            return available
        return None

    def run(task_id, attempt):
        client = OpenAIChatClient(
            model="deepseek-flash",
            base_url="https://api.deepseek.com",
            api_key=key,
            thinking=True,
            reasoning_effort="high",
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        return collect_for_task(
            {"task_id": task_id},
            client=client,
            base_url=args.base_url,
            max_steps=args.max_steps,
            attempt_index=attempt,
        )

    pending = {}
    failed = False
    started = time.monotonic()
    initial_attempts = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    api_totals = Counter()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            while True:
                while (
                    not failed
                    and len(pending) < args.workers
                    and memory_fraction() < args.memory_limit
                ):
                    completed_count = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
                    if (
                        args.probe_attempt_limit is not None
                        and completed_count + len(pending) >= args.probe_attempt_limit
                    ):
                        break
                    active = [
                        x[0]
                        for x in db.execute(
                            "SELECT task_id FROM tasks WHERE status='active' ORDER BY rowid"
                        )
                        if x[0] not in pending.values()
                    ]
                    task_id = active[0] if active else reserve()
                    if task_id is None:
                        break
                    attempt = db.execute(
                        "SELECT COUNT(*) FROM attempts WHERE task_id=?", (task_id,)
                    ).fetchone()[0]
                    pending[executor.submit(run, task_id, attempt)] = task_id
                if not pending:
                    if memory_fraction() >= args.memory_limit:
                        print("Memory threshold reached; resume after reducing load.", flush=True)
                        failed = True
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    task_id = pending.pop(future)
                    t = future.result()
                    api_totals.update(completion_usage(t))
                    swanlab.log({f"api/{name}": value for name, value in api_totals.items()})
                    if t.get("error") or t.get("release_error"):
                        with (args.output_dir / "infrastructure_errors.jsonl").open(
                            "a", encoding="utf-8"
                        ) as errors:
                            errors.write(
                                json.dumps(
                                    {
                                        "task_id": task_id,
                                        "attempt_index": t["attempt_index"],
                                        "error": t.get("error"),
                                        "release_error": t.get("release_error"),
                                        "trajectory": t,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            errors.flush()
                            os.fsync(errors.fileno())
                        swanlab.log(
                            {
                                "collection/infrastructure_error": 1,
                                "collection/memory_fraction": memory_fraction(),
                            }
                        )
                        print(
                            "Infrastructure error, pausing:",
                            t.get("error") or t.get("release_error"),
                            flush=True,
                        )
                        failed = True
                        if not t.get("done"):
                            continue
                    db.execute(
                        "INSERT INTO attempts VALUES (?, ?, ?)",
                        (task_id, t["attempt_index"], json.dumps(t, ensure_ascii=False)),
                    )
                    attempts = db.execute(
                        "SELECT trajectory FROM attempts WHERE task_id=? ORDER BY attempt",
                        (task_id,),
                    ).fetchall()
                    if valid_reward(t):
                        db.execute("UPDATE tasks SET status='accepted' WHERE task_id=?", (task_id,))
                    elif len(attempts) == 3:
                        db.execute(
                            "UPDATE tasks SET status='exhausted' WHERE task_id=?", (task_id,)
                        )
                    db.commit()
                    if valid_reward(t):
                        successful_attempts += 1
                        successful_task_ids.add(task_id)
                    total = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
                    summary = {
                        "accepted": db.execute(
                            "SELECT COUNT(*) FROM tasks WHERE status='accepted'"
                        ).fetchone()[0],
                        "attempts": total,
                        "exhausted": db.execute(
                            "SELECT COUNT(*) FROM tasks WHERE status='exhausted'"
                        ).fetchone()[0],
                    }
                    if total % 100 == 0:
                        snapshot()
                    swanlab.log(
                        {
                            "collection/accepted": summary["accepted"],
                            "collection/successful_attempts": successful_attempts,
                            "collection/successful_tasks": len(successful_task_ids),
                            "collection/attempts": summary["attempts"],
                            "collection/exhausted": summary["exhausted"],
                            "collection/memory_fraction": memory_fraction(),
                            "collection/elapsed_seconds": time.monotonic() - started,
                            "collection/last_reward": t.get("final_reward", 0),
                            "collection/attempts_per_hour": 3600
                            * (summary["attempts"] - initial_attempts)
                            / max(time.monotonic() - started, 1),
                        }
                    )
                    print(
                        f"task={task_id} attempt={t['attempt_index'] + 1}/3 "
                        f"reward={t.get('final_reward', 0)} "
                        f"successful_attempts={successful_attempts} "
                        f"successful_tasks={len(successful_task_ids)} "
                        f"accepted={summary['accepted']} attempts={summary['attempts']} "
                        f"exhausted={summary['exhausted']}",
                        flush=True,
                    )
    finally:
        summary = snapshot()
        swanlab.finish()
        db.close()
    if failed or summary["shortfalls"]:
        print("Collection incomplete; see summary.json for shortfalls.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
