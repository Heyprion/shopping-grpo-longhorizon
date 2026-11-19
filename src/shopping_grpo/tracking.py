"""Unified SwanLab project, grouping, job-type, naming, and description helpers."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

DEFAULT_SWANLAB_PROJECT = "shopping-grpo-longhorizon"

SWANLAB_GROUPS = {
    "data": "01-数据清洗与采集",
    "baseline": "02-基线评测",
    "sft_train": "03-SFT训练与调优",
    "sft_eval": "04-SFT评测",
    "grpo_probe": "05-GRPO工程摸底与排障",
    "grpo_train": "06-GRPO正式训练",
    "grpo_eval": "07-GRPO评测",
}

SWANLAB_JOB_TYPES = ("data", "train", "eval", "probe")


def build_swanlab_run_name(
    stage: str,
    purpose: str,
    *,
    run_date: str | None = None,
) -> str:
    """Build a canonical `YYYY-MM-DD-阶段-目的` run name."""
    date_part = run_date or datetime.now().astimezone().strftime("%Y-%m-%d")
    stage_part = stage.strip().upper()
    purpose_part = purpose.strip()
    return f"{date_part}-{stage_part}-{purpose_part}"


def swanlab_env_vars(
    *,
    group: str | None = None,
    job_type: str | None = None,
    description: str | None = None,
    tags: Iterable[str] | None = None,
) -> dict[str, str]:
    """Return SwanLab 0.9+ metadata env vars picked up automatically by swanlab.init."""
    env: dict[str, str] = {}
    if group:
        env["SWANLAB_GROUP"] = str(group)
    if job_type:
        env["SWANLAB_JOB_TYPE"] = str(job_type)
    if description:
        env["SWANLAB_DESCRIPTION"] = str(description)[:1024]
    if tags:
        cleaned = [str(tag).strip()[:20] for tag in tags if str(tag).strip()]
        if cleaned:
            env["SWANLAB_TAGS"] = ",".join(cleaned)
    return env


def swanlab_init_kwargs(
    *,
    project: str = DEFAULT_SWANLAB_PROJECT,
    name: str,
    group: str | None = None,
    job_type: str | None = None,
    description: str | None = None,
    tags: Iterable[str] | None = None,
    config: dict[str, Any] | None = None,
    mode: str | None = None,
    logdir: str | None = None,
) -> dict[str, Any]:
    """Build clean kwargs for `swanlab.init(...)`."""
    kwargs: dict[str, Any] = {
        "project": project,
        "experiment_name": name,
    }
    if group:
        kwargs["group"] = group
    if job_type:
        kwargs["job_type"] = job_type
    if description:
        kwargs["description"] = description[:1024]
    if tags:
        cleaned = [str(tag).strip()[:20] for tag in tags if str(tag).strip()]
        if cleaned:
            kwargs["tags"] = cleaned
    if config is not None:
        kwargs["config"] = config
    if mode is not None:
        kwargs["mode"] = mode
    if logdir is not None:
        kwargs["logdir"] = logdir
    return kwargs
