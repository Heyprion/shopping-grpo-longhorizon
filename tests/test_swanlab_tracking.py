"""Verify unified single-project SwanLab tracking, grouping, job types, and descriptions."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import patch

from scripts import evaluate_base_model, evaluate_shop_benchmark, run_grpo_pipeline, train_grpo
from shopping_grpo.tracking import (
    DEFAULT_SWANLAB_PROJECT,
    SWANLAB_GROUPS,
    build_swanlab_run_name,
    swanlab_env_vars,
    swanlab_init_kwargs,
)


def test_tracking_helpers_enforce_single_project_and_naming():
    assert DEFAULT_SWANLAB_PROJECT == "shopping-grpo-longhorizon"
    assert build_swanlab_run_name("rl", "GRPO前100步", run_date="2026-11-19") == "2026-11-19-RL-GRPO前100步"
    env = swanlab_env_vars(
        group=SWANLAB_GROUPS["grpo_train"],
        job_type="train",
        description="GRPO 正式训练",
        tags=["GRPO", "StrictReward"],
    )
    assert env == {
        "SWANLAB_GROUP": "06-GRPO正式训练",
        "SWANLAB_JOB_TYPE": "train",
        "SWANLAB_DESCRIPTION": "GRPO 正式训练",
        "SWANLAB_TAGS": "GRPO,StrictReward",
    }
    kwargs = swanlab_init_kwargs(
        name="2026-11-19-EVAL-测试",
        group=SWANLAB_GROUPS["grpo_eval"],
        job_type="eval",
        description="固定 200 题测试",
    )
    assert kwargs["project"] == "shopping-grpo-longhorizon"
    assert kwargs["group"] == "07-GRPO评测"
    assert kwargs["job_type"] == "eval"
    assert kwargs["description"] == "固定 200 题测试"


def test_grpo_train_propagates_swanlab_group_job_type_and_description_to_ray(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").touch()
    data = tmp_path / "tasks.parquet"
    data.touch()
    monkeypatch.setenv("SWANLAB_API_KEY", "test-key")
    args = Namespace(
        model=model,
        train_data=data,
        val_data=data,
        env_url="http://127.0.0.1:15705",
        output=tmp_path / "output",
        logger="swanlab",
        experiment_name="2026-11-19-RL-测试分组",
        swanlab_group="06-GRPO正式训练",
        swanlab_job_type="train",
        swanlab_description="测试描述",
        swanlab_tags="GRPO,Test",
        config=train_grpo.ROOT / "configs/grpo.yaml",
        hydra_overrides=run_grpo_pipeline.OVERRIDES,
    )
    command, environment = train_grpo.build_command(args)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    config = run_grpo_pipeline.compose_runtime_config(command[5:])
    assert config.trainer.project_name == "shopping-grpo-longhorizon"
    worker_env = OmegaConf.to_container(config.ray_kwargs.ray_init.runtime_env)["env_vars"]
    assert worker_env["SWANLAB_GROUP"] == "06-GRPO正式训练"
    assert worker_env["SWANLAB_JOB_TYPE"] == "train"
    assert worker_env["SWANLAB_DESCRIPTION"] == "测试描述"
    assert worker_env["SWANLAB_TAGS"] == "GRPO,Test"


def test_evaluation_entrypoints_use_unified_project_and_eval_job_type(tmp_path):
    with patch(
        "sys.argv",
        [
            "evaluate_shop_benchmark.py",
            "--benchmark",
            "data/evaluation/tasks.jsonl",
            "--output",
            str(tmp_path / "raw.jsonl"),
            "--summary",
            str(tmp_path / "summary.json"),
            "--model",
            "test-model",
            "--llm-base-url",
            "http://127.0.0.1:8000/v1",
            "--api-key",
            "EMPTY",
        ],
    ):
        eval_args = evaluate_shop_benchmark.parse_args()
    assert eval_args.swanlab_project == "shopping-grpo-longhorizon"
    assert eval_args.swanlab_job_type == "eval"

    base_args = evaluate_base_model.parse_args(["--output-dir", str(tmp_path / "base-run"), "--model-size", "4b"])
    _, env, _ = evaluate_base_model.prepare(base_args, 12345, 12346)
    assert env["EVAL_SWANLAB_PROJECT"] == "shopping-grpo-longhorizon"
    assert env["EVAL_SWANLAB_GROUP"] == "02-基线评测"
    assert env["EVAL_SWANLAB_JOB_TYPE"] == "eval"
    assert "-BASE-Qwen3.5-4B同条件200题评测" in env["EVAL_SWANLAB_RUN_NAME"]
    assert "Qwen3.5-4B" in env["EVAL_SWANLAB_DESCRIPTION"]
