"""Verify orchestration and checkpoint selection without training or model export."""

import json
import subprocess
from argparse import Namespace
from unittest.mock import MagicMock, Mock

import pytest

from scripts import run_grpo_pipeline as pipeline
from scripts.check_grpo_runtime import validate_reward_runtime_files
from shopping_grpo.environment.manifest import validate_manifest


def test_frozen_runtime_manifest_matches_current_strict_engine():
    manifest = pipeline.read_json(pipeline.ROOT / "data/environment.json")
    engine = pipeline.read_json(
        pipeline.ROOT / "environments/ShopSimulator/shop_env/configs/environment.json"
    )
    validate_manifest(manifest)
    validate_reward_runtime_files(manifest, pipeline.ROOT)
    assert manifest["reward"] == engine["reward"]
    assert manifest["reward"]["formula"] == "r_type*r_att*r_option*r_price"


def make_checkpoints(output):
    rows = []
    for step in range(1, 501):
        metrics = {"val-aux/shopsimulator/reward/mean@1": 1.0}
        if step % 50 == 0:
            # Two equal best checkpoints: choose the earlier, not the final one.
            metrics[pipeline.SELECTION_METRIC] = 0.8 if step in (100, 200) else 0.6
            actor = output / f"global_step_{step}/actor"
            (actor / "huggingface").mkdir(parents=True)
            (actor / "huggingface/config.json").write_text("{}")
            for rank in range(2):
                (actor / f"model_world_size_2_rank_{rank}.pt").touch()
            (actor / "fsdp_config.json").write_text('{"FSDP_version":2,"world_size":2}')
            (actor / "lora_train_meta.json").write_text('{"r":16,"lora_alpha":32}')
        rows.append({"event": "optimizer_step", "global_step": step, "metrics": metrics})
    path = output / "training_diagnostics.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path, rows


def test_best_checkpoint_uses_strict_validation_and_earliest_tie(tmp_path):
    make_checkpoints(tmp_path)
    selected = pipeline.select_checkpoint(tmp_path)
    assert selected["best"]["step"] == 100
    assert selected["best"]["value"] == 0.8
    assert len(selected["candidates"]) == 10


@pytest.mark.parametrize("problem", ["incomplete", "missing_validation", "nonfinite", "missing_actor", "wrong_alpha"])
def test_incomplete_or_unverifiable_training_cannot_reach_test(tmp_path, problem):
    path, rows = make_checkpoints(tmp_path)
    if problem == "incomplete":
        rows.pop()
    elif problem == "missing_validation":
        del rows[49]["metrics"][pipeline.SELECTION_METRIC]
    elif problem == "nonfinite":
        rows[49]["metrics"][pipeline.SELECTION_METRIC] = float("nan")
    elif problem == "missing_actor":
        (tmp_path / "global_step_100/actor/huggingface/config.json").unlink()
    else:
        (tmp_path / "global_step_100/actor/lora_train_meta.json").write_text('{"r":16,"lora_alpha":0}')
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        pipeline.select_checkpoint(tmp_path)


def test_recipe_composes_with_string_worker_env_and_fixed_learning_settings(monkeypatch):
    # Resolve the installed veRL schema, with no Ray, CUDA initialization or model load.
    for key, value in {
        "GRPO_MODEL_PATH": "/model", "GRPO_TRAIN_FILE": "/train.parquet",
        "GRPO_VAL_FILE": "/val.parquet", "GRPO_OUTPUT_DIR": "/output",
        "SHOPPING_GRPO_ROOT": str(pipeline.ROOT), "SHOPPING_TASK_BUDGETS": "/budgets.jsonl",
    }.items():
        monkeypatch.setenv(key, value)
    config = pipeline.compose_runtime_config(pipeline.OVERRIDES)
    actor, model, rollout = (config.actor_rollout_ref[k] for k in ("actor", "model", "rollout"))
    assert (config.data.train_batch_size, config.data.val_batch_size) == (2, 50)
    assert (actor.ppo_mini_batch_size, actor.ppo_micro_batch_size_per_gpu, rollout.n) == (2, 4, 4)
    assert (model.lora_rank, model.lora_alpha) == (16, 32)
    assert actor.optim.lr == 1e-6 and actor.optim.lr_warmup_steps == 15
    assert not actor.fsdp_config.param_offload and not actor.fsdp_config.optimizer_offload
    assert not actor.fsdp_config.reshard_after_forward
    assert actor.fsdp_config.dtype == "bfloat16" and actor.strategy == "fsdp2"
    assert model.use_fused_kernels and model.fused_kernel_options.impl_backend == "triton"
    assert not model.use_liger and not model.use_remove_padding and not actor.use_dynamic_bsz
    assert model.override_config.attn_implementation == "sdpa"
    assert not actor.use_kl_loss and actor.entropy_coeff == 0 and actor.calculate_entropy
    assert not config.algorithm.use_kl_in_reward and not config.algorithm.norm_adv_by_std_in_grpo
    assert not config.shopping_trace.enable
    assert (rollout.tensor_model_parallel_size, rollout.max_num_seqs, rollout.gpu_memory_utilization) == (1, 8, 0.45)
    assert rollout.mode == "async" and rollout.max_model_len == 24576
    assert (rollout.temperature, rollout.top_p, rollout.top_k) == (1.0, 1.0, -1)
    assert rollout.val_kwargs.temperature == 0 and rollout.val_kwargs.top_p == 1
    assert list(config.shopping_task_sampling.difficulty_weights) == [0.1, 0.3, 0.6]
    assert config.shopping_task_sampling.enable and config.shopping_task_sampling.seed == 42
    assert config.shopping_dynamic_sampling.max_num_gen_batches == 3
    assert config.shopping_dynamic_sampling.max_consecutive_skipped_updates == 10
    assert config.trainer.total_training_steps == 500
    assert config.trainer.test_freq == config.trainer.save_freq == 50
    assert config.trainer.val_before_train and config.trainer.max_actor_ckpt_to_keep is None
    env = config.ray_kwargs.ray_init.runtime_env.env_vars
    assert env.SHOPPING_GRPO_CHECKPOINTED_LAYERS == "20"
    assert env.SHOPPING_GRPO_FAST_KERNELS == "fla_conv,liger_norm"
    assert env.SHOPPING_TASK_BUDGETS == "/budgets.jsonl"


def test_export_merges_rl_adapter_instead_of_serving_frozen_base(tmp_path, monkeypatch):
    args = pipeline.parse_args(["--output", str(tmp_path / "checkpoints")])
    adapter = args.output / "best-export/lora_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"r":16,"lora_alpha":32}')
    calls = []
    monkeypatch.setattr(pipeline, "run_logged", lambda command, *_: calls.append(command))
    pipeline.export_best(args, {"best": {"actor": "/selected/actor"}}, {})
    assert calls[0][:2] == ["bash", "scripts/export_grpo.sh"]
    merge = calls[1]
    assert merge[1] == "scripts/merge_lora_adapter.py"
    assert merge[merge.index("--adapter") + 1] == str(adapter)
    assert merge[merge.index("--base-model") + 1] == str(adapter.parent)
    assert merge[merge.index("--output") + 1] == str(args.merged)


def test_training_name_survives_hydra_unicode_and_punctuation(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").touch()
    data = tmp_path / "tasks.parquet"
    data.touch()
    monkeypatch.setenv("SWANLAB_API_KEY", "test-key")
    budget_path = str(tmp_path / "canonical budgets.jsonl")
    monkeypatch.setenv("SHOPPING_TASK_BUDGETS", budget_path)
    name = "2026-11-16-RL-第一版正式训练 [batch=8]"
    args = Namespace(model=model, train_data=data, val_data=data, env_url="http://127.0.0.1:15705",
                     output=tmp_path / "output", logger="swanlab", experiment_name=name,
                     config=pipeline.ROOT / "configs/grpo.yaml", hydra_overrides=pipeline.OVERRIDES)
    command, environment = pipeline.build_command(args)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    config = pipeline.compose_runtime_config(command[5:])
    assert config.trainer.experiment_name == name
    assert list(config.trainer.logger) == ["console", "swanlab"]
    # Match veRL's unresolved conversion when handing the environment to Ray.
    worker_env = OmegaConf.to_container(config.ray_kwargs.ray_init.runtime_env)["env_vars"]
    assert worker_env["SHOPPING_TASK_BUDGETS"] == budget_path


def test_dry_run_never_starts_services_or_writes_outputs(tmp_path, monkeypatch):
    output = tmp_path / "new-output"
    monkeypatch.setattr(pipeline, "prepare", lambda args: ([], {}, {}, {}))
    popen = Mock(side_effect=AssertionError("dry-run launched a child"))
    monkeypatch.setattr(pipeline.subprocess, "Popen", popen)
    pipeline.main(["--output", str(output), "--dry-run"])
    popen.assert_not_called()
    assert not output.exists()


def test_training_failure_stops_before_selection_export_and_evaluation(tmp_path, monkeypatch):
    args = pipeline.parse_args(["--output", str(tmp_path / "checkpoints")])
    args.logs = tmp_path / "logs"
    args.evaluation = tmp_path / "evaluation"
    monkeypatch.setattr(pipeline, "parse_args", lambda _: args)
    monkeypatch.setattr(pipeline, "prepare", lambda _: (["train"], {}, {}, {}))
    monkeypatch.setattr(pipeline, "start_environment", lambda *_: {})
    monkeypatch.setattr(pipeline.signal, "signal", lambda *_: None)
    monkeypatch.setattr(pipeline.socket, "socket", MagicMock())
    monkeypatch.setattr(pipeline, "run_logged", Mock(side_effect=subprocess.CalledProcessError(7, "train")))
    select = Mock(side_effect=AssertionError("selected after failed training"))
    monkeypatch.setattr(pipeline, "select_checkpoint", select)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.main([])
    select.assert_not_called()
    assert not args.merged.exists()


def test_successful_pipeline_serves_merged_best_and_tests_only_once(tmp_path, monkeypatch):
    args = pipeline.parse_args(["--output", str(tmp_path / "checkpoints")])
    args.logs, args.evaluation = tmp_path / "logs", tmp_path / "evaluation"
    selection = {"best": {"step": 100, "actor": "actor"}}
    env = {"CUDA_VISIBLE_DEVICES": "1,0"}
    calls = []
    monkeypatch.setattr(pipeline, "parse_args", lambda _: args)
    monkeypatch.setattr(pipeline, "prepare", lambda _: (["train"], env, {}, {"data_sha256": {}}))
    monkeypatch.setattr(pipeline, "start_environment", lambda *_: env)
    monkeypatch.setattr(pipeline, "select_checkpoint", lambda *_: selection)
    monkeypatch.setattr(pipeline, "export_best", lambda *_: calls.append("export_best"))
    monkeypatch.setattr(pipeline, "wait_service", lambda *_: None)
    monkeypatch.setattr(pipeline, "stop_owned", lambda *_: None)
    monkeypatch.setattr(pipeline.signal, "signal", lambda *_: None)
    monkeypatch.setattr(pipeline.socket, "socket", MagicMock())
    server = Mock()
    monkeypatch.setattr(pipeline.subprocess, "Popen", server)

    def run(command, environment, _log):
        calls.append(command)
        if command == ["train"]:
            args.output.mkdir()
        else:
            assert command[:2] == ["bash", "scripts/evaluate.sh"]
            assert environment["EVAL_OUTPUT_DIR"] == str(args.evaluation)
            pipeline.write_json(args.evaluation / "summary.json", {"expected_tasks": 200, "completed_tasks": 200})

    monkeypatch.setattr(pipeline, "run_logged", run)
    pipeline.main([])
    assert calls[0] == ["train"] and calls[1] == "export_best" and len(calls) == 3
    assert server.call_args.args[0][-1] == str(args.merged)
    assert server.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert pipeline.read_json(args.evaluation / "final_report.json")["model_selection"] == selection


def test_pilot_accepts_only_the_completed_50_step_checkpoint(tmp_path):
    path, rows = make_checkpoints(tmp_path)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows[:50]))
    selection = pipeline.select_checkpoint(tmp_path, total_steps=50)
    assert selection["best"]["step"] == 50
    with pytest.raises(ValueError, match="500"):
        pipeline.select_checkpoint(tmp_path)


def validation_row(task_id, success=False, alternative=False):
    score = float(success or alternative)
    return {
        "task_id": task_id, "done": True, "reward_valid": True,
        "reward_version": "shopsimulator-reward-strict-v1",
        "reward_type": "valid_alternative_purchase" if alternative else "gold_purchase" if success else "wrong_purchase",
        "reward": {"total": score, "purchase_success": score,
                   "r_type": 1.0, "r_att": 1.0, "r_option": score, "r_price": 1.0},
    }


def test_pilot_report_tracks_gains_losses_and_excludes_alternative_success(tmp_path):
    rows = [
        {"event": "validation_batch", "global_step": 0,
         "rollouts": [validation_row(1, success=True), validation_row(2), validation_row(3)]},
        {"event": "validation_batch", "global_step": 50,
         "rollouts": [validation_row(1), validation_row(2, success=True), validation_row(3, alternative=True)]},
    ]
    path = tmp_path / "validation_diagnostics.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    selection = {"best": {"value": 1 / 3, "actor": "/checkpoint/actor"}}
    report = pipeline.pilot_validation_report(tmp_path, selection, [1, 2, 3])
    assert report["baseline"]["strict_successes"] == 1
    assert report["step_50"]["strict_successes"] == 1
    assert report["step_50"]["mean_strict_reward"] == 2 / 3
    assert report["gained_task_ids"] == [2] and report["lost_task_ids"] == [1]
    assert not report["heldout_test_executed"]
    rows[1]["rollouts"][2] = validation_row(2)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="固定名单"):
        pipeline.pilot_validation_report(tmp_path, selection, [1, 2, 3])


@pytest.mark.parametrize("batch_size", [2, 4])
def test_pilot_finishes_after_validation_without_export_or_test(tmp_path, monkeypatch, batch_size):
    args = pipeline.parse_args(["--pilot-steps", "50", "--task-batch-size", str(batch_size),
                                "--output", str(tmp_path / "pilot")])
    args.logs = tmp_path / "logs"
    monkeypatch.setattr(pipeline, "parse_args", lambda _: args)
    monkeypatch.setattr(pipeline, "prepare", lambda _: (["train"], {}, {}, {"data_sha256": {}}))
    monkeypatch.setattr(pipeline, "start_environment", lambda *_: {})
    monkeypatch.setattr(pipeline.signal, "signal", lambda *_: None)
    monkeypatch.setattr(pipeline.socket, "socket", MagicMock())
    monkeypatch.setattr(pipeline.subprocess, "Popen", Mock(side_effect=AssertionError("unexpected service")))
    monkeypatch.setattr(pipeline, "export_best", Mock(side_effect=AssertionError("pilot exported model")))
    selection = {"best": {"step": 50, "value": 0.8, "actor": "/actor"}}
    selected = Mock(return_value=selection)
    monkeypatch.setattr(pipeline, "select_checkpoint", selected)
    report = {"effective_updates": 50, "heldout_test_executed": False}
    reporter = Mock(return_value=report)
    monkeypatch.setattr(pipeline, "pilot_validation_report", reporter)

    def train(command, *_):
        assert command == ["train"]
        args.output.mkdir()

    monkeypatch.setattr(pipeline, "run_logged", train)
    pipeline.main([])
    selected.assert_called_once_with(args.output, total_steps=50)
    assert len(reporter.call_args.args[2]) == 50
    assert reporter.call_args.kwargs["task_batch_size"] == batch_size
    assert reporter.call_args.kwargs["learning_rate"] == 1e-6
    assert pipeline.read_json(args.output / "validation_report.json") == report
    assert not args.evaluation.exists() and not args.merged.exists()


def test_batch4_pilot_keeps_learning_settings_and_uses_separate_outputs(monkeypatch):
    for key, value in {
        "GRPO_MODEL_PATH": "/model", "GRPO_TRAIN_FILE": "/train.parquet",
        "GRPO_VAL_FILE": "/val.parquet", "GRPO_OUTPUT_DIR": "/output",
        "SHOPPING_GRPO_ROOT": str(pipeline.ROOT), "SHOPPING_TASK_BUDGETS": "/budgets.jsonl",
    }.items():
        monkeypatch.setenv(key, value)
    control = pipeline.parse_args(["--pilot-steps", "50"])
    args = pipeline.parse_args(["--pilot-steps", "50", "--task-batch-size", "4"])
    config = pipeline.compose_runtime_config(pipeline.training_overrides(args))
    assert args.output != control.output and args.logs != control.logs
    assert args.output.name.endswith("-batch4")
    assert config.data.train_batch_size == config.actor_rollout_ref.actor.ppo_mini_batch_size == 4
    assert config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 4
    assert config.actor_rollout_ref.rollout.n == 4
    assert config.actor_rollout_ref.actor.optim.lr == 1e-6
    assert config.actor_rollout_ref.actor.optim.lr_warmup_steps == 15
    assert config.algorithm.rollout_correction.bypass_mode
    assert config.shopping_pilot.max_steps == 50
    assert config.trainer.total_training_steps == 500
    assert config.trainer.val_before_train and config.trainer.test_freq == 50
    with pytest.raises(SystemExit):
        pipeline.parse_args(["--task-batch-size", "4"])


def test_lr_pilot_changes_only_learning_rate_and_preserves_batch2_budget(monkeypatch):
    from omegaconf import OmegaConf

    for key, value in {
        "GRPO_MODEL_PATH": "/model", "GRPO_TRAIN_FILE": "/train.parquet",
        "GRPO_VAL_FILE": "/val.parquet", "GRPO_OUTPUT_DIR": "/output",
        "SHOPPING_GRPO_ROOT": str(pipeline.ROOT), "SHOPPING_TASK_BUDGETS": "/budgets.jsonl",
    }.items():
        monkeypatch.setenv(key, value)
    control = pipeline.parse_args(["--pilot-steps", "50"])
    args = pipeline.parse_args(["--pilot-steps", "50", "--learning-rate", "3e-6"])
    old = OmegaConf.to_container(pipeline.compose_runtime_config(pipeline.training_overrides(control)), resolve=True)
    new = OmegaConf.to_container(pipeline.compose_runtime_config(pipeline.training_overrides(args)), resolve=True)
    assert args.output != control.output and args.logs != control.logs
    assert args.output.name.endswith("-lr3e-6")
    assert new["actor_rollout_ref"]["actor"]["optim"]["lr"] == 3e-6
    new["actor_rollout_ref"]["actor"]["optim"]["lr"] = 1e-6
    assert new == old
    with pytest.raises(SystemExit):
        pipeline.parse_args(["--learning-rate", "3e-6"])
