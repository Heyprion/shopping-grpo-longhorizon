"""Frozen-policy sampling diagnostics: reward semantics, isolation, pairing and deadlines."""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import probe_rl_sampling as launcher
from shopping_grpo.training.grpo import sampling_study as study

ROOT = Path(__file__).resolve().parents[1]


def info(reward=1.0, **changes):
    return {
        "done": True,
        "reward_valid": True,
        "reward_version": "shopsimulator-reward-strict-v1",
        "infrastructure_invalid": False,
        "reward_type": "gold_purchase" if reward == 1 else "partial_alternative_purchase",
        "actions": [],
        "reward": {
            "terminal_utility": reward,
            "purchase_success": float(reward == 1),
            "r_type": 1,
            "r_att": reward,
            "r_option": 1,
            "r_price": 1,
        },
        **changes,
    }


def group(task, setting="A", phase="screen", effective=True, seconds=1.0, invalid=False):
    rewards = [0, 1, 1, 1] if effective else [1, 1, 1, 1]
    infos = [info(r, infrastructure_invalid=invalid) for r in rewards]
    return study.describe_group(phase, setting, task, infos, seconds, [100] * 4)


def panels(n=16):
    return [
        {"task_id": i + index * n, "difficulty": level, "domain": "domain"}
        for index, level in enumerate(study.LEVELS)
        for i in range(n)
    ]


def test_panel_is_fixed_balanced_disjoint_and_only_uses_current_training_pool():
    first = study.choose_tasks(ROOT)
    assert first == study.choose_tasks(ROOT)
    train = {r["task_id"] for r in study.read_rows(ROOT / "data/grpo/train.jsonl")}
    labels = study.read_rows(ROOT / "data/splits/labels.jsonl")
    protected_groups = {r["group_id"] for r in labels if r["source_tag"] == "eval"}
    all_rows = [r for values in first.values() for r in values]
    assert len(all_rows) == len({r["task_id"] for r in all_rows}) == 74
    assert len({r["group_id"] for r in all_rows}) == 74
    assert {r["task_id"] for r in all_rows} <= train
    assert not {r["group_id"] for r in all_rows} & protected_groups
    for key, n in [("screen", 16), ("confirmation", 8)]:
        assert {r["domain"] for r in first[key]} == {r["domain"] for r in labels}
        assert all(sum(r["difficulty"] == d for r in first[key]) == n for d in study.LEVELS)
        assert all(
            len(pair) == 2 and pair[0]["difficulty"] == pair[1]["difficulty"]
            for pair in study.task_pairs(first[key])
        )


def test_partial_rewards_are_learning_signal_even_without_strict_success():
    task = panels(1)[0]
    row = study.describe_group(
        "screen", "A", task, [info(r) for r in [0.1, 0.2, 0.3, 0.4]], 2, [10] * 4
    )
    assert row["effective"] and row["strict_success_count"] == 0
    assert row["reward_variance"] > 0
    identical = study.describe_group("screen", "A", task, [info(0.5)] * 4, 2, [10] * 4)
    assert not identical["effective"]


def test_unverifiable_group_is_never_counted_as_valid_reward_contrast():
    task = panels(1)[0]
    rows = [info(1), info(0), info(1), info(0, reward_unverifiable=True)]
    result = study.describe_group("screen", "A", task, rows, 2, [10] * 4)
    assert result["invalid"] and not result["effective"]
    assert result["drop_reason"] == "sampling_invalid"


def test_sample_streams_are_distinct_reproducible_and_fixed_across_settings():
    assert study.sample_seed(12, 0) == study.sample_seed(12, 0)
    assert len({study.sample_seed(12, i) for i in range(8)}) == 8
    assert study.sample_seed(13, 0) != study.sample_seed(12, 0)
    assert study.sample_seed(12, 0, 1) != study.sample_seed(12, 0, 0)


def test_native_client_uses_independent_turn_streams_and_does_not_mutate_settings():
    calls = []

    class Client:
        async def generate(self, request_id, **kwargs):
            calls.append(kwargs["sampling_params"])
            return "tokens"

    async def run():
        params = {"temperature": 0.7}
        client = launcher.TrajectoryClient(Client(), 123, 0)
        await client.generate("request", sampling_params=params)
        await client.generate("request", sampling_params=params)
        paired = launcher.TrajectoryClient(Client(), 123, 0)
        await paired.generate("request", sampling_params={"temperature": 1.0})
        assert params == {"temperature": 0.7}

    asyncio.run(run())
    assert calls[0]["seed"] != calls[1]["seed"]
    assert calls[0]["seed"] == calls[2]["seed"]


def test_cost_aware_ranking_does_not_choose_more_signal_at_ten_times_the_cost():
    tasks = panels()
    groups = []
    for setting in study.SETTINGS:
        for task in tasks:
            effective = task["task_id"] % 2 == 0 if setting == "A" else True
            groups.append(
                group(
                    task, setting=setting, effective=effective, seconds=1 if setting == "A" else 10
                )
            )
    selected = study.select_candidate(groups)
    assert selected["setting"] == "A" and selected["mixture"] == "20-50-30"
    assert selected["effective"] == 0.5


def test_matched_panel_excludes_tasks_missing_from_one_setting():
    tasks = panels(2)
    groups = [group(task, setting=arm) for task in tasks for arm in study.SETTINGS]
    groups.pop()
    ranked, _ = study.candidates(groups)
    assert all(r["tasks"] == 5 for r in ranked)
    assert study.select_candidate(groups) is None  # <8 per difficulty, no recommendation


def test_baseline_is_retained_when_estimated_gains_are_tied():
    groups = [group(task, setting=arm) for task in panels() for arm in study.SETTINGS]
    selected = study.select_candidate(groups)
    assert selected["setting"] == "A" and selected["mixture"] == "20-50-30"


def test_group8_is_conditional_and_confirmation_uses_new_tasks():
    selection = {"setting": "B", "mixture": "10-30-60"}
    groups = [
        group(task, arm, "confirmation", effective=(task["task_id"] % 4 == 0))
        for task in panels(8)
        for arm in ["A", "B"]
    ]
    assert study.group8_needed(groups, selection)
    all_effective = [
        group(task, arm, "confirmation", effective=True) for task in panels(8) for arm in ["A", "B"]
    ]
    assert not study.group8_needed(all_effective, selection)
    comparison = study.paired_confirmation(groups, selection)
    assert comparison["tasks"] == 24
    assert comparison["paired_bootstrap_repeats"] == 400


def test_partial_report_survives_interruption_and_does_not_claim_completed_evidence(tmp_path):
    study.write_json(tmp_path / "plan.json", study.PLAN)
    g = group(panels(1)[0])
    (tmp_path / "groups.jsonl").write_text(json.dumps(g) + '\n{"partial":')
    report = study.build_report(tmp_path, "time_budget_reached")
    assert report["completed_groups"] == 1 and report["evidence"] == "incomplete"
    assert not report["training_started"]
    assert (tmp_path / "report.md").exists()
    assert study.read_rows(tmp_path / "groups.jsonl", allow_partial=True) == [g]
    with pytest.raises(json.JSONDecodeError):
        study.read_rows(tmp_path / "groups.jsonl")


def test_group8_requires_eight_valid_samples_and_reports_native_sampling_cost(tmp_path):
    task_rows = panels(8)
    groups = [group(task, arm, "confirmation") for task in task_rows for arm in ["A", "B"]]
    for task in task_rows:
        groups.append(
            study.describe_group("group8", "B", task, [info(1), info(0)] * 4, 1.5, [100] * 8)
        )
    study.write_json(tmp_path / "plan.json", study.PLAN)
    study.write_json(tmp_path / "selection.json", {"setting": "B", "mixture": "10-30-60"})
    (tmp_path / "groups.jsonl").write_text("".join(json.dumps(g) + "\n" for g in groups))
    report = study.build_report(tmp_path, "completed")
    measured = report["native_group4_vs_group8"]
    assert measured["4"]["effective"] == measured["8"]["effective"] == 1
    assert measured["8"]["generated_tokens"] == 800
    assert (
        measured["8"]["effective_groups_per_minute"] < measured["4"]["effective_groups_per_minute"]
    )


@pytest.mark.parametrize("minutes", [0, 4, 121])
def test_budget_rejects_values_outside_five_to_120_minutes(minutes):
    with pytest.raises(SystemExit):
        launcher.parse_args(["--budget-minutes", str(minutes)])


def test_dry_run_never_launches_a_service_or_creates_output(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "preflight", lambda _: ({"model_updates": 0}, {}))
    popen = Mock(side_effect=AssertionError("dry run started a child"))
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    launcher.main(["--dry-run", "--output", str(tmp_path / "output")])
    popen.assert_not_called()
    assert not (tmp_path / "output").exists()


def test_supervisor_deadline_stops_its_worker_and_leaves_unrelated_process_alive(
    tmp_path, monkeypatch
):
    args = launcher.parse_args(["--output", str(tmp_path)])
    original_popen = subprocess.Popen
    owned = []
    unrelated = original_popen(
        [sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True
    )

    def popen(_command, **kwargs):
        worker = original_popen([sys.executable, "-c", "import time;time.sleep(60)"], **kwargs)
        owned.append(worker)
        return worker

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    try:
        status = launcher.supervise(args, time.time() + 29)
        assert status == "time_budget_reached"
        assert all(process.poll() is not None for process in owned)
        assert unrelated.poll() is None
    finally:
        os.killpg(unrelated.pid, 15)
        unrelated.wait(timeout=5)


def test_native_configuration_is_standalone_frozen_and_uses_real_weights(monkeypatch):
    args = launcher.parse_args(["--dry-run"])
    for key in (
        "GRPO_MODEL_PATH",
        "GRPO_TRAIN_FILE",
        "GRPO_VAL_FILE",
        "GRPO_OUTPUT_DIR",
        "SHOPPING_GRPO_ROOT",
        "SHOPPING_TASK_BUDGETS",
    ):
        monkeypatch.setenv(key, str(ROOT / "dummy"))
    cfg = launcher.compose_runtime_config(launcher.STUDY_OVERRIDES)
    assert cfg.actor_rollout_ref.model.lora_rank == 0
    assert cfg.actor_rollout_ref.model.lora.rank == 0
    assert cfg.actor_rollout_ref.rollout.load_format == "auto"
    assert cfg.actor_rollout_ref.rollout.nnodes == 1
    assert cfg.actor_rollout_ref.rollout.n_gpus_per_node == 2
    assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size == 1
    assert cfg.shopping_dynamic_sampling.max_num_gen_batches == 30
    assert args.budget_minutes == 90
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.workers.rollout.llm_server import LLMServerManager

    assert LLMServerManager(cfg).worker_group is None
    assert ToolListWrap([]).tools == [] and DictConfigWrap(cfg).config is cfg


def test_worker_collects_all_paired_arms_then_confirms_on_fresh_tasks(tmp_path, monkeypatch):
    import ray
    import swanlab

    task_panels = study.choose_tasks(ROOT)
    study.write_json(tmp_path / "plan.json", {**study.PLAN, "data_sha256": {}})
    study.write_json(tmp_path / "panels.json", task_panels)
    monkeypatch.setenv("SHOPPING_STUDY_DEADLINE_UNIX", str(time.time() + 600))
    monkeypatch.setattr(launcher, "configure_environment", lambda _: None)
    monkeypatch.setattr(launcher.subprocess, "run", Mock())
    monkeypatch.setattr(launcher, "compose_runtime_config", lambda _: None)
    shutdown = Mock()
    monkeypatch.setattr(ray, "shutdown", shutdown)
    for name in ["init", "log", "finish"]:
        monkeypatch.setattr(swanlab, name, Mock())

    class Sampler:
        async def sample(self, row, index, setting):
            task_id = row["extra_info"]["task_id"]
            # B has useful contrast, A is saturated, C is explicitly invalid.
            reward = float(index % 2) if setting == "B" else 1.0
            return {
                **info(reward, infrastructure_invalid=(setting == "C")),
                "task_id": task_id,
            }, 100

    async def initialize(_):
        return Sampler()

    monkeypatch.setattr(launcher, "initialize_sampler", initialize)
    args = launcher.parse_args(["--output", str(tmp_path)])
    asyncio.run(launcher.study_worker(args))
    rows = study.read_rows(tmp_path / "groups.jsonl")
    trajectories = study.read_rows(tmp_path / "trajectories.jsonl")
    selected = json.loads((tmp_path / "selection.json").read_text())
    assert selected["setting"] == "B"
    assert len([r for r in rows if r["phase"] == "screen"]) == 144
    assert len([r for r in rows if r["phase"] == "confirmation"]) == 48
    assert len(trajectories) == 776  # 8 warmup + 576 screen + 192 confirmation
    assert not any(r["phase"] == "group8" for r in rows)
    assert json.loads((tmp_path / "summary.json").read_text())["confirmation_complete"]
    assert not json.loads((tmp_path / "group8_decision.json").read_text())["run"]
    assert shutdown.call_count == 1


def test_native_agent_uses_real_sft_tokenizer_and_production_tools_without_model_load(monkeypatch):
    import hydra
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.tools.tool_registry import load_all_tools
    from verl.utils.config import omega_conf_to_dataclass
    from verl.utils.dataset.rl_dataset import RLHFDataset

    for key, value in {
        "GRPO_MODEL_PATH": str(launcher.MODEL),
        "GRPO_TRAIN_FILE": str(ROOT / "data/grpo/train.parquet"),
        "GRPO_VAL_FILE": str(ROOT / "data/grpo/validation.parquet"),
        "GRPO_OUTPUT_DIR": "/unused",
        "SHOPPING_GRPO_ROOT": str(ROOT),
        "SHOPPING_TASK_BUDGETS": str(ROOT / "data/cleaned/task_budgets.jsonl"),
        "SHOPSIM_BASE_URL": "http://127.0.0.1:15706",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        monkeypatch.setenv(key, value)
    cfg = launcher.compose_runtime_config(launcher.STUDY_OVERRIDES)
    model = omega_conf_to_dataclass(cfg.actor_rollout_ref.model)
    tools = load_all_tools(str(ROOT / "configs/tools.json"), None)
    row = pq.read_table(ROOT / "data/grpo/train.parquet").to_pylist()[0]

    async def construct_and_tokenize():
        agent = hydra.utils.instantiate(
            OmegaConf.load(ROOT / "configs/agent_loop.yaml")[0],
            trainer_config=DictConfigWrap(cfg),
            server_manager=None,
            tokenizer=model.tokenizer,
            processor=model.processor,
            dataset_cls=RLHFDataset,
            data_config=DictConfigWrap(cfg.data),
            tools=ToolListWrap(tools),
        )
        ids = await agent.apply_chat_template(row["prompt"], tools=agent.tool_schemas)
        assert 0 < len(ids) <= 4096
        assert "buy_now" in agent.tools and "finish_without_purchase" in agent.tools
        assert agent.max_steps == 35 and agent.context_window_tokens == 24576

    asyncio.run(construct_and_tokenize())
