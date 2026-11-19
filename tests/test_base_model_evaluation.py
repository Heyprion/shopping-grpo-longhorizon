"""Exercise one-command service ownership without GPUs or benchmark rollouts."""

import json
import signal
import sys
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest

from scripts import evaluate_base_model as pipeline


def test_dry_run_never_imports_cuda_starts_services_or_creates_outputs(tmp_path, monkeypatch, capsys):
    output = tmp_path / "new-output"
    start = Mock(side_effect=AssertionError("must not start services"))
    monkeypatch.setattr(pipeline.subprocess, "Popen", start)
    monkeypatch.setitem(sys.modules, "torch", None)
    pipeline.main(["--dry-run", "--output-dir", str(output)])
    assert "未启动服务" in capsys.readouterr().out
    start.assert_not_called()
    assert not output.exists()


def test_plan_pins_two_gpu_recipe_and_passes_budget_to_all_services(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    args = pipeline.parse_args(["--output-dir", str(tmp_path / "run")])
    commands, env, contract = pipeline.prepare(args, 12345, 12346)
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["SHOPPING_TASK_BUDGETS"] == str(pipeline.ROOT / "data/cleaned/task_budgets.jsonl")
    assert env["SHOPSIM_PORT"] == "12345" and env["LLM_PORT"] == "12346"
    assert env["LLM_BASE_URL"] == "http://127.0.0.1:12346/v1"
    assert commands[0] == ["bash", "scripts/start_environment.sh"]
    assert commands[2][:2] == ["bash", "scripts/evaluate.sh"]
    model = commands[1]
    for flag, value in (("--tensor-parallel-size", "2"), ("--dtype", "bfloat16"), ("--seed", "42")):
        assert model[model.index(flag) + 1] == value
    assert json.loads(model[model.index("--default-chat-template-kwargs") + 1]) == {"enable_thinking": False}
    assert contract["tasks"] == 200 and contract["reward"] == "shopsimulator-reward-strict-v1"


@pytest.mark.parametrize("failure", [None, RuntimeError("model failed"), KeyboardInterrupt()])
def test_owned_processes_are_cleaned_on_success_startup_failure_and_interrupt(tmp_path, monkeypatch, failure):
    output = tmp_path / "evaluation"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2)))
    simulator, server, evaluator = Mock(), Mock(), Mock()
    evaluator.wait.return_value = 0
    start = Mock(side_effect=[simulator, server, evaluator])
    stop = Mock()
    monkeypatch.setattr(pipeline.subprocess, "Popen", start)
    monkeypatch.setattr(pipeline, "stop_owned", stop)

    def ready(*args, **kwargs):
        if failure is not None:
            raise failure
        (output / "summary.json").write_text(json.dumps({
            "expected_tasks": 200, "completed_tasks": 200, "strict_success_rate": 0.5,
        }))

    monkeypatch.setattr(pipeline, "wait_service", ready)
    if failure is not None:
        with pytest.raises(type(failure)):
            pipeline.main(["--output-dir", str(output)])
        assert [call.args[0] for call in stop.call_args_list] == [server, simulator]
    else:
        pipeline.main(["--output-dir", str(output)])
        assert [call.args[0] for call in stop.call_args_list] == [evaluator, server, simulator]
        assert (output / "run_contract.json").exists()
    assert all(call.kwargs["start_new_session"] for call in start.call_args_list)


def test_readiness_probe_does_not_lease_tasks(tmp_path, monkeypatch):
    process = Mock()
    process.poll.return_value = None
    request = Mock(side_effect=HTTPError("http://localhost/api/shop_agent", 405, "POST only", {}, None))
    monkeypatch.setattr(pipeline, "urlopen", request)
    pipeline.wait_service("http://localhost/api/shop_agent", [process], tmp_path / "env.log")
    request.assert_called_once_with("http://localhost/api/shop_agent", timeout=3)


def test_missing_gpus_stop_before_services_or_output(tmp_path, monkeypatch):
    output = tmp_path / "evaluation"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 0)))
    start = Mock()
    monkeypatch.setattr(pipeline.subprocess, "Popen", start)
    with pytest.raises(RuntimeError, match="两张可用 GPU"):
        pipeline.main(["--output-dir", str(output)])
    start.assert_not_called()
    assert not output.exists()


def test_cleanup_stops_owned_group_even_if_leader_exits_first(monkeypatch):
    process = Mock(pid=12345)
    process.poll.return_value = 0
    kill = Mock()
    monkeypatch.setattr(pipeline.os, "killpg", kill)
    pipeline.stop_owned(process)
    assert [call.args for call in kill.call_args_list] == [
        (12345, signal.SIGTERM), (12345, signal.SIGKILL),
    ]
