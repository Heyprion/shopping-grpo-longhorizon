"""验证 benchmark 清单与评测入口的最小 CLI 行为。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

from scripts.evaluate_shop_benchmark import check_evaluation_services, parse_args
from scripts.evaluate_shop_benchmark import main as evaluate_main


class BenchmarkCliTest(unittest.TestCase):
    def test_swanlab_records_progress_and_summary_without_logging_api_key(self):
        logger = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "tasks.jsonl"
            benchmark.write_text('{"task_id": 1}\n')
            output = root / "trajectories.jsonl"

            def collect(*args, **kwargs):
                row = {"task_id": 1}
                output.write_text(json.dumps(row) + "\n")
                kwargs["on_trajectory"](row)
                return [row]

            argv = ["evaluate_shop_benchmark.py", "--benchmark", str(benchmark),
                    "--output", str(output), "--summary", str(root / "summary.json"),
                    "--model", "test", "--llm-base-url", "http://localhost:8000/v1",
                    "--api-key", "secret-test-key", "--swanlab"]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"swanlab": logger}), \
                patch("scripts.evaluate_shop_benchmark.load_tasks", return_value=[{"task_id": 1}]), \
                patch("scripts.evaluate_shop_benchmark.OpenAIChatClient"), \
                patch("scripts.evaluate_shop_benchmark.check_evaluation_services"), \
                patch("scripts.evaluate_shop_benchmark.collect_tasks", side_effect=collect), \
                patch("scripts.evaluate_shop_benchmark.summarize_trajectories",
                      return_value={"strict_gold_success_count": 1}), patch("builtins.print"):
                evaluate_main()
            self.assertNotIn("api_key", logger.init.call_args.kwargs["config"])
            self.assertEqual(logger.init.call_args.kwargs["config"]["expected_tasks"], 1)
            self.assertEqual(logger.log.call_args_list[0].args[0]["eval/completed_tasks"], 1)
            self.assertEqual(logger.log.call_args_list[-1].args[0]["eval/strict_gold_success_count"], 1)
            logger.finish.assert_called_once()

    def test_service_check_accepts_ready_model_and_post_only_environment(self):
        args = SimpleNamespace(timeout=180, llm_base_url="http://127.0.0.1:8000/v1",
                               model="shopping-agent", api_key="EMPTY", context_window=24576,
                               observation_token_budget=1536, base_url="http://127.0.0.1:5700")
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"data":[{"id":"shopping-agent"}]}'
        route_error = HTTPError("http://127.0.0.1:5700/api/shop_agent", 405, "Method Not Allowed", {}, None)
        with patch("scripts.evaluate_shop_benchmark.urlopen", side_effect=[response, route_error]) as request, \
            patch("scripts.evaluate_shop_benchmark.VllmChatTokenCounter") as counter, patch("builtins.print"):
            check_evaluation_services(args)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[-1].args[0], args.base_url + "/api/shop_agent")
        counter.return_value.assert_called_once()

    def test_unready_model_stops_before_reading_tasks_or_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trajectories.jsonl"
            argv = ["evaluate_shop_benchmark.py", "--benchmark", "unused.jsonl",
                    "--output", str(output), "--summary", str(output.with_suffix('.summary.json')),
                    "--model", "shopping-agent", "--llm-base-url", "http://127.0.0.1:8000/v1",
                    "--api-key", "EMPTY", "--swanlab"]
            with patch.object(sys, "argv", argv), \
                patch("scripts.evaluate_shop_benchmark.urlopen", side_effect=URLError("Connection refused")), \
                patch("scripts.evaluate_shop_benchmark.load_tasks") as load, \
                patch("scripts.evaluate_shop_benchmark.collect_tasks") as collect, \
                self.assertRaisesRegex(SystemExit, "vLLM.*尚未就绪"):
                evaluate_main()
            load.assert_not_called()
            collect.assert_not_called()
            self.assertFalse(output.exists())

    def test_evaluation_defaults_match_frozen_protocol(self):
        """Base、SFT、GRPO 必须默认使用同一 35 步上限。"""
        with patch.object(
            sys,
            "argv",
            [
                "evaluate_shop_benchmark.py",
                "--benchmark",
                "data/evaluation/tasks.jsonl",
                "--output",
                "outputs/eval/base/raw.jsonl",
                "--summary",
                "outputs/eval/base/summary.json",
                "--model",
                "Qwen/Qwen3.5-2B",
                "--llm-base-url",
                "http://127.0.0.1:8000/v1",
                "--api-key",
                "EMPTY",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.max_steps, 35)
        self.assertEqual(args.max_tokens, 512)
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.context_window, 24576)
        self.assertEqual(args.context_safety_margin, 512)
        self.assertFalse(args.context_compaction)
        self.assertEqual(args.observation_token_budget, 1536)
        self.assertEqual(args.observation_detail_token_budget, 4096)
        self.assertEqual(args.observation_generic_token_budget, 768)
        self.assertEqual(args.observation_search_top_k, 20)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
