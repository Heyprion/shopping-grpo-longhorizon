#!/usr/bin/env python3
"""在固定 ShopSimulator benchmark 上评测 OpenAI-compatible 本地或远端模型。"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from shopping_grpo.environment.context import VllmChatTokenCounter
from shopping_grpo.evaluation.rollout import (
    OpenAIChatClient,
    _is_infrastructure_failure,
    collect_tasks,
    completed_task_attempts,
    load_tasks,
)
from shopping_grpo.evaluation.summary import summarize_trajectories


def parse_args():
    parser = argparse.ArgumentParser(description="评测 Base、SFT 或 GRPO Shopping Agent")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="原始评测轨迹 JSONL")
    parser.add_argument("--summary", type=Path, required=True, help="汇总指标 JSON")
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--model", required=True)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--api-key", required=True, help="本地 vLLM 可传 EMPTY")
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="单次模型生成上限；防止未调用工具时耗尽完整上下文。",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=24576,
        help="上下文窗口；传 0 禁用 vLLM /tokenize 依赖。",
    )
    parser.add_argument("--context-safety-margin", type=int, default=512)
    parser.add_argument(
        "--context-compaction",
        action="store_true",
        help="上下文接近上限时压缩较早的交互；默认关闭。",
    )
    parser.add_argument(
        "--observation-token-budget",
        type=int,
        default=1536,
        help="Observation token 预算；传 0 禁用 vLLM /tokenize 依赖。",
    )
    parser.add_argument("--observation-detail-token-budget", type=int, default=4096)
    parser.add_argument("--observation-generic-token-budget", type=int, default=768)
    parser.add_argument("--observation-search-top-k", type=int, default=20)
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="shopping-grpo-evaluation")
    parser.add_argument("--swanlab-run-name", default="shopping-evaluation")
    return parser.parse_args()


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_evaluation_services(args):
    """Probe serving/tokenization and the environment route before leasing any task."""
    timeout = min(args.timeout, 5)
    model_url = f"{args.llm_base_url.rstrip('/')}/models"
    request = Request(model_url, headers={"Authorization": f"Bearer {args.api_key}"})
    try:
        with urlopen(request, timeout=timeout) as response:
            models = json.loads(response.read().decode("utf-8"))
    except (URLError, OSError, ValueError) as exc:
        raise SystemExit(
            f"vLLM 模型服务尚未就绪或无法访问：{model_url}\n"
            f"原因：{exc}\n请等待模型服务启动完成后重新运行评估。"
        ) from exc
    available = [row.get("id") for row in models.get("data", []) if isinstance(row, dict)]
    if args.model not in available:
        raise SystemExit(f"模型 {args.model!r} 不在服务列表中：{available}")
    if args.context_window or args.observation_token_budget:
        counter = VllmChatTokenCounter(args.model, args.llm_base_url, args.api_key, timeout=timeout)
        try:
            counter([{"role": "user", "content": "readiness check"}], [])
        except (URLError, OSError, ValueError) as exc:
            raise SystemExit(f"vLLM 分词接口尚未就绪：{counter.url}\n原因：{exc}") from exc

    environment_url = f"{args.base_url.rstrip('/')}/api/shop_agent"
    try:
        with urlopen(environment_url, timeout=timeout):
            pass
    except HTTPError as exc:
        # This API is POST-only; GET=405 proves the route exists without leasing an env.
        if exc.code != 405:
            raise SystemExit(f"购物环境接口不可用：{environment_url}\n原因：{exc}") from exc
    except (URLError, OSError) as exc:
        raise SystemExit(f"购物环境服务尚未就绪：{environment_url}\n原因：{exc}") from exc
    print(f"服务检查通过：模型 {args.model}、分词接口和购物环境接口均可访问。")


def main():
    args = parse_args()
    if args.max_steps < 1:
        raise SystemExit("--max-steps 必须为正数")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens 必须为正数")
    if args.context_window < 0:
        raise SystemExit("--context-window 不能为负数")
    if args.context_window and args.context_window <= args.max_tokens + args.context_safety_margin:
        raise SystemExit("--context-window 必须大于 --max-tokens 与安全余量之和")
    if args.observation_token_budget < 0:
        raise SystemExit("--observation-token-budget 不能为负数")
    check_evaluation_services(args)
    tasks = load_tasks(args.benchmark)
    swanlab_logger = None
    if args.swanlab:
        import swanlab

        swanlab_logger = swanlab
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items() if key != "api_key"}
        config.update(expected_tasks=len(tasks), reward_contract="shopsimulator-reward-v3",
                      benchmark_sha256=hashlib.sha256(args.benchmark.read_bytes()).hexdigest())
        swanlab.init(project=args.swanlab_project, experiment_name=args.swanlab_run_name,
                     config=config, logdir=str(args.output.parent / "swanlab"))
    expected_ids = {int(task["task_id"]) for task in tasks}
    completed = {task_id for task_id, _ in completed_task_attempts(args.output)
                 if task_id in expected_ids}
    started = time.monotonic()

    def on_trajectory(trajectory):
        if swanlab_logger is not None:
            if not _is_infrastructure_failure(trajectory):
                completed.add(int(trajectory["task_id"]))
            swanlab_logger.log({"eval/completed_tasks": len(completed),
                                "eval/last_task_id": int(trajectory["task_id"]),
                                "eval/elapsed_seconds": time.monotonic() - started})

    client = OpenAIChatClient(
        model=args.model,
        base_url=args.llm_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        context_window=args.context_window,
        context_safety_margin=args.context_safety_margin,
        context_compaction_enable=args.context_compaction,
        observation_token_budget=args.observation_token_budget,
        observation_detail_token_budget=args.observation_detail_token_budget,
        observation_generic_token_budget=args.observation_generic_token_budget,
        observation_search_top_k=args.observation_search_top_k,
    )
    collect_tasks(
        tasks,
        client=client,
        output_path=args.output,
        base_url=args.base_url,
        max_steps=args.max_steps,
        on_trajectory=on_trajectory if swanlab_logger is not None else None,
    )
    summary = summarize_trajectories(
        [task["task_id"] for task in tasks], _read_jsonl(args.output)
    )
    summary["protocol"] = {
        "benchmark": str(args.benchmark),
        "model": args.model,
        "reward_contract": "shopsimulator-reward-v3",
        "max_steps": args.max_steps,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "context_window": args.context_window,
        "context_safety_margin": args.context_safety_margin,
        "context_compaction": args.context_compaction,
        "observation_token_budget": args.observation_token_budget,
        "observation_detail_token_budget": args.observation_detail_token_budget,
        "observation_generic_token_budget": args.observation_generic_token_budget,
        "observation_search_top_k": args.observation_search_top_k,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if swanlab_logger is not None:
        swanlab_logger.log({f"eval/{key}": value for key, value in summary.items()
                            if isinstance(value, (int, float))})
        swanlab_logger.finish()


if __name__ == "__main__":
    main()
