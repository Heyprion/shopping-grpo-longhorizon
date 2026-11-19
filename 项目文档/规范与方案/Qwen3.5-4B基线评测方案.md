# Qwen3.5-4B 同条件评测准备

使用官方后训练模型 [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)，
固定 revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，原始 BF16 权重。
本地入口为 `models/Qwen3.5-4B`，指向已下载的 Hugging Face snapshot，不重复存储权重。
权重、tokenizer、模板和当前测试名单的准备核验见本地
`outputs/preparation/qwen3.5-4b.json`；准备过程不运行测试任务。

使用当前 [Strict-Stratified-200](Strict-Stratified-200测试集说明.md)，不是旧 Final-200 Clean。
固定文件 `data/evaluation/tasks.jsonl`，200 道不同任务，难度为 21/119/60，
SHA-256 为 `e19a03a52d12ecd0635b3227ff11a713b8fe0a271f14c5ccdf07c45bcb2e9913`。
环境为 ShopSimulator v2.1 / Strict Reward / observation v2 / tools v2。
沿用公共 evaluator、system prompt、工具、观察预算与成功判定：每题一次 rollout，
temperature=0、top_p=1、最多 35 步、每次生成最多 512 tokens、上下文 24576、
安全余量 512、不压缩上下文。Strict Success 要求完整 `gold_purchase` 终局、
`reward_valid=true` 和完整 Strict Reward=1，以固定 200 题为分母。

模型服务固定两卡 tensor parallel（TP=2）、BF16、seed=42、关闭 thinking；
`--generation-config vllm` 使用中性生成默认值，避免模型自带采样配置介入。
`--language-model-only` 用于当前纯文本购物环境。
关闭 thinking 同时作用于 chat 与 `/tokenize` 模板，见
[vLLM 官方说明](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html#configuration-tips)。

从仓库根目录只需运行一个命令：

```bash
bash scripts/baseline.sh --qwen35-4b
```

入口会自动设置规范预算、选择空闲端口、启动购物环境及双卡模型服务、等待就绪、
运行固定 200 题评测并生成报告。默认使用 GPU 0、1，已有 `CUDA_VISIBLE_DEVICES`
设置时使用该设置中的前两张卡。结束或中断时清理本次启动的服务与模型 worker。
每次运行创建独立结果目录，不复用已有服务或轨迹；无需预先打开其他终端启动服务。

需要两张可见且空闲的 GPU；当前准备会话没有可见 GPU，尚未验证模型在双卡上的
实际加载，也未执行 200 题评测。在命令末尾加 `--dry-run` 只检查准备文件并显示命令，
不启动服务或加载模型。

评测入口会在租用任务前检查模型名、分词接口与环境路由，并记录 SwanLab。
结果目录为 `outputs/evaluation/qwen3.5-4b-strict200-tp2-<时间>/`，
包含 `run_contract.json`、`environment.log`、`model-server.log`、
`trajectories.jsonl`、`summary.json` 和 `report.html`。
终端会打印本次结果的完整路径；报告在本地生成，不额外调用外部 Judge。
重点查看 `strict_success_rate`、`mean_final_reward`、`average_steps`、
`reward_type_counts`、`status_counts` 与 `missing_tasks`。
也可用 `--output-dir` 显式指定新的空目录。

若在其他机器准备相同权重，可将固定版本下载到本地目录后复用上述命令：

```bash
.venv/bin/hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir models/Qwen3.5-4B
```
