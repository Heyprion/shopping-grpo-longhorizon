# Repository contract

This repository supports one workflow only:

```text
Baseline → SFT → GRPO → Evaluation
```

The runtime contract is ShopSimulator Environment v2.1, Strict Reward
(`shopsimulator-reward-strict-v1`), observation v2 and tool schema v2.
Reward is exactly `r_type * r_att * r_option * r_price`; do not add identity
discounts, bonuses, negative terminal utilities or length shaping.
Do not add compatibility launchers, historical datasets,
old benchmarks, machine-specific paths or experiment journals.

Full-corpus cleaning included the previous evaluation tasks, as authorized.
The new stratification, pools and initial selections are materialized in
`data/splits/manifest.json`. Training must never overlap the complete held-out
eval pool or its cross-tag instruction duplicates. Strict success
requires a complete `gold_purchase` terminal result with `reward_valid=true`.

Do not start training, merge models or run the 200-task evaluation unless the
user explicitly requests execution.

## 当前数据方案（启动训练前必读）

统一依据是 [docs/data-plan.md](docs/data-plan.md)，清洗实现细节见 [docs/task-cleaning.md](docs/task-cleaning.md)。本节及数据方案优先于旧数据 README、旧课程方案和脚本的数据默认值。

- **清洗已完成并固定**：23,421 条全量任务含旧测试集均已检查；16,741 条通过、6,680 条淘汰。唯一来源为 `data/cleaned/` 及其 metadata，不为调整训练规模或配比重跑清洗，不设人工复核队列。
- **分层及划分已落盘**：16,741 条正式标签在 `data/splits/labels.jsonl`，manifest 在 `data/splits/manifest.json`，逐题登记在 `data/splits/registry.jsonl`。按原始 `domain_zh` 9 大类 × 三档结构复杂度，需求数来自已提取的 `requirements`；不重新清洗、打标签或抽取首批名单。`prepare_data_splits.py verify` 核验后直接复用。
- **固定池边界**：原始 `tag=eval` 的 944 条全部保留测试用途，首批测试 200；与 eval 指令重复的 3 条 train 任务封存。其余 15,794 条分为 SFT 训练/验证候选池 4,500/500，RL 训练/验证候选池 10,494/300；seed=42、重复组不跨池。各阶段训练、阶段验证和测试相互隔离。
- **暂定首次用量与配比**：SFT 训练 800（simple/medium/hard=30%/50%/20%，240/400/160），验证 200；RL 训练 1,000（20%/50%/30%，200/500/300），验证 50；测试 200。训练数量和训练配比可根据训练、验证结果在本阶段训练池内调整，清洗结果和分层定义不随之变动；验证/测试保持固定自然分层分布，测试不用于调参。
- **首批题目已确定**：SFT 使用 `data/sft/train_tasks.jsonl` / `validation_tasks.jsonl`，RL 使用 `data/grpo/train.jsonl` / `validation.jsonl` 及已同步 Parquet，测试使用新 `data/evaluation/tasks.jsonl`。训练难度配额精确匹配，品类保持自然比例（逐类整数取整误差小于 1 条），各份名单均覆盖 9 大类。SFT 验证、RL 验证、测试的 simple/medium/hard 分别为 30/121/49、8/30/12、21/119/60。
- **补题及续训**：SFT / RL 训练剩余 3,700 / 9,494 题，见 `data/splits/reserves/`。用 `PYTHONPATH=src .venv/bin/python scripts/prepare_data_splits.py extend --stage rl --count N` 从本阶段训练池分配新批次（可 `--dry-run`，可 `--ratios A B C`），登记后不再被其他批次抽到；SFT 用 `--stage sft`。不得重抽首批、借用验证/测试题或把入选误报为已训练。续训按用户授权另行执行。
- **启动训练前**：固定任务已准备完，RL Parquet 已可用；SFT 仍需分别为已固定的 800/200 题采集新的 Strict Reward 示范。训练入口会拒绝不匹配的旧 SFT 示范。不能把旧 Reward-v3 轨迹只改 metadata 当作新数据，也不能在采集完成后重新随机划分训练/验证。
- **预算一致性**：所有环境进程（含 Ray worker）使用 `SHOPPING_TASK_BUDGETS` 指向 `data/cleaned/task_budgets.jsonl`，从仓库根目录可用 `export SHOPPING_TASK_BUDGETS="$PWD/data/cleaned/task_budgets.jsonl"`。清洗与 Strict Reward 共用结构化预算，不能让旧正则或无预算回退代替该标注。

## 当前采用的训练方案

后续用户要求启动 SFT / RL 时，默认遵循以下方案；旧脚本或 YAML 的裸默认值不代表当前配方。共同设置：双卡 FSDP2、BF16/SDPA、seed=42（不搜索 seed）、最长上下文 24576、SwanLab 记录。

- **SFT**：直接单阶段，入口 `bash scripts/sft.sh --balanced`；LoRA r32/alpha64/dropout0.05，有效 batch8（每卡 micro1×累积4），LR=1e-4、linear、warmup3%、3 epoch。按验证集 assistant-token 加权 NLL 最低选模。
- **SFT 优化**：`configs/fsdp2-balanced.yaml`，12/24 层隔层重算并保留 RNG、`reshard_after_forward=false`、`PYTORCH_ALLOC_CONF=expandable_segments:True`；开启 FLA 线性注意力和 Liger 融合交叉熵/RMSNorm/SwiGLU，loss 按全局有效 assistant tokens 归一化。
- **RL（GRPO）**：入口 `scripts/train_grpo.py`，默认从验证最佳新 SFT `outputs/models/sft-balanced-best-merged` 起步；新建 LoRA r16/alpha32，LR=1e-6、warmup15 步、500 次有效更新；group=4、每批2任务/PPO mini2任务，即有效8轨迹；每卡 micro4、累积1。其余学习设置沿用 `configs/grpo.yaml`（temperature0.7/top_p0.9、clip0.2、组内奖励减均值且不除标准差、KL/熵奖励关闭，保留熵诊断）。
- **RL 优化**：actor 参数/optimizer 不 offload，`reshard_after_forward=false`；`SHOPPING_GRPO_CHECKPOINTED_LAYERS=18`，`SHOPPING_GRPO_FAST_KERNELS=fla_conv,liger_norm`；开启 Triton 融合 PPO log-prob/entropy 输出头，保持 `actor_rollout_ref.model.use_liger=false`，单独启用 Norm。上述两个环境变量也须以字符串传给 Ray `runtime_env.env_vars`；不启用去 padding、动态 batch、额外 SwiGLU 或冻结权重同步裁剪。
- **RL 采样/验证**：异步 vLLM，TP1×2副本、显存比例0.45、`max_num_seqs=8`；验证50题/batch50、环境64槽，训练前验证，每50步验证/保存。完整启动覆盖参数见 `outputs/grpo-selected-config/recommended_command.txt` 和 `config.json`；启动前 dry-run 核对上述配方。RL 全部算子组合的整流程速度及长期效果尚未实测。
