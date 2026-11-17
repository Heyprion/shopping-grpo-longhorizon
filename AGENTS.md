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

统一依据是 [docs/data-plan.md](docs/data-plan.md)，SFT 采集见 [data/sft/README.md](data/sft/README.md) 和 `data/sft/collection_plan.json`。新方案优先于旧数据 README、旧课程方案、旧 SFT 分块和脚本默认值。

- **清洗和标签固定**：23,421 条全量任务中 16,741 条通过、6,680 条淘汰。清洗产物在 `data/cleaned/`，逐题标签在 `data/splits/labels.jsonl`；不为本次调整重跑清洗或打标签，不设人工复核队列。
- **用途边界固定**：SFT 候选总计 5,000 题，RL 训练/验证池为 10,494/300；原始 eval 的 944 题保留测试用途，3 条跨来源重复 train 题封存。各阶段与测试用途隔离，重复指令不跨最终训练/验证。
- **SFT 第一版已定稿**：统一采集 1,000 条后，仅应用第一阶段长度与合法性筛选，正式保留 **988 条**，不应用第二阶段。固定 seed=42、验证:训练约 1:9，已冻结 **99 条验证 / 889 条训练**；正式名单见 `manifest.sft_final` 和 `data/sft/final_*_tasks.jsonl`。旧 4,500/500 来源分块、800/200 文件与 `initial.sft_*` 仅作来源记录。
- **SFT 当前目标**：先获得 **1,000 道不同任务各一条合格轨迹**，作为统一的未划分集合；整体难度 simple/medium/hard=30%/50%/20%，即 **300/500/200**，品类按统一 SFT 池自然比例覆盖 9 大类。以实际接受轨迹填联合配额，不仅控制尝试分布。
- **SFT 拒绝采样**：每个候选任务最多安排 3 次独立采样，首次成功即停止该题，每次只按真实环境终局 Strict Reward==1 验收，不附加 gold ASIN、轨迹长度、guard 等质量过滤。同题首次成功立即接受且不再采样，只留 attempt 序号最早的一条合格轨迹，三次均失败则在统一 SFT 池同品类×难度层补题，不受旧 4,500/500 分块限制。接受 1,000 条且配额填满后停止，不用 RL/测试或重复轨迹凑数。
- **筛选与划分已执行**：`data/sft/trajectories.jsonl` 为格式化后的正式 988 条集合，`train.jsonl` / `validation.jsonl` 为 889 / 99；第一阶段排除 12 条，第二阶段暂不处理，不补采。`quality_report.json`、`metadata.json` 及登记表记录规则、来源与校验和。
- **RL / 测试不变**：RL 首批训练 1,000（simple/medium/hard=200/500/300），验证 50（8/30/12），测试 200（21/119/60）。数据位于 `data/grpo/` 的 JSONL/Parquet 和 `data/evaluation/tasks.jsonl`，品类自然分布覆盖 9 大类；不得用测试结果调训练参数。
- **记录和扩量**：候选来源、标签及登记见 `data/splits/manifest.json` / `registry.jsonl`。SFT 执行采集时记录尝试、接受、三次耗尽、替补和统一成功名单，不能把旧入选标志当作成功或已训练。RL 训练尚余 9,494 题，用 `prepare_data_splits.py extend --stage rl` 按授权追加，保持验证/测试不变；旧 `--stage sft` 训练池补题在统一采集期间停用。
- **执行状态**：教师采集、第一阶段筛选及正式划分已完成，训练前检查已接入最终名单；尚未启动训练、合并或测试。`prepare_data_splits.py verify` 核验来源、最终 SFT 归属及未改动的 RL / 测试文件。旧 Reward-v3 示范不能改 metadata 冒充新示范。
- **预算一致性**：所有环境进程（含 Ray worker）使用 `SHOPPING_TASK_BUDGETS` 指向 `data/cleaned/task_budgets.jsonl`，可从仓库根目录设置 `export SHOPPING_TASK_BUDGETS="$PWD/data/cleaned/task_budgets.jsonl"`；不能以旧正则或无预算回退代替规范预算。

## 当前采用的训练方案

后续用户要求启动 SFT / RL 时，默认遵循以下方案；旧脚本或 YAML 的裸默认值不代表当前配方。共同设置：双卡 FSDP2、BF16/SDPA、seed=42（不搜索 seed）、最长上下文 24576、SwanLab 记录。

- **SFT**：直接单阶段，入口 `bash scripts/sft.sh --balanced --evaluate-after`；LoRA r32/alpha64/dropout0.05，有效 batch8（每卡 micro1×累积4），LR=1e-4、linear、warmup3%、3 epoch。按固定 99 条验证集的 assistant-token 加权 NLL 最低选模，三 epoch 结束加载最佳检查点。加 `--evaluate-after` 后自动合并最佳模型并执行固定 200 题测试，结果只作最终报告，不参与选模或调参。最佳 adapter / merged 目录为 `outputs/models/sft-stage1-v1-balanced` / `sft-stage1-v1-best-merged`。
- **SFT 优化**：`configs/fsdp2-balanced.yaml`，12/24 层隔层重算并保留 RNG、`reshard_after_forward=false`、`PYTORCH_ALLOC_CONF=expandable_segments:True`；开启 FLA 线性注意力和 Liger 融合交叉熵/RMSNorm/SwiGLU，loss 按全局有效 assistant tokens 归一化。
- **RL（GRPO，配置 C）**：一键训练评测入口 `bash scripts/grpo.sh`，底层 `scripts/train_grpo.py`；从验证最佳新 SFT `outputs/models/sft-stage1-v1-best-merged` 起步，新输出 `outputs/models/grpo-stage1-v2-config-c-cp20`。新建 LoRA r16/alpha32，LR=1e-6、warmup15 步、500 次有效更新；group=4、每批2任务/PPO mini2任务，即有效8轨迹；每卡 micro4、累积1。训练采样 temperature=1.0/top_p=1.0/top_k=-1；当前 1,000 题源清单保持不变，候选题按 simple/medium/hard=1:3:6 抽样，保持九大品类自然边际，seed42、有状态分层采样，同批指令组去重。补采上限仍为3批、连续跳过上限10；验证解码不变。其余沿用 `configs/grpo.yaml`（clip0.2、组内奖励减均值且不除标准差、KL/熵奖励关闭，保留熵诊断）。
- **RL 优化**：actor 参数/optimizer 不 offload，`reshard_after_forward=false`；`SHOPPING_GRPO_CHECKPOINTED_LAYERS=20`，`SHOPPING_GRPO_FAST_KERNELS=fla_conv,liger_norm`；开启 Triton 融合 PPO log-prob/entropy 输出头，保持 `actor_rollout_ref.model.use_liger=false`，单独启用 Norm。上述两个环境变量也须以字符串传给 Ray `runtime_env.env_vars`；不启用去 padding、动态 batch、额外 SwiGLU 或冻结权重同步裁剪。
- **RL 采样/验证**：异步 vLLM，TP1×2副本、显存比例0.45、`max_num_seqs=8`；验证50题/batch50、环境64槽，训练前验证，每50步验证/保存。当前完整配置用 `bash scripts/grpo.sh --dry-run` 核对，实际启动后保存到对应日志目录的 `resolved_config.json` 和 `run_contract.json`；配置 C 优先于旧资源推荐命令中的模型路径及采样默认值。RL 全部算子组合的整流程速度及长期效果尚未实测。
