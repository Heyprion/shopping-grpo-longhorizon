# Repository contract

This repository supports one workflow only:

```text
Baseline → SFT → GRPO → Evaluation
```

The runtime contract is ShopSimulator Environment v2.1, Reward v3, observation
v2 and tool schema v2. Do not add compatibility launchers, historical datasets,
old benchmarks, machine-specific paths or experiment journals.

Training data must never overlap `data/evaluation/tasks.jsonl`. Strict success
requires a complete `gold_purchase` terminal result with `reward_valid=true`.

Do not start training, merge models or run the 200-task evaluation unless the
user explicitly requests execution.

## 当前采用的训练方案

后续用户要求启动 SFT / RL 时，默认遵循以下方案；旧脚本或 YAML 的裸默认值不代表当前配方。共同设置：双卡 FSDP2、BF16/SDPA、seed=42（不搜索 seed）、最长上下文 24576、SwanLab 记录。

- **SFT**：直接单阶段，入口 `bash scripts/sft.sh --balanced`；LoRA r32/alpha64/dropout0.05，有效 batch8（每卡 micro1×累积4），LR=1e-4、linear、warmup3%、3 epoch。按验证集 assistant-token 加权 NLL 最低选模。
- **SFT 优化**：`configs/fsdp2-balanced.yaml`，12/24 层隔层重算并保留 RNG、`reshard_after_forward=false`、`PYTORCH_ALLOC_CONF=expandable_segments:True`；开启 FLA 线性注意力和 Liger 融合交叉熵/RMSNorm/SwiGLU，loss 按全局有效 assistant tokens 归一化。
- **RL（GRPO）**：入口 `scripts/train_grpo.py`，默认从验证最佳新 SFT `outputs/models/sft-balanced-best-merged` 起步；新建 LoRA r16/alpha32，LR=1e-6、warmup15 步、500 次有效更新；group=4、每批2任务/PPO mini2任务，即有效8轨迹；每卡 micro4、累积1。其余学习设置沿用 `configs/grpo.yaml`（temperature0.7/top_p0.9、clip0.2、组内奖励减均值且不除标准差、KL/熵奖励关闭，保留熵诊断）。
- **RL 优化**：actor 参数/optimizer 不 offload，`reshard_after_forward=false`；`SHOPPING_GRPO_CHECKPOINTED_LAYERS=18`，`SHOPPING_GRPO_FAST_KERNELS=fla_conv,liger_norm`；开启 Triton 融合 PPO log-prob/entropy 输出头，保持 `actor_rollout_ref.model.use_liger=false`，单独启用 Norm。上述两个环境变量也须以字符串传给 Ray `runtime_env.env_vars`；不启用去 padding、动态 batch、额外 SwiGLU 或冻结权重同步裁剪。
- **RL 采样/验证**：异步 vLLM，TP1×2副本、显存比例0.45、`max_num_seqs=8`；验证50题/batch50、环境64槽，训练前验证，每50步验证/保存。完整启动覆盖参数见 `outputs/grpo-selected-config/recommended_command.txt` 和 `config.json`；启动前 dry-run 核对上述配方。RL 全部算子组合的整流程速度及长期效果尚未实测。
