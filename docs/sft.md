# 第一版 SFT：第一阶段筛选、三轮训练与最佳模型测试

当前正式数据采用 `shopping-sft-stage1-v1`。1,000 条 reward-only 教师轨迹经第一阶段长度与合法性筛选后保留 **988 条**；未应用第二阶段过程质量筛选。固定 seed=42，按品类×难度分层及重复指令组隔离划分为 **889 条训练、99 条验证**，验证数量按 `round(988×0.1)` 取整。正式名单和校验和见 `data/splits/manifest.json` 的 `sft_final` 及 `data/sft/metadata.json`。旧 800/200 名单只保留来源用途。

用户在仓库根目录执行：

```bash
bash scripts/sft.sh --balanced --evaluate-after
```

只检查完整启动命令、不加载模型或执行训练：

```bash
bash scripts/sft.sh --balanced --evaluate-after --dry-run
```

流水线依次执行：

1. 校验正式 SFT 数据及防泄漏约束，检查双卡 FSDP2 依赖。
2. 训练 3 个 epoch，每个 epoch 后对固定 99 条验证轨迹评估并保存检查点。
3. 按验证集所有有效 assistant token 加权的 NLL 最低选择检查点。分布式验证补齐产生的重复样本先去重；测试集不参与选模。
4. 训练结束时重新加载最佳检查点，将最佳 LoRA adapter 保存至 `outputs/models/sft-stage1-v1-balanced/`，`train_summary.json` 记录指标和来源检查点。
5. 自动合并该 adapter 和同 revision 基座，写至 `outputs/models/sft-stage1-v1-best-merged/`。
6. 启动专用 ShopSimulator（规范预算）及 vLLM，按既有固定协议执行 `data/evaluation/tasks.jsonl` 的 200 题测试，生成报告并关闭本次启动的服务。

| 训练设置 | 当前配方 |
|---|---|
| 基座 | Qwen/Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| 并行与精度 | 双卡 FSDP2，BF16 / SDPA |
| 长度 / seed | 24,576 / 42 |
| LoRA | r32 / alpha64 / dropout0.05 |
| 有效 batch | 8：每卡 micro1 × 两卡 × 累积4 |
| 学习率 / 调度 | 1e-4 / linear / warmup3% |
| epoch | 3 |
| 计算优化 | FLA；Liger 融合交叉熵、RMSNorm、SwiGLU |
| 重计算 | 24 层中 12 层隔层重算，保留 RNG |
| FSDP 参数 | `reshard_after_forward=false`，无参数 offload |
| 选模 | 最低验证 assistant-token 加权 NLL |

仅 assistant 输出 token 参与 loss，system、用户及工具观察作为上下文。完整序列已按 Student 模板检查；训练预处理如发生额外丢弃会中止。

训练 SwanLab 项目为 `shopping-grpo-sft`，测试项目为 `shopping-grpo-evaluation`。测试不用于选择 epoch 或调整训练参数。测试输出在 `outputs/evaluation/sft-stage1-v1/`，其中 `summary.json` 为指标、`report.html` 为既有报告生成器产物、`final_report.json` 关联最佳模型选择与测试结果。服务日志同目录保存。

训练脚本默认写新目录，拒绝无意覆盖已有 adapter。如训练已经完成、仅自动测试被中断，可以续跑后处理：

```bash
.venv/bin/python scripts/evaluate_sft_checkpoint.py --adapter outputs/models/sft-stage1-v1-balanced
```

后处理会核对训练数据、最佳模型及已合并模型的身份；同一模型的固定 200 题测试支持接续已记录任务。训练恢复使用：

```bash
bash scripts/sft.sh --balanced --evaluate-after --resume-from-checkpoint outputs/models/sft-stage1-v1-balanced/checkpoint-STEP
```

本文提供可执行命令；准备数据与启动检查不代表已启动训练、模型合并或测试。
