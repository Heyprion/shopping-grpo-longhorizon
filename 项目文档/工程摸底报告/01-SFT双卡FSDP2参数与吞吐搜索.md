# 一阶段 SFT 双卡参数搜索

> 记录日期：2026-11-14 ｜ 原始实验目录：`outputs/sft-grid`

状态：已完成。

训练路径为直接一阶段 LoRA SFT，不使用 curriculum。训练集 800 条，24,576-token 上限保留 799 条；验证集 200 条全部保留。训练/验证/Final-200 task_id 两两不重叠。验证不参与反向传播。

SwanLab 统一项目（已归并汇总至 `03-SFT训练与调优`）：https://swanlab.cn/@WangZIyu/shopping-grpo-longhorizon

## 资源测试

两张 RTX PRO 6000 Blackwell，BF16、SDPA、FSDP2 activation checkpointing。统一安装 FLA 0.6.0；Liger 0.8.5 为搜索开关。有效 batch 固定为 8，按真实样本长度降序测试前 64 条，覆盖最长 24,359-token 样本。

| Liger | 每卡 batch | 累积 | 峰值 reserved GiB（单卡最大） | 结果 |
|---|---:|---:|---:|---|
| 关闭 | 1 | 4 | 83.91 | ok |
| 关闭 | 2 | 2 | — | oom |
| 关闭 | 4 | 1 | — | oom |
| 开启 | 1 | 4 | 18.23 | ok |
| 开启 | 2 | 2 | 35.12 | ok |
| 开启 | 4 | 1 | 63.31 | ok |

随机顺序复测（全局非 padding tokens/s，去掉前两次更新）：

| 每卡 batch | 累积 | tokens/s | padding 比例 |
|---:|---:|---:|---:|
| 1 | 4 | 17836 | 0.0% |
| 2 | 2 | 14061 | 10.9% |

短测可能包含新形状 Triton JIT 开销；batch=4 通过显存压力测试，但未进入随机顺序复测的前两名。这里给出的是本轮搜索建议，不声称覆盖所有 batch/FSDP 参数组合。

## 学习参数初筛

相同基座 revision 15852e8c16360a2fea060d615a32b45270f8a8fc、相同 seed=42、相同 799 条训练数据。双卡采样每轮处理800条（1条用于均分补齐），各候选一致。每组使用原定 3 epoch 的线性调度及 warmup=0.03，在第 1 epoch 验证并暂停。有效 batch 不同时更新次数不同，但已见样本量相同。

Liger 训练统一按全局有效 assistant token 归一化，验证也按有效监督 token 加权；表内为 NLL，越低越好。

| 学习率 | 有效 batch=8 | 有效 batch=16 |
|---|---:|---:|
| 5e-05 | 0.409747 | 0.423860 |
| 0.0001 | 0.400964 | 0.409352 |
| 0.00015 | 0.397816 | 0.403566 |

## 入围复核

按首轮 NLL 选前两组，从包含 optimizer/scheduler/RNG 的检查点恢复到第 3 epoch。按用户要求仅使用 seed=42；额外 seed43 已停止，其结果不参与结论。

| 配置 | seed | epoch 1 NLL | epoch 2 NLL | epoch 3 NLL |
|---|---:|---:|---:|---:|
| quality-eb8-lr0.00015 | 42 | 0.397816 | 0.389498 | 0.391602 |
| quality-eb8-lr0.0001 | 42 | 0.400964 | 0.392172 | 0.389319 |

## 建议配置

```json
{
  "status": "complete",
  "seed": 42,
  "base_model": "Qwen/Qwen3.5-2B",
  "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
  "learning_rate": 0.0001,
  "effective_batch": 8,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "epochs_planned": 3,
  "selected_checkpoint_epoch": 3.0,
  "warmup_ratio": 0.03,
  "stop_after_epoch": null,
  "lr_scheduler_type": "linear",
  "max_grad_norm": 1.0,
  "max_length": 24576,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "dtype": "bf16",
  "attention_implementation": "sdpa",
  "liger_kernel": true,
  "fsdp_version": 2,
  "activation_checkpointing": true,
  "loss_normalization": "global_assistant_tokens",
  "best_adapter": "outputs/sft-grid/quality-eb8-lr0.0001/checkpoint-300",
  "software": {
    "torch": "2.11.0",
    "transformers": "5.15.0.dev0",
    "accelerate": "1.14.0",
    "peft": "0.20.0",
    "swanlab": "0.9.1",
    "liger-kernel": "0.8.5",
    "flash-linear-attention": "0.6.0",
    "fla-core": "0.6.0"
  },
  "data_sha256": {
    "data/sft/train.jsonl": "8c3a6ff0033f6ea672af609891e747d60652ddc17e8d3c8eacb19e9d96dd9477",
    "data/sft/validation.jsonl": "9525cc2fb04a1d8d38ae2db959397da908dde3fea766f580fdcf77d1239533cc"
  }
}
```

复现训练命令（直接一阶段 Trainer）：

```bash
.venv/bin/python -m accelerate.commands.launch --config_file configs/fsdp2.yaml scripts/train_lora_sft.py --model Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc --train data/sft/train.jsonl --validation data/sft/validation.jsonl --output outputs/models/sft-lora --fsdp2 --dtype bf16 --attention-implementation sdpa --liger-kernel --profile --epochs 3 --max-length 24576 --per-device-train-batch-size 1 --gradient-accumulation-steps 4 --per-device-eval-batch-size 1 --learning-rate 0.0001 --warmup-ratio 0.03 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --seed 42 --logging-steps 1 --preprocessing-cache outputs/sft-grid/cache --swanlab --swanlab-project shopping-grpo-sft-grid --swanlab-run-name selected-final
```

## 边界与可复现性

- 本轮以 SFT validation loss/NLL 选择参数，未运行购物 rollout，不能据此声称严格购物成功率提升。
- 未执行模型合并、GRPO 或 Final-200 评测。保存的模型均为 LoRA adapter/训练检查点。
- LoRA 16/32/0.05、最大长度 24,576、梯度裁剪 1.0 和 warmup 0.03 固定，未穷举这些参数。
- 若选择第2轮检查点，复现时使用 --epochs 3 --stop-after-epoch 2 --profile，保留本轮已测的学习率曲线。
- 运行前修复 Qwen3.5 嵌套 text_config.use_cache 未关闭导致的 FSDP 重计算错误；Liger 路径启用全局监督 token 分母，避免微批切分改变样本权重。
- 线性注意力实际使用 fla.ops.gated_delta_rule.chunk；causal-conv1d 未安装，卷积部分保留 PyTorch 实现。
- 当前分支 experiment/sft-grid-seed42-only。操作记录见仓库 operation_log.md；原实验代码快照及 SHA256 见 source_snapshot/manifest.json，当前搜索器已移除额外 seed 执行。
- 每组 console.log、command.json、step_profile.jsonl、train_summary.json 均保留。后续组另保存逐样本 validation_epoch_*.json，可交叉校验分布式指标。
- 暂停/续训时 Trainer 的单值 train_loss、train_samples_per_second 会受到总计划预算或历史步数分母影响；本报告只用真实逐步记录计算吞吐，用完整验证集 NLL 比较学习效果。
- 相关 30 项回归测试通过；新代码 Ruff 检查通过（忽略既有训练脚本 EXE001 权限告警）。
