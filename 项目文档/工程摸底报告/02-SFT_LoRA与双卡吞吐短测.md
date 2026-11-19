# 一阶段 SFT：LoRA 与双卡吞吐短测

> 记录日期：2026-11-14 ｜ 原始实验目录：`outputs/sft-lora-throughput`

本轮仅 seed=42；SFT 训练/验证/Final-200 task_id 两两不重叠。原有 adapter、训练摘要和数据 SHA256 核对未改变。

## 小网格（相同 20 更新）

固定全局 batch8（双卡×每卡1×累积4）、LR1e-4、原定3epoch/300更新线性调度、warmup0.03、LoRA dropout0.05、Liger、BF16和24576长度。每组看到相同160样本、967827输入tokens及54188监督tokens，验证200条/66267监督tokens。没有重搜seed、学习率或batch。

| r | alpha=r 的验证 NLL | alpha=2r 的验证 NLL |
|---:|---:|---:|
| 8 | 0.523946 | 0.489052 |
| 16 | 0.494093 | 0.468044 |
| 32 | 0.472355 | 0.459243 |

## 同一资源设置下的第1epoch复核

| LoRA | 验证 NLL（token 加权） | eval_loss（样本均值） | 稳态 tokens/s | 单卡 allocated 峰值 |
|---|---:|---:|---:|---:|
| r=16, alpha=32 | 0.400862 | 0.360068 | 21935 | 60.95 GiB |
| r=32, alpha=64 | 0.395891 | 0.355215 | 21947 | 61.20 GiB |

r32/alpha64 相对 r16/alpha32 的配对 token 加权 NLL 差为 -0.004971；任务级配对 bootstrap 95% CI [-0.006051, -0.003931]。这是当前验证集和seed42的区间，不包含训练种子变化的不确定性。两组100更新中loss/梯度均有限；372个LoRA tensor保存名称规范。

## 显存和速度

- 资源配置：12/24文本层保留重计算并保存Dropout RNG；FSDP reshard_after_forward=false；Liger保持开启；expandable_segments减少变长序列分配碎片。
- 同长度/同step样本的历史一轮基线：18137 tokens/s；新套餐 21947 tokens/s，吞吐 +21.0%。逐更新训练时间合计 267.3→220.6 秒（不含模型加载/验证/保存/上传）。
- 新套餐完整一轮 allocated峰值 61.20 GiB，reserved峰值 63.38 GiB，1秒间隔设备占用采样最大 64.24 GiB；单卡容量约95.6GiB。另通过包含最长24359-token样本的3更新压力测试。设备采样不是连续时间精确峰值。
- 完全关闭重计算的最长样本测试超过78GiB保护线，已停止，不推荐。普通allocator的整轮复核因碎片占用主动中止并保留记录；expandable_segments版完成整轮。

## 续训检查

- 短测第1轮保留了完整 optimizer/scheduler/双rank RNG，trainer_state显示step100/300、epoch1/3。
- 排查并修复当前Accelerate在FSDP2 adapter-only FULL_STATE_DICT恢复时绕过完整state广播的问题：原路径恢复后一步验证NLL异常升至0.4262，标为invalid_resume，不纳入选择。
- 仓库Trainer现从已gather的adapter_model.safetensors通过PyTorch分布式state loader给两rank恢复完整LoRA；optimizer/scheduler/RNG仍由Trainer恢复。修复后恢复到step101的验证NLL为0.395221，与checkpoint第1轮0.395891连续。原始失败及修复后日志均保留。

## 推荐启动方式

从基座重训：

```bash
bash scripts/sft.sh --balanced
```

复用本次已验证的第1轮检查点，继续剩下两轮（仍使用原3轮学习率调度）：

```bash
bash scripts/sft.sh --balanced --resume-from-checkpoint outputs/sft-lora-throughput/confirm-r32-a64-partial2-expandable/checkpoint-100
```

默认输出 outputs/models/sft-lora-balanced；SwanLab shopping-grpo-sft / sft-balanced-seed42。评估推理服务现保持停止，双卡已释放，环境服务仍运行。

## 适用边界

- 本轮选型依据为SFT验证NLL、有限loss/梯度、吞吐及显存压力检查；6组初筛并不等于6组完整收敛比较。
- 两入围配置只完成第1轮，未自动执行完整3轮训练或模型合并；未运行新的Final-200，不能声称69.5%购物成功率已经提高。
- 预期完整训练比原配置省时，具体三轮总耗时需实际续训确认。LoRA rank不用于强行填显存。
- 26项相关测试通过；Ruff、shell语法及diff空白检查通过。原始命令、各run配置、曲线、SwanLab本地记录、设备采样及source SHA均留在本目录。
