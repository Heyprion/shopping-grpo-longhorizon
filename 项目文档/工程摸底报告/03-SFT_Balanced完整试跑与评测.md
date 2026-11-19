# Balanced SFT 完整训练与 Final-200 评估

> 记录日期：2026-11-14 ｜ 原始实验目录：`outputs/runs/sft-balanced-seed42`

在当前AutoDL训练机的两张RTX PRO6000 Blackwell上完成正式一阶段SFT，seed42；800条源训练数据中799条符合24576长度限制，全部参与训练，无额外子集。训练/验证/最终测试task IDs不重叠。

## 实际训练及选模

- LoRA r32/alpha64/dropout0.05；双卡FSDP2、BF16、Liger、FLA；每卡batch1/累积4/有效batch8；LR1e-4、warmup0.03、线性调度、梯度裁剪1；隔层重计算、保留参数、expandable allocator。
- 接续此前完整第1轮的100步检查点，完成到300步/3轮；保留optimizer/scheduler/RNG，全程loss/梯度有限。每步输入/监督token计数与旧配方300步全部匹配。
- 三轮验证token加权NLL依次0.395891 / 0.386052 / 0.401951。第三轮回升，在任何测试轨迹产生前按验证NLL最低选择第2轮checkpoint-200；完整3轮训练、最后一轮模型也保留。

| 训练时间口径 | 实测 |
|---|---:|
| 本次续训端到端墙钟（后两轮） | 8分37.3秒 |
| 第1轮试训墙钟 | 4分23.3秒 |
| 三轮累计墙钟 | 13分0.6秒 |
| 三轮累计Trainer运行时间 | 12分10.5秒 |
| 300次训练更新计时之和 | 11分2.0秒 |

累计墙钟将两次训练会话相加，含各次启动、验证、保存和上传，排除中途等待用户及此前其它网格候选耗时。Trainer运行时间包含训练循环/验证/保存，排除前置模型加载等；与原配方同口径876.9015秒相比减少约16.7%。续训Trainer单值train_loss的历史步数分母不适合直接跨运行比较，选模使用完整验证集NLL。

单卡张量allocated峰值61.20GiB，reserved峰值63.38GiB；双卡训练已退出。

## 测试结果

| 指标 | 原r16配方 | 新r32配方 |
|---|---:|---:|
| 严格成功 | 139/200 | 136/200 |
| 严格成功率 | 69.5% | 68.0% |
| 环境终局数 | 198 | 198 |
| 平均步数 | 10.11 | 10.23 |
| 200题任务循环时间 | 8分41.6秒 | 8分47.5秒 |
| 平均每题 | 2.608秒 | 2.638秒 |

严格成功率变化-1.5个百分点。严格成功必须为完整gold_purchase且reward_valid=true；所有错误计入200题分母。错误类型：{'ContextBudgetError': 2}。这是固定200题的一次配方比较，不据此声称统计显著或不同训练种子均有改善。

新配方每题耗时中位数1.945秒、P95 7.909秒、最大12.806秒。时间从SwanLab逐题累计elapsed_seconds差分得到，包含模型、分词和环境交互。

评估命令总墙钟8分50.5秒，另包含前置就绪检查、SwanLab初始化/最终上传和HTML报告构建。所选模型合并8.68秒、服务启动/预热2分12.1秒，不计入任务循环时间。选模前最终第3轮模型的准备记录另留final_epoch3_pipeline_execution.json，未执行其测试任务。

## 产物

- 正式训练：outputs/models/sft-lora-balanced；测试所用adapter：其中checkpoint-200。
- 测试所用独立模型：outputs/models/sft-balanced-best-merged。
- 测试轨迹、summary.json、report.html、逐题耗时：outputs/evaluation/sft-balanced-seed42。
- 配置、模型来源、验证曲线与选模先后：model_selection.json、run_manifest.json及pipeline_execution.json。
- 数据、原模型和旧评估摘要SHA核对未变；原模型结果保留。
