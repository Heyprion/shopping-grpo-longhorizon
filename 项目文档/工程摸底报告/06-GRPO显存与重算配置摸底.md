# GRPO 双卡显存与重算配置摸底汇总

> 记录日期：2026-11-14 ｜ 原始实验目录：`outputs/grpo-quick-memory/`、`outputs/grpo-high-memory/`、`outputs/grpo-micro2-no-recompute/`

## 一、Group4 快速资源测试（`outputs/grpo-quick-memory`）

建议每卡microbatch4、累积1，保留20/24层梯度重算（第5/11/17/23层不重算，索引从0开始）。
group4、每次2个任务、有效batch8、seed42、LR1e-6、LoRA16/32、最大24576均不变。
正常训练通过SHOPPING_GRPO_CHECKPOINTED_LAYERS=20启用此重算预算，默认不设置时仍全层重算。
正式命令见recommended_command.txt；本次没有启动完整RL训练。

## 同一最长压力输入的热身后计算

下列是同一新SFT、同一FSDP2融合输出层的actor计算压力输入：每卡4条、每条24576，
全局8条，固定重复prompt token，SFT log-prob loss，仅做正反向、没有优化器更新。
它不是实际在线PPO训练速度或学习效果评测；各设置先热身一次，再计时一次。

| 设置 | 微批 | 重算层 | 正反向秒 | 张量峰值GiB |
|---|---:|---:|---:|---:|
| 原设置 | 2 | 24 | 10.967 | 21.01 |
| 仅增加微批 | 4 | 24 | 10.081 | 32.16 |
| 推荐资源设置 | 4 | 20 | 9.746 | 64.13 |

计算阶段缩短11.1%，不等于整轮GRPO提速这个比例。
同时驻留rollout运行时的独立24576压力检查中，推荐配置张量峰值63.41GiB，
allocator reserved约66.32GiB，设备一秒采样最高72.44GiB。
短轨迹实际占用会更低，未增加KV cache来填显存。

18/24层重算：设备采样85.52GiB，触发人为82GiB保护而中断，未发生CUDA OOM。
这不能证明显存不足或该档不可用；应归类为尚未完成验证，本次暂不推荐未经完整验证的档位。
真实PPO固定batch测试已完成micro2/micro4冷计时及梯度比较；micro4首次含形状编译，
29–32秒不能当作稳态速度。复核受600秒原限时停止，未取得真实PPO的热身后partial计时。
因此最后直接加载actor省去Ray/vLLM启动，用上述固定最长压力计算完成热身速度检查。
两次服务探针分别以显存保护和时间上限结束；完整原始状态保存在对应execution.json，不算完整训练。

本次所有优化器step均禁用/未调用、scheduler未推进，未保存新模型，没有验证或Final-200。
实际PPO同批micro2→4的梯度范数接近，但相对梯度L2差约4.5–5.6%，不能宣称逐bit等价或质量已验证。
20/24与全层重算使用相同HF非重入checkpoint机制，保留RNG；新增dropout输出/梯度一致性及
既有兼容/融合输出检查共5项通过，Ruff通过。全局batch、学习率和奖励目标保持不变。

正常训练配置已准备为相同有效batch8、group4。重算开关仅opt-in，不改变默认模型参数键名。
SwanLab日志与完整配置见swanlab_summary.json；源码、测试和脚本SHA另存source_sha256.json。

## 二、Group4 高显存快速测试（`outputs/grpo-high-memory`）

本轮按用户确认将88–90GiB视为正常候选范围，短测硬保护92GiB，未再使用82GiB否决档位。
新SFT、seed42、group4、每批2个任务、global8、micro4/累积1、LR1e-6、LoRA16/32保持。

## 最长上下文压力

直接加载同一veRL FSDP2 actor，不启动环境/rollout。全局8×24576固定重复prompt token，
SFT log-prob loss正反向，与实际PPO任务不同；只用于同形状计算/显存比较。
optimizer.step实际执行以创建Adam状态，但lr0，scheduler不推进；两rank逐LoRA参数torch.equal校验权重未变。
每档先热身一次再计时一次，已创建568项optimizer状态。

| 重算层数 | 热身后秒 | 张量峰值GiB | 设备采样峰值GiB |
|---|---:|---:|---:|
| 20/24 | 9.687 | 64.22 | 73.85 |
| 18/24 | 9.502 | 79.59 | 89.76 |

18层档完整通过，没有OOM或保护中断。相对20层仅额外缩短1.9%计算时间；
相对上轮micro2/full压力基线约缩短13.4%，非整轮RL加速承诺。
首个运行仅因诊断输出字段类型错误退出，保留failed-output-schema日志；修正后完整压力复测
69.2秒完成。这不是模型数值或显存失败。

## 一轮真实GRPO

通过标准train_grpo.py启用18层重算，真实完成1次更新、权重同步及模式切换；
总墙钟219.8秒，设备采样峰值77.88GiB。
真实该batch actor更新10.016秒，权重同步
5.235秒，grad_norm=0.27450。
使用原500步/15步warmup调度，仅pilot停止于1步，显式skip_validation；0验证，不执行Final-200。
第一步处于warmup，不能证明学习质量。真实批次和最长压力输入分别测量，不将两者峰值相加。

建议18层作为当前快速资源候选；20层仍是低占用备选，额外约16GiB只换来约2%的计算提速。
长样本峰值不代表每个短轨迹都会用满这些显存。没有继续降低重算层数或扩大有效batch。
完整RL未启动、模型未合并；正常启动命令只做dry-run，见recommended_command.txt。
5项pilot/重算测试与Ruff通过。SwanLab汇总见swanlab_summary.json。

## 三、Microbatch=2 完全关闭重算测试（`outputs/grpo-micro2-no-recompute`）

group4、global8、seed42、新SFT、FSDP2/融合输出层/常驻参数固定。每卡4条固定token序列，
用SFT log-prob loss做正反向压力检查，比较micro4/cp18与micro2/cp0；两者累积分别1/2，
总轨迹数均8。不是实际在线PPO吞吐或学习质量评测。

只加载一次actor，运行墙钟52.9秒；每档先热身一次再计时。
optimizer.step用于包含Adam状态，但lr0、不推进scheduler，不保存任何模型文件。

| token长度 | 配置 | 热身后秒 | 张量峰值GiB | 设备采样峰值GiB |
|---|---|---:|---:|---:|
| 8192 | current_micro4_cp18 | 3.122 | 32.79 | 40.33 |
| 8192 | micro2_no_recompute | 2.740 | 54.74 | 61.87 |
| 16384 | current_micro4_cp18 | 6.203 | 56.18 | 64.62 |

8192长度下，micro2完全关闭重算比当前micro4/cp18耗时缩短12.3%。
16384长度的micro2/cp0尚未完成即触发92GiB硬保护，实际采样最高94.53GiB，
未观察到CUDA OOM；这是保护中断，不能写成已发生OOM，也没有取得此档完整热身后计时。
24576长度未执行，不能把8192长度可行推广为整个24576上下文预算可行。

结论：关闭重算在短输入上能加速，但当前不能全局关闭。保留micro4/cp18为已通过最长压力
检查的配置；未实现自动按长度切换，未追加其它超参搜索。

没有启动Ray/vLLM、真实GRPO训练、验证、Final-200或模型合并。原始设备采样、逐rank记录、
保护退出与完整控制台日志全部保留，SwanLab汇总见swanlab_summary.json。
