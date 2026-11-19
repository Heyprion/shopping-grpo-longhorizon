# 双卡 GRPO 参数短测

> 记录日期：2026-11-14 ｜ 原始实验目录：`outputs/grpo-grid`

本轮只做用户授权的约一小时短测。起点固定为新 SFT 的验证最佳合并模型
`outputs/models/sft-balanced-best-merged`，所有实验 seed42。
没有启动完整500步训练、没有导出或合并RL模型、没有运行最终200题测试。

## 当前建议

资源配置采用融合 Triton 输出层、actor 参数和 optimizer 常驻 GPU、
`reshard_after_forward=false`、每卡 microbatch2。保留完整梯度检查点和
`use_remove_padding=false`；最大上下文24576、最大response20480不变。
使用现有标准入口和 `recommended_command.txt` 中的完整命令；该命令尚未执行。
`recommended_config.json` 保存全部参数。

学习配置暂保留 LR1e-6、LoRA r16/alpha32、温度0.7/top_p0.9、每prompt4条轨迹、
每次更新2个prompt/8条轨迹、clip0.2、entropy_coeff0、KL项关闭。
microbatch1→2只改变本地分批，不改变有效batch。RL的LoRA是新建在已合并SFT之上的
适配器；SFT原有r32/alpha64已包含在起始权重中。

## 实测资源和时间

默认非融合精确熵路径在第一次actor更新时就触发82GiB设备保护阈值，
两卡2秒采样最高88.92/94.69GiB，0次完成更新，不能用作学习效果基线。
可运行的资源基线只关闭熵诊断，保留entropy_coeff0及其余学习设置。

| 指标 | 关闭熵诊断的资源基线 | 融合头、常驻GPU、micro2，LR1e-6 |
|---|---:|---:|
| 已完成更新 | 3 | 18 |
| actor更新耗时中位数 | 18.92s | 6.65s |
| actor输入token吞吐 | 3920/s | 8661/s |
| 张量allocated峰值 | 39.85GiB | 20.77GiB |
| allocator reserved峰值 | 50.99GiB | 28.04GiB |
| 设备采样最高占用 | 55.59GiB | 53.23GiB |
| 含全部补采和权重同步、扣除验证的更新间隔均值 | 40.43s | 32.92s |

最后一行来自SwanLab本地actor/grad_norm时间戳之间的真实墙钟，扣掉该区间验证耗时，
不包含初始化和第一个更新；样本量只有基线2个间隔和候选17个间隔，不能当作严格配对测速。
`timing_s/step`不能完整计入所有被丢弃的补采轮次，不能单独作为真实整轮耗时。
独立报告每个命令的总墙钟，包含冷启动/预热、验证以及停止时间。

融合头仍计算精确全词表熵，避免构建完整sequence×248320词表logits。
独立BF16数值核对：log-prob最大绝对差8.58e-6、熵7.63e-6、梯度相对L2差0.00986。
异长序列布局与padding梯度测试通过；真实更新中各项标量有限。
短测遇到约21293 token输入上下文，但没有单独完成完整24576长度的训练压力测试。
设备占用包括vLLM缓存，不能与actor allocated或GPU算力利用率混为一谈。

## 验证与试验状态

所有验证为独立50题的贪心单轨迹，严格要求终局gold_purchase、reward_valid=true、
purchase_success=1、Reward v3且无基础设施或不可验证标记。
50题与GRPO1000题训练、SFT训练/验证和Final-200 IDs两两不重叠。
一题等于2个百分点，短测不能证明长期优劣或保证最终效果不降。

第二学习率尚无共同step10验证，不能比较效果。

| 试验 | 退出状态 | 已记录更新 | 命令墙钟秒 | 设备最高GiB | 严格成功 |
|---|---|---:|---:|---:|---|
| baseline | failed | 0 | 69.8 | 14.14 | 无严格验证记录 |
| baseline-device-map | failed | 0 | 207.9 | 43.01 | 无严格验证记录 |
| baseline-ready | device_memory_limit_82_gib | 0 | 221.0 | 94.69 | 无严格验证记录 |
| baseline-v50 | failed | 0 | 201.3 | 45.79 | 无严格验证记录 |
| fused-lr1e-6 | failed | 0 | 4.3 | 0.00 | 无严格验证记录 |
| fused-lr1e6 | time_limit | 18 | 850.0 | 53.23 | step0: 29/50, step10: 28/50 |
| fused-lr3e6 | time_limit | 9 | 615.1 | 53.09 | 无严格验证记录 |
| profile-no-entropy | complete | 3 | 335.9 | 55.59 | 无严格验证记录 |

`time_limit`是预设短测时间上限，不等同于数值发散或CUDA OOM；以已经完成并保存的
验证点作为质量证据。0更新的启动失败和取消记录不纳入效果对比。
LR3e-6在第10步验证过程中到达本轮总上限，验证未完成；第10次actor更新后尚未写入
optimizer_step整步诊断，因此表中9次为已经保存的更新记录下界，不能当作学习失败。
两学习率均保留500步总调度/15步warmup，pilot只改变fit停止条件。
step10仍处在warmup，证据只覆盖早期学习；大部分长期LR行为尚未覆盖。
异步在线采样并非两组逐token一致，验证也只有单seed/单次，不能宣称统计显著。

本轮没有对rank/alpha、rollout数、温度、clip或KL做独立网格；这些值沿用原设置。
不把增加rank或KV缓存作为占显存手段。当前主要耗时来自相同奖励组的补采，
后续若要扩大搜索，温度与有效组比例比继续增加缓存更值得优先检查。

## 运行修复和复现

- 修复Ray提前初始化CUDA后双rank落同一卡的问题；由veRL现有Worker按Ray分配ID选择设备。
- 8个agent worker不是轨迹并发上限；50题验证补齐56条。环境池已扩至64槽并验证同时租用56槽。
- 验证batch2会为每组补齐8条，50题产生200条轨迹；改为batch50只产生56条，评分仍只有50题。
- 固定FlashInfer采样器关闭，保持现有注意力后端；使用已有SwanLab登录，密钥不写命令或日志。
- 修复当前固定veRL版本融合输出层在不去padding时的逐序列布局，并增加严格验证指标；奖励不变。
- 19项相关单测、Ruff及diff空白检查通过。初次直接pytest入口因导入路径失败，改为python -m pytest后通过。

原始命令、退出状态、设备采样、逐步指标和逐题验证保存在各试验目录；
`results.json`为机器可读汇总，`final_source_sha256.json`记录最终代码/数据摘要。
所有真正进入训练fit的试验记录在SwanLab shopping-grpo项目；启动失败在汇总run补录。
两卡95.6GiB，保持安全余量；本轮内存和质量证据不能代替完整训练的持续监控。
