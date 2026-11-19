# Strict-Stratified-200 测试集

当前测试名单已从清洗通过且原始 `tag=eval` 的 **944 条候选池**重新分层抽取，名称为 **Strict-Stratified-200**。它替换旧 Final-200-Clean；旧名单中只有 8 条原本属于 eval，旧测试结果不能直接视作本测试集结果。

| 项目 | 当前值 |
|---|---|
| 固定名单 | `data/evaluation/tasks.jsonl` |
| 数量 | 200，均来自原始 eval 标签 |
| 环境 / 奖励 | ShopSimulator v2.1 / `shopsimulator-reward-strict-v1` |
| 分层 | 9 个商品大类 × simple / medium / hard，按 eval 池自然分布 |
| 难度数量 | 21 / 119 / 60 |
| 随机种子 | 42 |
| 样本及 metadata 校验和 | `data/evaluation/metadata.json`、`data/splits/manifest.json` |
| 正式评测状态 | 未执行 |

类别与难度两组边际按自然比例取整；联合配额使用 IPF 和带库存上限的整数分配。所有 9 大类均有题目。分布与每条标签可在 [划分 manifest](../data/splits/manifest.json) 和 [分层索引](../data/splits/README.md) 中查看。

整个测试候选池的 944 条题，以及与其指令重复的 3 条 train 任务，均禁止进入 SFT/RL 训练或阶段验证。未入选的 744 条继续保留测试用途。打包的 `blind_final_task_ids.json` 固定本次 200 题，`held_out_task_ids.json` 保护全部 947 条，`blind_guard.json` 固定当前契约和校验和。

训练规模或配比调整不能改变这份测试名单，不能用测试结果选择训练配比、补题或 checkpoint。将来补充训练题目只能从对应训练池的未入选题目取。不得混用旧模型、旧测试名单或旧 Reward-v3 报告作为本轮同条件对比。

这次仅完成测试题目选择和文件审计，没有运行正式 200 题购买轨迹评测。评测入口仍为 `bash scripts/evaluate.sh NAME`，须在用户明确要求评测时执行。完整来源及决策见 [数据方案](data-plan.md)。

官方 Qwen3.5-4B 的固定权重与双卡同条件评测命令见
[Qwen3.5-4B 评测准备](evaluation-qwen3.5-4b.md)。
