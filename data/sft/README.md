# 当前 SFT 任务选择

已经固定 **800 道训练题和 200 道验证题**：

- `train_tasks.jsonl`：训练题，simple/medium/hard 为 240/400/160。
- `validation_tasks.jsonl`：验证题，按自然分布为 30/121/49。

两份均覆盖 9 大品类，并保持品类自然比例。固定池和补题登记见 [`../splits/`](../splits/README.md)，训练池还有 3,700 道未入选题。

本次完成的是题目选择，未调用教师采集示范、未开始训练。目录中原来的 `train.jsonl`、`validation.jsonl` 和 `metadata.json` 仍是旧 Reward-v3 示范的原始记录，**不能用于本次训练**。SFT 启动检查会拒绝旧奖励版本或与新固定名单不一致的示范。

后续须分别按两份已固定任务名单采集并审计 Strict Reward 示范，将对应的训练和验证示范发布至 `train.jsonl` / `validation.jsonl`，同步 metadata。采集过程中不能把训练、验证题混在一起再次随机划分，也不能仅改旧 metadata 冒充新示范。详见 [数据方案](../../docs/data-plan.md)。
