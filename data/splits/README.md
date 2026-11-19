# 固定来源、统一 SFT 采集与题目登记

16,741 条清洗通过任务的标签和阶段用途已经落盘，seed=42。最新 SFT 流程已完成：**统一采 1,000 条合格轨迹 → 第一阶段筛选保留 988 条 → 固定 889 条训练 / 99 条验证**（seed=42，验证:训练约 1:9）；第二阶段暂不应用。完整规则见 [数据方案](../../项目文档/规范与方案/数据清洗与任务池划分方案.md)。

| 当前用途 | 候选数量 | 当前目标 / 固定首批 | simple / medium / hard |
|---|---:|---:|---|
| SFT 统一采集 | 5,000 | 1,000 条合格轨迹 | 300 / 500 / 200 |
| RL 训练 | 10,494 | 1,000 题，尚余 9,494 | 200 / 500 / 300 |
| RL 验证 | 300 | 50 题 | 8 / 30 / 12 |
| 测试 | 944 | 200 题 | 21 / 119 / 60 |
| 跨来源重复封存 | 3 | 0 | 不参与抽样 |

SFT 统一队列为 [`../sft/candidate_tasks.jsonl`](../sft/candidate_tasks.jsonl)，合并原先 4,500/500 两个来源分块。旧 `pools/sft_train.jsonl`、`sft_validation.jsonl`、原 800/200 首批文件及 registry 的旧 pool 字段仅保留来源记录，**不代表当前或将来的 SFT 训练/验证归属**。

## 采集与分布

SFT 每题最多安排 3 次独立拒绝采样，首次成功立即停止该题，单次只按真实环境终局 Strict Reward=1 验收，同题最多保留一条。失败时在统一 SFT 池同品类 × 难度层补位，收满 1,000 条不同任务轨迹、所有配额满足后停止。第一阶段筛选及 train/validation 划分已执行，正式名单见 `manifest.sft_final`，原始来源分块保持原样。

SFT 整体难度为 3:5:2，商品大类按统一 5,000 题自然比例。当前完整联合配额在 [`../sft/collection_plan.json`](../sft/collection_plan.json)。RL 和测试仍使用固定名单和各自分布，均覆盖 9 大类。

| 商品大类 | SFT 统一接受目标 | RL 训 | RL 验 | 测试 |
|---|---:|---:|---:|---:|
| 休闲娱乐文教 | 146 | 145 | 7 | 27 |
| 家居家装 | 170 | 170 | 8 | 36 |
| 家用电器数码 | 112 | 112 | 6 | 25 |
| 服饰鞋包饰品 | 149 | 149 | 8 | 25 |
| 母婴儿童 | 47 | 48 | 2 | 10 |
| 生产材料农用品 | 117 | 117 | 6 | 29 |
| 美妆个护健康 | 135 | 135 | 7 | 27 |
| 运动户外交通 | 74 | 74 | 4 | 10 |
| 食品饮品 | 50 | 50 | 2 | 11 |

## 记录与核验

- `labels.jsonl`：固定标签、原始来源、分数和重复指令组。
- `pools/` / `initial` / `reserves/`：原生成器的来源分块与分配记录。RL / 测试记录仍为当前名单；SFT 两个旧分块只作来源追踪，使用统一 candidate 文件采集。
- `registry.jsonl`：生成时的题目登记，`selected` 不表示采集成功、已训练或完成未来 SFT 划分。实际采集须另记每题尝试、接受、耗尽状态和统一成功名单。
- `manifest.json`：来源校验和、固定 RL / 测试以及最新 SFT 统一采集计划。`sft_demonstrations.split_status=finalized_after_stage1_v1`，`sft_final` 记录当前 889/99 正式名单和哈希。
- `rounds/`：实际追加 RL 题目后才产生，不能重复占用已有题目。

```bash
# 核验统一 SFT 来源、目标配额及固定 RL / 测试，不重新分层。
PYTHONPATH=src .venv/bin/python scripts/prepare_data_splits.py verify

# 预览再补 1,000 道 RL 新题，不修改登记。
PYTHONPATH=src .venv/bin/python scripts/prepare_data_splits.py extend --stage rl --count 1000 --dry-run
```

明确要求补 RL 题时去掉 `--dry-run`，可通过 `--ratios A B C` 指定难度权重。旧 `--stage sft` 按训练分块补题的行为在统一采集模式下禁用。任何补题都不能借用测试或封存题，也不自动启动训练。
