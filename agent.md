# 项目约定

1. 本项目基于他人的 AT 卡项目。我们将在此基础上开展一系列实验和修改，最终形成自己的项目。
2. SwanLab 已登录；每次实验必须使用 SwanLab 记录配置、过程和结果。
3. 每次修改前，自动使用 `git branch` 创建工作分支，切换到该分支后再修改。
4. 数据工程的来源、决策、结果和划分方案统一维护在 [docs/data-plan.md](docs/data-plan.md)，不新增实验流水账。运行时及训练约定以 [AGENTS.md](AGENTS.md) 为准。

## 当前数据方案索引

- 启动 SFT / RL / 评估前必须先读 [数据方案](docs/data-plan.md) 和 [AGENTS.md](AGENTS.md)，优先于旧数据文件和旧课程文档的默认设置。
- 全量清洗、逐条分层、候选池、首批训练/验证/测试题目都已经完成并落盘：23,421 条中保留 16,741 条；清洗在 `data/cleaned/`，分层与划分总清单在 `data/splits/manifest.json`，登记表在 `data/splits/registry.jsonl`。启动训练前核验后直接使用，不能再次抽取首批题目。
- 初始 SFT 为 800 训练 / 200 验证，训练难度比例 3:5:2；RL 为 1,000 训练 / 50 验证，比例 2:5:3；测试从原始 `tag=eval` 的清洗通过池抽 200 条。
- 训练样本数量和难度比例是暂定参数，可根据训练与验证结果在固定的本阶段训练池内调整；清洗结果、分层定义及验证/测试隔离不能随训练调参改变。
- SFT 首批任务在 `data/sft/train_tasks.jsonl` / `validation_tasks.jsonl`；RL JSONL / Parquet 已同步；SFT 新示范尚未采集，不能使用旧对话冒充。各份题目均覆盖 9 个品类，品类按自然分布，验证和测试也按自然难度分布。
- 后续补题从 `data/splits/reserves/` 取，SFT / RL 训练尚余 3,700 / 9,494 题。通过 `scripts/prepare_data_splits.py extend` 分配独立批次并更新登记表，不重用已入选题，不动验证/测试。
