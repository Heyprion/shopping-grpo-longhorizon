# 当前 SFT 第一版正式数据

当前已完成 1,000 条教师轨迹采集，仅应用第一阶段长度与合法性筛选，排除 12 条后正式保留 **988 条 SFT 轨迹**。第二阶段过程质量筛选暂不执行。固定 seed=42，按验证:训练约 1:9 划分为 **99 条验证、889 条训练**，名单已冻结。

## 来源与分布

统一来源是 [`candidate_tasks.jsonl`](candidate_tasks.jsonl)，共 **5,000 道 SFT 候选题**。它合并原来的 4,500 / 500 两个 SFT 来源分块；原分块名称和旧 `train_tasks.jsonl` / `validation_tasks.jsonl` 只保留来源记录，**不再表示当前训练或验证归属**。

| 本阶段目标 | 数量 |
|---|---:|
| 不同任务的合格轨迹总数 | **1,000** |
| simple | **300** |
| medium | **500** |
| hard | **200** |
| 正式 SFT / 训练 / 验证 | **988 / 889 / 99** |

统一采集仍按 SFT 难度 **3:5:2**，品类遵循这 5,000 题的自然比例、覆盖 9 大类。联合配额已保存在 [`collection_plan.json`](collection_plan.json)，按最终接受轨迹核对，不仅核对尝试分布。不使用 RL、测试或封存任务。

## 拒绝采样与停止

1. 每个候选任务最多安排 **3 次独立拒绝采样**，同题串行尝试，首次成功立即接受并停止该题，每次从该任务初始状态运行完整轨迹。
2. 每次只按真实环境终局 **Strict Reward == 1** 验收，不额外筛选长度、步数、guard、重复搜索、教师解释、gold ASIN 或 `gold_purchase` 标签。
3. 同题首次成功立即保留并停止后续尝试；已有多条成功记录时只留 attempt 序号最早的一条。三次都失败时，从统一 SFT 池的同品类 × 难度层补下一题，不受旧 4,500/500 分块限制。
4. 实际接受 **1,000 条不同任务轨迹且所有联合配额满足后立即停止**；失败、同题额外成功和并发返回的超额结果不计入目标。达到目标后不再启动剩余尝试。
5. 某层池子耗尽时记录缺额，不跨入 RL / 测试、不降低奖励门槛、不复制同题凑数。

## 记录和后续处理

采集时记录 task ID、来源、品类、难度、attempt、终局 reward、接受 / 三次耗尽 / 未尝试状态和替补关系。输出一份**未划分的 1,000 条轨迹集合**及对应成功任务名单，支持断点续采。当前任务候选队列不等于最终成功名单，也不等于未来训练 / 验证名单。

采集时仍只按 reward=1 接受；完成采集后，已按用户决定仅执行第一阶段筛选，未应用第二阶段。正式集合 988 条，未补采；训练/验证按 889/99 固定。相同任务和重复指令组不跨集合。

## 当前状态

统一采集入口已接入本计划。此前采用单次输出上限 2,048 tokens 的采集已暂停，记录保留在 `outputs/sft-unified-collection/`；该批次没有保存 API 截断原因，需另行审计。当前 64K 批次已完成 1,000 道不同任务的成功轨迹采集，难度 300/500/200、联合配额全部满足。第一阶段筛选和正式 889/99 划分均已执行，训练入口已核验新名单；**尚未启动训练、合并或 200 题测试**。

当前 `train.jsonl`、`validation.jsonl`、`metadata.json` 已替换为本次正式数据，旧版本可在 `outputs/sft-stage1-v1/previous/` 找到。旧 `train_tasks.jsonl` / `validation_tasks.jsonl` 保留来源；本次正式任务名单为 `final_train_tasks.jsonl` / `final_validation_tasks.jsonl`。RL 和测试名单保持原样。详细说明见 [数据方案](../../项目文档/规范与方案/数据清洗与任务池划分方案.md)。

## 统一采集入口

正式采集由用户在终端启动：`bash scripts/start_sft_collection.sh`。脚本交互式读取 DeepSeek API key（也可预先在环境中设置 `DEEPSEEK_API_KEY`），启动带规范预算文件的独立环境服务，并在线记录 SwanLab。默认并发 64，可通过环境变量 `SFT_COLLECTION_WORKERS` 调整；这是同时运行的轨迹数，不能超过服务的环境槽数及 DeepSeek 账户可用并发。进程在 cgroup 内存占用达到 90% 时停止提交新任务，待已有任务结束后暂停。

教师为 `deepseek-flash`，开启 `high` 思考，单次 API 输出上限为 **65,536 tokens（包含思考和输出）**，请求等待上限为 600 秒。逐次响应的 `finish_reason`、token 用量及截断标志保存于轨迹的 `model_completions`，SwanLab 同步记录用量。若返回 `finish_reason=length`，不执行不完整的工具调用；记录基础设施错误并暂停，不将该次截断记为普通失败或消耗三次拒绝采样名额。终局奖励验收仍只看实际 Strict Reward=1。

进度中 `reward` 是刚完成尝试的真实终局奖励，`successful_attempts` 是奖励为 1 的尝试次数，`successful_tasks` 是至少成功一次的不同任务数；`accepted` 在同题首次成功后立即增加，不等待三次全部结束。同题最多三次且串行执行，只有三次均失败才计入 `exhausted` 并同层补题。

新采集状态和完整尝试保存在 `outputs/sft-unified-collection-64k/collection.sqlite3`，每次尝试完成后立即提交；接受轨迹和成功名单每 100 次尝试及退出时分别导出为 `accepted.jsonl`、`successful_tasks.jsonl`；`task_status.jsonl` 和 `summary.json` 记录进度及缺额。中断后重新运行同一命令续采；错误及其完整轨迹记录位于 `infrastructure_errors.jsonl`。配置不符时拒绝在同一目录继续，旧 2,048-token 批次不自动并入。最终导出不划分训练/验证。独立并发试验和三次采样验证保存在 `outputs/sft-collection-probe/` 及 `outputs/sft-collection-probe-dual/`，不并入正式采集数据。

首次成功停止规则已写入 collection plan。对于本次已有的 64K 状态，仅允许在模型、输出上限、预算、候选池等配置完全相同的条件下转换此采样规则；转换前备份为 `collection.before-first-success.sqlite3`，旧配置保留在数据库 metadata，所有原始尝试保持原样。已有同题额外成功仅作审计，导出仍取最早的一条。

## 轨迹质量审计

当前审计按 Student 实际训练格式计算完整序列长度：复用 `build_sft_row` 去除教师私有思考与守卫拒绝消息，再使用固定 revision 的 Qwen3.5-2B processor/chat template，阈值为 24,576 tokens。1,000 条中 11 条超长，2 条存在实际规格动作与结构化页面选项的空格不一致，其中 1 条也超长；第一阶段建议排除并集为 12 条，预览剩余 988 条（simple/medium/hard=300/496/192）。该第一阶段规则现已应用，正式保留 988 条并完成 889/99 划分。

原始 344 条含守卫拒绝，99 条含重复同状态动作的诊断信号，均不能直接作为整条删除标准：守卫错误会在格式化时移除，重复动作多为重开商品后恢复规格。第二阶段建议检查动作是否带来新信息或必要状态变化、购买依据是否可见，并保留有用的搜索、比较和翻页技能。详见[数据方案](../../项目文档/规范与方案/数据清洗与任务池划分方案.md)与[审计报告](../../项目文档/专题诊断报告/01-SFT教师轨迹质量审计报告.md)。

## 正式数据与训练

- `trajectories.jsonl`：格式化后的 988 条正式 SFT 示范，已去掉教师私有思考、守卫拒绝消息及终局奖励详情。
- `train.jsonl` / `validation.jsonl`：889 / 99 条正式对话；`final_train_tasks.jsonl` / `final_validation_tasks.jsonl` 为固定任务名单。
- `quality_report.json`：第一阶段规则、12 条排除原因及保留轨迹长度；`rejected_trajectories.jsonl` 为排除索引，原始教师记录不改动。
- `metadata.json`：版本、来源、正式名单和文件校验和。

训练/验证难度分别为 simple/medium/hard=270/446/173 与 30/50/19。划分同时保持品类自然边际及重复指令组隔离。第二阶段诊断标记不影响本版成员。

执行 `bash scripts/sft.sh --balanced --evaluate-after`，自动训练三轮、每轮验证、加载最低验证加权 NLL 的模型、合并并运行固定 200 题测试。详细路径与恢复方式见[训练说明](../../项目文档/规范与方案/SFT第一版训练指南.md)。
