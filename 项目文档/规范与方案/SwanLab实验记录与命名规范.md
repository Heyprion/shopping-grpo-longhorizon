# SwanLab 实验记录与命名规范

为避免多项目分散、训练与推理评测混杂、以及曲线无法从名称分辨用途，本项目（2026 年 11 月起）统一采用**单 Project + 阶段分组（`group`）+ 任务类型分离（`job_type`）+ 规范命名与中文说明**的记录方式。

## 1. 唯一项目（Project）

- 统一项目名：**`shopping-grpo-longhorizon`**
- 所有阶段（数据清洗/采集、基线评测、SFT、GRPO、模型评测、工程摸底）全部写入此唯一项目，不再拆分 `shopping-grpo-sft`、`shopping-grpo-evaluation`、`shopping-grpo-sft-grid` 等子项目。
- 公共配置常量与辅助函数统一维护在 [`src/shopping_grpo/tracking.py`](../../src/shopping_grpo/tracking.py)。

## 2. 阶段分组（`group`）与类型分离（`job_type`）

在 SwanLab 页面中，使用 `group` 按实验阶段聚合，使用 `job_type` 明确区分**训练**与**推理/评测**（以及数据与性能诊断）：

| 阶段分组（`group`） | 典型 `job_type` | 覆盖内容 |
| :--- | :---: | :--- |
| `01-数据清洗与采集` | `data` | 全量任务清洗、教师轨迹拒绝采样采集、轨迹质量审计、正式 SFT 划分与校验 |
| `02-基线评测` | `eval` | Base 模型（如 Qwen3.5-2B / 4B）固定 200 题同条件评测 |
| `03-SFT训练与调优` | `train` / `probe` | SFT 双卡 FSDP2/LoRA 吞吐摸底与正式 SFT 训练 |
| `04-SFT评测` | `eval` | SFT 合并模型的固定 200 题评测 |
| `05-GRPO工程摸底与排障` | `probe` / `train` / `eval` | GRPO 双卡显存/算子摸底、SFT 采样诊断、权重同步验证、50 步瓶颈消融与动量反事实对照 |
| `06-GRPO正式训练` | `train` | GRPO 正式训练与续训（配置 C、Step 0→100、100→200、200→300、group8 / GiGPO 探索等） |
| `07-GRPO评测` | `eval` | GRPO 导出合并检查点的固定 200 题评测 |

### `job_type` 固定取值

- `train`：模型参数更新训练（SFT 训练、GRPO 训练）
- `eval`：模型推理与评测（固定 200 题正式评测、固定训练题抽检）
- `data`：数据清洗、教师轨迹采集、审计与划分
- `probe`：显存/吞吐网格摸底、采样诊断、权重同步与更新方向探针

## 3. 实验命名规则（`name` / `experiment_name`）

统一格式：`YYYY-MM-DD-阶段-简明目的`

- **日期**：实际启动日期（`YYYY-MM-DD`）
- **阶段标识**：`DATA`（清洗）、`COLLECT`（采集/筛选）、`BASE`（基线评测）、`SFT`（SFT 训练）、`RL`（GRPO 训练）、`EVAL`（正式/专项评测）、`PROBE`（性能与诊断摸底）
- **目的**：简短直白的中文短语，一眼能看出实验内容；同日同阶段同目的的独立重跑可追加 `-02`、`-03`
- **示例**：
  - `2026-11-15-SFT-第一版正式训练`
  - `2026-11-16-EVAL-SFT第一版200题评测`
  - `2026-11-18-RL-GRPO前100步LR3e-5无动量`
  - `2026-11-18-EVAL-GRPO-Step200-200题评测`

## 4. 实验简短描述（`description`）

每次启动实验必须填写 `description`（1–2 句中文，不超过 1024 字），交代：
1. 模型/检查点来源与数据集；
2. 关键训练或评测配方（如 LR、batch、group、beta1、温度等）；
3. 实验目的（或事后补充核心结论）。

## 5. 脚本接入方式

- 直接调用 `swanlab.init(...)` 的脚本（如 `train_lora_sft.py`、`evaluate_shop_benchmark.py`、`clean_tasks.py`、`collect_sft_unified.py`、`audit_sft_trajectories.py`、`prepare_sft_dataset.py`、`probe_rl_sampling.py`、`search_sft.py`）统一支持并默认传入 `project`、`group`、`job_type`、`description`、`tags`。
- veRL GRPO 入口（`scripts/train_grpo.py`、`scripts/run_grpo_pipeline.py`）通过设置环境变量 `SWANLAB_GROUP`、`SWANLAB_JOB_TYPE`、`SWANLAB_DESCRIPTION`、`SWANLAB_TAGS`（并传入 Ray `runtime_env.env_vars`），由 SwanLab 0.9.1 SDK 自动写入每个 Run 的元数据。
