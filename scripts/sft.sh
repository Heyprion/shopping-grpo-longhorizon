#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"
sft_python="${SFT_PYTHON:-$ROOT/.venv/bin/python}"
dry_run=false
balanced=false
resume_from_checkpoint=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fsdp2) shift ;; # This one-stage recipe always uses the two-GPU config.
    --balanced) balanced=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    --resume-from-checkpoint)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "--resume-from-checkpoint requires a checkpoint directory" >&2
        exit 2
      fi
      resume_from_checkpoint="$2"
      shift 2
      ;;
    *) echo "Usage: bash scripts/sft.sh [--balanced] [--dry-run] [--resume-from-checkpoint DIR]" >&2; exit 2 ;;
  esac
done

cd "$ROOT"
if [[ ! -x "$sft_python" ]]; then
  echo "找不到 Python 环境：$sft_python；请先配置环境或设置 SFT_PYTHON。" >&2
  exit 2
fi

# Match the measured seed-42 recipe; these defaults do not allocate a GPU.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

extra_args=()
fsdp_config="$ROOT/configs/fsdp2.yaml"
lora_r=16
lora_alpha=32
default_adapter="sft-lora"
default_run="sft-single-stage-seed42"
if "$balanced"; then
  fsdp_config="$ROOT/configs/fsdp2-balanced.yaml"
  lora_r=32
  lora_alpha=64
  default_adapter="sft-lora-balanced"
  default_run="sft-balanced-seed42"
  extra_args+=(--no-fsdp-activation-checkpointing --fsdp-checkpoint-stride 2)
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
fi
ADAPTER_DIR="${SFT_ADAPTER_DIR:-$ROOT/outputs/models/$default_adapter}"
if [[ "$BASE_MODEL" == "Qwen/Qwen3.5-2B" ]]; then
  extra_args+=(--revision "15852e8c16360a2fea060d615a32b45270f8a8fc")
fi
if [[ -n "$resume_from_checkpoint" ]]; then
  extra_args+=(--resume-from-checkpoint "$resume_from_checkpoint")
fi
train=("$sft_python" -m accelerate.commands.launch \
  --config_file "$fsdp_config" scripts/train_lora_sft.py \
  --model "$BASE_MODEL" \
  --train data/sft/train.jsonl \
  --validation data/sft/validation.jsonl \
  --output "$ADAPTER_DIR" \
  --fsdp2 --dtype bf16 --attention-implementation sdpa --liger-kernel \
  --epochs 3 --learning-rate 1e-4 --warmup-ratio 0.03 \
  --max-length 24576 \
  --per-device-train-batch-size 1 --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --lora-r "$lora_r" --lora-alpha "$lora_alpha" --lora-dropout 0.05 \
  --seed 42 --logging-steps 1 --save-total-limit 3 --profile \
  --preprocessing-cache "$ROOT/outputs/cache/sft-tokenized" \
  --swanlab --swanlab-mode online \
  --swanlab-project "${SFT_SWANLAB_PROJECT:-shopping-grpo-sft}" \
  --swanlab-run-name "${SFT_SWANLAB_RUN_NAME:-$default_run}" \
  "${extra_args[@]}")

if "$dry_run"; then
  if "$balanced"; then
    printf 'env %q ' "PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF"
  fi
  printf '%q ' "${train[@]}"; printf '\n'
  exit 0
fi

"$sft_python" scripts/check_fsdp2_runtime.py
"$sft_python" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for package in ("liger-kernel", "flash-linear-attention", "fla-core", "swanlab"):
    try:
        print(f"{package}={version(package)}")
    except PackageNotFoundError as exc:
        raise SystemExit(f"缺少 {package}，请先安装 SFT 加速与监控依赖。") from exc

from transformers.models.qwen3_5.modeling_qwen3_5 import chunk_gated_delta_rule

if chunk_gated_delta_rule is None or not chunk_gated_delta_rule.__module__.startswith("fla."):
    raise SystemExit("FLA 线性注意力内核未启用，请先检查 flash-linear-attention 安装。")
PY

mkdir -p "$ADAPTER_DIR"
printf '启动单阶段 SFT：seed=42，LR=1e-4，3 epoch，有效 batch=8\n'
printf 'LoRA r=%s alpha=%s；FSDP 配置：%s\n' "$lora_r" "$lora_alpha" "$fsdp_config"
printf '训练日志：%s/console.log\n' "$ADAPTER_DIR"
"${train[@]}" 2>&1 | tee -a "$ADAPTER_DIR/console.log"
