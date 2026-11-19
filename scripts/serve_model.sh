#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-Qwen/Qwen3.5-2B}"
if [[ $# -gt 0 ]]; then
  shift
fi
# Forward native vLLM options so evaluation can explicitly select two GPUs,
# precision, seed and chat-template behavior without a separate launcher.
DRY_RUN=false
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    *) EXTRA_ARGS+=("$arg") ;;
  esac
done
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-shopping-agent}"
LLM_PORT="${LLM_PORT:-8000}"

# Use the supported native sampler; FlashInfer sampling otherwise needs a
# matching CUDA JIT toolchain even when evaluation uses greedy decoding.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ ! -x "$ROOT/.venv/bin/vllm" ]]; then
  echo "vLLM is not installed. Run: bash scripts/setup.sh" >&2
  exit 1
fi

cd "$ROOT"
COMMAND=(
  "$ROOT/.venv/bin/vllm" serve "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --port "$LLM_PORT"
  --max-model-len 24576
  --max-num-seqs "${LLM_MAX_NUM_SEQS:-8}"
  --gpu-memory-utilization "${LLM_GPU_MEMORY_UTILIZATION:-0.6}"
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  "${EXTRA_ARGS[@]}"
)
if [[ "$DRY_RUN" == true ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
exec "${COMMAND[@]}"
