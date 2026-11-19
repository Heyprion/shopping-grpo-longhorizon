#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--qwen35-4b" ]]; then
  shift
  exec "$ROOT/.venv/bin/python" "$ROOT/scripts/evaluate_base_model.py" "$@"
fi
if [[ "${1:-}" == "--qwen35-2b" ]]; then
  shift
  exec "$ROOT/.venv/bin/python" "$ROOT/scripts/evaluate_base_model.py" --model-size 2b "$@"
fi
exec "$ROOT/scripts/evaluate.sh" baseline
