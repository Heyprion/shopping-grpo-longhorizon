#!/usr/bin/env bash
set -euo pipefail

# Configuration C; batch2, with fixed 30-task SFT-failure evaluation at 50/100 updates.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec "$ROOT/.venv/bin/python" scripts/run_grpo_pipeline.py "$@"
