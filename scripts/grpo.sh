#!/usr/bin/env bash
set -euo pipefail

# Configuration C; --pilot-steps 50 stops after the fixed validation comparison.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec "$ROOT/.venv/bin/python" scripts/run_grpo_pipeline.py "$@"
