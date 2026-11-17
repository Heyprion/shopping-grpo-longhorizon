#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${SFT_COLLECTION_PYTHON:-$ROOT/.venv/bin/python}"
WORKERS="${SFT_COLLECTION_WORKERS:-64}"
PORT="${SFT_COLLECTION_PORT:-15702}"
OUTPUT="${SFT_COLLECTION_OUTPUT:-$ROOT/outputs/sft-unified-collection-64k}"
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  read -r -s -p 'DeepSeek API key: ' DEEPSEEK_API_KEY
  printf '\n'
  export DEEPSEEK_API_KEY
fi
[[ -n "$DEEPSEEK_API_KEY" ]] || { echo 'Missing DeepSeek API key' >&2; exit 1; }
"$PYTHON" - "$PORT" <<'PY'
import socket, sys
s = socket.socket()
s.settimeout(1)
if s.connect_ex(('127.0.0.1', int(sys.argv[1]))) == 0:
    raise SystemExit('Collection port already in use; choose SFT_COLLECTION_PORT.')
PY
export SHOPPING_TASK_BUDGETS="$ROOT/data/cleaned/task_budgets.jsonl"
export SHOPSIM_ENV_SLOTS="$WORKERS" SHOPSIM_PORT="$PORT"
mkdir -p "$OUTPUT"
bash scripts/start_environment.sh > "$OUTPUT/environment.log" 2>&1 &
ENV_PID=$!
cleanup() { kill "$ENV_PID" 2>/dev/null || true; wait "$ENV_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
"$PYTHON" - "$PORT" "$ENV_PID" <<'PY'
import os, socket, sys, time
port, pid = map(int, sys.argv[1:])
for _ in range(180):
    try:
        os.kill(pid, 0)
        with socket.create_connection(('127.0.0.1', port), timeout=1):
            break
    except ConnectionRefusedError:
        time.sleep(1)
    except ProcessLookupError:
        raise SystemExit('Environment exited; inspect environment.log')
else:
    raise SystemExit('Environment failed to start; inspect environment.log')
PY
"$PYTHON" scripts/collect_sft_unified.py --workers "$WORKERS" --base-url "http://127.0.0.1:$PORT" --output-dir "$OUTPUT" "$@"
