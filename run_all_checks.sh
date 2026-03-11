#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

python -m pytest -q

FUNC_DUMP_MODE=dump FUNC_DUMP_DIR="${ROOT_DIR}/func_dumps_tmp" python demo_auto_dump.py

bash "${ROOT_DIR}/verify_replay.sh" "${ROOT_DIR}/func_dumps_tmp/calls.jsonl"
