#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

python -m pytest -q

rm -rf "${ROOT_DIR}/func_dumps_tmp"

FUNC_DUMP_MODE=dump FUNC_DUMP_DIR="${ROOT_DIR}/func_dumps_tmp" python demo_auto_dump.py

python func_dump.py replay "${ROOT_DIR}/func_dumps_tmp/calls.jsonl" -m ./demo_auto_dump.py

python func_dump.py replay "${ROOT_DIR}/test_data/fixed_calls.jsonl" -m ./demo_auto_dump.py
