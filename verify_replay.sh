#!/usr/bin/env bash
set -euo pipefail

JSONL_PATH="${1:-./func_dumps/calls.jsonl}"
python func_dump.py replay "${JSONL_PATH}" -m ./demo_auto_dump.py | tee /tmp/func_dump_replay.log

if grep -q "validate_args: success" /tmp/func_dump_replay.log && ! grep -q "FAIL" /tmp/func_dump_replay.log; then
  echo "replay validate: OK"
  exit 0
else
  echo "replay validate: FAIL"
  exit 1
fi
