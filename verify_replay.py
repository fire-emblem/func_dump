"""
Verify that replayed results match recorded metadata in func_dumps/calls.jsonl.

Usage:
    python verify_replay.py
    python verify_replay.py --jsonl ./func_dumps/calls.jsonl
"""

import argparse
import importlib
import importlib.util
import os
import sys

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from func_dump import load_records, replay_call


def _assert_match(serialized, actual):
    if isinstance(serialized, dict):
        if serialized.get("__tensor__"):
            if not HAS_TORCH or not isinstance(actual, torch.Tensor):
                raise AssertionError("expected torch.Tensor")
            exp_shape = tuple(serialized["shape"])
            exp_dtype = serialized["dtype"]
            exp_device = serialized.get("device", "cpu")
            if tuple(actual.shape) != exp_shape:
                raise AssertionError(f"shape mismatch: {actual.shape} vs {exp_shape}")
            if str(actual.dtype) != exp_dtype:
                raise AssertionError(f"dtype mismatch: {actual.dtype} vs {exp_dtype}")
            if str(actual.device) != exp_device:
                raise AssertionError(f"device mismatch: {actual.device} vs {exp_device}")
            return
        if serialized.get("__tuple__"):
            items = serialized["items"]
            if not isinstance(actual, tuple) or len(actual) != len(items):
                raise AssertionError("tuple length mismatch")
            for s, a in zip(items, actual):
                _assert_match(s, a)
            return
        if serialized.get("__dict__"):
            items = serialized["items"]
            if not isinstance(actual, dict) or set(actual.keys()) != set(items.keys()):
                raise AssertionError("dict keys mismatch")
            for k in items:
                _assert_match(items[k], actual[k])
            return
        if "__repr__" in serialized:
            return
        if not isinstance(actual, dict):
            raise AssertionError("dict mismatch")
        for k, v in serialized.items():
            _assert_match(v, actual.get(k))
        return
    if isinstance(serialized, list):
        if not isinstance(actual, list) or len(actual) != len(serialized):
            raise AssertionError("list length mismatch")
        for s, a in zip(serialized, actual):
            _assert_match(s, a)
        return
    if serialized != actual:
        raise AssertionError(f"value mismatch: {serialized} vs {actual}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="./func_dumps/calls.jsonl")
    parser.add_argument(
        "--module",
        action="append",
        default=["demo_auto_dump"],
        help="module name or .py file path to import for registration",
    )
    args = parser.parse_args()

    for mod in args.module:
        if mod:
            if mod.endswith(".py") or os.path.sep in mod:
                module_name = f"_verify_mod_{len(sys.modules)}"
                spec = importlib.util.spec_from_file_location(module_name, mod)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot load module from path: {mod}")
                loaded = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = loaded
                spec.loader.exec_module(loaded)
            else:
                importlib.import_module(mod)

    records = load_records(args.jsonl)
    if not records:
        print("no records found")
        return 1

    failed = 0
    for rec in records:
        try:
            out = replay_call(rec, device_override=None)
            _assert_match(rec.get("ret"), out)
        except Exception as e:
            failed += 1
            print(f"[FAIL] idx={rec.get('idx')} func={rec.get('func')}: {e}")

    if failed:
        print(f"{failed} failed")
        return 1
    print(f"ok: {len(records)} records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
