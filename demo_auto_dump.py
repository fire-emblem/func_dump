"""
Simple demo for func_dump.auto_dump.

This demo does not perform real GEMM. It only verifies dump/print behavior.
"""

import os

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from func_dump import auto_dump, load_records, replay_call


@auto_dump
def fp8_gemm(a, b, d, backend="torch_bf16"):
    # No real compute; return a small dict to verify serialization.
    return {"backend": backend, "a": a, "b": b, "d": d}


def _make_inputs():
    if HAS_TORCH:
        a = torch.randn(2, 3, dtype=torch.float16)
        b = torch.randn(3, 4, dtype=torch.float16)
        d = torch.zeros(2, 4, dtype=torch.float16)
        return a, b, d
    # Fallback: non-tensor inputs still serialize.
    return "A", "B", "D"


def main():
    os.environ.setdefault("FUNC_DUMP_MODE", "both")
    os.environ.setdefault("FUNC_DUMP_SINK", "memory")

    a, b, d = _make_inputs()
    out = fp8_gemm(a, b, d, backend="torch_bf16")
    print("demo output:", out)
    records = load_records()
    if records:
        replayed = replay_call(records[0], device_override="cpu")
        print("replay output:", replayed)


if __name__ == "__main__":
    main()
