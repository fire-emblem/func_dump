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

from func_dump import auto_dump

os.environ.setdefault("FUNC_DUMP_MODE", "both")
os.environ.setdefault("FUNC_DUMP_DIR", "./func_dumps")


@auto_dump
def fp8_gemm(a, b, d, backend="torch_bf16"):
    # No real compute; return a small dict to verify serialization.
    return {"backend": backend, "a": a, "b": b, "d": d}


@auto_dump
def complex_op(
    tensor,
    scale=1.0,
    count=3,
    name="demo",
    flag=True,
    tags=None,
    extra=None,
    mode="sum",
    *extras,
    **options,
):
    # Exercise many argument types without real compute.
    payload = {
        "tensor": tensor,
        "scale": scale,
        "count": count,
        "name": name,
        "flag": flag,
        "tags": tags,
        "extra": extra,
        "mode": mode,
        "extras": extras,
        "options": options,
    }
    return payload


EXPECTED_SPEC = {
    "shape": (2, 3),
    "dtype": "torch.float16",
    "device": "cpu",
    "scale": 0.5,
    "count": 2,
    "name": "complex",
    "flag": False,
    "tags": ["x", "y"],
    "extra": {"k1": 1, "k2": (1, 2)},
    "mode": "mix",
    "extras": (123, 4.56, None),
    "options": {"backend": "torch_bf16", "device": "cpu"},
}


@auto_dump
def validate_args(
    tensor,
    scale=1.0,
    count=3,
    name="demo",
    flag=True,
    tags=None,
    extra=None,
    mode="sum",
    options=None,
    *extras,
):
    ok = True
    if HAS_TORCH and isinstance(tensor, torch.Tensor):
        ok = ok and tuple(tensor.shape) == EXPECTED_SPEC["shape"]
        ok = ok and str(tensor.dtype) == EXPECTED_SPEC["dtype"]
        ok = ok and str(tensor.device) == EXPECTED_SPEC["device"]
    else:
        ok = False
    ok = ok and scale == EXPECTED_SPEC["scale"]
    ok = ok and count == EXPECTED_SPEC["count"]
    ok = ok and name == EXPECTED_SPEC["name"]
    ok = ok and flag == EXPECTED_SPEC["flag"]
    ok = ok and tags == EXPECTED_SPEC["tags"]
    ok = ok and extra == EXPECTED_SPEC["extra"]
    ok = ok and mode == EXPECTED_SPEC["mode"]
    ok = ok and options == EXPECTED_SPEC["options"]
    ok = ok and extras == EXPECTED_SPEC["extras"]

    if not ok:
        raise AssertionError("validate_args: argument mismatch")
    print("validate_args: success")
    return "success"


@auto_dump
def mixed_types(
    tensor,
    cuda_tensor=None,
    number=1,
    ratio=0.1,
    text="hello",
    flag=True,
    tags=None,
    payload=None,
    pair=(1, 2),
    items=(1, "a", None),
):
    return {
        "tensor": tensor,
        "cuda_tensor": cuda_tensor,
        "number": number,
        "ratio": ratio,
        "text": text,
        "flag": flag,
        "tags": tags,
        "payload": payload,
        "pair": pair,
        "items": items,
    }


def _make_inputs():
    if HAS_TORCH:
        a = torch.randn(2, 3, dtype=torch.float16)
        b = torch.randn(3, 4, dtype=torch.float16)
        d = torch.zeros(2, 4, dtype=torch.float16)
        if torch.cuda.is_available():
            c = torch.randn(2, 3, dtype=torch.float16, device="cuda")
        else:
            c = None
        return a, b, d, c
    # Fallback: non-tensor inputs still serialize.
    return "A", "B", "D", None


def main():
    a, b, d, c = _make_inputs()
    out = fp8_gemm(a, b, d, backend="torch_bf16")
    print("demo output:", out)
    complex_out = complex_op(
        a,
        0.5,
        2,
        "complex",
        False,
        ["x", "y"],
        {"k1": 1, "k2": (1, 2)},
        "mix",
        123,
        4.56,
        None,
        backend="torch_bf16",
        device="cpu",
    )
    print("complex output:", complex_out)
    validate_out = validate_args(
        a,
        0.5,
        2,
        "complex",
        False,
        ["x", "y"],
        {"k1": 1, "k2": (1, 2)},
        "mix",
        {"backend": "torch_bf16", "device": "cpu"},
        123,
        4.56,
        None,
    )
    print("validate output:", validate_out)

    mixed_out = mixed_types(
        a,
        cuda_tensor=c,
        number=7,
        ratio=3.14,
        text="mixed",
        flag=False,
        tags=("x", "y"),
        payload={"k": [1, 2, 3], "obj": object()},
        pair=(9, 10),
        items=(True, 0, None),
    )
    print("mixed output:", mixed_out)


if __name__ == "__main__":
    main()
