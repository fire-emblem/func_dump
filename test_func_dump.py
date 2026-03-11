"""
func_dump 功能测试

运行方式:
    pytest test_func_dump.py -v
    python test_func_dump.py
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import torch
import pytest

# 确保能导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from deep_gemm.testing.func_dump import (
    register,
    _registry,
    _memory_records,
    _parse_modes,
    _bind_args,
    _serialize_value,
    _deserialize_value,
    _make_random_tensor,
    _format_short,
    _print_call,
    dump_call,
    load_records,
    replay_call,
    replay_all,
    auto_dump,
)


# =========================================================================
# Fixtures
# =========================================================================
@pytest.fixture(autouse=True)
def clean_env():
    """每个测试前后清理环境变量和 registry。"""
    saved_registry = dict(_registry)
    saved_mode = os.environ.pop("FUNC_DUMP_MODE", None)
    saved_dir = os.environ.pop("FUNC_DUMP_DIR", None)
    saved_sink = os.environ.pop("FUNC_DUMP_SINK", None)
    _memory_records.clear()
    yield
    _registry.clear()
    _registry.update(saved_registry)
    if saved_mode is not None:
        os.environ["FUNC_DUMP_MODE"] = saved_mode
    else:
        os.environ.pop("FUNC_DUMP_MODE", None)
    if saved_dir is not None:
        os.environ["FUNC_DUMP_DIR"] = saved_dir
    else:
        os.environ.pop("FUNC_DUMP_DIR", None)
    if saved_sink is not None:
        os.environ["FUNC_DUMP_SINK"] = saved_sink
    else:
        os.environ.pop("FUNC_DUMP_SINK", None)
    _memory_records.clear()


@pytest.fixture
def tmp_dir():
    """临时目录，测试结束后自动清理。"""
    d = tempfile.mkdtemp(prefix="func_dump_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# =========================================================================
# 辅助函数
# =========================================================================
def _sample_fn(a, b, scale=1.0):
    return a * scale + b


def _sample_fn_varargs(*tensors, mode="sum"):
    if mode == "sum":
        return sum(tensors)
    return tensors[0]


def _sample_fn_no_return(x):
    _ = x + 1
    return None


# =========================================================================
# Test: _parse_modes
# =========================================================================
class TestParseModes:
    def test_empty(self):
        os.environ.pop("FUNC_DUMP_MODE", None)
        assert _parse_modes() == set()

    def test_blank(self):
        os.environ["FUNC_DUMP_MODE"] = "   "
        assert _parse_modes() == set()

    def test_dump_only(self):
        os.environ["FUNC_DUMP_MODE"] = "dump"
        assert _parse_modes() == {"dump"}

    def test_print_only(self):
        os.environ["FUNC_DUMP_MODE"] = "print"
        assert _parse_modes() == {"print"}

    def test_dump_print(self):
        os.environ["FUNC_DUMP_MODE"] = "dump,print"
        assert _parse_modes() == {"dump", "print"}

    def test_both(self):
        os.environ["FUNC_DUMP_MODE"] = "both"
        assert _parse_modes() == {"dump", "print"}

    def test_both_case_insensitive(self):
        os.environ["FUNC_DUMP_MODE"] = "BOTH"
        assert _parse_modes() == {"dump", "print"}

    def test_dump_print_with_spaces(self):
        os.environ["FUNC_DUMP_MODE"] = " dump , print "
        assert _parse_modes() == {"dump", "print"}


# =========================================================================
# Test: register
# =========================================================================
class TestRegister:
    def test_register_returns_key(self):
        key = register(_sample_fn)
        assert "test_func_dump._sample_fn" in key

    def test_register_adds_to_registry(self):
        key = register(_sample_fn)
        assert key in _registry
        assert _registry[key] is _sample_fn


# =========================================================================
# Test: _bind_args
# =========================================================================
class TestBindArgs:
    def test_positional_and_keyword(self):
        result = _bind_args(_sample_fn, (1, 2), {"scale": 3.0})
        assert result == {"a": 1, "b": 2, "scale": 3.0}

    def test_default_applied(self):
        result = _bind_args(_sample_fn, (1, 2), {})
        assert result == {"a": 1, "b": 2, "scale": 1.0}

    def test_all_positional(self):
        result = _bind_args(_sample_fn, (1, 2, 5.0), {})
        assert result == {"a": 1, "b": 2, "scale": 5.0}

    def test_varargs(self):
        result = _bind_args(_sample_fn_varargs, (1, 2, 3), {"mode": "sum"})
        assert result["tensors"] == (1, 2, 3)
        assert result["mode"] == "sum"

    def test_fallback_for_uninspectable(self):
        # 用 lambda 模拟无法 inspect 的情况不太好，
        # 改用 mock 让 signature 抛异常
        fn = mock.MagicMock()
        fn.__module__ = "test"
        fn.__qualname__ = "mock_fn"
        result = _bind_args(fn, (10, 20), {"k": "v"})
        assert result["__pos_0"] == 10
        assert result["__pos_1"] == 20
        assert result["k"] == "v"


# =========================================================================
# Test: _serialize_value
# =========================================================================
class TestSerialize:
    def test_int(self):
        assert _serialize_value(42) == 42

    def test_float(self):
        assert _serialize_value(3.14) == 3.14

    def test_string(self):
        assert _serialize_value("hello") == "hello"

    def test_bool(self):
        assert _serialize_value(True) is True

    def test_none(self):
        assert _serialize_value(None) is None

    def test_tensor(self):
        t = torch.randn(3, 4)
        result = _serialize_value(t)
        assert result["__tensor__"] is True
        assert result["shape"] == [3, 4]
        assert result["dtype"] == "torch.float32"
        assert "device" in result
        assert "stride" in result

    def test_tensor_fp16(self):
        t = torch.randn(2, 5).half()
        result = _serialize_value(t)
        assert result["dtype"] == "torch.float16"
        assert result["shape"] == [2, 5]

    def test_tuple(self):
        result = _serialize_value((1, "a"))
        assert result["__tuple__"] is True
        assert result["items"] == [1, "a"]

    def test_list(self):
        result = _serialize_value([1, 2, 3])
        assert result == [1, 2, 3]

    def test_dict(self):
        result = _serialize_value({"x": 1, "y": 2})
        assert result["__dict__"] is True
        assert result["items"] == {"x": 1, "y": 2}

    def test_unsupported_type(self):
        result = _serialize_value(object())
        assert "__repr__" in result

    def test_nested(self):
        t = torch.zeros(2)
        val = (t, [1, 2], {"k": "v"})
        result = _serialize_value(val)
        assert result["__tuple__"] is True
        assert result["items"][0]["__tensor__"] is True
        assert result["items"][1] == [1, 2]
        assert result["items"][2]["__dict__"] is True


# =========================================================================
# Test: _make_random_tensor
# =========================================================================
class TestMakeRandomTensor:
    def test_float32(self):
        t = _make_random_tensor([3, 4], "torch.float32", "cpu")
        assert t.shape == (3, 4)
        assert t.dtype == torch.float32

    def test_float16(self):
        t = _make_random_tensor([2, 2], "torch.float16", "cpu")
        assert t.dtype == torch.float16

    def test_bfloat16(self):
        t = _make_random_tensor([5], "torch.bfloat16", "cpu")
        assert t.dtype == torch.bfloat16

    def test_int32(self):
        t = _make_random_tensor([10], "torch.int32", "cpu")
        assert t.dtype == torch.int32

    def test_int64(self):
        t = _make_random_tensor([3, 3], "torch.int64", "cpu")
        assert t.dtype == torch.int64

    def test_bool(self):
        t = _make_random_tensor([4], "torch.bool", "cpu")
        assert t.dtype == torch.bool

    def test_uint8(self):
        t = _make_random_tensor([8], "torch.uint8", "cpu")
        assert t.dtype == torch.uint8


# =========================================================================
# Test: _deserialize_value
# =========================================================================
class TestDeserialize:
    def test_scalar(self):
        assert _deserialize_value(42) == 42
        assert _deserialize_value("hi") == "hi"
        assert _deserialize_value(None) is None

    def test_tensor(self):
        meta = {
            "__tensor__": True,
            "dtype": "torch.float32",
            "shape": [3, 4],
            "stride": [4, 1],
            "device": "cpu",
        }
        t = _deserialize_value(meta)
        assert isinstance(t, torch.Tensor)
        assert t.shape == (3, 4)
        assert t.dtype == torch.float32

    def test_tensor_device_override(self):
        meta = {
            "__tensor__": True,
            "dtype": "torch.float32",
            "shape": [2],
            "stride": [1],
            "device": "cuda:0",
        }
        t = _deserialize_value(meta, device_override="cpu")
        assert str(t.device) == "cpu"

    def test_tuple(self):
        val = {"__tuple__": True, "items": [1, 2, 3]}
        result = _deserialize_value(val)
        assert result == (1, 2, 3)
        assert isinstance(result, tuple)

    def test_dict(self):
        val = {"__dict__": True, "items": {"a": 1, "b": 2}}
        result = _deserialize_value(val)
        assert result == {"a": 1, "b": 2}

    def test_repr(self):
        val = {"__repr__": "<SomeObject>"}
        result = _deserialize_value(val)
        assert result == "<SomeObject>"

    def test_list(self):
        val = [1, 2, 3]
        result = _deserialize_value(val)
        assert result == [1, 2, 3]


# =========================================================================
# Test: _format_short
# =========================================================================
class TestFormatShort:
    def test_tensor(self):
        t = torch.randn(3, 4)
        s = _format_short(t)
        assert "Tensor" in s
        assert "[3, 4]" in s
        assert "float32" in s

    def test_scalar(self):
        assert _format_short(42) == "42"

    def test_string(self):
        assert _format_short("abc") == "'abc'"

    def test_long_string(self):
        s = _format_short("x" * 100)
        assert "..." in s

    def test_tuple(self):
        s = _format_short((1, 2, 3))
        assert s.startswith("(")
        assert s.endswith(")")

    def test_list(self):
        s = _format_short([1, 2])
        assert s.startswith("[")

    def test_dict(self):
        s = _format_short({"a": 1})
        assert "dict" in s

    def test_none(self):
        assert _format_short(None) == "None"


# =========================================================================
# Test: _print_call
# =========================================================================
class TestPrintCall:
    def test_prints_to_stderr(self, capsys):
        named = {"a": torch.randn(2, 3), "scale": 1.0}
        _print_call("mod.fn", named, torch.randn(2, 3))
        captured = capsys.readouterr()
        assert "mod.fn" in captured.err
        assert "Tensor" in captured.err
        assert "scale" in captured.err

    def test_contains_timestamp(self, capsys):
        _print_call("test.fn", {"x": 1}, 2)
        captured = capsys.readouterr()
        assert "func_dump" in captured.err
        assert ":" in captured.err  # 时间戳 HH:MM:SS


# =========================================================================
# Test: dump_call + load_records
# =========================================================================
class TestDumpAndLoad:
    def test_basic(self, tmp_dir):
        out = tmp_dir / "calls.jsonl"
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        ret = a + b

        dump_call(_sample_fn, (a, b), {"scale": 2.0}, ret, out)

        records = load_records(out)
        assert len(records) == 1

        rec = records[0]
        assert rec["idx"] == 0
        assert "_sample_fn" in rec["func"]
        assert "a" in rec["args"]
        assert "b" in rec["args"]
        assert "scale" in rec["args"]
        assert rec["args"]["scale"] == 2.0
        assert rec["args"]["a"]["__tensor__"] is True
        assert rec["args"]["a"]["shape"] == [4, 4]

    def test_multiple_calls(self, tmp_dir):
        out = tmp_dir / "calls.jsonl"
        for i in range(5):
            a = torch.randn(i + 1, 3)
            b = torch.randn(i + 1, 3)
            dump_call(_sample_fn, (a, b), {}, a + b, out)

        records = load_records(out)
        assert len(records) == 5
        for i, rec in enumerate(records):
            assert rec["idx"] == i
            assert rec["args"]["a"]["shape"] == [i + 1, 3]

    def test_none_return(self, tmp_dir):
        out = tmp_dir / "calls.jsonl"
        x = torch.randn(3)
        dump_call(_sample_fn_no_return, (x,), {}, None, out)

        records = load_records(out)
        assert records[0]["ret"] is None

    def test_has_timestamp(self, tmp_dir):
        out = tmp_dir / "calls.jsonl"
        dump_call(_sample_fn, (1, 2), {}, 3, out)
        records = load_records(out)
        assert "ts" in records[0]
        assert "T" in records[0]["ts"]  # ISO format

    def test_no_pt_files(self, tmp_dir):
        """确认不会生成 .pt 文件。"""
        out = tmp_dir / "calls.jsonl"
        a = torch.randn(100, 100)
        dump_call(_sample_fn, (a, a), {}, a, out)

        pt_files = list(tmp_dir.glob("*.pt"))
        assert len(pt_files) == 0

    def test_nested_args(self, tmp_dir):
        out = tmp_dir / "calls.jsonl"

        def fn_nested(data):
            return data

        t = torch.randn(2)
        val = (t, [1, 2], {"key": "value"})
        dump_call(fn_nested, (val,), {}, val, out)

        records = load_records(out)
        arg = records[0]["args"]["data"]
        assert arg["__tuple__"] is True


# =========================================================================
# Test: replay_call / replay_all
# =========================================================================
class TestReplay:
    def test_replay_basic(self, tmp_dir):
        out = tmp_dir / "calls.jsonl"
        register(_sample_fn)

        a = torch.randn(3, 3)
        b = torch.randn(3, 3)
        dump_call(_sample_fn, (a, b), {"scale": 2.0}, a * 2.0 + b, out)

        records = load_records(out)
        result = replay_call(records[0], device_override="cpu")
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 3)

    def test_replay_all_ok(self, tmp_dir):
        out = tmp_dir / "calls.jsonl"
        register(_sample_fn)

        for _ in range(3):
            a = torch.randn(4, 4)
            b = torch.randn(4, 4)
            dump_call(_sample_fn, (a, b), {}, a + b, out)

        results = replay_all(out, device_override="cpu")
        assert len(results) == 3
        for idx, fn_key, ret in results:
            assert not isinstance(ret, Exception)
            assert isinstance(ret, torch.Tensor)

    def test_replay_unknown_function(self, tmp_dir):
        out = tmp_dir / "calls.jsonl"
        # 手动写一条记录，func 指向不存在的函数
        record = {
            "idx": 0,
            "func": "nonexistent.module.fn",
            "args": {"x": 1},
            "ret": 2,
            "ts": "2025-01-01T00:00:00",
        }
        with open(out, "w") as f:
            f.write(json.dumps(record) + "\n")

        results = replay_all(out)
        assert len(results) == 1
        assert isinstance(results[0][2], KeyError)

    def test_replay_generates_random_tensors(self, tmp_dir):
        """回放两次，结果不应完全相同（随机输入）。"""
        out = tmp_dir / "calls.jsonl"
        register(_sample_fn)

        a = torch.randn(5, 5)
        b = torch.randn(5, 5)
        dump_call(_sample_fn, (a, b), {"scale": 1.0}, a + b, out)

        records = load_records(out)
        r1 = replay_call(records[0], device_override="cpu")
        r2 = replay_call(records[0], device_override="cpu")
        # 极小概率两次随机完全相同，实际不可能
        assert not torch.equal(r1, r2)

    def test_replay_respects_shape_dtype(self, tmp_dir):
        out = tmp_dir / "calls.jsonl"

        def identity(x):
            return x

        register(identity)
        t = torch.randn(7, 13).half()
        dump_call(identity, (t,), {}, t, out)

        records = load_records(out)
        result = replay_call(records[0], device_override="cpu")
        assert result.shape == (7, 13)
        assert result.dtype == torch.float16


# =========================================================================
# Test: auto_dump decorator
# =========================================================================
class TestAutoDecorator:
    def test_no_mode_returns_original(self):
        """未设置 FUNC_DUMP_MODE 时返回原函数。"""
        os.environ.pop("FUNC_DUMP_MODE", None)

        def my_fn(x):
            return x + 1

        decorated = auto_dump(my_fn)
        assert decorated is my_fn

    def test_dump_mode_wraps(self, tmp_dir):
        os.environ["FUNC_DUMP_MODE"] = "dump"
        os.environ["FUNC_DUMP_DIR"] = str(tmp_dir)
        os.environ["FUNC_DUMP_SINK"] = "file"

        def add_fn(a, b):
            return a + b

        decorated = auto_dump(add_fn)
        assert decorated is not add_fn

        result = decorated(torch.randn(2), torch.randn(2))
        assert isinstance(result, torch.Tensor)

        jsonl = tmp_dir / "calls.jsonl"
        assert jsonl.exists()
        records = load_records(jsonl)
        assert len(records) == 1

    def test_print_mode_wraps(self, capsys):
        os.environ["FUNC_DUMP_MODE"] = "print"

        def sub_fn(a, b):
            return a - b

        decorated = auto_dump(sub_fn)
        assert decorated is not sub_fn

        result = decorated(3.0, 1.0)
        assert result == 2.0

        captured = capsys.readouterr()
        assert "sub_fn" in captured.err
        assert "func_dump" in captured.err

    def test_both_mode(self, tmp_dir, capsys):
        os.environ["FUNC_DUMP_MODE"] = "both"
        os.environ["FUNC_DUMP_DIR"] = str(tmp_dir)
        os.environ["FUNC_DUMP_SINK"] = "file"

        def mul_fn(a, b):
            return a * b

        decorated = auto_dump(mul_fn)
        decorated(2.0, 3.0)

        # 检查 print
        captured = capsys.readouterr()
        assert "mul_fn" in captured.err

        # 检查 dump
        jsonl = tmp_dir / "calls.jsonl"
        assert jsonl.exists()
        records = load_records(jsonl)
        assert len(records) == 1

    def test_dump_print_comma(self, tmp_dir, capsys):
        os.environ["FUNC_DUMP_MODE"] = "dump,print"
        os.environ["FUNC_DUMP_DIR"] = str(tmp_dir)
        os.environ["FUNC_DUMP_SINK"] = "file"

        def div_fn(a, b):
            return a / b

        decorated = auto_dump(div_fn)
        decorated(10.0, 2.0)

        captured = capsys.readouterr()
        assert "div_fn" in captured.err

        jsonl = tmp_dir / "calls.jsonl"
        records = load_records(jsonl)
        assert len(records) == 1

    def test_preserves_function_name(self):
        os.environ["FUNC_DUMP_MODE"] = "print"

        def my_special_fn(x):
            """My docstring."""
            return x

        decorated = auto_dump(my_special_fn)
        assert decorated.__name__ == "my_special_fn"
        assert decorated.__doc__ == "My docstring."

    def test_registers_even_without_mode(self):
        os.environ.pop("FUNC_DUMP_MODE", None)

        def reg_fn(x):
            return x

        auto_dump(reg_fn)
        key = f"{reg_fn.__module__}.{reg_fn.__qualname__}"
        assert key in _registry

    def test_multiple_calls_dump(self, tmp_dir):
        os.environ["FUNC_DUMP_MODE"] = "dump"
        os.environ["FUNC_DUMP_DIR"] = str(tmp_dir)
        os.environ["FUNC_DUMP_SINK"] = "file"

        def inc_fn(x):
            return x + 1

        decorated = auto_dump(inc_fn)
        for i in range(10):
            decorated(i)

        jsonl = tmp_dir / "calls.jsonl"
        records = load_records(jsonl)
        assert len(records) == 10
        for i, rec in enumerate(records):
            assert rec["idx"] == i


# =========================================================================
# Test: 端到端（dump → load → replay）
# =========================================================================
class TestEndToEnd:
    def test_full_cycle_tensor(self, tmp_dir):
        os.environ["FUNC_DUMP_MODE"] = "dump"
        os.environ["FUNC_DUMP_DIR"] = str(tmp_dir)
        os.environ["FUNC_DUMP_SINK"] = "file"

        def matmul_fn(a, b):
            return a @ b

        decorated = auto_dump(matmul_fn)
        a = torch.randn(8, 16)
        b = torch.randn(16, 4)
        original_result = decorated(a, b)
        assert original_result.shape == (8, 4)

        # replay
        jsonl = tmp_dir / "calls.jsonl"
        records = load_records(jsonl)
        replay_result = replay_call(records[0], device_override="cpu")
        assert replay_result.shape == (8, 4)
        assert replay_result.dtype == torch.float32

    def test_full_cycle_mixed_args(self, tmp_dir):
        os.environ["FUNC_DUMP_MODE"] = "dump"
        os.environ["FUNC_DUMP_DIR"] = str(tmp_dir)
        os.environ["FUNC_DUMP_SINK"] = "file"

        def mixed_fn(tensor, count, name="default", flag=True):
            if flag:
                return tensor * count
            return tensor

        decorated = auto_dump(mixed_fn)
        t = torch.randn(3, 3)
        decorated(t, 5, name="test", flag=True)

        jsonl = tmp_dir / "calls.jsonl"
        records = load_records(jsonl)
        rec = records[0]
        assert rec["args"]["count"] == 5
        assert rec["args"]["name"] == "test"
        assert rec["args"]["flag"] is True
        assert rec["args"]["tensor"]["__tensor__"] is True

        result = replay_call(rec, device_override="cpu")
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 3)

    def test_full_cycle_replay_all(self, tmp_dir):
        os.environ["FUNC_DUMP_MODE"] = "dump"
        os.environ["FUNC_DUMP_DIR"] = str(tmp_dir)
        os.environ["FUNC_DUMP_SINK"] = "file"

        def add2(a, b):
            return a + b

        decorated = auto_dump(add2)
        shapes = [(2, 3), (4, 5), (1, 10)]
        for s in shapes:
            decorated(torch.randn(*s), torch.randn(*s))

        jsonl = tmp_dir / "calls.jsonl"
        results = replay_all(jsonl, device_override="cpu")
        assert len(results) == 3
        for i, (idx, fn_key, ret) in enumerate(results):
            assert not isinstance(ret, Exception)
            assert ret.shape == shapes[i]


# =========================================================================
# Test: JSONL 格式正确性
# =========================================================================
class TestJSONLFormat:
    def test_each_line_valid_json(self, tmp_dir):
        out = tmp_dir / "calls.jsonl"
        for i in range(5):
            dump_call(_sample_fn, (i, i + 1), {}, i * 2, out)

        with open(out, "r") as f:
            for line in f:
                obj = json.loads(line.strip())
                assert "idx" in obj
                assert "func" in obj
                assert "args" in obj
                assert "ret" in obj
                assert "ts" in obj

    def test_no_tensor_file_references(self, tmp_dir):
        """确认 JSONL 中没有 file 字段（不保存 .pt）。"""
        out = tmp_dir / "calls.jsonl"
        t = torch.randn(10, 10)
        dump_call(_sample_fn, (t, t), {}, t, out)

        with open(out, "r") as f:
            content = f.read()
        assert '"file"' not in content


# =========================================================================
# Test: 边界情况
# =========================================================================
class TestEdgeCases:
    def test_empty_tensor(self, tmp_dir):
        out = tmp_dir / "calls.jsonl"
        t = torch.tensor([])
        dump_call(_sample_fn, (t, t), {}, t, out)

        records = load_records(out)
        assert records[0]["args"]["a"]["shape"] == [0]

    def test_scalar_tensor(self, tmp_dir):
        out = tmp_dir / "calls.jsonl"
        t = torch.tensor(3.14)
        dump_call(_sample_fn, (t, t), {}, t, out)

        records = load_records(out)
        assert records[0]["args"]["a"]["shape"] == []

    def test_high_dim_tensor(self, tmp_dir):
        out = tmp_dir / "calls.jsonl"
        t = torch.randn(2, 3, 4, 5, 6)
        dump_call(_sample_fn, (t, t), {}, t, out)

        records = load_records(out)
        assert records[0]["args"]["a"]["shape"] == [2, 3, 4, 5, 6]

        register(_sample_fn)
        result = replay_call(records[0], device_override="cpu")
        assert result.shape == (2, 3, 4, 5, 6)

    def test_non_serializable_arg(self, tmp_dir):
        out = tmp_dir / "calls.jsonl"

        def fn_with_obj(x, obj):
            return x

        register(fn_with_obj)
        t = torch.randn(2)
        dump_call(fn_with_obj, (t, object()), {}, t, out)

        records = load_records(out)
        assert "__repr__" in records[0]["args"]["obj"]

    def test_load_empty_file(self, tmp_dir):
        out = tmp_dir / "calls.jsonl"
        out.touch()
        records = load_records(out)
        assert records == []


# =========================================================================
# 手动测试入口
# =========================================================================
def _run_manual_test():
    """不依赖 pytest 的快速验证。"""
    import tempfile
    import shutil

    tmp = Path(tempfile.mkdtemp(prefix="func_dump_manual_"))
    print(f"临时目录: {tmp}")

    try:
        # 1. 注册
        key = register(_sample_fn)
        print(f"[PASS] register -> {key}")

        # 2. 序列化
        t = torch.randn(4, 4)
        s = _serialize_value(t)
        assert s["__tensor__"] is True
        assert s["shape"] == [4, 4]
        assert "file" not in s
        print("[PASS] serialize tensor (no .pt file)")

        # 3. 反序列化
        t2 = _deserialize_value(s)
        assert t2.shape == (4, 4)
        assert t2.dtype == torch.float32
        print("[PASS] deserialize tensor (random)")

        # 4. dump
        out = tmp / "calls.jsonl"
        a, b = torch.randn(3, 3), torch.randn(3, 3)
        dump_call(_sample_fn, (a, b), {"scale": 2.0}, a * 2 + b, out)
        assert out.exists()
        assert len(list(tmp.glob("*.pt"))) == 0
        print("[PASS] dump_call (no .pt files)")

        # 5. load
        records = load_records(out)
        assert len(records) == 1
        assert records[0]["args"]["scale"] == 2.0
        print("[PASS] load_records")

        # 6. replay
        result = replay_call(records[0], device_override="cpu")
        assert result.shape == (3, 3)
        print(f"[PASS] replay_call -> shape={list(result.shape)}")

        # 7. print mode
        named = {"a": torch.randn(2, 3), "s": 1.0}
        _print_call("test.fn", named, torch.randn(2, 3))
        print("[PASS] _print_call")

        # 8. parse modes
        os.environ["FUNC_DUMP_MODE"] = "both"
        assert _parse_modes() == {"dump", "print"}
        os.environ["FUNC_DUMP_MODE"] = "dump,print"
        assert _parse_modes() == {"dump", "print"}
        os.environ.pop("FUNC_DUMP_MODE", None)
        assert _parse_modes() == set()
        print("[PASS] _parse_modes (both / dump,print / empty)")

        print("\n===== ALL MANUAL TESTS PASSED =====")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    _run_manual_test()
