"""
func_dump — 函数调用录制(dump)与打印(print)工具

录制时只保存 tensor 的元信息（shape/dtype/stride/device），不保存实际数据。
回放时根据元信息生成随机 tensor 作为输入。

环境变量:
    FUNC_DUMP_MODE : 支持 dump / print / dump,print / both
    FUNC_DUMP_DIR  : 录制输出目录，默认 ./func_dumps
    FUNC_DUMP_SINK : 输出位置，memory 或 file（默认 memory）
"""

import os
import sys
import json
import inspect
import functools
from pathlib import Path
from datetime import datetime

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# =========================================================================
# Registry
# =========================================================================
_registry = {}
_memory_records = []


def register(fn):
    """手动注册函数到 registry，返回 key。"""
    key = f"{fn.__module__}.{fn.__qualname__}"
    _registry[key] = fn
    return key


# =========================================================================
# 环境变量解析
# =========================================================================
def _parse_modes():
    raw = os.environ.get("FUNC_DUMP_MODE", "").strip().lower()
    if not raw:
        return set()
    if raw == "both":
        return {"dump", "print"}
    return {m.strip() for m in raw.split(",") if m.strip()}


def _get_dump_dir():
    return Path(os.environ.get("FUNC_DUMP_DIR", "./func_dumps"))


# =========================================================================
# 参数绑定
# =========================================================================
def _bind_args(fn, args, kwargs):
    """将位置参数映射为形参名，返回 dict。"""
    try:
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):
        named = {}
        for i, a in enumerate(args):
            named[f"__pos_{i}"] = a
        named.update(kwargs)
        return named


# =========================================================================
# 序列化（只保存元信息，不保存 tensor 数据）
# =========================================================================
def _serialize_value(val):
    """将值序列化为 JSON 可存储的结构（tensor 只存元信息）。"""
    if HAS_TORCH and isinstance(val, torch.Tensor):
        return {
            "__tensor__": True,
            "dtype": str(val.dtype),
            "shape": list(val.shape),
            "stride": list(val.stride()),
            "device": str(val.device),
        }
    elif isinstance(val, tuple):
        return {"__tuple__": True, "items": [_serialize_value(v) for v in val]}
    elif isinstance(val, list):
        return [_serialize_value(v) for v in val]
    elif isinstance(val, dict):
        return {"__dict__": True, "items": {k: _serialize_value(v) for k, v in val.items()}}
    elif isinstance(val, (int, float, str, bool, type(None))):
        return val
    else:
        return {"__repr__": repr(val)}


# =========================================================================
# 反序列化（tensor 用随机数据填充）
# =========================================================================
_FLOAT_DTYPES = {
    "torch.float16", "torch.float32", "torch.float64",
    "torch.bfloat16", "torch.half", "torch.float",
    "torch.float8_e4m3fn", "torch.float8_e5m2",
    "torch.float8_e4m3fnuz", "torch.float8_e5m2fnuz",
}

_INT_DTYPES = {
    "torch.int8", "torch.int16", "torch.int32", "torch.int64",
    "torch.uint8",
}


def _make_random_tensor(shape, dtype_str, device):
    """根据元信息生成随机 tensor。"""
    dtype = getattr(torch, dtype_str.replace("torch.", ""))

    if dtype_str in _FLOAT_DTYPES:
        t = torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    elif dtype_str in _INT_DTYPES:
        t = torch.randint(-10, 10, shape, dtype=dtype, device=device)
    elif dtype_str == "torch.bool":
        t = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    else:
        t = torch.zeros(shape, dtype=dtype, device=device)

    return t


def _deserialize_value(val, device_override=None):
    """从序列化结构还原为 Python 对象（tensor 为随机生成）。"""
    if isinstance(val, dict):
        if val.get("__tensor__"):
            device = device_override or val.get("device", "cpu")
            return _make_random_tensor(val["shape"], val["dtype"], device)
        elif val.get("__tuple__"):
            return tuple(_deserialize_value(v, device_override) for v in val["items"])
        elif val.get("__dict__"):
            return {k: _deserialize_value(v, device_override) for k, v in val["items"].items()}
        elif "__repr__" in val:
            return val["__repr__"]
        else:
            return val
    elif isinstance(val, list):
        return [_deserialize_value(v, device_override) for v in val]
    else:
        return val


# =========================================================================
# Print — 打印调用摘要到 stderr
# =========================================================================
def _format_short(val):
    """返回值的简短描述字符串。"""
    if HAS_TORCH and isinstance(val, torch.Tensor):
        return f"Tensor{list(val.shape)} {val.dtype} ({val.device})"
    elif isinstance(val, tuple):
        inner = ", ".join(_format_short(v) for v in val[:4])
        suffix = ", ..." if len(val) > 4 else ""
        return f"({inner}{suffix})"
    elif isinstance(val, list):
        inner = ", ".join(_format_short(v) for v in val[:4])
        suffix = ", ..." if len(val) > 4 else ""
        return f"[{inner}{suffix}]"
    elif isinstance(val, dict):
        return f"dict(len={len(val)})"
    elif isinstance(val, str):
        return repr(val[:50] + "...") if len(val) > 50 else repr(val)
    else:
        r = repr(val)
        return r[:80] + "..." if len(r) > 80 else r


def _print_call(fn_key, named_args, ret):
    """将一次调用以可读格式打印到 stderr。"""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    lines = [f"[func_dump {ts}] {fn_key}("]
    for name, val in named_args.items():
        lines.append(f"    {name} = {_format_short(val)},")
    lines.append(f") -> {_format_short(ret)}")
    print("\n".join(lines), file=sys.stderr, flush=True)


# =========================================================================
# Dump — 序列化到 JSONL 文件
# =========================================================================
def dump_call(fn, args, kwargs, ret, out_path=None, sink=None):
    """将一次函数调用的元信息记录到内存或 JSONL 文件。"""
    fn_key = f"{fn.__module__}.{fn.__qualname__}"
    named_args = _bind_args(fn, args, kwargs)

    if sink is None:
        sink = os.environ.get("FUNC_DUMP_SINK", "memory").strip().lower()
    if sink not in {"memory", "file"}:
        raise ValueError("FUNC_DUMP_SINK must be 'memory' or 'file'")

    serialized_args = {name: _serialize_value(val) for name, val in named_args.items()}
    serialized_ret = _serialize_value(ret)

    if sink == "file":
        if out_path is None:
            dump_dir = _get_dump_dir()
            out_path = dump_dir / "calls.jsonl"
        else:
            out_path = Path(out_path)
            dump_dir = out_path.parent
        dump_dir.mkdir(parents=True, exist_ok=True)

        idx = 0
        if out_path.exists():
            with open(out_path, "r") as f:
                idx = sum(1 for _ in f)
    else:
        idx = len(_memory_records)

    record = {
        "idx": idx,
        "func": fn_key,
        "args": serialized_args,
        "ret": serialized_ret,
        "ts": datetime.now().isoformat(),
    }

    if sink == "file":
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        _memory_records.append(record)
    return record


# =========================================================================
# Load / Replay
# =========================================================================
def load_records(path=None):
    """加载调用记录。path=None 时从内存加载。"""
    if path is None:
        return list(_memory_records)
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def replay_call(record, device_override=None):
    """回放单条调用记录（使用随机 tensor），返回函数执行结果。"""
    fn_key = record["func"]
    fn = _registry.get(fn_key)
    if fn is None:
        raise KeyError(
            f"Function '{fn_key}' not found in registry. "
            f"Available: {list(_registry.keys())}"
        )
    kwargs = {
        name: _deserialize_value(val, device_override)
        for name, val in record["args"].items()
    }
    return fn(**kwargs)


def replay_all(path=None, device_override=None):
    """回放所有记录，返回 [(idx, func_key, result_or_exception), ...]。"""
    records = load_records(path)
    results = []
    for rec in records:
        try:
            ret = replay_call(rec, device_override)
            results.append((rec["idx"], rec["func"], ret))
        except Exception as e:
            results.append((rec["idx"], rec["func"], e))
    return results


# =========================================================================
# Decorator
# =========================================================================
def auto_dump(fn):
    """
    装饰器。根据 FUNC_DUMP_MODE 决定行为:
      - 未设置          : 直接返回原函数，零开销
      - dump            : 每次调用后将元信息写入文件
      - print           : 每次调用后打印摘要到 stderr
      - dump,print/both : 同时执行以上两者
    """
    register(fn)
    modes = _parse_modes()

    if not modes:
        return fn

    do_dump = "dump" in modes
    do_print = "print" in modes

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ret = fn(*args, **kwargs)

        if do_print:
            named_args = _bind_args(fn, args, kwargs)
            _print_call(f"{fn.__module__}.{fn.__qualname__}", named_args, ret)

        if do_dump:
            dump_call(fn, args, kwargs, ret)

        return ret

    return wrapper


# =========================================================================
# CLI
# =========================================================================
def _cli_show(jsonl_path):
    records = load_records(jsonl_path)
    for rec in records:
        ts = rec.get("ts", "")
        fn = rec["func"]
        idx = rec["idx"]
        print(f"[{idx}] {ts}  {fn}(")
        for name, val in rec["args"].items():
            if isinstance(val, dict) and val.get("__tensor__"):
                desc = f"Tensor{val['shape']} {val['dtype']} ({val.get('device', '?')})"
            elif isinstance(val, dict) and val.get("__tuple__"):
                desc = f"tuple(len={len(val['items'])})"
            elif isinstance(val, dict) and "__repr__" in val:
                desc = val["__repr__"]
            else:
                r = repr(val)
                desc = r[:80] + "..." if len(r) > 80 else r
            print(f"      {name}: {desc}")
        ret = rec.get("ret")
        if isinstance(ret, dict) and ret.get("__tensor__"):
            ret_desc = f"Tensor{ret['shape']} {ret['dtype']}"
        else:
            ret_desc = repr(ret)
        print(f"    ) -> {ret_desc}")
        print()


def _cli_replay(jsonl_path, modules, device):
    import importlib
    for mod_name in modules:
        importlib.import_module(mod_name)

    results = replay_all(jsonl_path, device_override=device)
    for idx, fn_key, ret in results:
        if isinstance(ret, Exception):
            print(f"[{idx}] {fn_key}  -> FAIL: {ret}")
        else:
            print(f"[{idx}] {fn_key}  -> OK")


def main():
    import argparse
    parser = argparse.ArgumentParser(prog="func_dump")
    sub = parser.add_subparsers(dest="command")

    p_show = sub.add_parser("show", help="查看录制内容")
    p_show.add_argument("jsonl", help="calls.jsonl 路径")

    p_replay = sub.add_parser("replay", help="回放录制的调用")
    p_replay.add_argument("jsonl", help="calls.jsonl 路径")
    p_replay.add_argument("-m", "--module", action="append", default=[],
                          help="需要导入的模块（可多次使用）")
    p_replay.add_argument("--device", default=None,
                          help="设备覆盖，如 cuda:0, cpu")

    args = parser.parse_args()
    if args.command == "show":
        _cli_show(args.jsonl)
    elif args.command == "replay":
        _cli_replay(args.jsonl, args.module, args.device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
