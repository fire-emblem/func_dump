# func_dump.py 使用示例

下面示例均基于当前目录运行。

## 1) 生成录制文件

```bash
FUNC_DUMP_MODE=dump FUNC_DUMP_DIR=./func_dumps python demo_auto_dump.py
```

## 2) 查看录制内容

```bash
python func_dump.py show ./func_dumps/calls.jsonl
```

## 3) 回放录制内容

需要先导入包含被装饰函数的模块，使其注册到 registry（此处用模块文件路径）。

```bash
python func_dump.py replay ./func_dumps/calls.jsonl -m ./demo_auto_dump.py
```

## 4) 同时打印与录制（可选）

```bash
FUNC_DUMP_MODE=both FUNC_DUMP_DIR=./func_dumps python demo_auto_dump.py
```
