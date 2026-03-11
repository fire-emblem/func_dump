# Repository Guidelines（仓库贡献指南）

## 项目结构与模块组织
- `func_dump.py`：库 + CLI；录制函数调用（仅保存 tensor 元信息：shape/dtype/stride/device），并可按元信息生成随机 tensor 进行回放。
- `test_func_dump.py`：`pytest` 测试（必要时使用 `unittest.mock`）。
- `func_dumps/`：默认输出目录（运行时创建；可用 `FUNC_DUMP_DIR` 覆盖）。

## 本地开发 / 构建 / 测试命令
- 创建虚拟环境：`python -m venv .venv && source .venv/bin/activate`
- 安装依赖：`python -m pip install -U pytest torch`
  - 运行库本身可不安装 PyTorch（会降级），但测试需要。
- 运行测试：`pytest -v`（或 `pytest test_func_dump.py -v`）
  - 注意：`test_func_dump.py` 当前从 `deep_gemm.testing.func_dump` 导入；若仅在本目录使用，请调整导入或按该包路径放置文件。
- CLI 使用：
  - 帮助：`python func_dump.py --help`
  - 查看录制：`python func_dump.py show ./func_dumps/calls.jsonl`
  - 回放录制：`python func_dump.py replay ./func_dumps/calls.jsonl -m your.module --device cpu`
  - 典型录制：`FUNC_DUMP_MODE=both FUNC_DUMP_DIR=./func_dumps python your_script.py`

## 代码风格与命名
- Python 4 空格缩进；尽量符合 PEP 8。
- 使用 `snake_case`；内部辅助函数以 `_` 前缀。
- 保持 `torch` 为可选依赖：导入失败不应影响基础功能。

## 测试约定
- 框架：`pytest`；命名：`test_*` / `Test*`。
- 断言优先验证 shape/dtype/device 等确定性信息，避免依赖随机值。

## 提交与 PR
- 本目录未包含 `.git/` 历史，暂无既定提交风格；建议使用 Conventional Commits（如 `fix: ...` / `test: ...` / `docs: ...`）。
- PR 请包含：改动说明、验证方式（例如 `pytest -v`）、以及必要的 CLI 输出或样例 `calls.jsonl` 片段。

## 安全与配置
- Dump 为 JSONL；对不可序列化值会写入 `repr()`，避免录制密钥/隐私数据。
- 环境变量：`FUNC_DUMP_MODE=dump|print|both`，`FUNC_DUMP_DIR=./path/to/output`，`FUNC_DUMP_SINK=memory|file`
