# 统一路径管理更新说明

## 概述

已完成项目路径管理的统一化，所有路径相关的配置现在都集中在 `src/config/path.py` 中管理。这解决了跨平台和不同环境下的路径问题。

## 主要变更

### 1. 新增文件

#### `src/config/path.py`
- 统一的项目路径管理模块
- 提供智能的项目根目录查找功能
- 支持环境变量 `QUANT_ML_ROOT` 覆盖
- 导出所有常用路径常量

#### `src/config/__init__.py`
- 导出路径管理模块的所有公共接口

### 2. 修改的文件

#### 核心模块 (`src/`)
- ✅ `src/backtest.py` - 使用统一路径管理
- ✅ `src/modeling.py` - 使用统一路径管理
- ✅ `src/factor_engine.py` - 使用统一路径管理
- ✅ `src/optimizer.py` - 使用统一路径管理

#### Web 应用 (`zhe_trading_strategy/`)
- ✅ `zhe_trading_strategy/app.py` - 替换 `find_project_root()` 函数，使用统一路径管理

#### 脚本 (`scripts/`)
- ✅ `scripts/daily_update.py` - 使用统一路径管理
- ✅ `scripts/fetch_ibkr_data.py` - 使用统一路径管理
- ✅ `scripts/generate_ic_ir_for_existing_factors.py` - 使用统一路径管理
- ✅ `scripts/daily_factor_update.py` - 使用统一路径管理
- ✅ `scripts/filter_factors_to_top100.py` - 使用统一路径管理
- ✅ `scripts/filter_weights_to_top100.py` - 使用统一路径管理

#### 其他脚本
- ✅ `generate_factor_precomputed.py` - 使用统一路径管理

## 使用方法

### 基本用法

```python
from src.config.path import ROOT_DIR, SETTINGS_FILE, OUTPUT_DIR, get_path

# 使用预定义的路径
settings = load_settings(str(SETTINGS_FILE))

# 将相对路径转换为绝对路径
data_file = get_path("data/processed/prices.parquet")
```

### 可用路径常量

- `ROOT_DIR` - 项目根目录
- `SETTINGS_FILE` - 配置文件路径 (`config/settings.yaml`)
- `CONFIG_DIR` - 配置目录
- `DATA_DIR` - 数据目录
- `DATA_PROCESSED_DIR` - 处理后的数据目录
- `DATA_FACTORS_DIR` - 因子数据目录
- `DATA_META_DIR` - 元数据目录
- `OUTPUT_DIR` - 输出目录
- `OUTPUT_BACKTESTS_DIR` - 回测结果目录
- `OUTPUT_PORTFOLIOS_DIR` - 组合权重目录
- `OUTPUT_REPORTS_DIR` - 报告目录
- `OUTPUT_MODELS_DIR` - 模型目录
- `OUTPUT_IBKR_DATA_DIR` - IBKR 数据目录
- `DUCKDB_DIR` - DuckDB 数据库目录

### 辅助函数

#### `get_path(relative_path, base=None)`
将相对路径转换为绝对路径（相对于项目根目录或指定基础路径）

```python
from src.config.path import get_path, DATA_FACTORS_DIR

# 相对于项目根目录
path1 = get_path("data/factors/factor_store.parquet")

# 相对于指定基础路径
path2 = get_path("factor_store.parquet", DATA_FACTORS_DIR)
```

#### `ensure_dir(path)`
确保目录存在，如果不存在则创建

```python
from src.config.path import ensure_dir, OUTPUT_DIR

output_path = ensure_dir(OUTPUT_DIR / "models")
```

### 环境变量支持

可以通过设置环境变量 `QUANT_ML_ROOT` 来覆盖项目根目录的自动查找：

```bash
export QUANT_ML_ROOT=/path/to/project/root
```

## 优势

1. ✅ **统一管理** - 所有路径在一个地方定义
2. ✅ **跨平台** - 使用 `pathlib.Path` 处理路径分隔符
3. ✅ **环境变量支持** - 可通过环境变量覆盖
4. ✅ **易于维护** - 修改路径只需改一个文件
5. ✅ **类型安全** - 使用 `Path` 对象而非字符串拼接
6. ✅ **智能查找** - 自动查找项目根目录，支持多种场景

## 迁移指南

如果你有自定义脚本需要迁移到新的路径管理：

### 旧代码
```python
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
settings_file = project_root / "config" / "settings.yaml"
data_file = project_root / "data" / "processed" / "prices.parquet"
```

### 新代码
```python
from src.config.path import SETTINGS_FILE, get_path
settings_file = SETTINGS_FILE
data_file = get_path("data/processed/prices.parquet")
```

## 测试

可以通过以下命令测试路径管理模块：

```bash
cd /path/to/quant-ml
python -c "from src.config.path import ROOT_DIR, SETTINGS_FILE; print(f'ROOT_DIR: {ROOT_DIR}'); print(f'SETTINGS_FILE exists: {SETTINGS_FILE.exists()}')"
```

## 注意事项

1. 所有使用路径的模块现在都依赖于 `src/config/path.py`
2. 确保在导入路径模块之前，项目根目录已添加到 `sys.path`
3. 路径查找失败时会抛出 `FileNotFoundError`，提供详细的错误信息

## 后续改进建议

1. 可以添加路径验证功能，确保关键目录存在
2. 可以添加路径缓存机制，提高性能
3. 可以考虑添加配置文件中的路径映射功能

