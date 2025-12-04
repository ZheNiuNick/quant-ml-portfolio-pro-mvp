#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一项目路径管理模块

所有路径相关的配置都集中在这里，确保跨平台和不同环境下的路径一致性。

使用方法：
    from src.config.path import ROOT_DIR, SETTINGS_FILE, OUTPUT_DIR, get_path
    
    # 使用预定义的路径
    settings = load_settings(str(SETTINGS_FILE))
    
    # 将相对路径转换为绝对路径
    data_file = get_path("data/processed/prices.parquet")
"""

import os
import sys
from pathlib import Path
from typing import Optional


def find_project_root(marker_files: Optional[list] = None) -> Path:
    """
    智能查找项目根目录
    
    查找策略（按优先级）：
    1. 检查环境变量 QUANT_ML_ROOT（如果设置）
    2. 从当前文件位置向上查找，直到找到包含 marker_files 的目录
    3. 从当前工作目录向上查找
    4. 从调用者文件位置向上查找（如果可用）
    
    Args:
        marker_files: 标记文件列表，用于识别项目根目录
                     默认: ["src", "config", "config/settings.yaml"]
    
    Returns:
        项目根目录的 Path 对象
    
    Raises:
        FileNotFoundError: 如果找不到项目根目录
    """
    if marker_files is None:
        marker_files = ["src", "config"]
    
    # 策略1: 检查环境变量
    env_root = os.getenv("QUANT_ML_ROOT")
    if env_root:
        root = Path(env_root).resolve()
        if root.exists() and all((root / marker).exists() for marker in marker_files):
            return root
    
    # 策略2: 从当前文件位置向上查找
    # 当前文件在 src/config/path.py，所以项目根目录应该是 current_file.parent.parent
    current_file = Path(__file__).resolve()
    for parent in [current_file.parent.parent, current_file.parent.parent.parent]:
        if parent.exists() and all((parent / marker).exists() for marker in marker_files):
            settings_file = parent / "config" / "settings.yaml"
            if settings_file.exists():
                return parent
    
    # 策略3: 从当前工作目录向上查找
    cwd = Path.cwd().resolve()
    for parent in [cwd, cwd.parent, cwd.parent.parent, cwd.parent.parent.parent]:
        if parent.exists() and all((parent / marker).exists() for marker in marker_files):
            settings_file = parent / "config" / "settings.yaml"
            if settings_file.exists():
                return parent
    
    # 策略4: 从调用栈中查找（适用于脚本文件）
    import inspect
    try:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_file = frame.f_back.f_globals.get('__file__', '')
            if caller_file:
                caller_path = Path(caller_file).resolve()
                if caller_path.exists():
                    for parent in [caller_path.parent, caller_path.parent.parent, caller_path.parent.parent.parent]:
                        if parent.exists() and all((parent / marker).exists() for marker in marker_files):
                            settings_file = parent / "config" / "settings.yaml"
                            if settings_file.exists():
                                return parent
    except Exception:
        pass
    
    # 如果都找不到，尝试最后一个方法：从 sys.path 中查找
    for path_str in sys.path:
        if path_str and path_str not in ['', '.']:
            try:
                path = Path(path_str).resolve()
                if path.exists() and all((path / marker).exists() for marker in marker_files):
                    settings_file = path / "config" / "settings.yaml"
                    if settings_file.exists():
                        return path
            except Exception:
                continue
    
    # 如果都找不到，抛出异常
    raise FileNotFoundError(
        f"无法找到项目根目录。\n"
        f"当前工作目录: {Path.cwd()}\n"
        f"当前文件位置: {Path(__file__).resolve()}\n"
        f"请确保项目根目录包含以下文件/目录: {marker_files}\n"
        f"或者设置环境变量 QUANT_ML_ROOT 指向项目根目录。"
    )


# 项目根目录（单例模式，只查找一次）
_ROOT_DIR: Optional[Path] = None


def get_root_dir() -> Path:
    """获取项目根目录（单例）"""
    global _ROOT_DIR
    if _ROOT_DIR is None:
        _ROOT_DIR = find_project_root()
    return _ROOT_DIR


# 导出项目根目录
ROOT_DIR = get_root_dir()

# ============================================================================
# 常用路径（相对于项目根目录）
# ============================================================================

# 配置路径
CONFIG_DIR = ROOT_DIR / "config"
SETTINGS_FILE = CONFIG_DIR / "settings.yaml"

# 数据路径
DATA_DIR = ROOT_DIR / "data"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_FACTORS_DIR = DATA_DIR / "factors"
DATA_META_DIR = DATA_DIR / "meta"

# 输出路径
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_BACKTESTS_DIR = OUTPUT_DIR / "backtests"
OUTPUT_PORTFOLIOS_DIR = OUTPUT_DIR / "portfolios"
OUTPUT_REPORTS_DIR = OUTPUT_DIR / "reports"
OUTPUT_MODELS_DIR = OUTPUT_DIR / "models"
OUTPUT_IBKR_DATA_DIR = OUTPUT_DIR / "ibkr_data"

# DuckDB 路径
DUCKDB_DIR = ROOT_DIR / "duckdb"

# ============================================================================
# 辅助函数
# ============================================================================

def get_path(relative_path: str, base: Optional[Path] = None) -> Path:
    """
    将相对路径转换为绝对路径（相对于项目根目录或指定基础路径）
    
    Args:
        relative_path: 相对路径字符串
        base: 基础路径，默认为项目根目录
    
    Returns:
        绝对路径的 Path 对象
    """
    if base is None:
        base = ROOT_DIR
    
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def ensure_dir(path: Path) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
    
    Returns:
        路径对象
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# 验证路径存在性
# ============================================================================

def validate_paths():
    """验证关键路径是否存在"""
    errors = []
    
    if not SETTINGS_FILE.exists():
        errors.append(f"配置文件不存在: {SETTINGS_FILE}")
    
    if not CONFIG_DIR.exists():
        errors.append(f"配置目录不存在: {CONFIG_DIR}")
    
    if not ROOT_DIR.exists():
        errors.append(f"项目根目录不存在: {ROOT_DIR}")
    
    if errors:
        raise FileNotFoundError("\n".join(errors))
    
    return True


# 初始化时验证（但不要因为验证失败就崩溃，只警告）
try:
    validate_paths()
except FileNotFoundError as e:
    import warnings
    warnings.warn(f"路径验证失败: {e}", UserWarning)

