#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置模块 - 导出统一的路径管理功能
"""

from src.config.path import (
    ROOT_DIR,
    SETTINGS_FILE,
    CONFIG_DIR,
    DATA_DIR,
    DATA_PROCESSED_DIR,
    DATA_FACTORS_DIR,
    DATA_META_DIR,
    OUTPUT_DIR,
    OUTPUT_BACKTESTS_DIR,
    OUTPUT_PORTFOLIOS_DIR,
    OUTPUT_REPORTS_DIR,
    OUTPUT_MODELS_DIR,
    OUTPUT_IBKR_DATA_DIR,
    DUCKDB_DIR,
    get_path,
    ensure_dir,
    get_root_dir,
    find_project_root,
)

__all__ = [
    'ROOT_DIR',
    'SETTINGS_FILE',
    'CONFIG_DIR',
    'DATA_DIR',
    'DATA_PROCESSED_DIR',
    'DATA_FACTORS_DIR',
    'DATA_META_DIR',
    'OUTPUT_DIR',
    'OUTPUT_BACKTESTS_DIR',
    'OUTPUT_PORTFOLIOS_DIR',
    'OUTPUT_REPORTS_DIR',
    'OUTPUT_MODELS_DIR',
    'OUTPUT_IBKR_DATA_DIR',
    'DUCKDB_DIR',
    'get_path',
    'ensure_dir',
    'get_root_dir',
    'find_project_root',
]

