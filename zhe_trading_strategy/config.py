#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dashboard 配置文件
统一管理 API 基础地址和配置
"""

import os
from pathlib import Path

# API 基础地址 - 从环境变量读取，如果没有则使用相对路径
API_BASE_URL = os.getenv('API_BASE_URL', '')  # 空字符串表示使用相对路径

# IBKR 连接配置
IBKR_CONFIG = {
    'host': os.getenv('IBKR_HOST', '127.0.0.1'),
    'port': int(os.getenv('IBKR_PORT', '7497')),  # 7497=Paper, 7496=Live
    'client_id': int(os.getenv('IBKR_CLIENT_ID', '777')),
    'enabled': os.getenv('IBKR_ENABLED', 'false').lower() == 'true'
}

