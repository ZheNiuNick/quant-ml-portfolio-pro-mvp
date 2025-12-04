#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据文件加载器 - 支持从 Hugging Face Datasets 自动下载
"""

import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Hugging Face 数据集配置
HF_DATASET_REPO = "NickNiu/quant-ml-data"
HF_DATASET_FILES = {
    "factor_store": "data/factors/factor_store.parquet",
}

def download_from_hf(
    filename: str,
    local_path: Path,
    repo_id: str = HF_DATASET_REPO,
    force_download: bool = False
) -> bool:
    """
    从 Hugging Face Datasets 下载文件
    
    Args:
        filename: 数据集中的文件路径
        local_path: 本地保存路径
        repo_id: Hugging Face 数据集仓库 ID
        force_download: 是否强制重新下载
    
    Returns:
        是否下载成功
    """
    # 如果文件已存在且不需要强制下载，直接返回
    if local_path.exists() and not force_download:
        logger.info(f"文件已存在，跳过下载: {local_path}")
        return True
    
    try:
        from huggingface_hub import hf_hub_download
        
        logger.info(f"开始从 Hugging Face 下载: {filename}")
        logger.info(f"  数据集: {repo_id}")
        logger.info(f"  保存到: {local_path}")
        
        # 确保目录存在
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 下载文件
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=local_path.parent,
            local_dir_use_symlinks=False,
            force_download=force_download,
        )
        
        # 如果下载路径与目标路径不同，重命名
        if Path(downloaded_path) != local_path:
            Path(downloaded_path).rename(local_path)
        
        logger.info(f"✅ 下载成功: {local_path}")
        return True
        
    except ImportError:
        logger.warning(
            "huggingface_hub 未安装，无法从 Hugging Face 下载文件。"
            "安装方法: pip install huggingface_hub"
        )
        return False
    except Exception as e:
        logger.error(f"❌ 下载失败: {e}")
        return False


def ensure_factor_store(
    local_path: Path,
    auto_download: bool = True
) -> bool:
    """
    确保 factor_store.parquet 文件存在，如果不存在则从 Hugging Face 下载
    
    Args:
        local_path: factor_store.parquet 的本地路径
        auto_download: 是否自动下载（如果文件不存在）
    
    Returns:
        文件是否存在（下载后或原本就存在）
    """
    # 如果文件已存在
    if local_path.exists():
        return True
    
    # 如果文件不存在且允许自动下载
    if auto_download:
        logger.info(f"factor_store.parquet 不存在，尝试从 Hugging Face 下载...")
        return download_from_hf(
            filename=HF_DATASET_FILES["factor_store"],
            local_path=local_path,
        )
    
    return False


def get_factor_store_path(project_root: Path) -> Path:
    """
    获取 factor_store.parquet 的路径，如果不存在则自动下载
    
    Args:
        project_root: 项目根目录
    
    Returns:
        factor_store.parquet 的路径
    """
    local_path = project_root / "data" / "factors" / "factor_store.parquet"
    
    # 尝试自动下载
    ensure_factor_store(local_path, auto_download=True)
    
    return local_path

