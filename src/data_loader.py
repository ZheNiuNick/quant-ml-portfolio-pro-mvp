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
        print(f"[INFO] 文件已存在，跳过下载: {local_path}")
        return True
    
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"[INFO] 开始从 Hugging Face 下载: {filename}")
        print(f"[INFO]   数据集: {repo_id}")
        print(f"[INFO]   保存到: {local_path}")
        
        # 确保目录存在
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 下载文件
        # hf_hub_download 会将文件下载到 local_dir/filename，保留目录结构
        # 例如：filename="data/factors/factor_store.parquet" 会下载到 local_dir/data/factors/factor_store.parquet
        # 我们需要将 local_dir 设置为项目根目录
        # 从 local_path 推断项目根目录：如果路径包含 "data/factors"，则项目根目录是上三级
        # local_path = /app/data/factors/factor_store.parquet
        # project_root 应该是 /app
        if "data/factors" in str(local_path) or str(local_path).endswith("factors/factor_store.parquet"):
            # 从 /app/data/factors/factor_store.parquet 推断 /app
            project_root = local_path.parent.parent.parent
        else:
            # 否则使用 local_path 的父目录
            project_root = local_path.parent
        
        print(f"[INFO] 推断的项目根目录: {project_root}")
        print(f"[INFO] 下载后预期路径: {project_root / filename}")
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(project_root),
            local_dir_use_symlinks=False,
            force_download=force_download,
        )
        
        downloaded_path_obj = Path(downloaded_path)
        print(f"[INFO] 实际下载路径: {downloaded_path_obj}")
        print(f"[INFO] 目标路径: {local_path}")
        
        # 如果下载路径与目标路径不同，需要移动文件
        if downloaded_path_obj != local_path:
            if downloaded_path_obj.exists():
                # 确保目标目录存在
                local_path.parent.mkdir(parents=True, exist_ok=True)
                # 如果目标文件已存在，先删除
                if local_path.exists():
                    local_path.unlink()
                # 移动文件到目标位置
                downloaded_path_obj.rename(local_path)
                print(f"[INFO] 文件已移动到: {local_path}")
            else:
                print(f"[ERROR] 下载的文件不存在: {downloaded_path_obj}")
                return False
        
        if local_path.exists():
            file_size = local_path.stat().st_size / (1024 * 1024)  # MB
            print(f"[INFO] 文件已下载，大小: {file_size:.2f} MB")
            
            # 验证文件是否为有效的 Parquet 文件
            try:
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(local_path)
                num_rows = parquet_file.metadata.num_rows
                num_columns = len(parquet_file.schema)
                print(f"[INFO] ✅ Parquet 文件验证成功: {num_rows} 行, {num_columns} 列")
                print(f"[INFO] ✅ 下载成功: {local_path} ({file_size:.2f} MB)")
                return True
            except Exception as e:
                print(f"[ERROR] ❌ Parquet 文件验证失败: {e}")
                print(f"[ERROR] 文件可能损坏，删除并重试...")
                local_path.unlink()
                return False
        else:
            print(f"[ERROR] 下载后文件不存在: {local_path}")
            return False
        
    except ImportError:
        print("[WARN] huggingface_hub 未安装，无法从 Hugging Face 下载文件。")
        print("[WARN] 安装方法: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"[ERROR] ❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
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
        print(f"[INFO] factor_store.parquet 不存在，尝试从 Hugging Face 下载...")
        print(f"[INFO] 文件路径: {local_path}")
        result = download_from_hf(
            filename=HF_DATASET_FILES["factor_store"],
            local_path=local_path,
        )
        print(f"[INFO] 下载结果: {result}, 文件存在: {local_path.exists()}")
        return result
    
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

