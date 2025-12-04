#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查 Hugging Face 数据集状态
"""

from huggingface_hub import list_repo_files, hf_hub_download
from pathlib import Path

REPO_ID = "NickNiu/quant-ml-data"
REPO_TYPE = "dataset"
FILE_PATH = "data/factors/factor_store.parquet"

print("=" * 60)
print("检查 Hugging Face 数据集状态")
print("=" * 60)

# 1. 列出所有文件
print("\n1. 数据集中的文件:")
try:
    files = list(list_repo_files(REPO_ID, repo_type=REPO_TYPE))
    if files:
        for f in files:
            print(f"   ✓ {f}")
    else:
        print("   ✗ 数据集为空")
except Exception as e:
    print(f"   ✗ 错误: {e}")

# 2. 检查目标文件是否存在
print(f"\n2. 检查目标文件: {FILE_PATH}")
if FILE_PATH in files:
    print(f"   ✓ 文件存在于数据集中")
else:
    print(f"   ✗ 文件不存在于数据集中")
    print(f"   请运行: python upload_to_hf.py")

# 3. 测试下载
print(f"\n3. 测试下载功能:")
try:
    test_path = Path("/tmp/test_factor_store.parquet")
    if test_path.exists():
        test_path.unlink()
    
    print(f"   开始下载到: {test_path}")
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=FILE_PATH,
        local_dir=str(test_path.parent),
        local_dir_use_symlinks=False,
    )
    
    # 检查实际文件位置
    actual_file = Path(downloaded)
    if actual_file.exists():
        size_mb = actual_file.stat().st_size / (1024 * 1024)
        print(f"   ✓ 下载成功: {actual_file}")
        print(f"   ✓ 文件大小: {size_mb:.2f} MB")
        # 清理测试文件
        actual_file.unlink()
        if test_path.exists():
            test_path.unlink()
    else:
        print(f"   ✗ 下载后文件不存在: {downloaded}")
        
except Exception as e:
    print(f"   ✗ 下载失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)

