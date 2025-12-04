#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上传 factor_store.parquet 到 Hugging Face Datasets
使用 Python API，不需要 CLI 命令
"""

from pathlib import Path
from huggingface_hub import HfApi
import os
import sys

# 使用统一的路径管理
sys.path.insert(0, str(Path(__file__).parent))
from src.config.path import DATA_FACTORS_DIR, get_path

# 配置
REPO_ID = "NickNiu/quant-ml-data"
REPO_TYPE = "dataset"
FILE_TO_UPLOAD = "data/factors/factor_store.parquet"
HF_FILENAME = "data/factors/factor_store.parquet"

def upload_factor_store(token=None):
    """上传 factor_store.parquet 到 Hugging Face"""
    
    # 检查文件是否存在
    file_path = get_path(FILE_TO_UPLOAD, DATA_FACTORS_DIR)
    if not file_path.exists():
        print(f"❌ 错误：文件不存在: {file_path}")
        print(f"   当前工作目录: {Path.cwd()}")
        return False
    
    file_size = file_path.stat().st_size / (1024 * 1024)  # MB
    print(f"📁 文件: {file_path}")
    print(f"📦 大小: {file_size:.2f} MB")
    
    # 获取访问令牌
    if not token:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    if not token:
        print("\n" + "=" * 60)
        print("🔐 需要 Hugging Face 访问令牌")
        print("=" * 60)
        print("\n获取令牌的步骤:")
        print("1. 访问: https://huggingface.co/settings/tokens")
        print("2. 点击 'New token'")
        print("3. 选择权限: 'Write'")
        print("4. 复制生成的令牌")
        print("\n使用方法:")
        print("   方法 1: 设置环境变量")
        print("   export HF_TOKEN=your_token_here")
        print("   python upload_to_hf.py")
        print("\n   方法 2: 作为参数传入")
        print("   python upload_to_hf.py your_token_here")
        print("\n   方法 3: 交互式输入（运行脚本后会提示）")
        print("=" * 60)
        
        # 交互式输入
        try:
            token = input("\n请输入你的 Hugging Face 令牌: ").strip()
            if not token:
                print("❌ 未输入令牌，退出")
                return False
        except (KeyboardInterrupt, EOFError):
            print("\n\n❌ 已取消")
            return False
    
    print(f"\n🚀 开始上传到: {REPO_ID}")
    print(f"   文件路径: {HF_FILENAME}")
    print(f"   文件大小: {file_size:.2f} MB")
    print(f"   这可能需要几分钟，请耐心等待...")
    
    try:
        api = HfApi(token=token)
        
        # 上传文件
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=HF_FILENAME,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"Upload factor_store.parquet ({file_size:.2f} MB)",
        )
        
        print(f"\n✅ 上传成功！")
        print(f"   数据集地址: https://huggingface.co/datasets/{REPO_ID}")
        print(f"\n📝 下一步:")
        print(f"   1. 提交代码更改: git add . && git commit -m 'Add HF download support' && git push")
        print(f"   2. Railway 会自动重新部署")
        print(f"   3. 应用会在首次访问 Factor Diagnostics 时自动下载文件")
        return True
        
    except Exception as e:
        print(f"\n❌ 上传失败: {e}")
        print(f"\n可能的原因:")
        print(f"   1. 访问令牌无效或没有 'write' 权限")
        print(f"   2. 数据集仓库不存在（需要先在 Hugging Face 创建）")
        print(f"   3. 网络连接问题")
        print(f"\n解决方案:")
        print(f"   1. 检查令牌权限: https://huggingface.co/settings/tokens")
        print(f"   2. 创建数据集: https://huggingface.co/new-dataset")
        print(f"      数据集名称: quant-ml-data")
        print(f"      可见性: Public")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Hugging Face Datasets 文件上传工具")
    print("=" * 60)
    print()
    
    # 支持从命令行参数传入令牌
    token = None
    if len(sys.argv) > 1:
        token = sys.argv[1]
        print(f"✅ 使用命令行参数中的令牌")
        print()
    
    success = upload_factor_store(token=token)
    
    if not success:
        print("\n" + "=" * 60)
        print("请检查错误信息并重试")
        print("=" * 60)
