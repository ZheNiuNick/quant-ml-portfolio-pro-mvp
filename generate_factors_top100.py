#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为市值前100股票生成因子（内存优化版本）
价格数据使用已下载的全S&P500数据，但因子生成仅处理市值前100只股票
"""

import argparse
from pathlib import Path
import pandas as pd
import yaml
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.factor_engine import build_and_evaluate, load_settings
from src.data_pipeline import fetch_daily_prices, load_to_duckdb, sanity_checks

def load_top100_tickers():
    """从文件加载市值前100股票列表（按市值从大到小排序）"""
    txt_file = Path("data/top100_stocks.txt")
    if txt_file.exists():
        try:
            with open(txt_file, 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
            print(f"[OK] 从文件加载 {len(tickers)} 只市值前100股票（按市值排序）")
            return tickers
        except Exception as e:
            print(f"[ERROR] 读取top100文件失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"[ERROR] 找不到top100股票文件: {txt_file}")
        print(f"       请确保文件存在: data/top100_stocks.txt")
        return None

def load_sp500_tickers():
    """从CSV文件加载全S&P500股票列表（用于获取价格数据）"""
    csv_file = Path("s&p500.csv")
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            if "Symbol" not in df.columns:
                print(f"[ERROR] CSV文件缺少'Symbol'列")
                print(f"       当前列: {list(df.columns)}")
                return None
            
            tickers = df["Symbol"].tolist()
            # 清理ticker符号（处理BRK.B -> BRK-B等，去除空格和NaN）
            tickers = [str(t).replace(".", "-").strip() for t in tickers if pd.notna(t) and str(t).strip()]
            # 去重（保持顺序）
            tickers = list(dict.fromkeys(tickers))
            print(f"[OK] 从CSV文件加载 {len(tickers)} 只S&P500股票")
            return tickers
        except Exception as e:
            print(f"[ERROR] 读取CSV文件失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"[ERROR] 找不到S&P500 CSV文件: {csv_file}")
        print(f"       请确保文件存在: s&p500.csv")
        return None

def main():
    parser = argparse.ArgumentParser(description="为市值前100股票生成因子（内存优化版本）")
    parser.add_argument("--fetch-prices", action="store_true", help="先获取价格数据（全S&P500）")
    parser.add_argument("--build-factors", action="store_true", help="生成因子（仅市值前100股票）")
    parser.add_argument("--all", action="store_true", help="执行完整流程（获取价格+生成因子）")
    args = parser.parse_args()
    
    if not (args.fetch_prices or args.build_factors or args.all):
        parser.print_help()
        return
    
    # 加载配置
    cfg = load_settings()
    
    # 步骤1：获取价格数据（使用全S&P500）
    if args.fetch_prices or args.all:
        print("\n[步骤1] 获取价格数据（全S&P500）...")
        # 加载全S&P500股票列表用于获取价格数据
        sp500_tickers = load_sp500_tickers()
        if not sp500_tickers:
            return
        
        try:
            start = cfg["data"]["start"]
            end = cfg["data"]["end"]
            output_path = Path(cfg["paths"]["prices_parquet"])
            
            print(f"  获取 {len(sp500_tickers)} 只股票的价格数据...")
            print(f"  日期范围: {start} 到 {end}")
            
            # 使用yfinance获取US市场数据
            df = fetch_daily_prices(sp500_tickers, start, end, output_path, region="us")
            
            # 数据检查
            sanity_checks(df)
            
            # 加载到DuckDB
            load_to_duckdb(output_path, cfg["database"]["duckdb_path"])
            
            print("[OK] 价格数据获取完成")
        except Exception as e:
            print(f"[ERROR] 价格数据获取失败: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # 步骤2：生成因子（仅使用市值前100股票，节省内存）
    if args.build_factors or args.all:
        print("\n[步骤2] 生成因子（市值前100股票，内存优化）...")
        # 加载市值前100股票列表用于生成因子
        top100_tickers = load_top100_tickers()
        if not top100_tickers:
            return
        
        # 更新配置中的股票列表（仅用于因子生成）
        if "data" not in cfg:
            cfg["data"] = {}
        cfg["data"]["universe"] = top100_tickers  # 只使用市值前100只股票生成因子
        cfg["data"]["region"] = "us"
        cfg["data"]["instruments"] = "top100"
        
        print("\n" + "="*60)
        print(f"为市值前100股票生成因子（内存优化版本）")
        print("="*60)
        print(f"股票数量: {len(top100_tickers)}（按市值排序）")
        print(f"日期范围: {cfg['data']['start']} 到 {cfg['data']['end']}")
        print(f"注意: 价格数据来自全S&P500，但因子仅生成市值前100只股票")
        print("="*60)
        
        try:
            build_and_evaluate(cfg)
            print("\n[OK] 因子生成完成！")
        except Exception as e:
            print(f"[ERROR] 因子生成失败: {e}")
            import traceback
            traceback.print_exc()
            return

if __name__ == "__main__":
    main()

