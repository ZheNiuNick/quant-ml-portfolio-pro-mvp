#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
获取前100市值的美股股票列表
使用yfinance批量接口，代码简洁高效
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

def get_sp500_tickers():
    """从CSV文件获取S&P500股票列表"""
    csv_file = Path("s&p500.csv")
    
    if not csv_file.exists():
        print(f"[ERROR] 找不到S&P500 CSV文件: {csv_file}")
        print(f"       请确保文件存在: s&p500.csv")
        return None
    
    try:
        print(f"[1] 从CSV文件读取S&P500成分股: {csv_file.name}")
        df = pd.read_csv(csv_file)
        
        # 检查Symbol列
        if "Symbol" not in df.columns:
            print(f"[ERROR] CSV文件缺少'Symbol'列")
            print(f"       当前列: {list(df.columns)}")
            return None
        
        tickers = df["Symbol"].tolist()
        # 清理ticker符号（处理BRK.B -> BRK-B等，去除空格和NaN）
        tickers = [str(t).replace(".", "-").strip() for t in tickers if pd.notna(t) and str(t).strip()]
        # 去重（保持顺序）
        tickers = list(dict.fromkeys(tickers))
        
        print(f"[OK] 从CSV文件获取到 {len(tickers)} 只S&P500股票")
        return tickers
    except Exception as e:
        print(f"[ERROR] 读取CSV文件失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_top100_market_cap_stocks():
    """
    获取前100市值的美股股票
    使用yfinance批量接口获取市值
    """
    print("="*60)
    print("获取前100市值美股股票（批量API）")
    print("="*60)
    
    # 步骤1：获取S&P500列表
    tickers = get_sp500_tickers()
    if not tickers:
        print("[ERROR] 无法获取S&P500列表")
        return None
    
    # 步骤2：使用yfinance批量获取市值
    print(f"\n[2] 批量获取 {len(tickers)} 只股票的市值信息...")
    print("    （使用yfinance批量接口，预计1-2分钟）")
    
    market_caps = {}
    
    # 使用yfinance的Tickers（复数）批量获取
    # 分批处理，避免一次性请求太多
    batch_size = 50
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        print(f"  批次 {batch_num}/{total_batches}: 处理 {len(batch)} 只股票...", end=" ", flush=True)
        
        try:
            # 使用Tickers批量获取（更高效）
            tickers_obj = yf.Tickers(" ".join(batch))
            
            for ticker in batch:
                try:
                    stock = tickers_obj.tickers.get(ticker)
                    if stock:
                        info = stock.info
                        if 'marketCap' in info and info['marketCap'] and info['marketCap'] > 0:
                            market_caps[ticker] = info['marketCap']
                except:
                    # 如果批量获取失败，尝试单独获取
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        if 'marketCap' in info and info['marketCap'] and info['marketCap'] > 0:
                            market_caps[ticker] = info['marketCap']
                    except:
                        pass
        except Exception as e:
            # 如果批量失败，逐个获取
            print(f"\n    批量获取失败，改为逐个获取...", end=" ", flush=True)
            for ticker in batch:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    if 'marketCap' in info and info['marketCap'] and info['marketCap'] > 0:
                        market_caps[ticker] = info['marketCap']
                except:
                    pass
        
        print(f"✓ (已获取 {len(market_caps)} 只)")
    
    print(f"\n[OK] 成功获取 {len(market_caps)} 只股票的市值")
    
    if len(market_caps) < 100:
        print(f"[WARN] 只获取到 {len(market_caps)} 只股票的市值，少于100只")
        print(f"      将使用所有可用的股票")
    
    # 步骤3：按市值排序，取前100
    sorted_tickers = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)
    top100 = [ticker for ticker, _ in sorted_tickers[:100]]
    
    print(f"\n[3] 前100市值股票结果:")
    print(f"   总股票数: {len(top100)}")
    if len(sorted_tickers) > 0:
        max_mcap = sorted_tickers[0][1]
        min_mcap = sorted_tickers[min(99, len(sorted_tickers)-1)][1]
        print(f"   市值范围: ${max_mcap/1e12:.2f}T - ${min_mcap/1e9:.2f}B")
        print(f"\n   前10只股票:")
        for i, (ticker, mcap) in enumerate(sorted_tickers[:10], 1):
            print(f"     {i:2d}. {ticker:6s} - ${mcap/1e12:.2f}T")
    
    # 保存到文件
    output_file = Path("data/top100_stocks.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for ticker in top100:
            f.write(f"{ticker}\n")
    
    print(f"\n[OK] 股票列表已保存: {output_file}")
    
    # 也保存为JSON格式（包含市值信息）
    output_json = Path("data/top100_stocks.json")
    data = {
        "tickers": top100,
        "market_caps": {ticker: mcap for ticker, mcap in sorted_tickers[:100]},
        "total": len(top100),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "source": "yfinance_batch_api"
    }
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[OK] 详细信息已保存: {output_json}")
    
    return top100

if __name__ == "__main__":
    tickers = get_top100_market_cap_stocks()
    if tickers:
        print(f"\n✅ 完成！共 {len(tickers)} 只股票")
    else:
        print("\n❌ 获取失败，请检查网络连接或重试")

