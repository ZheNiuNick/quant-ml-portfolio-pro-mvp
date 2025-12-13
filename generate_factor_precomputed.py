#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预计算因子诊断结果，避免在 Web 部署时读取大文件
生成：
- outputs/factor_long_short.json - Long-Short 收益（按因子）
- outputs/factor_corr.json - 因子相关性矩阵
- outputs/factor_exposure.json - 风险暴露（按日期）
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# 使用统一的路径管理
sys.path.insert(0, str(Path(__file__).parent))

from src.config.path import SETTINGS_FILE, OUTPUT_DIR, DATA_FACTORS_DIR, ROOT_DIR, get_path
from src.factor_engine import read_prices, forward_return, load_settings as load_factor_settings

SETTINGS = SETTINGS_FILE
OUTPUT_DIR.mkdir(exist_ok=True)

def generate_long_short_performance():
    """生成 Long-Short 收益数据（按因子）"""
    print("=" * 60)
    print("生成 Long-Short 收益数据...")
    print("=" * 60)
    
    try:
        from src.factor_engine import read_prices, forward_return
        
        # 读取配置
        factor_cfg = load_factor_settings(str(SETTINGS))
        factor_store_rel_path = factor_cfg["paths"].get("factors_store", "data/factors/factor_store.parquet")
        # 如果已经是绝对路径，直接使用；否则基于项目根目录解析
        if Path(factor_store_rel_path).is_absolute():
            factor_store_path = Path(factor_store_rel_path)
        else:
            from src.config.path import ROOT_DIR
            factor_store_path = (ROOT_DIR / factor_store_rel_path).resolve()
        
        if not factor_store_path.exists():
            print(f"❌ 文件不存在: {factor_store_path}")
            return False
        
        print(f"📖 读取因子数据: {factor_store_path}")
        factor_store = pd.read_parquet(factor_store_path)
        
        if not isinstance(factor_store.index, pd.MultiIndex):
            if "date" in factor_store.columns and "ticker" in factor_store.columns:
                factor_store["date"] = pd.to_datetime(factor_store["date"])
                factor_store = factor_store.set_index(["date", "ticker"]).sort_index()
        
        # 读取价格数据
        print("📖 读取价格数据...")
        if "paths" in factor_cfg and "prices_parquet" in factor_cfg["paths"]:
            parquet_path = factor_cfg["paths"]["prices_parquet"]
            factor_cfg["paths"]["prices_parquet"] = str(get_path(parquet_path))
        
        prices = read_prices(factor_cfg)
        if prices is None or len(prices) == 0:
            print("❌ 价格数据不存在或为空")
            return False
        
        # 处理重复索引（如果存在）
        if isinstance(prices.index, pd.MultiIndex):
            prices = prices[~prices.index.duplicated(keep='first')]
            print(f"📊 价格数据去重后: {len(prices)} 行")
        
        # 计算未来收益
        forward_ret = forward_return(prices, horizon=1)
        
        # 处理 forward_ret 的重复索引
        if isinstance(forward_ret.index, pd.MultiIndex):
            forward_ret = forward_ret[~forward_ret.index.duplicated(keep='first')]
        
        # 获取所有因子
        factors = [col for col in factor_store.columns if col not in ['date', 'ticker']]
        print(f"📊 处理 {len(factors)} 个因子...")
        
        results = {}
        
        for i, factor_name in enumerate(factors, 1):
            if i % 10 == 0:
                print(f"  处理进度: {i}/{len(factors)}")
            
            try:
                # 获取近12个月的数据
                latest_date = factor_store.index.get_level_values(0).max()
                start_date = latest_date - pd.DateOffset(months=12)
                date_range = factor_store.index.get_level_values(0).unique()
                date_range = date_range[date_range >= start_date]
                
                if len(date_range) == 0:
                    date_range = factor_store.index.get_level_values(0).unique()
                
                dates = []
                long_returns = []
                short_returns = []
                long_short_returns = []
                
                for date in sorted(date_range):
                    date_factors = factor_store.loc[factor_store.index.get_level_values(0) == date, factor_name]
                    # 处理重复索引
                    if isinstance(date_factors.index, pd.MultiIndex):
                        date_factors = date_factors[~date_factors.index.duplicated(keep='first')]
                    
                    date_forward_ret = forward_ret.loc[forward_ret.index.get_level_values(0) == date]
                    # 处理重复索引
                    if isinstance(date_forward_ret.index, pd.MultiIndex):
                        date_forward_ret = date_forward_ret[~date_forward_ret.index.duplicated(keep='first')]
                    
                    aligned = pd.concat([date_factors, date_forward_ret], axis=1).dropna()
                    if len(aligned) < 20:
                        continue
                    
                    aligned = aligned.sort_values(by=aligned.columns[0])
                    n = len(aligned)
                    long_portfolio = aligned.iloc[-n//5:]
                    short_portfolio = aligned.iloc[:n//5]
                    
                    long_ret = long_portfolio.iloc[:, 1].mean()
                    short_ret = short_portfolio.iloc[:, 1].mean()
                    ls_ret = long_ret - short_ret
                    
                    dates.append(date.strftime("%Y-%m-%d"))
                    long_returns.append(float(long_ret))
                    short_returns.append(float(short_ret))
                    long_short_returns.append(float(ls_ret))
                
                if len(dates) > 0:
                    # 计算累计收益
                    long_cum = (1 + pd.Series(long_returns)).cumprod().tolist()
                    short_cum = (1 + pd.Series(short_returns)).cumprod().tolist()
                    ls_cum = (1 + pd.Series(long_short_returns)).cumprod().tolist()
                    
                    # 计算统计指标
                    def calc_stats(returns):
                        returns_series = pd.Series(returns)
                        annual_return = returns_series.mean() * 252
                        sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
                        cum = (1 + returns_series).cumprod()
                        max_dd = (cum / cum.cummax() - 1).min()
                        return {
                            "annual_return": float(annual_return),
                            "sharpe": float(sharpe),
                            "max_dd": float(max_dd)
                        }
                    
                    results[factor_name] = {
                        "dates": dates,
                        "long_returns": long_cum,
                        "short_returns": short_cum,
                        "long_short_returns": ls_cum,
                        "stats": {
                            "long": calc_stats(long_returns),
                            "short": calc_stats(short_returns),
                            "long_short": calc_stats(long_short_returns)
                        }
                    }
            except Exception as e:
                print(f"  ⚠️  因子 {factor_name} 处理失败: {e}")
                continue
        
        # 保存结果
        output_file = OUTPUT_DIR / "factor_long_short.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        file_size = output_file.stat().st_size / (1024 * 1024)
        print(f"✅ 生成成功: {output_file} ({file_size:.2f} MB, {len(results)} 个因子)")
        return True
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_correlation_matrix():
    """生成因子相关性矩阵"""
    print("\n" + "=" * 60)
    print("生成因子相关性矩阵...")
    print("=" * 60)
    
    try:
        factor_cfg = load_factor_settings(str(SETTINGS))
        factor_store_rel_path = factor_cfg["paths"].get("factors_store", "data/factors/factor_store.parquet")
        # 如果已经是绝对路径，直接使用；否则基于项目根目录解析
        if Path(factor_store_rel_path).is_absolute():
            factor_store_path = Path(factor_store_rel_path)
        else:
            factor_store_path = (ROOT_DIR / factor_store_rel_path).resolve()
        
        if not factor_store_path.exists():
            print(f"❌ 文件不存在: {factor_store_path}")
            return False
        
        print(f"📖 读取因子数据: {factor_store_path}")
        factor_store = pd.read_parquet(factor_store_path)
        
        if not isinstance(factor_store.index, pd.MultiIndex):
            if "date" in factor_store.columns and "ticker" in factor_store.columns:
                factor_store["date"] = pd.to_datetime(factor_store["date"])
                factor_store = factor_store.set_index(["date", "ticker"]).sort_index()
        
        # 获取近12个月的数据
        latest_date = factor_store.index.get_level_values(0).max()
        start_date = latest_date - pd.DateOffset(months=12)
        recent_factors = factor_store.loc[factor_store.index.get_level_values(0) >= start_date]
        
        if len(recent_factors) == 0:
            recent_factors = factor_store
        
        # 选择部分因子（限制为50个）
        factors = list(recent_factors.columns)[:50]
        factor_subset = recent_factors[factors]
        
        # 计算相关性矩阵（按日期平均）
        dates = factor_subset.index.get_level_values(0).unique()
        corr_list = []
        
        print(f"📊 计算 {len(factors)} 个因子的相关性...")
        for i, date in enumerate(dates, 1):
            if i % 50 == 0:
                print(f"  处理进度: {i}/{len(dates)}")
            date_factors = factor_subset.loc[factor_subset.index.get_level_values(0) == date]
            if len(date_factors) > 10:
                corr = date_factors.corr(method='pearson')
                corr_list.append(corr)
        
        if len(corr_list) == 0:
            print("❌ 没有足够的数据计算相关性")
            return False
        
        # 平均相关性矩阵
        mean_corr = pd.concat(corr_list).groupby(level=0).mean()
        mean_corr = mean_corr.fillna(0)
        
        # 保存结果
        output_file = OUTPUT_DIR / "factor_corr.json"
        result = {
            "factors": factors,
            "correlation_matrix": mean_corr.values.tolist(),
            "method": "pearson"
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        file_size = output_file.stat().st_size / (1024 * 1024)
        print(f"✅ 生成成功: {output_file} ({file_size:.2f} MB, {len(factors)} 个因子)")
        return True
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_risk_exposure():
    """生成风险暴露数据（按日期）- 基于实际投资组合权重"""
    print("\n" + "=" * 60)
    print("生成风险暴露数据...")
    print("=" * 60)
    
    try:
        # 1. 读取因子数据
        factor_cfg = load_factor_settings(str(SETTINGS))
        factor_store_rel_path = factor_cfg["paths"].get("factors_store", "data/factors/factor_store.parquet")
        if Path(factor_store_rel_path).is_absolute():
            factor_store_path = Path(factor_store_rel_path)
        else:
            factor_store_path = (ROOT_DIR / factor_store_rel_path).resolve()
        
        if not factor_store_path.exists():
            print(f"❌ 文件不存在: {factor_store_path}")
            return False
        
        print(f"📖 读取因子数据: {factor_store_path}")
        factor_store = pd.read_parquet(factor_store_path)
        
        if not isinstance(factor_store.index, pd.MultiIndex):
            if "date" in factor_store.columns and "ticker" in factor_store.columns:
                factor_store["date"] = pd.to_datetime(factor_store["date"])
                factor_store = factor_store.set_index(["date", "ticker"]).sort_index()
        
        # 2. 读取投资组合权重
        portfolio_path_rel = factor_cfg["paths"].get("portfolio_path", "outputs/portfolios/weights.parquet")
        if Path(portfolio_path_rel).is_absolute():
            portfolio_path = Path(portfolio_path_rel)
        else:
            portfolio_path = (ROOT_DIR / portfolio_path_rel).resolve()
        
        if not portfolio_path.exists():
            print(f"⚠️  权重文件不存在: {portfolio_path}")
            print("   将使用所有股票的平均因子值（不推荐）")
            weights_df = None
        else:
            print(f"📖 读取投资组合权重: {portfolio_path}")
            weights_df = pd.read_parquet(portfolio_path)
            weights_df.index = pd.to_datetime(weights_df.index)
            print(f"   权重日期范围: {weights_df.index.min()} 到 {weights_df.index.max()}")
        
        # 获取所有可用日期（最近30个交易日）
        available_dates = pd.to_datetime(factor_store.index.get_level_values(0).unique()).sort_values()
        dates_to_process = available_dates[-30:]  # 只处理最近30个交易日
        
        print(f"📊 处理 {len(dates_to_process)} 个日期...")
        
        results = {}
        
        for i, date_obj in enumerate(dates_to_process, 1):
            if i % 10 == 0:
                print(f"  处理进度: {i}/{len(dates_to_process)}")
            
            try:
                date_factors = factor_store.loc[factor_store.index.get_level_values(0) == date_obj]
                # 重置索引，只保留 ticker 级别
                if isinstance(date_factors.index, pd.MultiIndex):
                    date_factors = date_factors.reset_index(level='date', drop=True)
                
                # 3. 获取当日投资组合权重
                portfolio_weights = None
                portfolio_tickers = None
                if weights_df is not None:
                    # 查找最接近的日期权重
                    if date_obj in weights_df.index:
                        portfolio_weights_series = weights_df.loc[date_obj].fillna(0.0)
                        portfolio_weights_series = portfolio_weights_series[portfolio_weights_series > 0]
                        if len(portfolio_weights_series) > 0:
                            portfolio_weights = portfolio_weights_series
                            portfolio_tickers = portfolio_weights.index.tolist()
                            # 归一化权重（确保和为1）
                            portfolio_weights = portfolio_weights / portfolio_weights.sum()
                
                # 4. 计算因子暴露度和风险贡献
                portfolio_exposures = {}
                portfolio_risk_contributions = {}
                
                for factor_name in date_factors.columns:
                    # date_factors 已经是该日期的数据，factor_series 是 Series(ticker -> value)
                    factor_series = date_factors[factor_name].dropna()
                    
                    if portfolio_weights is not None and portfolio_tickers:
                        # 方法1: 使用投资组合权重计算因子暴露度（Barra-style）
                        # 只考虑投资组合中的股票
                        portfolio_factor_values = factor_series.reindex(portfolio_tickers).dropna()
                        portfolio_weights_aligned = portfolio_weights.reindex(portfolio_factor_values.index).fillna(0.0)
                        
                        if len(portfolio_factor_values) > 0 and portfolio_weights_aligned.sum() > 0:
                            # 归一化对齐的权重
                            portfolio_weights_aligned = portfolio_weights_aligned / portfolio_weights_aligned.sum()
                            
                            # 计算基准（所有股票的平均值和标准差）
                            benchmark_mean = factor_series.mean()
                            benchmark_std = factor_series.std()
                            
                            # 计算投资组合的因子暴露度（加权平均）
                            portfolio_factor_mean = (portfolio_weights_aligned * portfolio_factor_values).sum()
                            
                            # Barra-style 暴露度：相对于基准的标准化暴露度
                            # 暴露度 = (投资组合因子值 - 基准因子值) / 基准因子标准差
                            if benchmark_std > 0:
                                portfolio_exposure = (portfolio_factor_mean - benchmark_mean) / benchmark_std
                            else:
                                portfolio_exposure = 0.0
                            
                            portfolio_exposures[factor_name] = float(portfolio_exposure)
                            
                            # 计算风险贡献：使用投资组合中股票因子值的加权方差
                            weighted_mean = (portfolio_weights_aligned * portfolio_factor_values).sum()
                            weighted_variance = ((portfolio_weights_aligned * (portfolio_factor_values - weighted_mean) ** 2).sum())
                            portfolio_risk_contributions[factor_name] = weighted_variance
                        else:
                            portfolio_exposures[factor_name] = 0.0
                            portfolio_risk_contributions[factor_name] = 0.0
                    else:
                        # 方法2: 如果没有权重文件，使用所有股票的平均（旧方法）
                        mean_val = factor_series.mean()
                        std_val = factor_series.std()
                        if std_val > 0:
                            normalized = (factor_series - mean_val) / std_val
                            portfolio_exposures[factor_name] = float(normalized.median())
                        else:
                            portfolio_exposures[factor_name] = 0.0
                        portfolio_risk_contributions[factor_name] = float(factor_series.var())
                
                # 5. 归一化风险贡献
                total_risk = sum(portfolio_risk_contributions.values())
                if total_risk > 0:
                    for factor_name in portfolio_risk_contributions:
                        portfolio_risk_contributions[factor_name] = portfolio_risk_contributions[factor_name] / total_risk
                
                # 6. 排序（按风险贡献，取前50个）
                sorted_factors = sorted(portfolio_risk_contributions.items(), key=lambda x: x[1], reverse=True)[:50]
                
                results[date_obj.strftime("%Y-%m-%d")] = {
                    "factors": [f[0] for f in sorted_factors],
                    "exposures": [round(portfolio_exposures.get(f[0], 0.0), 4) for f in sorted_factors],
                    "risk_contributions": [round(f[1] * 100, 2) for f in sorted_factors]
                }
            except Exception as e:
                print(f"  ⚠️  日期 {date_obj} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 保存结果
        output_file = OUTPUT_DIR / "factor_exposure.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        file_size = output_file.stat().st_size / (1024 * 1024)
        print(f"✅ 生成成功: {output_file} ({file_size:.2f} MB, {len(results)} 个日期)")
        return True
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 开始生成预计算的因子诊断结果...")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    
    success_count = 0
    
    if generate_long_short_performance():
        success_count += 1
    
    if generate_correlation_matrix():
        success_count += 1
    
    if generate_risk_exposure():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ 完成！成功生成 {success_count}/3 个文件")
    print("=" * 60)
    
    if success_count == 3:
        print("\n📝 下一步：")
        print("1. 检查生成的文件大小")
        print("2. 如果文件太大（>50MB），考虑上传到 Hugging Face")
        print("3. 提交文件到 Git 或上传到 Hugging Face")
        print("4. 部署后，API 将直接读取这些 JSON 文件")

