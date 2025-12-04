#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
每日更新脚本：获取最新数据、计算因子、生成预测和权重

功能：
1. 获取最近 N 天的价格数据（默认 60 天，因为因子计算需要历史数据）
2. 更新因子库（只计算最新日期的因子）
3. 加载已保存的模型进行预测
4. 生成最新日期的权重

使用方法：
    python scripts/daily_update.py

参数：
    --lookback-days: 获取多少天的历史数据（默认 60，因子计算需要）
    --model-type: 模型类型 (lightgbm/catboost/xgboost，默认 lightgbm)
    --skip-fetch: 跳过数据获取（如果已经手动更新了数据）
    --skip-factors: 跳过因子计算（如果因子已经计算好）
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings

import pandas as pd
import numpy as np
import yaml

# 使用统一的路径管理
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.path import SETTINGS_FILE, get_path
from src.data_pipeline import (
    load_settings,
    get_tickers_from_qlib,
    fetch_daily_prices,
    fetch_daily_prices_from_qlib,
)
from src.factor_engine import (
    read_prices,
    calculate_all_factors,
    qlib_style_processing,
)
from src.optimizer import (
    topk_dropout_strategy,
    full_rebalance_strategy,
    load_predictions,
    run_optimize,
)

warnings.filterwarnings("ignore")

SETTINGS = SETTINGS_FILE


def load_top100_tickers() -> list:
    """加载市值前100股票列表"""
    top100_file = get_path("data/top100_stocks.txt")
    if top100_file.exists():
        with open(top100_file, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
        print(f"[Info] 加载前100股票列表: {len(tickers)} 只股票")
        return tickers
    else:
        print(f"[Warn] 找不到前100股票文件: {top100_file}")
        print(f"       将使用所有股票（建议运行 python get_top100_stocks.py 生成列表）")
        return None


def get_latest_date_from_prices(prices_path: Path) -> pd.Timestamp:
    """从价格文件中获取最新日期"""
    if not prices_path.exists():
        return None
    
    prices = pd.read_parquet(prices_path)
    if isinstance(prices.index, pd.MultiIndex):
        dates = prices.index.get_level_values("date")
    else:
        # 尝试从索引中提取日期
        dates = pd.to_datetime(prices.index.get_level_values(0), errors='coerce')
    
    if len(dates) == 0:
        return None
    
    return pd.to_datetime(dates.max())


def update_prices(cfg, lookback_days: int = 60):
    """更新价格数据（只获取最近 N 天）"""
    print("=" * 60)
    print("[Step 1] 更新价格数据")
    print("=" * 60)
    
    prices_path = Path(cfg["paths"]["prices_parquet"])
    
    # 获取现有数据的最新日期
    latest_date = get_latest_date_from_prices(prices_path)
    
    if latest_date is not None:
        print(f"  现有数据最新日期: {latest_date.date()}")
        # 从最新日期的下一天开始获取
        start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        # 如果没有现有数据，获取最近 N 天
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        print(f"  未找到现有数据，获取最近 {lookback_days} 天的数据")
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # 检查是否需要更新
    if latest_date is not None:
        days_since_update = (datetime.now() - latest_date.to_pydatetime()).days
        if days_since_update == 0:
            print("  ✓ 数据已是最新，跳过更新")
            return
        print(f"  需要更新 {days_since_update} 天的数据")
    
    # 获取股票列表
    region = cfg.get("data", {}).get("region", "us")
    instruments = cfg.get("data", {}).get("instruments", "sp500")
    
    if prices_path.exists():
        # 从现有数据中获取股票列表
        existing_prices = pd.read_parquet(prices_path)
        if isinstance(existing_prices.index, pd.MultiIndex):
            tickers = sorted(existing_prices.index.get_level_values("ticker").unique().tolist())
        else:
            tickers = get_tickers_from_qlib(instruments, region)
        print(f"  从现有数据中获取 {len(tickers)} 只股票")
    else:
        tickers = get_tickers_from_qlib(instruments, region)
        print(f"  获取 {len(tickers)} 只股票")
    
    # 获取新数据
    print(f"  获取日期范围: {start_date} 到 {end_date}")
    
    try:
        # 创建临时文件路径用于获取新数据
        temp_path = prices_path.parent / f"temp_{prices_path.name}"
        
        if region == "cn":
            new_data = fetch_daily_prices_from_qlib(instruments, start_date, end_date, str(temp_path), region)
        else:
            new_data = fetch_daily_prices(tickers, start_date, end_date, str(temp_path), region)
        
        # 函数已经保存了数据到 temp_path，现在读取并合并
        if temp_path.exists():
            new_data = pd.read_parquet(temp_path)
            temp_path.unlink()  # 删除临时文件
            
            if new_data.empty:
                print("  ⚠️ 没有获取到新数据")
                return
            
            # 合并数据
            if prices_path.exists() and latest_date is not None:
                existing_prices = pd.read_parquet(prices_path)
                # 确保索引格式一致
                if isinstance(existing_prices.index, pd.MultiIndex):
                    existing_prices.index = pd.MultiIndex.from_tuples(
                        [(pd.to_datetime(d), t) for d, t in existing_prices.index],
                        names=["date", "ticker"]
                    )
                
                # 合并：保留旧数据，添加新数据
                combined = pd.concat([existing_prices, new_data]).drop_duplicates().sort_index()
                combined.to_parquet(prices_path)
                print(f"  ✓ 已合并数据，总行数: {len(combined)}")
            else:
                new_data.to_parquet(prices_path)
                print(f"  ✓ 已保存新数据，总行数: {len(new_data)}")
        else:
            print("  ⚠️ 数据获取失败，临时文件不存在")
            
    except ValueError as e:
        error_msg = str(e)
        # 检查是否是"没有数据"的错误（可能是非交易日）
        if "No data downloaded" in error_msg:
            print(f"  ⚠️ 数据获取失败: {error_msg}")
            print(f"\n  可能原因：")
            print(f"    1. {end_date} 可能是非交易日（节假日）")
            print(f"    2. yfinance API 暂时不可用")
            print(f"    3. 数据尚未准备好（通常需要交易日结束后）")
            print(f"\n  💡 解决方案：")
            if latest_date is not None:
                print(f"    - 当前数据最新日期: {latest_date.date()}")
            print(f"    - 跳过数据获取，使用现有数据继续：")
            print(f"      python scripts/daily_update.py --skip-fetch")
            print(f"    - 或等待下一个交易日再尝试完整更新")
            # 如果是数据不可用，不抛出异常，允许跳过继续
            return
        else:
            print(f"  ✗ 数据获取失败: {e}")
            raise
    except Exception as e:
        print(f"  ✗ 数据获取失败: {e}")
        print(f"\n  💡 提示：可以使用 --skip-fetch 跳过数据获取，继续使用现有数据")
        raise


def update_factors(cfg, lookback_days: int = 60):
    """更新因子（只计算最新日期的因子，但需要历史数据）"""
    print("\n" + "=" * 60)
    print("[Step 2] 更新因子")
    print("=" * 60)
    
    prices_path = Path(cfg["paths"]["prices_parquet"])
    factors_path = Path(cfg["paths"]["factors_store"])
    
    if not prices_path.exists():
        raise FileNotFoundError(f"价格文件不存在: {prices_path}")
    
    # 读取价格数据
    prices = read_prices(cfg)
    
    # 过滤到前100股票（如果配置了）
    top100_tickers = load_top100_tickers()
    if top100_tickers:
        available_tickers = set(prices.index.get_level_values("ticker").unique())
        valid_tickers = [t for t in top100_tickers if t in available_tickers]
        if len(valid_tickers) < len(top100_tickers):
            missing = set(top100_tickers) - available_tickers
            print(f"  [Warn] {len(missing)} 只前100股票在价格数据中不存在: {sorted(list(missing))[:10]}...")
        prices = prices.loc[prices.index.get_level_values("ticker").isin(valid_tickers)]
        print(f"  [Info] 过滤到前100股票: {len(valid_tickers)} 只股票")
    
    # 获取最新日期
    dates = prices.index.get_level_values("date")
    latest_date = pd.to_datetime(dates.max())
    
    # 检查因子库中是否已有最新日期的因子
    existing_factors = None
    if factors_path.exists():
        existing_factors = pd.read_parquet(factors_path)
        # 确保索引格式一致
        if isinstance(existing_factors.index, pd.MultiIndex):
            existing_factors.index = pd.MultiIndex.from_tuples(
                [(pd.to_datetime(d), t) for d, t in existing_factors.index],
                names=["date", "ticker"]
            )
        
        existing_dates = existing_factors.index.get_level_values("date")
        if len(existing_dates) > 0:
            latest_existing_date = pd.to_datetime(existing_dates.max())
            
            # 如果最新日期的因子已存在，跳过计算
            if latest_existing_date >= latest_date:
                print(f"  ✓ 因子已是最新（最新日期: {latest_date.date()}）")
                print(f"    因子库最新日期: {latest_existing_date.date()}")
                return
    
    # 计算需要的历史数据范围（因子计算需要历史数据）
    # 注意：虽然只保存最新日期，但计算时需要历史数据作为输入
    # 增加数据范围以确保 TA-Lib 因子计算有足够数据（TA-Lib 需要至少 30-50 天）
    extended_lookback = max(lookback_days, 90)  # 至少 90 天以确保 TA-Lib 有足够数据
    start_date = (latest_date - timedelta(days=extended_lookback)).strftime("%Y-%m-%d")
    end_date = latest_date.strftime("%Y-%m-%d")
    
    print(f"  计算因子日期范围: {start_date} 到 {end_date}")
    print(f"  （需要历史数据用于因子计算，但只保存最新日期 {latest_date.date()} 的因子）")
    print(f"  （扩展数据范围到 {extended_lookback} 天以确保 TA-Lib 因子计算有足够数据）")
    
    # 计算因子（会计算整个范围，但只保存最新日期）
    try:
        # 只使用需要的历史数据范围来计算因子
        # 这样虽然会计算整个范围，但因子计算函数内部需要历史数据
        new_factors = calculate_all_factors(prices, start_date, end_date)
        
        # 处理因子（对齐训练时的处理方式）
        print("  处理因子（winsorize + zscore）...")
        processed_factors = new_factors.copy()
        
        for col in processed_factors.columns:
            if processed_factors[col].notna().sum() > 0:
                processed_factors[col] = qlib_style_processing(processed_factors[col])
        
        # 只保留最新日期的因子（避免重复保存历史数据）
        latest_factors = processed_factors.loc[
            processed_factors.index.get_level_values("date") == latest_date
        ]
        
        if len(latest_factors) == 0:
            print(f"  ⚠️ 最新日期 {latest_date.date()} 的因子计算失败")
            return
        
        # 合并到现有因子库
        if factors_path.exists():
            # 移除现有数据中的最新日期（如果有，避免重复）
            existing_factors = existing_factors.loc[
                existing_factors.index.get_level_values("date") < latest_date
            ]
            
            # 过滤现有因子库到前100股票（如果配置了）
            if top100_tickers:
                existing_factors = existing_factors.loc[
                    existing_factors.index.get_level_values("ticker").isin(valid_tickers)
                ]
                print(f"  [Info] 过滤现有因子库到前100股票: {len(valid_tickers)} 只股票")
            
            # 合并
            combined = pd.concat([existing_factors, latest_factors]).sort_index()
            
            # 最终检查：确保只包含前100股票
            if top100_tickers:
                combined = combined.loc[
                    combined.index.get_level_values("ticker").isin(valid_tickers)
                ]
            
            combined.to_parquet(factors_path)
            print(f"  ✓ 已更新因子，最新日期: {latest_date.date()}")
            print(f"    总行数: {len(combined)}, 因子数: {len(combined.columns)}")
            print(f"    股票数: {combined.index.get_level_values('ticker').nunique()}")
            print(f"    日期范围: {combined.index.get_level_values('date').min().date()} 到 {combined.index.get_level_values('date').max().date()}")
        else:
            # 如果没有现有因子库，只保存最新日期
            latest_factors.to_parquet(factors_path)
            print(f"  ✓ 已保存因子，最新日期: {latest_date.date()}")
            print(f"    总行数: {len(latest_factors)}, 因子数: {len(latest_factors.columns)}")
            print(f"    股票数: {latest_factors.index.get_level_values('ticker').nunique()}")
            
    except Exception as e:
        print(f"  ✗ 因子计算失败: {e}")
        raise


def generate_daily_prediction(cfg, model_type: str = "lightgbm"):
    """生成最新日期的预测"""
    print("\n" + "=" * 60)
    print(f"[Step 3] 生成预测（模型: {model_type}）")
    print("=" * 60)
    
    model_dir = Path(cfg["paths"]["model_dir"])
    factors_path = Path(cfg["paths"]["factors_store"])
    
    # 检查模型文件
    ranker_path = model_dir / "lgbm_ranker.txt"
    reg_path = model_dir / "lgbm_regression.txt"
    
    if not ranker_path.exists() and not reg_path.exists():
        raise FileNotFoundError(
            f"模型文件不存在。请先运行训练: python src/modeling.py --train"
        )
    
    # 加载因子数据
    print("  加载因子数据...")
    factor_store = pd.read_parquet(factors_path)
    
    # 过滤到前100股票（如果配置了）
    top100_tickers = load_top100_tickers()
    if top100_tickers:
        available_tickers = set(factor_store.index.get_level_values("ticker").unique())
        valid_tickers = [t for t in top100_tickers if t in available_tickers]
        if len(valid_tickers) < len(top100_tickers):
            missing = set(top100_tickers) - available_tickers
            print(f"  [Warn] {len(missing)} 只前100股票在因子数据中不存在: {sorted(list(missing))[:10]}...")
        factor_store = factor_store.loc[factor_store.index.get_level_values("ticker").isin(valid_tickers)]
        print(f"  [Info] 过滤到前100股票: {len(valid_tickers)} 只股票")
    
    # 修复索引格式
    if isinstance(factor_store.index, pd.MultiIndex):
        level_0 = factor_store.index.get_level_values(0)
        level_1 = factor_store.index.get_level_values(1)
        if pd.api.types.is_datetime64_any_dtype(level_0):
            dates = pd.to_datetime(level_0, errors='coerce')
            tickers = pd.Series(level_1).astype(str).values
        elif pd.api.types.is_datetime64_any_dtype(level_1):
            dates = pd.to_datetime(level_1, errors='coerce')
            tickers = pd.Series(level_0).astype(str).values
        else:
            dates = pd.to_datetime(level_0, errors='coerce')
            tickers = pd.Series(level_1).astype(str).values
        factor_store.index = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])
    
    # 获取最新日期
    dates = factor_store.index.get_level_values("date")
    latest_date = pd.to_datetime(dates.max())
    print(f"  最新因子日期: {latest_date.date()}")
    
    # 获取最新日期的因子
    latest_factors = factor_store.loc[factor_store.index.get_level_values("date") == latest_date].copy()
    
    if len(latest_factors) == 0:
        raise ValueError(f"没有找到日期 {latest_date.date()} 的因子数据")
    
    print(f"  最新日期因子数据: {len(latest_factors)} 只股票")
    
    # 加载模型
    import lightgbm as lgb
    import json
    
    if ranker_path.exists():
        print(f"  加载模型: {ranker_path}")
        model = lgb.Booster(model_file=str(ranker_path))
        feature_list_path = model_dir / "feature_list_ranker.json"
    elif reg_path.exists():
        print(f"  加载模型: {reg_path}")
        model = lgb.Booster(model_file=str(reg_path))
        feature_list_path = model_dir / "feature_list_regression.json"
    else:
        raise FileNotFoundError("找不到模型文件")
    
    # 加载特征列表
    if feature_list_path.exists():
        with open(feature_list_path, "r") as f:
            feature_list = json.load(f)
        print(f"  加载特征列表: {len(feature_list)} 个特征")
    else:
        # 从模型获取特征名
        feature_list = model.feature_name()
        print(f"  从模型获取特征列表: {len(feature_list)} 个特征")
    
    # 特征对齐
    print("  对齐特征...")
    available_features = set(latest_factors.columns)
    required_features = set(feature_list)
    
    missing_features = required_features - available_features
    extra_features = available_features - required_features
    
    if missing_features:
        print(f"  ⚠️ 缺失特征 {len(missing_features)} 个，用中位数填充")
        # 尝试从历史数据获取中位数
        if factors_path.exists():
            hist_factors = pd.read_parquet(factors_path)
            for feat in missing_features:
                if feat in hist_factors.columns:
                    median_val = hist_factors[feat].median()
                    latest_factors[feat] = median_val
                else:
                    latest_factors[feat] = 0.0
    
    # 只保留需要的特征，按顺序排列
    X_pred = latest_factors.reindex(columns=feature_list, fill_value=0.0)
    
    # 填充缺失值（使用中位数）
    for col in X_pred.columns:
        if X_pred[col].isna().any():
            median_val = X_pred[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            X_pred[col] = X_pred[col].fillna(median_val)
    
    print(f"  预测数据形状: {X_pred.shape}")
    
    # 生成预测
    print("  生成预测...")
    pred_values = model.predict(X_pred.values)
    
    # 创建预测 Series
    pred_series = pd.Series(
        pred_values,
        index=pd.MultiIndex.from_tuples(
            [(latest_date, ticker) for ticker in latest_factors.index.get_level_values("ticker")],
            names=["date", "ticker"]
        ),
        name="prediction"
    )
    
    print(f"  ✓ 预测完成: {len(pred_series)} 个样本")
    print(f"    预测值范围: [{pred_series.min():.4f}, {pred_series.max():.4f}]")
    
    # 保存预测（追加到现有预测文件）
    pred_file = model_dir / f"{model_type}_predictions.pkl"
    if pred_file.exists():
        import pickle
        with open(pred_file, "rb") as f:
            existing_pred = pickle.load(f)
        
        if isinstance(existing_pred, pd.Series):
            # 移除同一天的旧预测（如果有）
            existing_pred = existing_pred.loc[
                existing_pred.index.get_level_values("date") != latest_date
            ]
            # 合并
            combined_pred = pd.concat([existing_pred, pred_series]).sort_index()
        else:
            combined_pred = pred_series
    else:
        combined_pred = pred_series
    
    import pickle
    with open(pred_file, "wb") as f:
        pickle.dump(combined_pred, f)
    
    print(f"  ✓ 已保存预测到 {pred_file}")
    
    return pred_series


def generate_daily_weights(cfg, pred_series: pd.Series):
    """生成最新日期的权重"""
    print("\n" + "=" * 60)
    print("[Step 4] 生成权重")
    print("=" * 60)
    
    weights_path = Path(cfg["paths"]["portfolio_path"])
    strategy_config = cfg.get("strategy", {})
    
    # 获取最新日期的预测
    latest_date = pred_series.index.get_level_values("date").max()
    pred_day = pred_series.xs(latest_date, level="date").dropna()
    
    # 过滤到前100股票（如果配置了）
    top100_tickers = load_top100_tickers()
    if top100_tickers:
        available_tickers = set(pred_day.index)
        valid_tickers = [t for t in top100_tickers if t in available_tickers]
        if len(valid_tickers) < len(top100_tickers):
            missing = set(top100_tickers) - available_tickers
            print(f"  [Warn] {len(missing)} 只前100股票在预测数据中不存在: {sorted(list(missing))[:10]}...")
        pred_day = pred_day.reindex(valid_tickers).dropna()
        print(f"  [Info] 过滤到前100股票: {len(pred_day)} 只股票")
    
    print(f"  最新日期: {latest_date.date()}")
    print(f"  预测股票数: {len(pred_day)}")
    
    # 获取当前持仓（从现有权重文件）
    current_positions = pd.Series(dtype=float)
    if weights_path.exists():
        existing_weights = pd.read_parquet(weights_path)
        if latest_date in existing_weights.index:
            # 获取前一天的持仓
            prev_dates = existing_weights.index[existing_weights.index < latest_date]
            if len(prev_dates) > 0:
                prev_date = prev_dates.max()
                current_positions = existing_weights.loc[prev_date].fillna(0.0)
                current_positions = current_positions[current_positions > 0]
                print(f"  前一日持仓: {len(current_positions)} 只股票")
    
    # 生成权重
    strategy_type = strategy_config.get("type", "topk_dropout")
    topk = strategy_config.get("topk", 20)
    
    if strategy_type == "full_rebalance":
        print(f"  策略: Full Rebalance (每日全量换仓, topk={topk})")
        
        new_weights = full_rebalance_strategy(
            pred_day,
            current_positions,
            topk=topk,
        )
    elif strategy_type == "topk_dropout":
        n_drop = strategy_config.get("n_drop", 3)
        method_sell = strategy_config.get("method_sell", "bottom")
        method_buy = strategy_config.get("method_buy", "top")
        
        print(f"  策略: TopK Dropout (topk={topk}, n_drop={n_drop})")
        
        new_weights = topk_dropout_strategy(
            pred_day,
            current_positions,
            topk=topk,
            n_drop=n_drop,
            method_sell=method_sell,
            method_buy=method_buy,
        )
    else:
        raise NotImplementedError(f"策略类型 {strategy_type} 暂不支持每日更新")
    
    # 归一化
    if new_weights.sum() > 0:
        new_weights = new_weights / new_weights.sum()
    
    print(f"  ✓ 生成权重: {len(new_weights[new_weights > 0])} 只股票")
    print(f"    权重和: {new_weights.sum():.6f}")
    
    # 保存权重（追加到现有权重文件）
    if weights_path.exists():
        existing_weights = pd.read_parquet(weights_path)
        # 移除同一天的旧权重（如果有）
        existing_weights = existing_weights.loc[existing_weights.index != latest_date]
        # 添加新权重
        new_weights_df = pd.DataFrame({latest_date: new_weights}).T
        combined_weights = pd.concat([existing_weights, new_weights_df]).sort_index()
    else:
        combined_weights = pd.DataFrame({latest_date: new_weights}).T
    
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    combined_weights.to_parquet(weights_path)
    
    print(f"  ✓ 已保存权重到 {weights_path}")
    print(f"    权重文件日期范围: {combined_weights.index.min().date()} 到 {combined_weights.index.max().date()}")
    
    return new_weights


def main():
    parser = argparse.ArgumentParser(description="每日更新：获取最新数据、计算因子、生成预测和权重")
    parser.add_argument("--lookback-days", type=int, default=60,
                       help="获取多少天的历史数据（默认 60，因子计算需要）")
    parser.add_argument("--model-type", default="lightgbm", choices=["lightgbm", "catboost", "xgboost"],
                       help="模型类型（默认 lightgbm）")
    parser.add_argument("--skip-fetch", action="store_true",
                       help="跳过数据获取（如果已经手动更新了数据）")
    parser.add_argument("--skip-factors", action="store_true",
                       help="跳过因子计算（如果因子已经计算好）")
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_settings(SETTINGS)
    
    try:
        # Step 1: 更新价格数据
        if not args.skip_fetch:
            update_prices(cfg, args.lookback_days)
        else:
            print("[跳过] 数据获取")
        
        # Step 2: 更新因子
        if not args.skip_factors:
            update_factors(cfg, args.lookback_days)
        else:
            print("[跳过] 因子计算")
        
        # Step 3: 生成预测
        pred_series = generate_daily_prediction(cfg, args.model_type)
        
        # Step 4: 生成权重
        weights = generate_daily_weights(cfg, pred_series)
        
        print("\n" + "=" * 60)
        print("✓ 每日更新完成！")
        print("=" * 60)
        print(f"  最新日期: {pred_series.index.get_level_values('date').max().date()}")
        print(f"  持仓股票数: {len(weights[weights > 0])}")
        print(f"  权重文件: {cfg['paths']['portfolio_path']}")
        print("\n下一步：")
        print("  1. 查看权重: python -c \"import pandas as pd; print(pd.read_parquet('outputs/portfolios/weights.parquet').tail(1))\"")
        print("  2. 运行回测: python src/backtest.py --run")
        print("  3. 实盘交易: python src/ibkr_live_trader.py --weights outputs/portfolios/weights.parquet")
        
    except Exception as e:
        print(f"\n✗ 更新失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

