#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantitative Trading Strategy Web Dashboard
展示量化交易策略的完整工作流程、因子分析、持仓和收益
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pytz

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from flask import Flask, render_template, jsonify, request
import yaml

# 使用统一的路径管理
# 添加项目根目录到路径（path.py 会处理项目根目录的查找）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.path import (
    ROOT_DIR as project_root,
    SETTINGS_FILE,
    OUTPUT_DIR,
    OUTPUT_BACKTESTS_DIR as BACKTEST_DIR,
    OUTPUT_PORTFOLIOS_DIR as PORTFOLIO_DIR,
    OUTPUT_REPORTS_DIR as REPORTS_DIR,
    OUTPUT_IBKR_DATA_DIR,
    DATA_FACTORS_DIR,
    get_path,
)

# 导入 src 模块的真实函数
from src.backtest import load_settings, risk_analysis
from src.factor_engine import read_prices, load_settings as load_factor_settings, daily_rank_ic, forward_return
from src.optimizer import load_predictions
from src.modeling import rank_ic_per_day

# 可选导入 IBKRLiveTrader（需要 ib_insync，这是可选依赖）
try:
    from src.ibkr_live_trader import IBKRLiveTrader
    IBKR_AVAILABLE = True
except ImportError as e:
    IBKRLiveTrader = None
    IBKR_AVAILABLE = False
    print(f"[WARN] IBKR Live Trader not available: {e}")
    print("[WARN] IBKR features will be disabled. Install ib_insync to enable: pip install ib_insync")

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 加载配置
SETTINGS = SETTINGS_FILE
cfg = load_settings(str(SETTINGS))

# 导入 Dashboard 配置
try:
    from config import API_BASE_URL, IBKR_CONFIG
except ImportError:
    import os
    API_BASE_URL = os.getenv('API_BASE_URL', '')
    IBKR_CONFIG = {
        'host': os.getenv('IBKR_HOST', '127.0.0.1'),
        'port': int(os.getenv('IBKR_PORT', '7497')),
        'client_id': int(os.getenv('IBKR_CLIENT_ID', '777')),
        'enabled': os.getenv('IBKR_ENABLED', 'false').lower() == 'true'
    }

# 数据路径已在 path.py 中定义，无需重复定义


def get_factor_store_path(factor_cfg=None):
    """
    获取 factor_store.parquet 的路径
    
    Args:
        factor_cfg: 因子配置（可选）
    
    Returns:
        factor_store.parquet 的路径
    """
    if factor_cfg is None:
        factor_cfg = load_factor_settings(str(SETTINGS))
    
    factor_store_path = get_path(
        factor_cfg["paths"].get("factors_store", "data/factors/factor_store.parquet"),
        DATA_FACTORS_DIR
    )
    
    return factor_store_path


def convert_to_ny_time(time_str, input_tz='UTC'):
    """
    将时间字符串转换为纽约时区
    
    Args:
        time_str: 时间字符串 (格式: "YYYY-MM-DD HH:MM:SS" 或 "YYYY-MM-DD")
        input_tz: 输入时区，默认 'UTC'
    
    Returns:
        转换后的时间字符串 (纽约时区，格式: "YYYY-MM-DD HH:MM:SS")
    """
    if not time_str:
        return time_str
    
    try:
        # 解析输入时间
        if len(time_str) == 19:  # "YYYY-MM-DD HH:MM:SS"
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        elif len(time_str) == 10:  # "YYYY-MM-DD"
            dt = datetime.strptime(time_str, "%Y-%m-%d")
        else:
            return time_str  # 无法解析，返回原值
        
        # 设置输入时区
        if input_tz == 'UTC':
            input_timezone = pytz.UTC
        else:
            input_timezone = pytz.timezone(input_tz)
        
        # 如果 datetime 是 naive，先设置为 UTC
        if dt.tzinfo is None:
            dt = input_timezone.localize(dt)
        
        # 转换到纽约时区
        ny_tz = pytz.timezone('America/New_York')
        ny_dt = dt.astimezone(ny_tz)
        
        # 返回格式化字符串
        return ny_dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"[Warn] Failed to convert time to NY timezone: {time_str}, error: {e}")
        return time_str  # 转换失败，返回原值


def load_json_safe(path: Path, default=None):
    """安全加载 JSON 文件"""
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return default
    return default


def clean_nan_for_json(obj):
    """清理数据中的NaN值，转换为None（JSON中的null）"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: clean_nan_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_for_json(item) for item in obj]
    elif isinstance(obj, (np.floating, float)) and (np.isnan(obj) or np.isinf(obj)):
        return None
    elif isinstance(obj, pd.Series):
        return [None if (isinstance(x, (np.floating, float)) and (np.isnan(x) or np.isinf(x))) else x for x in obj]
    elif isinstance(obj, np.ndarray):
        return [None if (isinstance(x, (np.floating, float)) and (np.isnan(x) or np.isinf(x))) else x for x in obj]
    else:
        return obj


def load_parquet_safe(path: Path):
    """安全加载 Parquet 文件"""
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as e:
            print(f"[Error] Failed to load parquet file {path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"[Warn] Parquet file does not exist: {path}")
        print(f"  Current working directory: {Path.cwd()}")
        print(f"  Project root: {project_root}")
    return None


@app.route('/')
def index():
    """首页 - 项目介绍"""
    return render_template('index.html', api_base_url=API_BASE_URL)


@app.route('/workflow')
def workflow():
    """详细工作流程页面"""
    return render_template('workflow.html', api_base_url=API_BASE_URL)


@app.route('/api/workflow/data')
def workflow_data():
    """获取工作流程数据"""
    # 加载模型指标
    metrics = load_json_safe(REPORTS_DIR / "metrics.json", {})
    backtest_summary = load_json_safe(BACKTEST_DIR / "summary.json", {})
    
    # 加载回测数据
    daily_returns = load_parquet_safe(BACKTEST_DIR / "daily_returns.parquet")
    
    return jsonify({
        "metrics": metrics,
        "backtest_summary": backtest_summary,
        "has_daily_returns": daily_returns is not None
    })


@app.route('/api/workflow/model_performance')
def workflow_model_performance():
    """工作流程页面 - 模型表现（基于 LightGBM 模型，不是回测 NAV）"""
    try:
        # 使用 src/optimizer.py 加载预测
        predictions = load_predictions(cfg, model_type="lightgbm")
        
        # 加载模型指标
        model_metrics = load_json_safe(REPORTS_DIR / "metrics.json", {})
        if not model_metrics:
            model_dir = Path(cfg["paths"]["model_dir"])
            metrics_path = model_dir / "metrics.json"
            if metrics_path.exists():
                model_metrics = load_json_safe(metrics_path, {})
        
        # 计算预测分布统计
        if predictions is not None and len(predictions) > 0:
            if isinstance(predictions.index, pd.MultiIndex):
                dates = predictions.index.get_level_values(0).unique()
                latest_date = dates.max()
                latest_predictions = predictions.loc[latest_date] if latest_date in dates else predictions.iloc[-100:]
            else:
                latest_predictions = predictions.iloc[-100:]
            
            pred_stats = {
                "mean": float(latest_predictions.mean()),
                "std": float(latest_predictions.std()),
                "min": float(latest_predictions.min()),
                "max": float(latest_predictions.max()),
                "count": len(latest_predictions)
            }
        else:
            pred_stats = {}
        
        return jsonify({
            "error": None,
            "model_metrics": model_metrics,
            "prediction_stats": pred_stats,
            "prediction_count": len(predictions) if predictions is not None else 0
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"模型数据加载失败: {str(e)}",
            "model_metrics": {},
            "prediction_stats": {},
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/api/workflow/backtest_chart')
def backtest_chart():
    """生成回测收益曲线图 - 规范化数据格式"""
    daily_returns = load_parquet_safe(BACKTEST_DIR / "daily_returns.parquet")
    
    if daily_returns is None:
        return jsonify({"error": "暂无数据", "data": None}), 200
    
    try:
        # 规范化列名：支持多种可能的列名
        strategy_col = None
        benchmark_col = None
        
        # 优先查找标准列名
        if 'strategy_return' in daily_returns.columns:
            strategy_col = 'strategy_return'
        elif 'net_return' in daily_returns.columns:
            strategy_col = 'net_return'  # backtest.py 生成的列名
        else:
            # 尝试其他可能的列名
            for col in daily_returns.columns:
                col_lower = col.lower()
                if 'strategy' in col_lower and ('return' in col_lower or 'ret' in col_lower):
                    strategy_col = col
                    break
                elif 'net' in col_lower and 'return' in col_lower:
                    strategy_col = col
                    break
        
        if strategy_col is None:
            return jsonify({"error": "数据加载失败，请检查后端服务：缺少策略收益列", "data": None}), 200
        
        # 计算累计收益
        cumulative_returns = (1 + daily_returns[strategy_col]).cumprod()
        
        # 尝试加载基准数据（从价格数据计算 S&P500 收益）
        benchmark_returns = None
        try:
            prices = load_parquet_safe(get_path("data/processed/prices.parquet"))
            if prices is not None:
                # 计算基准收益（所有股票的平均收益，近似 S&P500）
                if isinstance(prices.index, pd.MultiIndex):
                    close_prices = prices["Adj Close"].unstack("ticker")
                    benchmark_ret = close_prices.mean(axis=1).pct_change().fillna(0.0)
                    # 对齐日期
                    common_dates = daily_returns.index.intersection(benchmark_ret.index)
                    if len(common_dates) > 0:
                        benchmark_returns = (1 + benchmark_ret.loc[common_dates]).cumprod()
        except:
            pass
        
        if benchmark_returns is None:
            benchmark_returns = pd.Series([1.0] * len(cumulative_returns), index=cumulative_returns.index)
        
        # 确保索引是日期格式
        dates = daily_returns.index
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.to_datetime(dates)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative_returns * 100,
            mode='lines',
            name='Strategy',
            line=dict(color='#2E86AB', width=2)
        ))
        
        if len(benchmark_returns) > 0:
            fig.add_trace(go.Scatter(
                x=dates,
                y=benchmark_returns.values * 100,
                mode='lines',
                name='Benchmark (S&P500)',
                line=dict(color='#A23B72', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title="Strategy vs Benchmark Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return jsonify({"data": fig.to_json(), "error": None})
    except Exception as e:
        import traceback
        return jsonify({"error": f"数据加载失败，请检查后端服务：{str(e)}", "data": None, "traceback": traceback.format_exc()}), 200


@app.route('/api/workflow/performance_metrics')
def performance_metrics():
    """获取性能指标 - 使用 src/backtest.py 的真实函数计算"""
    backtest_summary = load_json_safe(BACKTEST_DIR / "summary.json", {})
    daily_returns = load_parquet_safe(BACKTEST_DIR / "daily_returns.parquet")
    
    if daily_returns is None:
        return jsonify({
            "error": "暂无数据",
            "metrics": {
                "annual_return": None,
                "sharpe": None,
                "max_drawdown": None,
                "information_ratio": None,
                "turnover": None
            }
        }), 200
    
    try:
        # 规范化列名 - 优先使用 backtest.py 生成的统一字段
        strategy_col = None
        if 'strategy_return' in daily_returns.columns:
            strategy_col = 'strategy_return'
        elif 'net_return' in daily_returns.columns:
            strategy_col = 'net_return'
        else:
            for col in daily_returns.columns:
                col_lower = col.lower()
                if 'strategy' in col_lower and ('return' in col_lower or 'ret' in col_lower):
                    strategy_col = col
                    break
                elif 'net' in col_lower and 'return' in col_lower:
                    strategy_col = col
                    break
        
        if strategy_col is None:
            return jsonify({
                "error": "数据加载失败，请检查后端服务：缺少策略收益列",
                "metrics": {
                    "annual_return": None,
                    "sharpe": None,
                    "max_drawdown": None,
                    "information_ratio": None,
                    "turnover": None
                }
            }), 200
        
        # 使用 src/backtest.py 的真实函数计算风险指标
        strategy_returns = daily_returns[strategy_col]
        risk_metrics = risk_analysis(strategy_returns)
        
        return jsonify({
            "error": None,
            "metrics": {
                "annual_return": backtest_summary.get("annualized_return", risk_metrics.get("annualized_return")),
                "sharpe": backtest_summary.get("sharpe_ratio", risk_metrics.get("information_ratio")),
                "max_drawdown": backtest_summary.get("max_drawdown", risk_metrics.get("max_drawdown")),
                "information_ratio": risk_metrics.get("information_ratio"),
                "turnover": backtest_summary.get("avg_turnover", 0)
            }
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"数据加载失败，请检查后端服务：{str(e)}",
            "metrics": {
                "annual_return": None,
                "sharpe": None,
                "max_drawdown": None,
                "information_ratio": None,
                "turnover": None
            },
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/factor-diagnostics')
def factor_diagnostics():
    """因子长期监控（Factor Diagnostics）"""
    return render_template('factor_diagnostics.html', api_base_url=API_BASE_URL)


@app.route('/api/factors/dates')
def factor_dates():
    """获取所有可用的因子日期 - 从 DuckDB/parquet 读取"""
    try:
        factor_cfg = load_factor_settings(str(SETTINGS))
        factor_store_path = Path(factor_cfg["paths"].get("factors_store", "data/factors/factor_store.parquet"))
        if not factor_store_path.is_absolute():
            factor_store_path = get_path(factor_store_path, DATA_FACTORS_DIR)
        
        factor_store = None
        if factor_store_path.exists():
            try:
                factor_store = pd.read_parquet(factor_store_path)
            except:
                pass
        
        # Fallback 到 DuckDB
        if factor_store is None:
            try:
                import duckdb
                db_path = factor_cfg["database"].get("duckdb_path", "data/duckdb/prices.db")
                if not Path(db_path).is_absolute():
                    db_path = get_path(db_path)
                con = duckdb.connect(str(db_path))
                try:
                    factor_store = con.execute("SELECT * FROM factor_store").df()
                except:
                    pass
                con.close()
            except:
                pass
        
        if factor_store is None or len(factor_store) == 0:
            return jsonify({"error": "暂无数据", "dates": []}), 200
        
        # 确保索引是 MultiIndex
        if not isinstance(factor_store.index, pd.MultiIndex):
            if "date" in factor_store.columns and "ticker" in factor_store.columns:
                factor_store["date"] = pd.to_datetime(factor_store["date"])
                factor_store = factor_store.set_index(["date", "ticker"]).sort_index()
        
        if isinstance(factor_store.index, pd.MultiIndex):
            dates = sorted(factor_store.index.get_level_values(0).unique().astype(str).tolist())
        else:
            dates = sorted(factor_store.index.unique().astype(str).tolist())
        
        return jsonify({"error": None, "dates": dates})
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"数据加载失败，请检查后端服务：{str(e)}",
            "dates": [],
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/api/factors')
def factors_by_date():
    """根据日期获取因子数据 - 从 DuckDB/parquet 读取，fallback 到最近交易日"""
    date = request.args.get('date')
    
    try:
        # 使用 src/factor_engine.py 的真实函数读取数据
        factor_cfg = load_factor_settings(str(SETTINGS))
        
        # 尝试从 parquet 读取
        factor_store_path = Path(factor_cfg["paths"].get("factors_store", "data/factors/factor_store.parquet"))
        if not factor_store_path.is_absolute():
            factor_store_path = get_path(factor_store_path, DATA_FACTORS_DIR)
        
        factor_store = None
        if factor_store_path.exists():
            try:
                factor_store = pd.read_parquet(factor_store_path)
            except Exception as e:
                print(f"[Warn] Failed to load factor_store from parquet: {e}")
        
        # Fallback 到 DuckDB
        if factor_store is None:
            try:
                import duckdb
                db_path = factor_cfg["database"].get("duckdb_path", "data/duckdb/prices.db")
                if not Path(db_path).is_absolute():
                    db_path = get_path(db_path)
                
                con = duckdb.connect(str(db_path))
                # 尝试从 DuckDB 读取因子数据（如果存在）
                try:
                    factor_store = con.execute("SELECT * FROM factor_store").df()
                    if "date" in factor_store.columns and "ticker" in factor_store.columns:
                        factor_store["date"] = pd.to_datetime(factor_store["date"])
                        factor_store = factor_store.set_index(["date", "ticker"]).sort_index()
                except:
                    # 如果 factor_store 表不存在，从 prices 计算（需要实时计算）
                    pass
                con.close()
            except Exception as e:
                print(f"[Warn] Failed to load from DuckDB: {e}")
        
        if factor_store is None or len(factor_store) == 0:
            return jsonify({
                "error": "暂无因子数据，请先运行因子计算",
                "factors": {},
                "date": date
            }), 200
        
        # 确保索引是 MultiIndex
        if not isinstance(factor_store.index, pd.MultiIndex):
            if "date" in factor_store.columns and "ticker" in factor_store.columns:
                factor_store["date"] = pd.to_datetime(factor_store["date"])
                factor_store = factor_store.set_index(["date", "ticker"]).sort_index()
            else:
                return jsonify({
                    "error": "因子数据格式不正确",
                    "factors": {},
                    "date": date
                }), 200
        
        # 获取所有可用日期
        available_dates = pd.to_datetime(factor_store.index.get_level_values(0).unique()).sort_values()
        
        # 如果指定了日期，使用该日期；否则使用最新日期
        if date:
            try:
                date_obj = pd.to_datetime(date)
            except:
                return jsonify({
                    "error": f"日期格式错误: {date}，请使用 YYYY-MM-DD 格式",
                    "factors": {},
                    "date": date
                }), 200
        else:
            # Fallback 到最近交易日
            date_obj = available_dates.max()
            date = date_obj.strftime("%Y-%m-%d")
        
        # 如果指定日期不存在，fallback 到最近交易日
        if date_obj not in available_dates:
            nearest_date = available_dates[available_dates <= date_obj]
            if len(nearest_date) > 0:
                date_obj = nearest_date.max()
                date = date_obj.strftime("%Y-%m-%d")
            else:
                return jsonify({
                    "error": f"日期 {date} 不存在，且没有更早的数据",
                    "factors": {},
                    "date": date
                }), 200
        
        # 获取该日期的因子数据
        date_factors = factor_store.loc[date_obj]
        
        # 转换为字典格式（每个 ticker 的因子值）
        if isinstance(date_factors, pd.DataFrame):
            # 如果是 DataFrame，转换为 {ticker: {factor: value}} 格式
            factors_dict = date_factors.to_dict('index')
        elif isinstance(date_factors, pd.Series):
            # 如果是 Series（单因子），转换为 {ticker: value} 格式
            factors_dict = date_factors.to_dict()
        else:
            factors_dict = {}
        
        return jsonify({
            "error": None,
            "factors": factors_dict,
            "date": date,
            "factor_count": len(factors_dict) if isinstance(factors_dict, dict) else 0,
            "ticker_count": len(date_factors.index.get_level_values(1).unique()) if isinstance(date_factors.index, pd.MultiIndex) else 1
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"数据加载失败，请检查后端服务：{str(e)}",
            "factors": {},
            "date": date,
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


# ============================================================================
# 因子长期监控（Factor Diagnostics）API
# ============================================================================

@app.route('/api/factor-diagnostics/rolling-ic')
def rolling_ic():
    """Rolling IC (60 trading-day window) - Single factor analysis
    
    Query params:
        factor: factor name (required)
    """
    factor_name = request.args.get('factor')
    
    if not factor_name:
        return jsonify({
            "error": "请提供因子名称 (factor parameter required)",
            "dates": [],
            "rolling_ic": [],
            "ic_upper": [],
            "ic_lower": []
        }), 200
    
    try:
        # 读取IC/ICIR数据 - 直接使用 DATA_FACTORS_DIR
        ic_store_path = DATA_FACTORS_DIR / "factor_ic_ir.parquet"
        
        if not ic_store_path.exists():
            return jsonify({
                "error": f"IC数据不存在: {ic_store_path}，请先运行因子IC计算",
                "dates": [],
                "rolling_ic": [],
                "ic_upper": [],
                "ic_lower": []
            }), 200
        
        ic_data = pd.read_parquet(ic_store_path)
        
        if not pd.api.types.is_datetime64_any_dtype(ic_data["date"]):
            ic_data["date"] = pd.to_datetime(ic_data["date"])
        
        # 筛选特定因子
        factor_ic = ic_data[ic_data["factor"] == factor_name].copy()
        
        if len(factor_ic) == 0:
            return jsonify({
                "error": f"因子 {factor_name} 的IC数据不存在",
                "dates": [],
                "rolling_ic": [],
                "ic_upper": [],
                "ic_lower": []
            }), 200
        
        # 按日期排序
        factor_ic = factor_ic.sort_values("date").reset_index(drop=True)
        
        # 使用60个交易日滚动窗口计算Rolling IC
        ROLLING_WINDOW = 60  # 60 trading days
        
        # 计算滚动均值（60天窗口）
        rolling_ic_mean = factor_ic["ic"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
        
        # 计算滚动标准差（用于置信区间）
        rolling_std = factor_ic["ic"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std()
        
        # 计算上下界（均值 ± 1.96 * 标准差）
        ic_upper = rolling_ic_mean + 1.96 * rolling_std
        ic_lower = rolling_ic_mean - 1.96 * rolling_std
        
        # 将NaN转换为None（在JSON中会序列化为null，前端可以处理）
        rolling_ic_list = [None if pd.isna(x) else float(x) for x in rolling_ic_mean]
        ic_upper_list = [None if pd.isna(x) else float(x) for x in ic_upper]
        ic_lower_list = [None if pd.isna(x) else float(x) for x in ic_lower]
        
        return jsonify({
            "error": None,
            "dates": [d.strftime("%Y-%m-%d") for d in factor_ic["date"]],
            "rolling_ic": rolling_ic_list,
            "ic_upper": ic_upper_list,
            "ic_lower": ic_lower_list,
            "factor": factor_name
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"计算Rolling IC失败: {str(e)}",
            "dates": [],
            "rolling_ic": [],
            "ic_upper": [],
            "ic_lower": [],
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/api/factor-diagnostics/rolling-icir')
def rolling_icir():
    """Rolling ICIR (60 trading-day window) - Single factor analysis
    
    ICIR = mean(IC_60d) / std(IC_60d)
    
    Query params:
        factor: factor name (required)
    """
    factor_name = request.args.get('factor')
    
    if not factor_name:
        return jsonify({
            "error": "请提供因子名称 (factor parameter required)",
            "dates": [],
            "rolling_icir": []
        }), 200
    
    try:
        # 读取IC/ICIR数据 - 直接使用 DATA_FACTORS_DIR
        ic_store_path = DATA_FACTORS_DIR / "factor_ic_ir.parquet"
        
        if not ic_store_path.exists():
            return jsonify({
                "error": "IC数据不存在",
                "dates": [],
                "rolling_icir": []
            }), 200
        
        ic_data = pd.read_parquet(ic_store_path)
        if not pd.api.types.is_datetime64_any_dtype(ic_data["date"]):
            ic_data["date"] = pd.to_datetime(ic_data["date"])
        
        # 筛选特定因子
        factor_ic = ic_data[ic_data["factor"] == factor_name].copy()
        
        if len(factor_ic) == 0:
            return jsonify({
                "error": f"因子 {factor_name} 的IC数据不存在",
                "dates": [],
                "rolling_icir": []
            }), 200
        
        # 按日期排序
        factor_ic = factor_ic.sort_values("date").reset_index(drop=True)
        
        # 使用60个交易日滚动窗口计算Rolling ICIR
        ROLLING_WINDOW = 60  # 60 trading days
        
        # 计算60天滚动均值和标准差
        rolling_mean = factor_ic["ic"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
        rolling_std = factor_ic["ic"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std()
        
        # 计算ICIR: ICIR = mean(IC_60d) / std(IC_60d)
        rolling_icir = rolling_mean / rolling_std
        # 将NaN转换为None（当std=0或数据不足60天时）
        rolling_icir = rolling_icir.where(pd.notna(rolling_icir), None)
        
        # 转换为列表，None会保持为None（JSON序列化为null）
        icir_list = [None if pd.isna(x) else float(x) if x is not None else None for x in rolling_icir]
        
        return jsonify({
            "error": None,
            "dates": [d.strftime("%Y-%m-%d") for d in factor_ic["date"]],
            "rolling_icir": icir_list,
            "factor": factor_name
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"计算Rolling ICIR失败: {str(e)}",
            "dates": [],
            "rolling_icir": [],
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/api/factor-diagnostics/rolling-tstat')
def rolling_tstat():
    """Rolling t-stat (60 trading-day window) - Single factor analysis
    
    t-stat = mean(IC_60d) / std(IC_60d) * sqrt(N)
    where N = number of IC observations in the 60-day window (N = 60)
    
    Query params:
        factor: factor name (required)
    """
    factor_name = request.args.get('factor')
    
    if not factor_name:
        return jsonify({
            "error": "请提供因子名称 (factor parameter required)",
            "dates": [],
            "rolling_tstat": []
        }), 200
    
    try:
        # 读取IC/ICIR数据 - 直接使用 DATA_FACTORS_DIR
        ic_store_path = DATA_FACTORS_DIR / "factor_ic_ir.parquet"
        
        if not ic_store_path.exists():
            return jsonify({
                "error": "IC数据不存在",
                "dates": [],
                "rolling_tstat": []
            }), 200
        
        ic_data = pd.read_parquet(ic_store_path)
        if not pd.api.types.is_datetime64_any_dtype(ic_data["date"]):
            ic_data["date"] = pd.to_datetime(ic_data["date"])
        
        ic_data = ic_data.sort_values("date")
        
        # 筛选特定因子
        factor_ic = ic_data[ic_data["factor"] == factor_name].copy()
        
        if len(factor_ic) == 0:
            return jsonify({
                "error": f"因子 {factor_name} 的IC数据不存在",
                "dates": [],
                "rolling_tstat": []
            }), 200
        
        # 按日期排序
        factor_ic = factor_ic.sort_values("date").reset_index(drop=True)
        
        # 使用60个交易日滚动窗口计算Rolling t-stat
        # t-stat = mean(IC_60d) / std(IC_60d) * sqrt(N)
        # where N = number of IC observations in the 60-day window (N = 60)
        ROLLING_WINDOW = 60  # 60 trading days
        
        # 计算60天滚动均值和标准差
        rolling_mean = factor_ic["ic"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
        rolling_std = factor_ic["ic"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std()
        
        # 计算60天窗口内的有效观察数（对于rolling window，N = 60）
        valid_count = factor_ic["ic"].notna().rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).sum()
        
        # 计算t-stat: t = mean / std * sqrt(N)
        # 只有当std > 0且有效观察数 >= 60时才计算，否则为NaN
        rolling_tstat = (rolling_mean / rolling_std) * np.sqrt(valid_count)
        # 将NaN转换为None（当std=0或数据不足60天时）
        rolling_tstat = rolling_tstat.where(pd.notna(rolling_tstat), None)
        
        # 转换为列表，None会保持为None（JSON序列化为null）
        tstat_list = [None if pd.isna(x) else float(x) if x is not None else None for x in rolling_tstat]
        
        return jsonify({
            "error": None,
            "dates": [d.strftime("%Y-%m-%d") for d in factor_ic["date"]],
            "rolling_tstat": tstat_list,
            "factor": factor_name
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"计算Rolling t-stat失败: {str(e)}",
            "dates": [],
            "rolling_tstat": [],
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/api/factor-diagnostics/factors')
def factor_diagnostics_factors():
    """获取所有因子列表 - 从预计算的 JSON 文件中读取"""
    try:
        # 尝试从预计算的 Long-Short 文件中读取因子列表
        precomputed_file = OUTPUT_DIR / "factor_long_short.json"
        
        if precomputed_file.exists():
            with open(precomputed_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            factors = list(data.keys())
            return jsonify({
                "error": None,
                "factors": factors
            })
        
        # Fallback: 尝试从相关性矩阵文件中读取
        corr_file = OUTPUT_DIR / "factor_corr.json"
        if corr_file.exists():
            with open(corr_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            factors = data.get("factors", [])
            return jsonify({
                "error": None,
                "factors": factors
            })
        
        # 如果预计算文件都不存在，返回错误
        return jsonify({
            "error": "预计算的因子数据不存在。请运行 generate_factor_precomputed.py 生成数据。",
            "factors": []
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"获取因子列表失败: {str(e)}",
            "factors": []
        }), 200


@app.route('/api/factor-diagnostics/long-short')
def long_short_performance():
    """Long-Short Portfolio Performance（按因子分层）- 使用预计算的 JSON 文件"""
    factor_name = request.args.get('factor')
    
    if not factor_name:
        return jsonify({
            "error": "请提供因子名称",
            "dates": [],
            "long_returns": [],
            "short_returns": [],
            "long_short_returns": [],
            "stats": {}
        }), 200
    
    try:
        # 读取预计算的 JSON 文件
        precomputed_file = OUTPUT_DIR / "factor_long_short.json"
        
        if not precomputed_file.exists():
            return jsonify({
                "error": "预计算的 Long-Short 数据不存在。请运行 generate_factor_precomputed.py 生成数据。",
                "dates": [],
                "long_returns": [],
                "short_returns": [],
                "long_short_returns": [],
                "stats": {}
            }), 200
        
        print(f"[INFO] 读取预计算的 Long-Short 数据: {precomputed_file}")
        with open(precomputed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if factor_name not in data:
            available_factors = list(data.keys())[:20]  # 显示前20个
            return jsonify({
                "error": f"因子 {factor_name} 不存在。可用因子示例: {available_factors}",
                "dates": [],
                "long_returns": [],
                "short_returns": [],
                "long_short_returns": [],
                "stats": {}
            }), 200
        
        result = data[factor_name]
        
        # 预计算文件现在同时包含daily和cumulative returns
        dates = result.get("dates", [])
        
        # 优先使用预计算的daily returns（如果存在）
        if "long_returns_daily" in result:
            long_returns_daily = result.get("long_returns_daily", [])
            short_returns_daily = result.get("short_returns_daily", [])
            ls_returns_daily = result.get("ls_returns_daily", [])
        else:
            # Fallback: 从cumulative计算daily returns
            long_cum = result.get("long_returns", [])
            short_cum = result.get("short_returns", [])
            ls_cum = result.get("long_short_returns", [])
            
            long_returns_daily = [0.0] if len(long_cum) > 0 else []
            short_returns_daily = [0.0] if len(short_cum) > 0 else []
            ls_returns_daily = [0.0] if len(ls_cum) > 0 else []
            
            for i in range(1, len(dates)):
                if i < len(long_cum) and long_cum[i-1] > 0:
                    long_returns_daily.append((long_cum[i] / long_cum[i-1]) - 1.0)
                else:
                    long_returns_daily.append(0.0)
                
                if i < len(short_cum) and short_cum[i-1] > 0:
                    short_returns_daily.append((short_cum[i] / short_cum[i-1]) - 1.0)
                else:
                    short_returns_daily.append(0.0)
                
                if i < len(ls_cum) and ls_cum[i-1] > 0:
                    ls_returns_daily.append((ls_cum[i] / ls_cum[i-1]) - 1.0)
                else:
                    ls_returns_daily.append(0.0)
        
        # Cumulative returns
        long_cum = result.get("long_returns", [])
        short_cum = result.get("short_returns", [])
        ls_cum = result.get("long_short_returns", [])
        
        # 转换 stats 结构：强调Long-Short的指标
        stats_flat = {}
        if "stats" in result:
            stats = result["stats"]
            # 保留Long和Short用于参考，但强调Long-Short
            if "long" in stats:
                stats_flat["long_annual_return"] = stats["long"].get("annual_return", 0.0)
                stats_flat["long_sharpe"] = stats["long"].get("sharpe", 0.0)
                stats_flat["long_max_dd"] = stats["long"].get("max_dd", 0.0)
            if "short" in stats:
                stats_flat["short_annual_return"] = stats["short"].get("annual_return", 0.0)
                stats_flat["short_sharpe"] = stats["short"].get("sharpe", 0.0)
                stats_flat["short_max_dd"] = stats["short"].get("max_dd", 0.0)
            if "long_short" in stats:
                # 强调Long-Short指标
                stats_flat["long_short_annual_return"] = stats["long_short"].get("annual_return", 0.0)
                stats_flat["long_short_sharpe"] = stats["long_short"].get("sharpe", 0.0)
                stats_flat["long_short_max_dd"] = stats["long_short"].get("max_dd", 0.0)
        
        return jsonify({
            "error": None,
            "dates": dates,
            "long_return": long_returns_daily,  # Daily returns
            "short_return": short_returns_daily,  # Daily returns
            "ls_return": ls_returns_daily,  # Daily returns (long - short)
            "cum_long": long_cum,  # Cumulative returns
            "cum_short": short_cum,  # Cumulative returns
            "cum_ls": ls_cum,  # Cumulative returns (long - short)
            "stats": stats_flat
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"计算Long-Short Performance失败: {str(e)}",
            "dates": [],
            "long_returns": [],
            "short_returns": [],
            "long_short_returns": [],
            "stats": {},
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/api/factor-diagnostics/clusters')
def factor_clusters():
    """Factor Clusters Analysis using Barra-style dominant style attribution
    
    Uses precomputed factor style attribution (from generate_factor_style_attribution.py)
    which computes dominant style based on average style exposure over a rolling window.
    This ensures Alpha factors are classified correctly by their exposure, not name matching.
    """
    try:
        # 读取IC/ICIR数据
        ic_store_path = DATA_FACTORS_DIR / "factor_ic_ir.parquet"
        
        if not ic_store_path.exists():
            return jsonify({
                "error": "IC数据不存在",
                "clusters": {}
            }), 200
        
        ic_data = pd.read_parquet(ic_store_path)
        if not pd.api.types.is_datetime64_any_dtype(ic_data["date"]):
            ic_data["date"] = pd.to_datetime(ic_data["date"])
        
        # 按因子分组，计算统计指标
        factor_stats = ic_data.groupby("factor").agg({
            "ic": ["mean", "std", "count"]
        }).reset_index()
        
        factor_stats.columns = ["factor", "ic_mean", "ic_std", "ic_count"]
        
        # 重新计算 ICIR (IC_mean / IC_std)
        factor_stats["icir"] = factor_stats.apply(
            lambda row: row["ic_mean"] / row["ic_std"] if row["ic_std"] > 0 else None,
            axis=1
        )
        # 将 None 转换为 NaN
        factor_stats["icir"] = factor_stats["icir"].replace([None], np.nan)
        
        # 计算t-stat
        factor_stats["tstat"] = factor_stats["ic_mean"] / (factor_stats["ic_std"] / np.sqrt(factor_stats["ic_count"]))
        factor_stats["tstat"] = factor_stats["tstat"].fillna(0)
        
        # Load canonical matrix (SINGLE SOURCE OF TRUTH - NO FALLBACK)
        canonical_matrix_path = DATA_FACTORS_DIR / "factor_style_exposure.parquet"
        
        if not canonical_matrix_path.exists():
            return jsonify({
                "error": f"Canonical matrix not found: {canonical_matrix_path}. Please run scripts/compute_raw_factor_style_exposure.py first.",
                "clusters": {}
            }), 200
        
        # Load canonical matrix (MANDATORY - no fallback)
        exposure_matrix_df = pd.read_parquet(canonical_matrix_path)
        print(f"[INFO] Loaded canonical matrix (SINGLE SOURCE OF TRUTH): {len(exposure_matrix_df)} exposure records")
        
        # Compute dominant_style and top_exposure for each factor from canonical matrix
        from src.barra_style_mapper import BarraStyleMapper
        mapper = BarraStyleMapper()
        CUSTOM_THRESHOLD = mapper.CUSTOM_THRESHOLD
        
        dominant_styles = []
        top_exposures_list = []
        top_exposures_dict_list = []  # For tooltip display
        
        for _, factor_row in factor_stats.iterrows():
            factor_name = factor_row["factor"]
            factor_exposures = exposure_matrix_df[exposure_matrix_df["factor"] == factor_name]
            
            if factor_exposures.empty:
                # Factor not in matrix, assign Custom
                dominant_styles.append("Custom")
                top_exposures_list.append(f"Below threshold (<{CUSTOM_THRESHOLD})")
                top_exposures_dict_list.append({})
            else:
                # Pivot exposures into dictionary: {style: exposure}
                exposures_dict = {row["style"]: float(row["exposure"]) for _, row in factor_exposures.iterrows()}
                
                # Compute dominant_style = argmax(abs(exposure))
                if exposures_dict:
                    max_abs_exposure = max(abs(v) for v in exposures_dict.values())
                    
                    # AUTHORITATIVE RULE: If max_abs_exposure < threshold, assign Custom
                    if max_abs_exposure < CUSTOM_THRESHOLD:
                        dominant_style = "Custom"
                        # Custom factors: TopStyleExposure MUST be "Below threshold (<threshold>)"
                        # NEVER show a style name or numeric exposure
                        top_exposures_list.append(f"Below threshold (<{CUSTOM_THRESHOLD})")
                        top_exposures_dict_list.append({})  # No style exposures for Custom
                    else:
                        # Valid style factor: assign dominant style and show exposure
                        dominant_style = max(exposures_dict.items(), key=lambda x: abs(x[1]))[0]
                        # Get top 2 exposures for tooltip (sorted by absolute value)
                        sorted_exposures = sorted(exposures_dict.items(), key=lambda x: abs(x[1]), reverse=True)
                        top_2_exposures = dict(sorted_exposures[:2])
                        top_exposures_list.append(f"{dominant_style}: {max_abs_exposure:.4f}")
                        top_exposures_dict_list.append(top_2_exposures)
                    
                    dominant_styles.append(dominant_style)
                else:
                    # Empty exposures_dict, assign Custom
                    dominant_styles.append("Custom")
                    top_exposures_list.append(f"Below threshold (<{CUSTOM_THRESHOLD})")
                    top_exposures_dict_list.append({})
        
        # Add computed columns to factor_stats
        factor_stats = factor_stats.copy()
        factor_stats["dominant_style"] = dominant_styles
        factor_stats["top_style_exposure"] = top_exposures_list
        factor_stats["top_exposures_dict"] = top_exposures_dict_list  # For API response
        
        # Initialize clusters with canonical Barra-style categories (SINGLE SOURCE OF TRUTH)
        from src.barra_style_mapper import BarraStyleMapper
        canonical_styles = BarraStyleMapper.CANONICAL_STYLES + ["Custom"]  # Custom is fallback only
        clusters = {style: [] for style in canonical_styles}
        
        # Group factors by dominant style
        for _, row in factor_stats.iterrows():
            dominant_style = row["dominant_style"]
            
            # Ensure style is in canonical list (fallback to Custom if not)
            if dominant_style not in canonical_styles:
                dominant_style = "Custom"
            
            # ICIR 如果是 NaN，不转换为 0，而是保持为 None（前端会处理）
            icir_value = row["icir"]
            if pd.isna(icir_value):
                icir_value = None
            else:
                icir_value = float(icir_value)
            
            # Get top exposures from precomputed dict (from canonical matrix)
            top_exposures = row.get("top_exposures_dict", {})
            
            # Ensure top_exposures is a dict (never N/A)
            if not isinstance(top_exposures, dict):
                top_exposures = {}
            
            # Get top_style_exposure (string format: "Style: value" or "Below threshold (<threshold>)")
            top_style_exposure_str = row.get("top_style_exposure", f"Below threshold (<{CUSTOM_THRESHOLD})")
            if not isinstance(top_style_exposure_str, str):
                # Legacy numeric format, convert to string
                if dominant_style == "Custom":
                    top_style_exposure_str = f"Below threshold (<{CUSTOM_THRESHOLD})"
                else:
                    top_style_exposure_str = f"{dominant_style}: {float(top_style_exposure_str):.4f}"
            
            # HARD VALIDATION: Ensure Custom factors NEVER show style name or number
            if dominant_style == "Custom":
                if ":" in top_style_exposure_str or any(char.isdigit() and float(char) >= CUSTOM_THRESHOLD for char in top_style_exposure_str.split() if char.replace('.', '').replace('-', '').isdigit()):
                    # Invalid: Custom with style name/number, enforce correct format
                    top_style_exposure_str = f"Below threshold (<{CUSTOM_THRESHOLD})"
                    top_exposures = {}  # Clear exposures dict
            
            # Detect degenerate factors (zero IC, N/A ICIR, zero exposure)
            ic_mean_val = float(row["ic_mean"]) if not pd.isna(row["ic_mean"]) else 0.0
            is_degenerate = False
            
            # Check if factor is degenerate:
            # - Mean IC is exactly 0 (constant/degenerate factor)
            # - ICIR is N/A (insufficient data)
            # - Dominant style is Custom AND max exposure is exactly 0
            if abs(ic_mean_val) < 1e-8:
                is_degenerate = True
            if icir_value is None:
                is_degenerate = True
            if dominant_style == "Custom" and top_exposures == {} and "Below threshold" not in str(top_style_exposure_str):
                is_degenerate = True
            
            # Store degenerate flag for filtering
            factor_data = {
                "name": row["factor"],
                "ic_mean": ic_mean_val,
                "icir": icir_value,  # May be None (NaN)
                "tstat": float(row["tstat"]) if not pd.isna(row["tstat"]) else 0.0,
                "dominant_style": dominant_style,
                "top_style_exposure": top_style_exposure_str,  # String format
                "top_exposures": top_exposures,  # Dict with top 2 exposures for tooltip
                "is_degenerate": is_degenerate  # Flag for filtering
            }
            
            clusters[dominant_style].append(factor_data)
        
        # Separate Custom factors and valid economic styles
        # Custom is NOT an economic style, so we separate it for presentation
        custom_factors = clusters.pop("Custom", [])
        
        # Filter out degenerate factors from economic style clusters
        valid_clusters = {}
        degenerate_factors = []
        
        for style, factors in clusters.items():
            valid_factors = []
            for factor in factors:
                if factor.get("is_degenerate", False):
                    degenerate_factors.append(factor)
                else:
                    valid_factors.append(factor)
            
            if valid_factors:  # Only include styles with valid factors
                valid_clusters[style] = valid_factors
        
        # Remove empty clusters
        valid_clusters = {k: v for k, v in valid_clusters.items() if len(v) > 0}
        
        return jsonify({
            "error": None,
            "clusters": valid_clusters,  # Only economic styles (no Custom)
            "custom_factors": custom_factors,  # Custom factors (separate, not plotted)
            "degenerate_factors": degenerate_factors,  # Degenerate factors (excluded)
            "metadata": {
                "custom_count": len(custom_factors),
                "degenerate_count": len(degenerate_factors),
                "total_factors": len(factor_stats)
            }
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"计算因子簇分析失败: {str(e)}",
            "clusters": {},
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/api/factor-diagnostics/correlation')
def factor_correlation():
    """因子相关性矩阵 - 支持 Top-K 因子选择（按 |ICIR| 排序）"""
    method = request.args.get('method', 'pearson')
    top_k_str = request.args.get('top_k', '50')
    
    # Parse and validate top_k parameter
    try:
        top_k = int(top_k_str)
    except (ValueError, TypeError):
        top_k = 50  # Default to 50 if invalid
    
    allowed_top_k = {20, 50, 100}
    if top_k not in allowed_top_k:
        return jsonify({
            "error": f"Invalid top_k={top_k}. Allowed values: {sorted(allowed_top_k)}",
            "factors": [],
            "correlation_matrix": []
        }), 400
    
    try:
        # 读取预计算的 JSON 文件
        precomputed_file = OUTPUT_DIR / "factor_corr.json"
        
        if not precomputed_file.exists():
            return jsonify({
                "error": "预计算的相关性数据不存在。请运行 generate_factor_precomputed.py 生成数据。",
                "factors": [],
                "correlation_matrix": []
            }), 200
        
        print(f"[INFO] 读取预计算的相关性数据: {precomputed_file}")
        with open(precomputed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查方法是否匹配（目前只支持 pearson）
        if data.get("method") != method:
            return jsonify({
                "error": f"预计算数据使用的方法 ({data.get('method')}) 与请求的方法 ({method}) 不匹配。目前只支持 pearson。",
                "factors": [],
                "correlation_matrix": []
            }), 200
        
        all_factors = data["factors"]
        all_corr_matrix = np.array(data["correlation_matrix"])
        
        # If top_k is None or >= all factors, return all factors
        if top_k >= len(all_factors):
            return jsonify({
                "error": None,
                "factors": all_factors,
                "correlation_matrix": data["correlation_matrix"],
                "top_k": len(all_factors),
                "method": method
            })
        
        # Load IC/ICIR data to rank factors by |ICIR|
        ic_store_path = DATA_FACTORS_DIR / "factor_ic_ir.parquet"
        
        if not ic_store_path.exists():
            # Fallback: use first top_k factors if IC data not available
            print(f"[WARN] IC data not found, using first {top_k} factors")
            selected_factors = all_factors[:top_k]
            factor_indices = [all_factors.index(f) for f in selected_factors]
            selected_corr_matrix = all_corr_matrix[np.ix_(factor_indices, factor_indices)].tolist()
            
            return jsonify({
                "error": None,
                "factors": selected_factors,
                "correlation_matrix": selected_corr_matrix,
                "top_k": top_k,
                "method": method,
                "warning": "IC data not available, factors selected by order"
            })
        
        # Load IC data and compute ICIR for ranking
        ic_data = pd.read_parquet(ic_store_path)
        if not pd.api.types.is_datetime64_any_dtype(ic_data["date"]):
            ic_data["date"] = pd.to_datetime(ic_data["date"])
        
        # Compute factor statistics (ICIR for ranking)
        factor_stats = ic_data.groupby("factor").agg({
            "ic": ["mean", "std"]
        }).reset_index()
        factor_stats.columns = ["factor", "ic_mean", "ic_std"]
        
        # Compute ICIR (IC_mean / IC_std), fallback to |IC_mean| if ICIR unavailable
        factor_stats["icir"] = factor_stats.apply(
            lambda row: row["ic_mean"] / row["ic_std"] if row["ic_std"] > 1e-8 else np.nan,
            axis=1
        )
        factor_stats["abs_icir"] = factor_stats["icir"].abs().fillna(factor_stats["ic_mean"].abs())
        
        # Create ranking dictionary: factor -> |ICIR| (or |IC_mean| as fallback)
        factor_ranking = dict(zip(factor_stats["factor"], factor_stats["abs_icir"]))
        
        # Rank all factors in the correlation matrix by |ICIR|
        factor_scores = []
        for factor in all_factors:
            score = factor_ranking.get(factor, 0.0)
            factor_scores.append((factor, score))
        
        # Sort by score descending and take actual_top_k
        factor_scores_sorted = sorted(factor_scores, key=lambda x: x[1], reverse=True)
        selected_factors = [f[0] for f in factor_scores_sorted[:actual_top_k]]
        
        # Get indices of selected factors in the original matrix
        factor_to_index = {f: i for i, f in enumerate(all_factors)}
        selected_indices = [factor_to_index[f] for f in selected_factors if f in factor_to_index]
        
        # Extract submatrix for selected factors
        if len(selected_indices) == actual_top_k:
            selected_corr_matrix = all_corr_matrix[np.ix_(selected_indices, selected_indices)].tolist()
        else:
            # Handle case where some factors are missing from correlation matrix
            print(f"[WARN] Only {len(selected_indices)}/{actual_top_k} factors found in correlation matrix")
            selected_factors = [all_factors[i] for i in selected_indices]
            selected_corr_matrix = all_corr_matrix[np.ix_(selected_indices, selected_indices)].tolist()
        
        print(f"[INFO] Selected top {len(selected_factors)} factors by |ICIR| for correlation matrix (requested: {top_k})")
        
        return jsonify({
            "error": None,
            "factors": selected_factors,
            "correlation_matrix": selected_corr_matrix,
            "top_k": len(selected_factors),
            "method": method
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"计算相关性矩阵失败: {str(e)}",
            "factors": [],
            "correlation_matrix": [],
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/api/factor-diagnostics/risk-exposure')
def risk_exposure():
    """多因子风险暴露（Barra-style）- 使用预计算的 JSON 文件"""
    date = request.args.get('date')
    
    try:
        # 读取预计算的 JSON 文件
        precomputed_file = OUTPUT_DIR / "factor_exposure.json"
        
        if not precomputed_file.exists():
            return jsonify({
                "error": "预计算的风险暴露数据不存在。请运行 generate_factor_precomputed.py 生成数据。",
                "factors": [],
                "exposures": [],
                "risk_contributions": []
            }), 200
        
        print(f"[INFO] 读取预计算的风险暴露数据: {precomputed_file}")
        with open(precomputed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 确定日期
        available_dates = sorted(data.keys())
        if date:
            # 查找最接近的日期
            date_obj = pd.to_datetime(date)
            nearest_date = None
            for d in available_dates:
                if pd.to_datetime(d) <= date_obj:
                    nearest_date = d
            if nearest_date:
                selected_date = nearest_date
            else:
                selected_date = available_dates[-1] if available_dates else None
        else:
            selected_date = available_dates[-1] if available_dates else None
        
        if not selected_date or selected_date not in data:
            return jsonify({
                "error": f"日期 {date} 的数据不存在。可用日期: {available_dates[-10:]}",
                "factors": [],
                "exposures": [],
                "risk_contributions": []
            }), 200
        
        result = data[selected_date]
        
        # VALIDATION: Filter out raw factor names - only allow style factor names
        # Style factors are: Price/Level, Trend, Momentum, Volatility, Liquidity, Quality/Stability, Custom
        valid_style_factors = {
            'Price/Level', 'Trend', 'Momentum', 'Volatility', 'Liquidity', 
            'Quality/Stability', 'Custom'
        }
        
        # Patterns that indicate raw factors (should be filtered out)
        raw_factor_patterns = [
            'Alpha', 'AD', 'OBV', 'ADOSC', 'BB_', 'SMA_', 'EMA_', 'WMA_', 
            'DEMA_', 'RSI_', 'CCI_', 'STOCH', 'WILLR_', 'AROON', 'MACD',
            'MOM_', 'ROC_', 'MFI_', 'ATR_', 'NATR_', 'BOP', 'CUSTOM_'
        ]
        
        def is_raw_factor(factor_name: str) -> bool:
            """Check if factor_name is a raw factor (not a style factor)"""
            if factor_name in valid_style_factors:
                return False
            # Check if it matches any raw factor pattern
            for pattern in raw_factor_patterns:
                if pattern in factor_name:
                    return True
            return False
        
        # Filter factors, exposures, and risk_contributions to only include style factors
        filtered_factors = []
        filtered_exposures = []
        filtered_risk_contributions = []
        
        for i, factor_name in enumerate(result["factors"]):
            if not is_raw_factor(factor_name):
                filtered_factors.append(factor_name)
                filtered_exposures.append(result["exposures"][i])
                filtered_risk_contributions.append(result["risk_contributions"][i])
            else:
                # Log warning for debugging
                print(f"[WARN] Filtering out raw factor '{factor_name}' from risk exposure results")
        
        # If no valid style factors found, return error
        if len(filtered_factors) == 0:
            return jsonify({
                "error": f"数据包含原始因子名称而非风格因子。请重新运行 generate_barra_risk_exposure.py 生成数据。",
                "factors": [],
                "exposures": [],
                "risk_contributions": []
            }), 200
        
        return jsonify({
            "error": None,
            "date": selected_date,
            "factors": filtered_factors,
            "exposures": filtered_exposures,
            "risk_contributions": filtered_risk_contributions,
            "specific_risk_contribution": result.get("specific_risk_contribution", 0),
            "specific_risk": result.get("specific_risk", None)
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"计算风险暴露失败: {str(e)}",
            "factors": [],
            "exposures": [],
            "risk_contributions": [],
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/api/factor-diagnostics/latest-date')
def factor_diagnostics_latest_date():
    """获取最新日期"""
    try:
        factor_cfg = load_factor_settings(str(SETTINGS))
        factor_store_path = Path(factor_cfg["paths"].get("factors_store", "data/factors/factor_store.parquet"))
        if not factor_store_path.is_absolute():
            factor_store_path = get_path(factor_store_path, DATA_FACTORS_DIR)
        
        if not factor_store_path.exists():
            return jsonify({"error": "因子数据不存在", "date": None}), 200
        
        factor_store = pd.read_parquet(factor_store_path)
        if not isinstance(factor_store.index, pd.MultiIndex):
            if "date" in factor_store.columns and "ticker" in factor_store.columns:
                factor_store["date"] = pd.to_datetime(factor_store["date"])
                factor_store = factor_store.set_index(["date", "ticker"]).sort_index()
        
        latest_date = factor_store.index.get_level_values(0).max()
        
        return jsonify({
            "error": None,
            "date": latest_date.strftime("%Y-%m-%d")
        })
    except Exception as e:
        return jsonify({
            "error": f"获取最新日期失败: {str(e)}",
            "date": None
        }), 200


# ============================================================================
# 旧的因子看报API（保留用于兼容，但不再使用）
# ============================================================================

@app.route('/api/factor-report/daily')
def factor_report_daily():
    """获取每日因子报告数据 - 使用每日因子更新模块生成的IC/ICIR数据"""
    date = request.args.get('date')
    
    try:
        # 读取因子数据
        factor_cfg = load_factor_settings(str(SETTINGS))
        factor_store_path = Path(factor_cfg["paths"].get("factors_store", "data/factors/factor_store.parquet"))
        if not factor_store_path.is_absolute():
            factor_store_path = get_path(factor_store_path, DATA_FACTORS_DIR)
        
        # 读取IC/ICIR数据（从每日因子更新模块生成）
        ic_store_path = factor_store_path.parent / "factor_ic_ir.parquet"
        
        factor_store = None
        if factor_store_path.exists():
            try:
                factor_store = pd.read_parquet(factor_store_path)
            except:
                pass
        
        if factor_store is None or len(factor_store) == 0:
            return jsonify({
                "error": "暂无因子数据，请先运行因子计算",
                "latest_date": None,
                "factor_summary": {},
                "available_dates": [],
                "date_factors": {}
            }), 200
        
        # 确保索引是 MultiIndex
        if not isinstance(factor_store.index, pd.MultiIndex):
            if "date" in factor_store.columns and "ticker" in factor_store.columns:
                factor_store["date"] = pd.to_datetime(factor_store["date"])
                factor_store = factor_store.set_index(["date", "ticker"]).sort_index()
        
        # 获取所有可用日期
        available_dates = pd.to_datetime(factor_store.index.get_level_values(0).unique()).sort_values()
        latest_date = available_dates.max()
        
        # 确定日期（自动 fallback 到最近日期）
        if date:
            try:
                date_obj = pd.to_datetime(date)
                if date_obj not in available_dates:
                    # Fallback 到最近日期
                    nearest_date = available_dates[available_dates <= date_obj]
                    if len(nearest_date) > 0:
                        date_obj = nearest_date.max()
                        date = date_obj.strftime("%Y-%m-%d")
                    else:
                        # 如果指定日期太早，使用最早日期
                        date_obj = available_dates.min()
                        date = date_obj.strftime("%Y-%m-%d")
            except:
                date_obj = latest_date
                date = latest_date.strftime("%Y-%m-%d")
        else:
            date_obj = latest_date
            date = latest_date.strftime("%Y-%m-%d")
        
        # 读取IC/ICIR数据
        date_factors = {}
        if ic_store_path.exists():
            try:
                ic_data = pd.read_parquet(ic_store_path)
                # 确保日期列是datetime类型
                if not pd.api.types.is_datetime64_any_dtype(ic_data["date"]):
                    ic_data["date"] = pd.to_datetime(ic_data["date"])
                
                # 尝试匹配日期（精确匹配或fallback到最近日期）
                date_ic_data = ic_data[ic_data["date"] == date_obj]
                
                # 如果精确匹配没有数据，fallback到最近日期
                if len(date_ic_data) == 0:
                    available_ic_dates = pd.to_datetime(ic_data["date"].unique()).sort_values()
                    nearest_ic_date = available_ic_dates[available_ic_dates <= date_obj]
                    if len(nearest_ic_date) > 0:
                        nearest_ic_date = nearest_ic_date.max()
                        date_ic_data = ic_data[ic_data["date"] == nearest_ic_date]
                        print(f"[Info] IC数据日期fallback: {date_obj.strftime('%Y-%m-%d')} -> {nearest_ic_date.strftime('%Y-%m-%d')}")
                
                for _, row in date_ic_data.iterrows():
                    factor_name = str(row["factor"])
                    date_factors[factor_name] = {
                        "ic": float(row["ic"]) if not pd.isna(row["ic"]) else 0.0,
                        "icir": float(row["icir"]) if not pd.isna(row["icir"]) else 0.0,
                        "win_rate": 0.5  # 需要从历史IC序列计算
                    }
            except Exception as e:
                print(f"[Warn] Failed to read IC/ICIR data: {e}")
                import traceback
                traceback.print_exc()
        
        # 如果没有IC/ICIR数据，尝试实时计算（fallback）
        if not date_factors:
            from src.factor_engine import read_prices, daily_rank_ic, forward_return
            
            prices = read_prices(cfg)
            if prices is not None and len(prices) > 0:
                forward_ret = forward_return(prices, horizon=1)
                date_forward_ret = forward_ret.loc[forward_ret.index.get_level_values(0) == date_obj]
                
                date_factors_data = factor_store.loc[date_obj]
                if isinstance(date_factors_data, pd.DataFrame):
                    for factor_name in date_factors_data.columns:
                        factor_series = date_factors_data[factor_name]
                        aligned = pd.concat([factor_series, date_forward_ret], axis=1).dropna()
                        if len(aligned) < 10:
                            continue
                        
                        try:
                            ic = aligned.iloc[:, 0].rank().corr(aligned.iloc[:, 1].rank(), method='spearman')
                            if pd.isna(ic):
                                continue
                            
                            factor_all = factor_store[factor_name]
                            ic_series = daily_rank_ic(factor_all, forward_ret)
                            ic_mean = float(ic_series.mean()) if len(ic_series) > 0 else float(ic)
                            ic_std = float(ic_series.std()) if len(ic_series) > 0 and ic_series.std() > 0 else 1.0
                            icir = ic_mean / ic_std if ic_std > 0 else 0.0
                            win_rate = float((ic_series > 0).mean()) if len(ic_series) > 0 else 0.5
                            
                            date_factors[str(factor_name)] = {
                                "ic": float(ic),
                                "icir": icir,
                                "win_rate": win_rate
                            }
                        except:
                            continue
        
        return jsonify({
            "error": None,
            "latest_date": latest_date.strftime("%Y-%m-%d"),
            "factor_summary": {},
            "available_dates": [d.strftime("%Y-%m-%d") for d in available_dates],
            "date_factors": date_factors,
            "selected_date": date
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"计算每日因子IC失败: {str(e)}",
            "latest_date": None,
            "factor_summary": {},
            "available_dates": [],
            "date_factors": {},
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/api/factor-report/top-factors')
def top_factors():
    """获取表现最好的因子 - 计算每日真实的IC/ICIR（不是历史整体回测）"""
    date = request.args.get('date')
    
    try:
        # 使用 src/factor_engine.py 的真实函数
        from src.factor_engine import read_prices, daily_rank_ic, forward_return
        
        # 读取价格数据
        prices = read_prices(cfg)
        if prices is None or len(prices) == 0:
            return jsonify({"error": "暂无价格数据", "top_factors": []}), 200
        
        # 读取因子数据
        factor_cfg = load_factor_settings(str(SETTINGS))
        factor_store_path = Path(factor_cfg["paths"].get("factors_store", "data/factors/factor_store.parquet"))
        if not factor_store_path.is_absolute():
            factor_store_path = get_path(factor_store_path, DATA_FACTORS_DIR)
        
        factor_store = None
        if factor_store_path.exists():
            try:
                factor_store = pd.read_parquet(factor_store_path)
            except:
                pass
        
        if factor_store is None or len(factor_store) == 0:
            return jsonify({"error": "暂无因子数据", "top_factors": []}), 200
        
        # 确保索引是 MultiIndex
        if not isinstance(factor_store.index, pd.MultiIndex):
            if "date" in factor_store.columns and "ticker" in factor_store.columns:
                factor_store["date"] = pd.to_datetime(factor_store["date"])
                factor_store = factor_store.set_index(["date", "ticker"]).sort_index()
        
        # 确定日期
        available_dates = pd.to_datetime(factor_store.index.get_level_values(0).unique()).sort_values()
        if date:
            try:
                date_obj = pd.to_datetime(date)
                if date_obj not in available_dates:
                    nearest_date = available_dates[available_dates <= date_obj]
                    if len(nearest_date) > 0:
                        date_obj = nearest_date.max()
                    else:
                        return jsonify({"error": f"日期 {date} 不存在", "top_factors": []}), 200
            except:
                date_obj = available_dates.max()
        else:
            date_obj = available_dates.max()
        
        # 计算该日期的未来1日收益（用于计算IC）
        close_prices = prices["Adj Close"].unstack("ticker")
        forward_ret = forward_return(prices, horizon=1)  # 未来1日收益
        
        # 对齐日期
        common_dates = factor_store.index.get_level_values(0).intersection(forward_ret.index)
        if date_obj not in common_dates:
            # 如果指定日期没有未来收益数据，使用最近有数据的日期
            common_dates_before = common_dates[common_dates <= date_obj]
            if len(common_dates_before) > 0:
                date_obj = common_dates_before.max()
            else:
                return jsonify({"error": f"日期 {date_obj} 没有未来收益数据", "top_factors": []}), 200
        
        # 获取该日期的因子数据
        date_factors = factor_store.loc[date_obj]
        date_forward_ret = forward_ret.loc[date_obj]
        
        # 计算每个因子在该日期的IC和ICIR
        factors_list = []
        for factor_name in date_factors.columns:
            if isinstance(date_factors, pd.DataFrame):
                factor_series = date_factors[factor_name]
            else:
                # 如果是Series，只有一个因子
                factor_series = date_factors
                factor_name = date_factors.name if hasattr(date_factors, 'name') else "factor"
            
            # 对齐索引
            aligned = pd.concat([factor_series, date_forward_ret], axis=1).dropna()
            if len(aligned) < 10:  # 至少需要10个样本
                continue
            
            # 计算该日期的Rank IC（Spearman相关系数）
            try:
                ic = aligned.iloc[:, 0].rank().corr(aligned.iloc[:, 1].rank(), method='spearman')
                if pd.isna(ic):
                    continue
                
                # 计算该因子在所有日期的IC序列（用于计算ICIR）
                factor_all = factor_store[factor_name]
                ic_series = daily_rank_ic(factor_all, forward_ret)
                
                # 计算ICIR（IC均值/IC标准差）
                ic_mean = float(ic_series.mean()) if len(ic_series) > 0 else float(ic)
                ic_std = float(ic_series.std()) if len(ic_series) > 0 and ic_series.std() > 0 else 1.0
                icir = ic_mean / ic_std if ic_std > 0 else 0.0
                
                # 计算胜率（IC > 0 的比例）
                win_rate = float((ic_series > 0).mean()) if len(ic_series) > 0 else 0.5
                
                factors_list.append({
                    "name": str(factor_name),
                    "ic": float(ic),  # 该日期的IC
                    "icir": icir,  # 历史ICIR
                    "win_rate": win_rate  # 历史胜率
                })
            except:
                continue
        
        # 按该日期的IC绝对值降序排序
        factors_list.sort(key=lambda x: abs(x["ic"]), reverse=True)
        
        return jsonify({
            "error": None,
            "top_factors": factors_list[:20] if factors_list else [],
            "date": date_obj.strftime("%Y-%m-%d")
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"计算因子IC失败: {str(e)}",
            "top_factors": [],
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/positions')
def positions():
    """每日真实持仓页面"""
    return render_template('positions.html', api_base_url=API_BASE_URL)


@app.route('/api/ibkr/positions')
def ibkr_positions():
    """获取 IBKR 真实持仓 - 使用 src/ibkr_live_trader.py 的真实函数"""
    if not IBKR_CONFIG.get('enabled', False):
        return jsonify({
            "error": "IBKR 未启用",
            "positions": [],
            "source": "backtest"
        }), 200
    
    if not IBKR_AVAILABLE:
        return jsonify({
            "error": "IBKR 功能不可用：ib_insync 未安装",
            "message": "请安装 ib_insync: pip install ib_insync",
            "positions": [],
            "source": "error"
        }), 503
    
    try:
        # 使用 src/ibkr_live_trader.py 的真实类
        trader = IBKRLiveTrader(
            host=IBKR_CONFIG['host'],
            port=IBKR_CONFIG['port'],
            client_id=IBKR_CONFIG['client_id']
        )
        
        # 连接并获取持仓
        trader.connect()
        positions_dict = trader.get_current_positions()
        
        # 获取实时价格和持仓详情
        positions_data = []
        total_value = 0.0
        
        for ticker, shares in positions_dict.items():
            # 获取实时价格（使用真实函数）
            price = trader.get_realtime_price(ticker)
            
            # 从 IBKR positions 获取更多信息
            for pos in trader.ib.positions():
                if pos.contract.secType == "STK" and pos.contract.symbol == ticker:
                    # 使用 IBKR 提供的平均成本
                    avg_cost = pos.avgCost if hasattr(pos, 'avgCost') else price
                    value = abs(shares) * price if price and price > 0 else abs(shares) * avg_cost
                    total_value += value
                    
                    positions_data.append({
                        "symbol": ticker,
                        "ticker": ticker,
                        "quantity": float(shares),
                        "price": float(price) if price and price > 0 else float(avg_cost),
                        "avg_cost": float(avg_cost),
                        "value": value,
                        "market_value": value,
                        "unrealized_pnl": (price - avg_cost) * shares if price and price > 0 else 0.0
                    })
                    break
            else:
                # 如果没有找到详细持仓信息，使用基本数据
                value = abs(shares) * price if price and price > 0 else 0.0
                total_value += value
                positions_data.append({
                    "symbol": ticker,
                    "ticker": ticker,
                    "quantity": float(shares),
                    "price": float(price) if price and price > 0 else 0.0,
                    "avg_cost": float(price) if price and price > 0 else 0.0,
                    "value": value,
                    "market_value": value,
                    "unrealized_pnl": 0.0
                })
        
        # 计算权重（基于总价值）
        for pos in positions_data:
            if total_value > 0 and pos.get("value", 0) > 0:
                pos["weight"] = pos["value"] / total_value
            else:
                pos["weight"] = 0.0
        
        trader.disconnect()
        
        return jsonify({
            "error": None,
            "positions": positions_data,
            "total_value": total_value,
            "source": "ibkr",
            "count": len(positions_data)
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"IBKR 连接失败: {str(e)}",
            "positions": [],
            "source": "backtest",
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/api/positions/current')
def current_positions():
    """获取当前持仓 - 优先使用 IBKR 文件数据，否则使用真实权重文件"""
    date = request.args.get('date')  # 支持日期参数
    
    # 先尝试从文件读取 IBKR 持仓数据
    ibkr_positions_file = OUTPUT_IBKR_DATA_DIR / "positions.json"
    if ibkr_positions_file.exists():
        try:
            with open(ibkr_positions_file, 'r', encoding='utf-8') as f:
                ibkr_data = json.load(f)
            if ibkr_data and ibkr_data.get("positions"):
                positions = ibkr_data.get("positions", [])
                total_value = ibkr_data.get("total_value", 0.0)
                
                # 为 IBKR 数据添加 weight 字段（基于 market_value）
                for pos in positions:
                    market_value = pos.get("market_value", 0.0) or pos.get("value", 0.0)
                    if total_value > 0 and market_value > 0:
                        pos["weight"] = market_value / total_value
                    else:
                        pos["weight"] = 0.0
                    # 确保有所有必要的字段
                    if "quantity" not in pos:
                        pos["quantity"] = pos.get("shares", 0.0)
                    if "price" not in pos:
                        pos["price"] = pos.get("current_price", 0.0)
                    if "value" not in pos:
                        pos["value"] = market_value
                
                return jsonify({
                    "error": None,
                    "date": ibkr_data.get("timestamp", datetime.now().strftime("%Y-%m-%d")).split('_')[0],
                    "positions": positions,
                    "total_stocks": ibkr_data.get("total_stocks", 0),
                    "total_value": total_value,
                    "source": "ibkr_file"
                })
        except Exception as e:
            print(f"[Warn] Failed to load IBKR positions from file: {e}")
    
    # 再尝试实时获取 IBKR 持仓
    if IBKR_CONFIG.get('enabled', False):
        try:
            ibkr_result = ibkr_positions()
            ibkr_data = ibkr_result.get_json()
            if ibkr_data and not ibkr_data.get("error") and ibkr_data.get("source") == "ibkr":
                return jsonify({
                    "error": None,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "positions": ibkr_data.get("positions", []),
                    "total_stocks": len(ibkr_data.get("positions", [])),
                    "total_value": ibkr_data.get("total_value", 0),
                    "source": "ibkr"
                })
        except Exception as e:
            print(f"[Warn] Failed to get IBKR positions in real-time: {e}")
    
    # 回退到真实权重文件（从策略优化器生成）
    weights_path = PORTFOLIO_DIR / "weights.parquet"
    if not weights_path.exists():
        weights_path = PORTFOLIO_DIR / "weights.parquet"
    
    weights = load_parquet_safe(weights_path)
    
    if weights is None:
        return jsonify({
            "error": "暂无权重数据，请先运行策略优化器生成权重",
            "date": None,
            "positions": [],
            "total_stocks": 0,
            "total_value": 0,
            "source": "backtest"
        }), 200
    
    # 确定日期
    if isinstance(weights.index, pd.MultiIndex):
        available_dates = pd.to_datetime(weights.index.get_level_values(0).unique()).sort_values()
    else:
        available_dates = pd.to_datetime(weights.index.unique()).sort_values()
    
    if date:
        try:
            date_obj = pd.to_datetime(date)
            if date_obj not in available_dates:
                nearest_date = available_dates[available_dates <= date_obj]
                if len(nearest_date) > 0:
                    date_obj = nearest_date.max()
                else:
                    return jsonify({
                        "error": f"日期 {date} 不存在",
                        "date": None,
                        "positions": [],
                        "total_stocks": 0,
                        "total_value": 0,
                        "source": "backtest"
                    }), 200
        except:
            date_obj = available_dates.max()
    else:
        date_obj = available_dates.max()
    
    # 获取该日期的权重
    if isinstance(weights.index, pd.MultiIndex):
        date_weights = weights.loc[date_obj]
        if isinstance(date_weights, pd.DataFrame):
            # 如果是DataFrame，取第一列或计算总和
            latest_weights = date_weights.iloc[:, 0] if len(date_weights.columns) > 0 else pd.Series(dtype=float)
        else:
            latest_weights = date_weights
    else:
        latest_weights = weights.loc[date_obj]
    
    # 只保留正权重
    if isinstance(latest_weights, pd.Series):
        latest_weights = latest_weights[latest_weights > 0].sort_values(ascending=False)
    else:
        latest_weights = pd.Series(dtype=float)
    
    # 加载价格数据计算持仓价值（使用 src/factor_engine.py 的真实函数）
    try:
        # 确保路径是绝对路径
        cfg_copy = cfg.copy()
        if "paths" in cfg_copy and "prices_parquet" in cfg_copy["paths"]:
            parquet_path = cfg_copy["paths"]["prices_parquet"]
            if not Path(parquet_path).is_absolute():
                cfg_copy["paths"]["prices_parquet"] = str(get_path(parquet_path))
        
        prices = read_prices(cfg_copy)
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"无法加载价格数据: {str(e)}",
            "date": date_obj.strftime("%Y-%m-%d") if 'date_obj' in locals() else None,
            "positions": [],
            "total_stocks": 0,
            "total_value": 0,
            "source": "backtest",
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200
    
    positions_data = []
    total_value = 0.0
    base_capital = 100000.0  # 默认资金
    
    # 使用配置中的初始资金
    base_capital = float(cfg.get("portfolio", {}).get("initial_capital", 100000.0))
    
    if prices is not None and len(prices) > 0:
        # 处理 MultiIndex - 获取最新可用日期的价格
        if isinstance(prices.index, pd.MultiIndex):
            date_level = pd.to_datetime(prices.index.get_level_values(0))
            ticker_level = prices.index.get_level_values(1)
            
            # 找到最接近的日期（<= date_obj）
            available_dates = date_level.unique()
            nearest_date = available_dates[available_dates <= pd.to_datetime(date_obj)]
            
            if len(nearest_date) > 0:
                target_date = pd.to_datetime(nearest_date.max())
                print(f"[Debug] Using price date: {target_date} for position date: {date_obj}")
                
                # 获取该日期的所有价格数据
                date_mask = date_level == target_date
                date_prices = prices.loc[date_mask]
                
                if isinstance(date_prices, pd.DataFrame) and "Adj Close" in date_prices.columns:
                    price_dict = date_prices["Adj Close"].to_dict()
                    print(f"[Debug] Found {len(price_dict)} prices for date {target_date}")
                else:
                    price_dict = {}
            else:
                # 如果没有找到合适的日期，使用最新日期
                latest_date = pd.to_datetime(available_dates.max())
                print(f"[Debug] No date <= {date_obj}, using latest date: {latest_date}")
                date_mask = date_level == latest_date
                date_prices = prices.loc[date_mask]
                
                if isinstance(date_prices, pd.DataFrame) and "Adj Close" in date_prices.columns:
                    price_dict = date_prices["Adj Close"].to_dict()
                    print(f"[Debug] Found {len(price_dict)} prices for latest date {latest_date}")
                else:
                    price_dict = {}
        else:
            price_dict = {}
            print("[Warn] Prices data is not MultiIndex format")
        
        # 获取所有可用的 ticker（用于调试）
        available_tickers = set(price_dict.keys()) if price_dict else set()
        print(f"[Debug] Available tickers in price data: {sorted(list(available_tickers))[:10]}...")
        print(f"[Debug] Position tickers: {sorted(list(latest_weights.keys()))}")
        
        for ticker, weight in latest_weights.items():
            # 尝试多种方式获取价格
            price = 0.0
            
            # 方法1：直接匹配（精确）
            if ticker in price_dict:
                price = float(price_dict[ticker])
            # 方法2：大小写不敏感匹配
            elif price_dict:
                ticker_upper = ticker.upper()
                for key, value in price_dict.items():
                    if str(key).upper() == ticker_upper:
                        price = float(value)
                        break
            # 方法3：从 prices DataFrame 直接查询（如果前面都失败）
            if price == 0.0 and prices is not None:
                try:
                    if isinstance(prices.index, pd.MultiIndex):
                        # 尝试所有可用日期
                        for check_date in sorted(available_dates, reverse=True)[:5]:  # 只检查最近5个日期
                            try:
                                date_mask = prices.index.get_level_values(0) == check_date
                                ticker_mask = prices.index.get_level_values(1) == ticker
                                combined_mask = date_mask & ticker_mask
                                
                                if combined_mask.any():
                                    price_data = prices.loc[combined_mask]
                                    if isinstance(price_data, pd.Series) and "Adj Close" in price_data.index:
                                        price = float(price_data["Adj Close"])
                                    elif isinstance(price_data, pd.DataFrame) and "Adj Close" in price_data.columns:
                                        price = float(price_data["Adj Close"].iloc[0])
                                    
                                    if price > 0:
                                        print(f"[Debug] Found price for {ticker} on {check_date}: {price}")
                                        break
                            except Exception as e:
                                continue
                except Exception as e:
                    print(f"[Debug] Error getting price for {ticker}: {e}")
            
            position_value = base_capital * weight if price > 0 else 0.0
            total_value += position_value
            
            # 计算股数（基于权重和价格）
            quantity = (base_capital * weight / price) if price > 0 else 0.0
            
            positions_data.append({
                "symbol": ticker,
                "ticker": ticker,  # 兼容前端可能使用的字段名
                "quantity": float(quantity),
                "weight": float(weight),
                "price": float(price) if price > 0 else 0.0,  # 确保是 float
                "value": float(position_value),
                "market_value": float(position_value)  # 兼容字段名
            })
            
            if price == 0.0:
                print(f"[Warn] Price not found for {ticker} (weight={weight:.4f})")
    else:
        print("[Error] Prices data is None or empty")
        # 即使没有价格数据，也返回持仓信息（价格显示为0）
        for ticker, weight in latest_weights.items():
            positions_data.append({
                "symbol": ticker,
                "ticker": ticker,
                "quantity": 0.0,
                "weight": float(weight),
                "price": 0.0,
                "value": 0.0,
                "market_value": 0.0
            })
    
    return jsonify({
        "error": None,
        "date": date_obj.strftime("%Y-%m-%d") if 'date_obj' in locals() else datetime.now().strftime("%Y-%m-%d"),
        "positions": positions_data,
        "total_stocks": len(latest_weights),
        "total_value": total_value,
        "source": "backtest"
    })


@app.route('/api/positions/explanation')
def positions_explanation():
    """Explain how positions are generated"""
    return jsonify({
        "workflow": [
            {
                "step": 1,
                "title": "Data Acquisition",
                "description": "Fetch S&P500 constituent daily data (OHLCV) from yfinance"
            },
            {
                "step": 2,
                "title": "Factor Calculation",
                "description": "Calculate Alpha101 (101 factors) + TA-Lib (50-80 factors) + Custom factors (5 factors), totaling ~160+ factors"
            },
            {
                "step": 3,
                "title": "Factor Processing",
                "description": "Use MAD winsorize to handle outliers while preserving original factor distribution information"
            },
            {
                "step": 4,
                "title": "Model Training",
                "description": "Use LightGBM Ranker model to predict future 5-day return rankings"
            },
            {
                "step": 5,
                "title": "Prediction Generation",
                "description": "Generate prediction scores (predicted return rankings) for all stocks on the latest date"
            },
            {
                "step": 6,
                "title": "Portfolio Optimization",
                "description": "Use TopK Dropout Strategy: select 20 stocks with highest prediction scores, rebalance all 20 stocks daily (full rebalance strategy)"
            },
            {
                "step": 7,
                "title": "Weight Allocation",
                "description": "Equal weight allocation for the selected 20 stocks (5% each)"
            }
        ]
    })


@app.route('/blotter')
def blotter():
    """交易记录（Blotter）页面"""
    return render_template('blotter.html', api_base_url=API_BASE_URL)


@app.route('/api/ibkr/trades')
def ibkr_trades():
    """获取 IBKR 真实交易记录（Executions API）- 使用 src/ibkr_live_trader.py，支持日期过滤"""
    if not IBKR_CONFIG.get('enabled', False):
        return jsonify({
            "error": "IBKR 未启用",
            "trades": []
        }), 200
    
    if not IBKR_AVAILABLE:
        return jsonify({
            "error": "IBKR 功能不可用：ib_insync 未安装",
            "message": "请安装 ib_insync: pip install ib_insync",
            "trades": []
        }), 503
    
    try:
        # 使用 src/ibkr_live_trader.py 的真实类
        trader = IBKRLiveTrader(
            host=IBKR_CONFIG['host'],
            port=IBKR_CONFIG['port'],
            client_id=IBKR_CONFIG['client_id']
        )
        
        trader.connect()
        
        # 等待一下确保连接稳定
        trader.ib.sleep(1)
        
        # 获取所有成交记录（Executions）- 优先使用 executions()
        trades = []
        
        try:
            # 方法1: 从 executions 获取（最可靠的方法）
            executions = trader.ib.executions()
            for exec_item in executions:
                if exec_item.contract.secType == "STK":
                    exec_time = ""
                    if hasattr(exec_item, 'time') and exec_item.time:
                        exec_time = exec_item.time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    shares = float(exec_item.shares) if hasattr(exec_item, 'shares') and exec_item.shares else 0.0
                    price = float(exec_item.price) if hasattr(exec_item, 'price') and exec_item.price else 0.0
                    side = exec_item.side if hasattr(exec_item, 'side') else "UNKNOWN"
                    commission = float(exec_item.commission) if hasattr(exec_item, 'commission') and exec_item.commission else 0.0
                    
                    if shares > 0 and price > 0:  # 只添加有效的交易
                        # 转换时间到纽约时区
                        ny_time = convert_to_ny_time(exec_time, input_tz='UTC') if exec_time else ""
                        trades.append({
                            "time": ny_time,
                            "symbol": exec_item.contract.symbol,
                            "side": side,
                            "qty": shares,
                            "price": price,
                            "amount": shares * price,
                            "commission": commission,
                            "status": "FILLED"  # executions 都是已成交的
                        })
        except Exception as e:
            print(f"[Warn] Failed to get executions: {e}")
        
        # 方法2: 从 fills 获取（如果 executions 为空）
        if not trades:
            try:
                all_trades = trader.ib.trades()
                for trade in all_trades:
                    if trade.contract.secType == "STK":
                        fills = trade.fills if hasattr(trade, 'fills') and trade.fills else []
                        for fill in fills:
                            exec_time = ""
                            if hasattr(fill, 'time') and fill.time:
                                exec_time = fill.time.strftime("%Y-%m-%d %H:%M:%S")
                            elif hasattr(fill, 'execution') and hasattr(fill.execution, 'time'):
                                exec_time = fill.execution.time.strftime("%Y-%m-%d %H:%M:%S")
                            
                            shares = 0.0
                            price = 0.0
                            if hasattr(fill, 'execution'):
                                shares = float(fill.execution.shares) if hasattr(fill.execution, 'shares') else 0.0
                                price = float(fill.execution.price) if hasattr(fill.execution, 'price') else 0.0
                            
                            if shares > 0 and price > 0:
                                # 转换时间到纽约时区
                                ny_time = convert_to_ny_time(exec_time, input_tz='UTC') if exec_time else ""
                                trades.append({
                                    "time": ny_time,
                                    "symbol": trade.contract.symbol,
                                    "side": trade.order.action if hasattr(trade, 'order') and trade.order else "UNKNOWN",
                                    "qty": shares,
                                    "price": price,
                                    "amount": shares * price,
                                    "commission": float(fill.commission) if hasattr(fill, 'commission') else 0.0,
                                    "status": "FILLED"
                                })
            except Exception as e:
                print(f"[Warn] Failed to get fills: {e}")
        
        trader.disconnect()
        
        # 按时间排序（最新的在前）
        trades.sort(key=lambda x: x.get("time", ""), reverse=True)
        
        return jsonify({
            "error": None,
            "trades": trades,
            "count": len(trades)
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"IBKR 连接失败: {str(e)}",
            "trades": [],
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/api/blotter/trades')
def blotter_trades():
    """获取交易记录 - 优先使用 IBKR 文件数据，否则使用文件"""
    # 获取筛选参数
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    ticker_filter = request.args.get('ticker', '').upper()
    
    # 先尝试从文件读取 IBKR 交易记录
    ibkr_trades_file = OUTPUT_IBKR_DATA_DIR / "trades.json"
    if ibkr_trades_file.exists():
        try:
            with open(ibkr_trades_file, 'r', encoding='utf-8') as f:
                ibkr_data = json.load(f)
            
            # 处理两种格式：直接是列表，或包含"trades"字段的字典
            if isinstance(ibkr_data, list):
                trades = ibkr_data
            elif isinstance(ibkr_data, dict):
                # 可能是 {"trades": [...]} 或直接是交易记录字典
                if "trades" in ibkr_data:
                    trades = ibkr_data.get("trades", [])
                elif len(ibkr_data) > 0 and isinstance(list(ibkr_data.values())[0], dict):
                    # 可能是按日期组织的字典 {"2024-01-01": [...], ...}
                    trades = []
                    for key, value in ibkr_data.items():
                        if isinstance(value, list):
                            trades.extend(value)
                        elif isinstance(value, dict) and "trades" in value:
                            trades.extend(value.get("trades", []))
                else:
                    trades = []
            else:
                trades = []
            
            if trades:
                # 转换所有交易记录的时间到纽约时区
                for trade in trades:
                    if trade.get("time"):
                        trade["time"] = convert_to_ny_time(trade["time"], input_tz='UTC')
                
                # 应用筛选
                if date_from:
                    trades = [t for t in trades if t.get("time", "") >= date_from]
                if date_to:
                    trades = [t for t in trades if t.get("time", "") <= date_to]
                if ticker_filter:
                    trades = [t for t in trades if t.get("symbol", "").upper() == ticker_filter.upper()]
                
                return jsonify({
                    "error": None,
                    "trades": trades,
                    "total_trades": len(trades),
                    "source": "ibkr_file"
                })
        except Exception as e:
            print(f"[Warn] Failed to load IBKR trades from file: {e}")
            import traceback
            traceback.print_exc()
    
    # 再尝试实时获取 IBKR 交易记录
    if IBKR_CONFIG.get('enabled', False):
        try:
            ibkr_result = ibkr_trades()
            ibkr_data = ibkr_result.get_json()
            if ibkr_data and not ibkr_data.get("error") and ibkr_data.get("trades"):
                trades = ibkr_data.get("trades", [])
                
                # 转换所有交易记录的时间到纽约时区（ibkr_trades 已经转换，但确保一致性）
                for trade in trades:
                    if trade.get("time"):
                        trade["time"] = convert_to_ny_time(trade["time"], input_tz='UTC')
                
                # 应用筛选
                if date_from:
                    trades = [t for t in trades if t.get("time", "") >= date_from]
                if date_to:
                    trades = [t for t in trades if t.get("time", "") <= date_to]
                if ticker_filter:
                    trades = [t for t in trades if t.get("symbol", "").upper() == ticker_filter]
                
                return jsonify({
                    "error": None,
                    "trades": trades,
                    "total_trades": len(trades),
                    "source": "ibkr"
                })
        except Exception as e:
            print(f"[Warn] Failed to get IBKR trades in real-time: {e}")
    
    # 回退到文件读取
    trade_files = []
    reports_dir = REPORTS_DIR
    if reports_dir.exists():
        trade_files.extend(list(reports_dir.glob("*order*.json")))
        trade_files.extend(list(reports_dir.glob("*trade*.json")))
        trade_files.extend(list(reports_dir.glob("*execution*.json")))
    
    trades = []
    
    for trade_file in trade_files:
        try:
            trade_data = load_json_safe(trade_file, {})
            if isinstance(trade_data, dict) and "orders" in trade_data:
                for order in trade_data["orders"]:
                    trade_time = trade_data.get("date", "")
                    symbol = order.get("symbol", "")
                    
                    # 应用筛选
                    if date_from and trade_time < date_from:
                        continue
                    if date_to and trade_time > date_to:
                        continue
                    if ticker_filter and symbol.upper() != ticker_filter:
                        continue
                    
                    # 转换时间到纽约时区（假设文件中的时间是 UTC）
                    ny_time = convert_to_ny_time(trade_time, input_tz='UTC') if trade_time else ""
                    trades.append({
                        "time": ny_time,
                        "symbol": symbol,
                        "side": "BUY" if order.get("target_weight", 0) > 0 else "SELL",
                        "qty": 0.0,
                        "price": 0.0,
                        "amount": 0.0
                    })
        except:
            continue
    
    # 转换所有交易记录的时间到纽约时区
    for trade in trades:
        if trade.get("time"):
            trade["time"] = convert_to_ny_time(trade["time"], input_tz='UTC')
    
    return jsonify({
        "error": None if trades else "暂无数据",
        "trades": trades,
        "source": "file"
    })


@app.route('/backtest-results')
def backtest_results():
    """回测结果页面（原真实收益展示）"""
    return render_template('performance.html', api_base_url=API_BASE_URL)


@app.route('/api/performance/summary')
def performance_summary():
    """获取性能摘要 - 用于首页"""
    backtest_summary = load_json_safe(BACKTEST_DIR / "summary.json", {})
    
    return jsonify({
        "error": None,
        "annual_return": backtest_summary.get("annualized_return"),
        "sharpe": backtest_summary.get("sharpe_ratio"),
        "max_drawdown": backtest_summary.get("max_drawdown"),
        "positions_count": 20  # 固定持仓数量
    })


@app.route('/api/ibkr/pnl')
def ibkr_pnl():
    """获取 IBKR 真实收益数据（PnL/Ledger）- 优先从文件读取，否则使用实时连接"""
    # 优先从文件读取（如果存在）
    ibkr_pnl_file = OUTPUT_IBKR_DATA_DIR / "pnl.json"
    if ibkr_pnl_file.exists():
        try:
            with open(ibkr_pnl_file, 'r', encoding='utf-8') as f:
                pnl_data = json.load(f)
            # 确保包含所有必需字段
            if "account_type" not in pnl_data:
                pnl_data["account_type"] = "Unknown"
            if "account_id" not in pnl_data:
                pnl_data["account_id"] = None
            if "total_profit_loss" not in pnl_data:
                total_profit_loss = pnl_data.get("realized_pnl", 0.0) + pnl_data.get("unrealized_pnl", 0.0)
                pnl_data["total_profit_loss"] = total_profit_loss
            if "profit_loss_percent" not in pnl_data:
                net_liq = pnl_data.get("net_liquidation", 0.0)
                profit_loss_percent = (pnl_data["total_profit_loss"] / net_liq * 100) if net_liq > 0 else 0.0
                pnl_data["profit_loss_percent"] = profit_loss_percent
            return jsonify({
                "error": None,
                **pnl_data,
                "source": "file"
            })
        except Exception as e:
            print(f"[Warn] Failed to load IBKR PnL from file: {e}")
            # 继续尝试实时连接
    
    # 如果没有文件或读取失败，尝试实时连接
    if not IBKR_CONFIG.get('enabled', False):
        return jsonify({
            "error": "IBKR 未启用",
            "account_type": "Unknown",
            "account_id": None,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "total_profit_loss": 0.0,
            "profit_loss_percent": 0.0,
            "daily_pnl": [],
            "source": "backtest"
        }), 200
    
    if not IBKR_AVAILABLE:
        return jsonify({
            "error": "IBKR 功能不可用：ib_insync 未安装",
            "message": "请安装 ib_insync: pip install ib_insync",
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "daily_pnl": []
        }), 503
    
    try:
        # 使用 src/ibkr_live_trader.py 的真实类
        trader = IBKRLiveTrader(
            host=IBKR_CONFIG['host'],
            port=IBKR_CONFIG['port'],
            client_id=IBKR_CONFIG['client_id']
        )
        
        trader.connect()
        
        # 检测账户类型（Paper Trading vs Real Money）
        account_type = "Unknown"
        account_id = None
        
        # 通过端口判断（7497=Paper, 7496=Live）
        if IBKR_CONFIG.get('port', 7497) == 7497:
            account_type = "Paper Trading"
        elif IBKR_CONFIG.get('port', 7497) == 7496:
            account_type = "Real Money"
        else:
            account_type = f"Port {IBKR_CONFIG.get('port', 7497)}"
        
        # 尝试从账户信息中获取账户ID
        try:
            account_summary = trader.ib.accountSummary()
            if account_summary:
                account_id = account_summary[0].account
                # 检查账户ID是否包含paper/demo等关键词
                if account_id and ('paper' in account_id.lower() or 'demo' in account_id.lower() or 'sim' in account_id.lower()):
                    account_type = "Paper Trading"
        except:
            pass
        
        # 获取账户价值（从 IBKR 账本）
        account_values = trader.ib.accountValues()
        
        # 提取关键指标
        realized_pnl = 0.0
        unrealized_pnl = 0.0
        total_pnl = 0.0
        net_liquidation = 0.0
        buying_power = 0.0
        cash = 0.0
        
        for av in account_values:
            tag = av.tag
            value = float(av.value) if av.value else 0.0
            
            if tag == "RealizedPnL":
                realized_pnl = value
            elif tag == "UnrealizedPnL":
                unrealized_pnl = value
            elif tag == "TotalPnL":
                total_pnl = value
            elif tag == "NetLiquidation":
                net_liquidation = value
            elif tag == "BuyingPower":
                buying_power = value
            elif tag == "CashBalance":
                cash = value
        
        # 获取持仓的未实现盈亏（使用真实价格）
        positions_pnl = []
        for pos in trader.ib.positions():
            if pos.contract.secType == "STK":
                ticker = pos.contract.symbol
                shares = pos.position
                avg_cost = pos.avgCost if hasattr(pos, 'avgCost') else 0.0
                current_price = trader.get_realtime_price(ticker)
                
                if current_price and current_price > 0 and avg_cost > 0:
                    pnl = (current_price - avg_cost) * shares
                    positions_pnl.append({
                        "symbol": ticker,
                        "shares": float(shares),
                        "avg_cost": float(avg_cost),
                        "current_price": float(current_price),
                        "unrealized_pnl": float(pnl),
                        "value": float(abs(shares) * current_price)
                    })
        
        # 计算累计收益曲线（从历史交易记录和账户价值变化）
        daily_pnl = []
        try:
            # 获取历史交易记录计算每日 PnL
            executions = trader.ib.executions()
            # 按日期分组计算每日收益
            from collections import defaultdict
            daily_trades = defaultdict(lambda: {"realized": 0.0, "unrealized": 0.0, "count": 0, "volume": 0.0})
            
            # 获取持仓信息用于计算未实现盈亏
            positions_dict = {}
            for pos in trader.ib.positions():
                if pos.contract.secType == "STK":
                    ticker = pos.contract.symbol
                    positions_dict[ticker] = {
                        "shares": pos.position,
                        "avg_cost": pos.avgCost if hasattr(pos, 'avgCost') else 0.0
                    }
            
            # 计算每日已实现和未实现盈亏
            for exec_item in executions:
                if exec_item.contract.secType == "STK" and hasattr(exec_item, 'time') and exec_item.time:
                    trade_date = exec_item.time.date()
                    shares = float(exec_item.shares) if hasattr(exec_item, 'shares') else 0.0
                    price = float(exec_item.price) if hasattr(exec_item, 'price') else 0.0
                    commission = float(exec_item.commission) if hasattr(exec_item, 'commission') else 0.0
                    
                    daily_trades[trade_date]["count"] += 1
                    daily_trades[trade_date]["volume"] += shares * price
                    # 简化：已实现盈亏需要知道买入价，这里暂时用0
            
            # 计算当前未实现盈亏
            current_unrealized = 0.0
            for ticker, pos_info in positions_dict.items():
                current_price = trader.get_realtime_price(ticker)
                if current_price and current_price > 0 and pos_info["avg_cost"] > 0:
                    current_unrealized += (current_price - pos_info["avg_cost"]) * pos_info["shares"]
            
            # 转换为列表格式（按日期）
            today = datetime.now().date()
            for date, data in sorted(daily_trades.items()):
                daily_pnl.append({
                    "date": str(date),
                    "realized_pnl": data["realized"],
                    "unrealized_pnl": current_unrealized if date == today else 0.0,
                    "total_pnl": data["realized"] + (current_unrealized if date == today else 0.0),
                    "trade_count": data["count"],
                    "volume": data["volume"]
                })
            
            # 如果没有历史交易，至少返回今天的PnL
            if not daily_pnl:
                daily_pnl.append({
                    "date": str(today),
                    "realized_pnl": realized_pnl,
                    "unrealized_pnl": unrealized_pnl,
                    "total_pnl": total_pnl,
                    "trade_count": 0,
                    "volume": 0.0
                })
        except Exception as e:
            print(f"[Warn] Failed to calculate daily PnL: {e}")
            # 至少返回今天的PnL
            daily_pnl.append({
                "date": str(datetime.now().date()),
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": total_pnl,
                "trade_count": 0,
                "volume": 0.0
            })
        
        trader.disconnect()
        
        # 计算总盈亏和盈亏百分比
        total_profit_loss = realized_pnl + unrealized_pnl
        profit_loss_percent = (total_profit_loss / net_liquidation * 100) if net_liquidation > 0 else 0.0
        
        return jsonify({
            "error": None,
            "account_type": account_type,
            "account_id": account_id,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "total_profit_loss": total_profit_loss,
            "profit_loss_percent": profit_loss_percent,
            "net_liquidation": net_liquidation,
            "buying_power": buying_power,
            "cash": cash,
            "positions_pnl": positions_pnl,
            "daily_pnl": daily_pnl,
            "source": "ibkr"
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"IBKR 连接失败: {str(e)}",
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "source": "backtest",
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/api/backtest/analysis')
def backtest_analysis():
    """生成回测分析数据（月度收益、回撤、行业暴露）"""
    daily_returns = load_parquet_safe(BACKTEST_DIR / "daily_returns.parquet")
    
    if daily_returns is None:
        return jsonify({
            "error": "暂无数据",
            "monthly_returns": [],
            "drawdown_analysis": {},
            "sector_exposure": {}
        }), 200
    
    try:
        # 规范化列名
        strategy_col = None
        if 'strategy_return' in daily_returns.columns:
            strategy_col = 'strategy_return'
        elif 'net_return' in daily_returns.columns:
            strategy_col = 'net_return'
        
        if strategy_col is None:
            return jsonify({
                "error": "数据加载失败，请检查后端服务：缺少策略收益列",
                "monthly_returns": [],
                "drawdown_analysis": {},
                "sector_exposure": {}
            }), 200
        
        strategy_returns = daily_returns[strategy_col]
        
        # 1. 月度收益分析
        monthly_returns = []
        strategy_returns_series = pd.Series(strategy_returns, index=daily_returns.index)
        monthly_ret = strategy_returns_series.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        
        for date, ret in monthly_ret.items():
            monthly_returns.append({
                "month": date.strftime("%Y-%m"),
                "return": float(ret),
                "return_pct": float(ret * 100)
            })
        
        # 2. 回撤分析
        equity_curve = (1 + strategy_returns_series).cumprod()
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max - 1) * 100
        
        max_dd = float(drawdown.min())
        max_dd_date = drawdown.idxmin()
        current_dd = float(drawdown.iloc[-1])
        
        # 清理NaN值
        drawdown_clean = clean_nan_for_json(drawdown.fillna(0).tolist())
        
        drawdown_analysis = {
            "max_drawdown": float(max_dd) if not pd.isna(max_dd) else 0.0,
            "max_drawdown_date": str(max_dd_date) if not pd.isna(max_dd_date) else "",
            "current_drawdown": float(current_dd) if not pd.isna(current_dd) else 0.0,
            "drawdown_series": drawdown_clean,
            "dates": [str(d) for d in drawdown.index]
        }
        
        # 3. 行业暴露（从权重数据计算）
        sector_exposure = {}
        try:
            # 读取权重数据
            weights_path = PORTFOLIO_DIR / "weights.parquet"
            if not weights_path.exists():
                weights_path = PORTFOLIO_DIR / "weights.parquet"
            
            weights = load_parquet_safe(weights_path)
            if weights is not None:
                # 简化：按股票代码前缀分组（实际应该使用行业分类数据）
                # 这里暂时返回空，需要真实的行业分类数据
                pass
        except:
            pass
        
        # 4. 计算Turnover（换手率）
        turnover = []
        try:
            if 'turnover' in daily_returns.columns:
                turnover_series = pd.Series(daily_returns['turnover'], index=daily_returns.index)
                monthly_turnover = turnover_series.resample('ME').mean()
                for date, t in monthly_turnover.items():
                    turnover.append({
                        "month": date.strftime("%Y-%m"),
                        "turnover": float(t) if not pd.isna(t) else 0.0
                    })
        except:
            pass
        
        # 5. 计算日度PnL
        daily_pnl = []
        try:
            if 'nav' in daily_returns.columns:
                nav_series = pd.Series(daily_returns['nav'], index=daily_returns.index)
                daily_pnl_series = nav_series.diff().fillna(0.0)
                for date, pnl in daily_pnl_series.items():
                    daily_pnl.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "pnl": float(pnl) if not pd.isna(pnl) else 0.0
                    })
            else:
                # 从收益计算
                daily_pnl_series = strategy_returns_series * 100  # 假设初始资金100万
                for date, pnl in daily_pnl_series.items():
                    daily_pnl.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "pnl": float(pnl) if not pd.isna(pnl) else 0.0
                    })
        except:
            pass
        
        # 6. Winners / Losers（最佳和最差月份）
        winners_losers = {
            "best_month": None,
            "worst_month": None
        }
        if monthly_returns:
            sorted_months = sorted(monthly_returns, key=lambda x: x["return"])
            winners_losers["best_month"] = sorted_months[-1] if sorted_months else None
            winners_losers["worst_month"] = sorted_months[0] if sorted_months else None
        
        # 保存分析结果
        analysis_data = {
            "monthly_returns": monthly_returns,
            "drawdown_analysis": drawdown_analysis,
            "sector_exposure": sector_exposure,
            "turnover": turnover,
            "daily_pnl": daily_pnl,
            "winners_losers": winners_losers,
            "generated_at": datetime.now().isoformat()
        }
        
        # 保存到文件
        analysis_path = BACKTEST_DIR / "backtest_analysis.json"
        try:
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(clean_nan_for_json(analysis_data), f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"[Warn] Failed to save analysis: {e}")
        
        return jsonify(clean_nan_for_json({
            "error": None,
            **analysis_data
        }))
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"数据加载失败，请检查后端服务：{str(e)}",
            "monthly_returns": [],
            "drawdown_analysis": {},
            "sector_exposure": {},
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/api/model/performance')
def model_performance():
    """获取 ML 模型性能指标（不是策略 NAV）- 展示LightGBM Ranker的真实指标"""
    try:
        # 使用 src/optimizer.py 的真实函数加载预测
        predictions = load_predictions(cfg, model_type="lightgbm")
        
        # 加载模型指标（从训练输出）
        model_metrics = load_json_safe(REPORTS_DIR / "metrics.json", {})
        
        # 如果没有 metrics.json，尝试从模型目录加载
        if not model_metrics:
            model_dir = Path(cfg["paths"]["model_dir"])
            metrics_path = model_dir / "metrics.json"
            if metrics_path.exists():
                model_metrics = load_json_safe(metrics_path, {})
        
        # 计算每日Rank IC（使用 src/modeling.py 的函数）
        from src.modeling import rank_ic_per_day
        from src.factor_engine import read_prices, forward_return
        
        daily_rank_ic_data = []
        if predictions is not None and len(predictions) > 0:
            try:
                # 读取价格数据计算未来收益
                prices = read_prices(cfg)
                if prices is not None:
                    # 计算未来收益（使用horizon_days配置）
                    horizon_days = cfg.get("data", {}).get("horizon_days", 5)
                    forward_ret = forward_return(prices, horizon=horizon_days)
                    
                    # 对齐索引
                    common_idx = predictions.index.intersection(forward_ret.index)
                    if len(common_idx) > 0:
                        pred_aligned = predictions.loc[common_idx]
                        ret_aligned = forward_ret.loc[common_idx]
                        
                        # 计算每日Rank IC
                        ric_daily = rank_ic_per_day(pred_aligned, ret_aligned)
                        
                        # 转换为列表格式
                        for date, ic in ric_daily.items():
                            daily_rank_ic_data.append({
                                "date": str(date),
                                "rank_ic": float(ic) if not pd.isna(ic) else 0.0
                            })
            except Exception as e:
                print(f"[Warn] Failed to calculate daily Rank IC: {e}")
        
        # 计算模型指标
        model_perf = {
            "ndcg": model_metrics.get("ndcg", model_metrics.get("oof_mean_ndcg", None)),
            "precision_at_k": model_metrics.get("precision_at_k", None),
            "recall_at_k": model_metrics.get("recall_at_k", None),
            "mean_rank_ic": model_metrics.get("oof_mean_rank_ic", model_metrics.get("mean_rank_ic", None)),
            "rank_icir": model_metrics.get("rank_icir", None),
            "feature_importance": model_metrics.get("feature_importance", {}),
            "train_metrics": model_metrics.get("train_metrics", {}),
            "validation_metrics": model_metrics.get("validation_metrics", {}),
            "daily_rank_ic": daily_rank_ic_data,  # 每日Rank IC
            "prediction_count": len(predictions) if predictions is not None else 0,
            "prediction_date_range": {
                "start": str(predictions.index.get_level_values(0).min()) if isinstance(predictions.index, pd.MultiIndex) and len(predictions) > 0 else None,
                "end": str(predictions.index.get_level_values(0).max()) if isinstance(predictions.index, pd.MultiIndex) and len(predictions) > 0 else None
            } if predictions is not None and len(predictions) > 0 else None
        }
        
        return jsonify({
            "error": None,
            "model_metrics": model_perf
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"模型数据加载失败: {str(e)}",
            "model_metrics": {},
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/api/performance/real-time')
def performance_real_time():
    """获取实时收益数据 - 优先使用 IBKR 文件数据，否则使用回测数据"""
    # 先尝试从文件读取 IBKR 收益数据（但IBKR数据没有历史曲线，所以需要fallback到回测数据）
    # IBKR数据只用于显示当前PnL，历史曲线使用回测数据
    ibkr_pnl_data = None
    ibkr_pnl_file = OUTPUT_IBKR_DATA_DIR / "pnl.json"
    if ibkr_pnl_file.exists():
        try:
            with open(ibkr_pnl_file, 'r', encoding='utf-8') as f:
                ibkr_pnl_data = json.load(f)
        except Exception as e:
            print(f"[Warn] Failed to load IBKR PnL from file: {e}")
    
    # 再尝试实时获取 IBKR 实盘收益
    if IBKR_CONFIG.get('enabled', False):
        try:
            ibkr_result = ibkr_pnl()
            ibkr_data = ibkr_result.get_json()
            if ibkr_data and not ibkr_data.get("error") and ibkr_data.get("source") == "ibkr":
                return jsonify({
                    "error": None,
                    "source": "ibkr",
                    "realized_pnl": ibkr_data.get("realized_pnl", 0.0),
                    "unrealized_pnl": ibkr_data.get("unrealized_pnl", 0.0),
                    "total_pnl": ibkr_data.get("total_pnl", 0.0),
                    "net_liquidation": ibkr_data.get("net_liquidation", 0.0),
                    "buying_power": ibkr_data.get("buying_power", 0.0),
                    "cash": ibkr_data.get("cash", 0.0),
                    "positions_pnl": ibkr_data.get("positions_pnl", []),
                    "daily_pnl": ibkr_data.get("daily_pnl", [])
                })
        except Exception as e:
            print(f"[Warn] Failed to get IBKR PnL in real-time: {e}")
            pass
    
    # 回退到回测数据
    backtest_summary = load_json_safe(BACKTEST_DIR / "summary.json", {})
    
    # 调试：打印路径信息
    daily_returns_path = BACKTEST_DIR / "daily_returns.parquet"
    print(f"[Debug] Looking for daily_returns at: {daily_returns_path}")
    print(f"[Debug] File exists: {daily_returns_path.exists()}")
    print(f"[Debug] BACKTEST_DIR: {BACKTEST_DIR}")
    print(f"[Debug] Project root: {project_root}")
    
    daily_returns = load_parquet_safe(daily_returns_path)
    
    if daily_returns is None:
        return jsonify({
            "error": "暂无数据",
            "source": "backtest",
            "dates": [],
            "strategy_returns": [],
            "benchmark_returns": [],
            "drawdown": [],
            "nav": [],
            "benchmark_nav": [],
            "summary": backtest_summary,
            "metrics": {}
        }), 200
    
    try:
        # 规范化列名 - 优先使用 nav 字段（如果存在）
        if 'nav' in daily_returns.columns:
            nav = daily_returns['nav']
            strategy_returns = daily_returns['strategy_return'] if 'strategy_return' in daily_returns.columns else None
        else:
            # 如果没有 nav，从 returns 计算
            strategy_col = None
            if 'strategy_return' in daily_returns.columns:
                strategy_col = 'strategy_return'
            elif 'net_return' in daily_returns.columns:
                strategy_col = 'net_return'
            else:
                for col in daily_returns.columns:
                    col_lower = col.lower()
                    if 'strategy' in col_lower and ('return' in col_lower or 'ret' in col_lower):
                        strategy_col = col
                        break
                    elif 'net' in col_lower and 'return' in col_lower:
                        strategy_col = col
                        break
            
            if strategy_col is None:
                return jsonify({
                    "error": "数据加载失败，请检查后端服务：缺少策略收益列",
                    "source": "backtest",
                    "dates": [],
                    "strategy_returns": [],
                    "benchmark_returns": [],
                    "drawdown": [],
                    "nav": [],
                    "benchmark_nav": [],
                    "summary": backtest_summary,
                    "metrics": {}
                }), 200
            
            # 修复 NAV 计算：确保从 100 开始，不复制数据
            strategy_returns_series = daily_returns[strategy_col]
            cumulative = (1 + strategy_returns_series).cumprod()
            nav = cumulative * 100  # 从 100 开始
            strategy_returns = strategy_returns_series
        
        # 获取基准 NAV
        if 'benchmark_nav' in daily_returns.columns:
            benchmark_nav = daily_returns['benchmark_nav']
            benchmark_returns = daily_returns['benchmark_return'] if 'benchmark_return' in daily_returns.columns else None
        else:
            # 计算基准收益
            benchmark_returns_series = None
            if 'benchmark_return' in daily_returns.columns:
                benchmark_returns_series = daily_returns['benchmark_return']
            else:
                # 从价格数据计算（使用 src/factor_engine.py 的真实函数）
                try:
                    # 使用 src/factor_engine.py 的真实函数读取价格
                    prices = read_prices(cfg)
                    if prices is not None and isinstance(prices.index, pd.MultiIndex):
                        close_prices = prices["Adj Close"].unstack("ticker")
                        benchmark_ret = close_prices.mean(axis=1).pct_change().fillna(0.0)
                        common_dates = daily_returns.index.intersection(benchmark_ret.index)
                        if len(common_dates) > 0:
                            benchmark_returns_series = benchmark_ret.loc[common_dates].reindex(daily_returns.index, method='ffill').fillna(0.0)
                except Exception as e:
                    print(f"[Warn] Failed to load benchmark data: {e}")
                    pass
            
            if benchmark_returns_series is not None:
                benchmark_cumulative = (1 + benchmark_returns_series).cumprod()
                benchmark_nav = benchmark_cumulative * 100
                benchmark_returns = benchmark_returns_series
            else:
                benchmark_nav = pd.Series([100.0] * len(nav), index=nav.index)
                benchmark_returns = pd.Series([0.0] * len(strategy_returns), index=strategy_returns.index)
        
        # 计算回撤（基于 NAV）
        equity_curve = nav / 100.0  # 转换回累计收益
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max - 1) * 100
        
        # 计算指标
        if strategy_returns is not None:
            risk_metrics = risk_analysis(strategy_returns)
        else:
            risk_metrics = {}
        
        # 确保索引是日期格式
        dates = daily_returns.index
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.to_datetime(dates)
        
        # 清理NaN值
        strategy_returns_clean = clean_nan_for_json(strategy_returns.fillna(0).tolist() if strategy_returns is not None else [])
        benchmark_returns_clean = clean_nan_for_json(benchmark_returns.fillna(0).tolist() if benchmark_returns is not None else [])
        drawdown_clean = clean_nan_for_json(drawdown.fillna(0).tolist() if drawdown is not None else [])
        nav_clean = clean_nan_for_json(nav.fillna(100).tolist() if nav is not None else [])
        benchmark_nav_clean = clean_nan_for_json(benchmark_nav.fillna(100).tolist() if benchmark_nav is not None else [])
        
        # 合并IBKR PnL数据（如果有）
        response_data = {
            "error": None,
            "source": "backtest",
            "dates": dates.astype(str).tolist(),
            "strategy_returns": strategy_returns_clean,
            "benchmark_returns": benchmark_returns_clean,
            "drawdown": drawdown_clean,
            "nav": nav_clean,
            "benchmark_nav": benchmark_nav_clean,
            "summary": backtest_summary,
            "metrics": {
                "annual_return": backtest_summary.get("annualized_return", risk_metrics.get("annualized_return", 0.0)) or 0.0,
                "sharpe": backtest_summary.get("sharpe_ratio", risk_metrics.get("information_ratio", 0.0)) or 0.0,
                "max_drawdown": backtest_summary.get("max_drawdown", risk_metrics.get("max_drawdown", 0.0)) or 0.0,
                "information_ratio": risk_metrics.get("information_ratio", 0.0) or 0.0,
                "turnover": backtest_summary.get("avg_turnover", 0) or 0,
                "total_return": backtest_summary.get("total_return", 0.0) or 0.0
            }
        }
        
        # 如果有IBKR数据，添加当前PnL信息
        if ibkr_pnl_data:
            response_data["ibkr_pnl"] = {
                "realized_pnl": ibkr_pnl_data.get("realized_pnl", 0.0),
                "unrealized_pnl": ibkr_pnl_data.get("unrealized_pnl", 0.0),
                "total_pnl": ibkr_pnl_data.get("total_pnl", 0.0),
                "net_liquidation": ibkr_pnl_data.get("net_liquidation", 0.0)
            }
        
        return jsonify(response_data)
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"数据加载失败，请检查后端服务：{str(e)}",
            "source": "backtest",
            "dates": [],
            "strategy_returns": [],
            "benchmark_returns": [],
            "drawdown": [],
            "nav": [],
            "benchmark_nav": [],
            "summary": backtest_summary,
            "metrics": {},
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/api/performance/monthly')
def performance_monthly():
    """获取月度收益数据 - 从回测分析 API 获取"""
    try:
        # 调用 backtest_analysis API 获取月度收益
        daily_returns = load_parquet_safe(BACKTEST_DIR / "daily_returns.parquet")
        
        if daily_returns is None:
            return jsonify({
                "error": "暂无数据",
                "data": []
            }), 200
        
        # 规范化列名
        strategy_col = None
        if 'strategy_return' in daily_returns.columns:
            strategy_col = 'strategy_return'
        elif 'net_return' in daily_returns.columns:
            strategy_col = 'net_return'
        
        if strategy_col is None:
            return jsonify({
                "error": "数据加载失败，请检查后端服务：缺少策略收益列",
                "data": []
            }), 200
        
        strategy_returns = daily_returns[strategy_col]
        
        # 计算月度收益
        strategy_returns_series = pd.Series(strategy_returns, index=daily_returns.index)
        monthly_ret = strategy_returns_series.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        
        monthly_data = []
        for date, ret in monthly_ret.items():
            monthly_data.append({
                "month": date.strftime("%Y-%m"),
                "date": date.strftime("%Y-%m-%d"),
                "return": float(ret),
                "return_pct": float(ret * 100)
            })
        
        return jsonify({
            "error": None,
            "data": monthly_data
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"数据加载失败：{str(e)}",
            "data": [],
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

