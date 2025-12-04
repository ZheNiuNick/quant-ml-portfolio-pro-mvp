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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from flask import Flask, render_template, jsonify, request
import yaml

# 智能查找项目根目录
# 项目根目录应该包含 src/ 和 config/ 目录
def find_project_root():
    """查找项目根目录（包含 src/ 和 config/ 的目录）"""
    current = Path(__file__).resolve()
    
    # 尝试从 app.py 的位置向上查找
    for path in [current.parent.parent, current.parent, Path.cwd()]:
        if (path / "src").exists() and (path / "config").exists():
            return path
    
    # 如果都找不到，尝试从当前工作目录查找
    cwd = Path.cwd()
    if (cwd / "src").exists() and (cwd / "config").exists():
        return cwd
    
    # 如果还是找不到，尝试从 app.py 的父目录的父目录
    # 这适用于 app.py 在 zhe_trading_strategy/ 目录下的情况
    app_dir = current.parent
    parent = app_dir.parent
    if (parent / "src").exists() and (parent / "config").exists():
        return parent
    
    # 最后尝试：如果当前目录就是项目根
    if (current / "src").exists() and (current / "config").exists():
        return current
    
    # 如果都找不到，返回 app.py 的父目录的父目录（默认行为）
    return app_dir.parent

# 添加项目根目录到路径
project_root = find_project_root()
sys.path.insert(0, str(project_root))

# 验证项目根目录
if not (project_root / "src").exists():
    raise ImportError(
        f"无法找到项目根目录。当前工作目录: {Path.cwd()}, "
        f"app.py 位置: {Path(__file__).resolve()}, "
        f"尝试的项目根: {project_root}. "
        f"请确保 src/ 目录存在。"
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
SETTINGS = project_root / "config" / "settings.yaml"
if not SETTINGS.exists():
    raise FileNotFoundError(
        f"无法找到配置文件: {SETTINGS}. "
        f"项目根目录: {project_root}"
    )
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

# 数据路径
DATA_DIR = project_root / "outputs"
BACKTEST_DIR = DATA_DIR / "backtests"
PORTFOLIO_DIR = DATA_DIR / "portfolios"
REPORTS_DIR = DATA_DIR / "reports"


def get_factor_store_path(factor_cfg=None, auto_download=True):
    """
    获取 factor_store.parquet 的路径，如果不存在则尝试从 Hugging Face 自动下载
    
    Args:
        factor_cfg: 因子配置（可选）
        auto_download: 是否自动下载（如果文件不存在）
    
    Returns:
        factor_store.parquet 的路径
    """
    if factor_cfg is None:
        factor_cfg = load_factor_settings(str(SETTINGS))
    
    factor_store_path = Path(factor_cfg["paths"].get("factors_store", "data/factors/factor_store.parquet"))
    if not factor_store_path.is_absolute():
        factor_store_path = project_root / factor_store_path
    
    # 尝试自动从 Hugging Face 下载（如果文件不存在）
    if auto_download and not factor_store_path.exists():
        try:
            from src.data_loader import ensure_factor_store
            ensure_factor_store(factor_store_path, auto_download=True)
        except ImportError:
            pass  # data_loader 模块不存在时忽略
    
    return factor_store_path


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
        except:
            return None
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
            prices = load_parquet_safe(project_root / "data" / "processed" / "prices.parquet")
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
            factor_store_path = project_root / factor_store_path
        
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
                    db_path = project_root / db_path
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
            factor_store_path = project_root / factor_store_path
        
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
                    db_path = project_root / db_path
                
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
    """近12个月 Rolling IC 曲线"""
    try:
        # 读取IC/ICIR数据
        factor_cfg = load_factor_settings(str(SETTINGS))
        factor_store_path = Path(factor_cfg["paths"].get("factors_store", "data/factors/factor_store.parquet"))
        if not factor_store_path.is_absolute():
            factor_store_path = project_root / factor_store_path
        
        ic_store_path = factor_store_path.parent / "factor_ic_ir.parquet"
        
        if not ic_store_path.exists():
            return jsonify({
                "error": "IC数据不存在，请先运行因子IC计算",
                "dates": [],
                "ic_mean": [],
                "ic_upper": [],
                "ic_lower": []
            }), 200
        
        ic_data = pd.read_parquet(ic_store_path)
        if not pd.api.types.is_datetime64_any_dtype(ic_data["date"]):
            ic_data["date"] = pd.to_datetime(ic_data["date"])
        
        # 获取近12个月的数据（如果数据不足12个月，使用所有可用数据）
        latest_date = ic_data["date"].max()
        start_date = latest_date - pd.DateOffset(months=12)
        recent_ic = ic_data[ic_data["date"] >= start_date].copy()
        
        # 如果近12个月数据不足，使用所有可用数据
        if len(recent_ic) == 0:
            recent_ic = ic_data.copy()
        
        if len(recent_ic) == 0:
            return jsonify({
                "error": "没有IC数据",
                "dates": [],
                "ic_mean": [],
                "ic_upper": [],
                "ic_lower": []
            }), 200
        
        # 按日期分组，计算每日平均IC和上下界
        daily_ic = recent_ic.groupby("date")["ic"].agg(['mean', 'std', 'count']).reset_index()
        daily_ic = daily_ic.sort_values("date")
        
        # 计算上下界（均值 ± 1.96 * 标准差）
        daily_ic["ic_mean"] = daily_ic["mean"]
        daily_ic["ic_upper"] = daily_ic["mean"] + 1.96 * daily_ic["std"] / np.sqrt(daily_ic["count"])
        daily_ic["ic_lower"] = daily_ic["mean"] - 1.96 * daily_ic["std"] / np.sqrt(daily_ic["count"])
        
        return jsonify({
            "error": None,
            "dates": [d.strftime("%Y-%m-%d") for d in daily_ic["date"]],
            "ic_mean": daily_ic["ic_mean"].fillna(0).tolist(),
            "ic_upper": daily_ic["ic_upper"].fillna(0).tolist(),
            "ic_lower": daily_ic["ic_lower"].fillna(0).tolist()
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"计算Rolling IC失败: {str(e)}",
            "dates": [],
            "ic_mean": [],
            "ic_upper": [],
            "ic_lower": [],
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/api/factor-diagnostics/rolling-icir')
def rolling_icir():
    """Rolling ICIR"""
    try:
        # 读取IC/ICIR数据
        factor_cfg = load_factor_settings(str(SETTINGS))
        factor_store_path = Path(factor_cfg["paths"].get("factors_store", "data/factors/factor_store.parquet"))
        if not factor_store_path.is_absolute():
            factor_store_path = project_root / factor_store_path
        
        ic_store_path = factor_store_path.parent / "factor_ic_ir.parquet"
        
        if not ic_store_path.exists():
            return jsonify({
                "error": "IC数据不存在",
                "dates": [],
                "icir": []
            }), 200
        
        ic_data = pd.read_parquet(ic_store_path)
        if not pd.api.types.is_datetime64_any_dtype(ic_data["date"]):
            ic_data["date"] = pd.to_datetime(ic_data["date"])
        
        # 获取近12个月的数据（如果数据不足12个月，使用所有可用数据）
        latest_date = ic_data["date"].max()
        start_date = latest_date - pd.DateOffset(months=12)
        recent_ic = ic_data[ic_data["date"] >= start_date].copy()
        
        # 如果近12个月数据不足，使用所有可用数据
        if len(recent_ic) == 0:
            recent_ic = ic_data.copy()
        
        if len(recent_ic) == 0:
            return jsonify({
                "error": "没有IC数据",
                "dates": [],
                "icir": []
            }), 200
        
        # 按日期分组，重新计算ICIR (IC_mean / IC_std)
        # 因为factor_ic_ir.parquet中的icir可能都是0，需要重新计算
        daily_stats = recent_ic.groupby("date")["ic"].agg(['mean', 'std', 'count']).reset_index()
        daily_stats = daily_stats.sort_values("date")
        
        # 计算ICIR: ICIR = IC_mean / IC_std
        # 只有当std > 0时才计算ICIR，否则为NaN（不填充为0，因为0没有意义）
        daily_stats["icir"] = daily_stats.apply(
            lambda row: row["mean"] / row["std"] if row["std"] > 0 else None, 
            axis=1
        )
        # 将None转换为NaN，然后前端会处理NaN显示
        daily_stats["icir"] = daily_stats["icir"].replace([None], np.nan)
        
        return jsonify({
            "error": None,
            "dates": [d.strftime("%Y-%m-%d") for d in daily_stats["date"]],
            "icir": daily_stats["icir"].tolist()
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"计算Rolling ICIR失败: {str(e)}",
            "dates": [],
            "icir": [],
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/api/factor-diagnostics/rolling-tstat')
def rolling_tstat():
    """Rolling t-stat (IC / IC_std)"""
    try:
        # 读取IC/ICIR数据
        factor_cfg = load_factor_settings(str(SETTINGS))
        factor_store_path = Path(factor_cfg["paths"].get("factors_store", "data/factors/factor_store.parquet"))
        if not factor_store_path.is_absolute():
            factor_store_path = project_root / factor_store_path
        
        ic_store_path = factor_store_path.parent / "factor_ic_ir.parquet"
        
        if not ic_store_path.exists():
            return jsonify({
                "error": "IC数据不存在",
                "dates": [],
                "tstat": []
            }), 200
        
        ic_data = pd.read_parquet(ic_store_path)
        if not pd.api.types.is_datetime64_any_dtype(ic_data["date"]):
            ic_data["date"] = pd.to_datetime(ic_data["date"])
        
        # 获取近12个月的数据（如果数据不足12个月，使用所有可用数据）
        latest_date = ic_data["date"].max()
        start_date = latest_date - pd.DateOffset(months=12)
        recent_ic = ic_data[ic_data["date"] >= start_date].copy()
        
        # 如果近12个月数据不足，使用所有可用数据
        if len(recent_ic) == 0:
            recent_ic = ic_data.copy()
        
        if len(recent_ic) == 0:
            return jsonify({
                "error": "没有IC数据",
                "dates": [],
                "tstat": []
            }), 200
        
        # 按日期分组，计算每日t-stat (IC_mean / IC_std)
        daily_stats = recent_ic.groupby("date")["ic"].agg(['mean', 'std', 'count']).reset_index()
        daily_stats = daily_stats.sort_values("date")
        
        # 计算t-stat: t = mean / (std / sqrt(n))
        daily_stats["tstat"] = daily_stats["mean"] / (daily_stats["std"] / np.sqrt(daily_stats["count"]))
        daily_stats["tstat"] = daily_stats["tstat"].fillna(0)
        
        return jsonify({
            "error": None,
            "dates": [d.strftime("%Y-%m-%d") for d in daily_stats["date"]],
            "tstat": daily_stats["tstat"].tolist()
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"计算Rolling t-stat失败: {str(e)}",
            "dates": [],
            "tstat": [],
            "traceback": traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
        }), 200


@app.route('/api/factor-diagnostics/factors')
def factor_diagnostics_factors():
    """获取所有因子列表"""
    try:
        factor_cfg = load_factor_settings(str(SETTINGS))
        factor_store_path = Path(factor_cfg["paths"].get("factors_store", "data/factors/factor_store.parquet"))
        if not factor_store_path.is_absolute():
            factor_store_path = project_root / factor_store_path
        
        # 尝试自动从 Hugging Face 下载（如果文件不存在）
        if not factor_store_path.exists():
            try:
                from src.data_loader import ensure_factor_store
                if not ensure_factor_store(factor_store_path, auto_download=True):
                    return jsonify({
                        "error": "因子数据不存在，正在尝试从 Hugging Face 下载...",
                        "factors": [],
                        "downloading": True
                    }), 200
            except ImportError:
                pass
        
        if not factor_store_path.exists():
            return jsonify({"error": "因子数据不存在", "factors": []}), 200
        
        factor_store = pd.read_parquet(factor_store_path)
        factors = list(factor_store.columns) if isinstance(factor_store, pd.DataFrame) else []
        
        return jsonify({
            "error": None,
            "factors": factors
        })
    except Exception as e:
        return jsonify({
            "error": f"获取因子列表失败: {str(e)}",
            "factors": []
        }), 200


@app.route('/api/factor-diagnostics/long-short')
def long_short_performance():
    """Long-Short Portfolio Performance（按因子分层）"""
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
        from src.factor_engine import read_prices, forward_return
        
        # 读取因子数据
        factor_cfg = load_factor_settings(str(SETTINGS))
        factor_store_path = Path(factor_cfg["paths"].get("factors_store", "data/factors/factor_store.parquet"))
        if not factor_store_path.is_absolute():
            factor_store_path = project_root / factor_store_path
        
        factor_store = pd.read_parquet(factor_store_path)
        if not isinstance(factor_store.index, pd.MultiIndex):
            if "date" in factor_store.columns and "ticker" in factor_store.columns:
                factor_store["date"] = pd.to_datetime(factor_store["date"])
                factor_store = factor_store.set_index(["date", "ticker"]).sort_index()
        
        if factor_name not in factor_store.columns:
            return jsonify({
                "error": f"因子 {factor_name} 不存在",
                "dates": [],
                "long_returns": [],
                "short_returns": [],
                "long_short_returns": [],
                "stats": {}
            }), 200
        
        # 读取价格数据 - 使用factor_cfg，并确保路径解析正确
        try:
            # 确保使用正确的配置对象（包含paths和database）
            # 如果路径是相对路径，需要转换为绝对路径
            if "paths" in factor_cfg and "prices_parquet" in factor_cfg["paths"]:
                parquet_path = factor_cfg["paths"]["prices_parquet"]
                if not Path(parquet_path).is_absolute():
                    factor_cfg["paths"]["prices_parquet"] = str(project_root / parquet_path)
            
            prices = read_prices(factor_cfg)
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback_str = traceback.format_exc() if IBKR_CONFIG.get('debug', False) else None
            return jsonify({
                "error": f"无法加载价格数据: {error_msg}",
                "dates": [],
                "long_returns": [],
                "short_returns": [],
                "long_short_returns": [],
                "stats": {},
                "traceback": traceback_str
            }), 200
        
        if prices is None or len(prices) == 0:
            return jsonify({
                "error": "价格数据不存在或为空",
                "dates": [],
                "long_returns": [],
                "short_returns": [],
                "long_short_returns": [],
                "stats": {}
            }), 200
        
        # 计算未来收益 - 先处理重复索引
        # 移除prices中的重复索引
        if isinstance(prices.index, pd.MultiIndex):
            prices = prices[~prices.index.duplicated(keep='first')]
        
        forward_ret = forward_return(prices, horizon=1)
        
        # 移除forward_ret中的重复索引
        if isinstance(forward_ret.index, pd.MultiIndex):
            forward_ret = forward_ret[~forward_ret.index.duplicated(keep='first')]
        
        # 获取近12个月的数据（如果数据不足12个月，使用所有可用数据）
        latest_date = factor_store.index.get_level_values(0).max()
        start_date = latest_date - pd.DateOffset(months=12)
        date_range = factor_store.index.get_level_values(0).unique()
        date_range = date_range[date_range >= start_date]
        
        # 如果近12个月数据不足，使用所有可用数据
        if len(date_range) == 0:
            date_range = factor_store.index.get_level_values(0).unique()
        
        # 按日期计算long-short收益
        dates = []
        long_returns = []
        short_returns = []
        long_short_returns = []
        
        for date in sorted(date_range):
            date_factors = factor_store.loc[factor_store.index.get_level_values(0) == date, factor_name]
            # 移除重复索引
            if isinstance(date_factors.index, pd.MultiIndex):
                date_factors = date_factors[~date_factors.index.duplicated(keep='first')]
            
            date_forward_ret = forward_ret.loc[forward_ret.index.get_level_values(0) == date]
            # 移除重复索引
            if isinstance(date_forward_ret.index, pd.MultiIndex):
                date_forward_ret = date_forward_ret[~date_forward_ret.index.duplicated(keep='first')]
            
            # 对齐索引
            aligned = pd.concat([date_factors, date_forward_ret], axis=1).dropna()
            if len(aligned) < 20:  # 至少需要20个股票
                continue
            
            # 按因子值排序，分成5层
            aligned = aligned.sort_values(by=aligned.columns[0])
            n = len(aligned)
            long_portfolio = aligned.iloc[-n//5:]  # Top 20%
            short_portfolio = aligned.iloc[:n//5]  # Bottom 20%
            
            # 计算收益
            long_ret = long_portfolio.iloc[:, 1].mean()
            short_ret = short_portfolio.iloc[:, 1].mean()
            ls_ret = long_ret - short_ret
            
            dates.append(date.strftime("%Y-%m-%d"))
            long_returns.append(float(long_ret))
            short_returns.append(float(short_ret))
            long_short_returns.append(float(ls_ret))
        
        # 计算累计收益
        long_cum = (1 + pd.Series(long_returns)).cumprod()
        short_cum = (1 + pd.Series(short_returns)).cumprod()
        ls_cum = (1 + pd.Series(long_short_returns)).cumprod()
        
        # 计算统计指标
        long_returns_series = pd.Series(long_returns)
        short_returns_series = pd.Series(short_returns)
        ls_returns_series = pd.Series(long_short_returns)
        
        def calc_stats(returns):
            annual_return = returns.mean() * 252
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            cum = (1 + returns).cumprod()
            max_dd = (cum / cum.cummax() - 1).min()
            return {
                "annual_return": float(annual_return),
                "sharpe": float(sharpe),
                "max_dd": float(max_dd)
            }
        
        stats = {
            "long_annual_return": calc_stats(long_returns_series)["annual_return"],
            "long_sharpe": calc_stats(long_returns_series)["sharpe"],
            "long_max_dd": calc_stats(long_returns_series)["max_dd"],
            "short_annual_return": calc_stats(short_returns_series)["annual_return"],
            "short_sharpe": calc_stats(short_returns_series)["sharpe"],
            "short_max_dd": calc_stats(short_returns_series)["max_dd"],
            "long_short_annual_return": calc_stats(ls_returns_series)["annual_return"],
            "long_short_sharpe": calc_stats(ls_returns_series)["sharpe"],
            "long_short_max_dd": calc_stats(ls_returns_series)["max_dd"]
        }
        
        return jsonify({
            "error": None,
            "dates": dates,
            "long_returns": long_cum.tolist(),
            "short_returns": short_cum.tolist(),
            "long_short_returns": ls_cum.tolist(),
            "stats": stats
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
    """因子簇分析（Momentum / Quality / Volatility）"""
    try:
        # 读取IC/ICIR数据
        factor_cfg = load_factor_settings(str(SETTINGS))
        factor_store_path = Path(factor_cfg["paths"].get("factors_store", "data/factors/factor_store.parquet"))
        if not factor_store_path.is_absolute():
            factor_store_path = project_root / factor_store_path
        
        ic_store_path = factor_store_path.parent / "factor_ic_ir.parquet"
        
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
        
        # 重新计算 ICIR (IC_mean / IC_std)，而不是从文件中读取（因为文件中的 icir 可能都是 0）
        factor_stats["icir"] = factor_stats.apply(
            lambda row: row["ic_mean"] / row["ic_std"] if row["ic_std"] > 0 else None,
            axis=1
        )
        # 将 None 转换为 NaN
        factor_stats["icir"] = factor_stats["icir"].replace([None], np.nan)
        
        # 计算t-stat
        factor_stats["tstat"] = factor_stats["ic_mean"] / (factor_stats["ic_std"] / np.sqrt(factor_stats["ic_count"]))
        factor_stats["tstat"] = factor_stats["tstat"].fillna(0)
        
        # 因子分类（基于因子名称）
        def classify_factor(factor_name):
            name_lower = str(factor_name).lower()
            if any(x in name_lower for x in ['momentum', 'mom', 'ret', 'return', 'alpha']):
                return 'momentum'
            elif any(x in name_lower for x in ['quality', 'roe', 'roa', 'profit', 'margin']):
                return 'quality'
            elif any(x in name_lower for x in ['vol', 'volatility', 'std', 'var', 'risk']):
                return 'volatility'
            elif any(x in name_lower for x in ['value', 'pe', 'pb', 'price']):
                return 'value'
            elif any(x in name_lower for x in ['size', 'market', 'cap', 'mkt']):
                return 'size'
            else:
                return 'other'
        
        factor_stats["category"] = factor_stats["factor"].apply(classify_factor)
        
        # 按类别分组
        clusters = {
            "momentum": [],
            "quality": [],
            "volatility": [],
            "value": [],
            "size": [],
            "other": []
        }
        
        for _, row in factor_stats.iterrows():
            # ICIR 如果是 NaN，不转换为 0，而是保持为 None（前端会处理）
            icir_value = row["icir"]
            if pd.isna(icir_value):
                icir_value = None
            else:
                icir_value = float(icir_value)
            
            clusters[row["category"]].append({
                "name": row["factor"],
                "ic_mean": float(row["ic_mean"]) if not pd.isna(row["ic_mean"]) else 0.0,
                "icir": icir_value,  # 可能是 None（NaN）
                "tstat": float(row["tstat"]) if not pd.isna(row["tstat"]) else 0.0
            })
        
        return jsonify({
            "error": None,
            "clusters": clusters
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
    """因子相关性矩阵"""
    method = request.args.get('method', 'pearson')
    
    try:
        # 读取因子数据
        factor_cfg = load_factor_settings(str(SETTINGS))
        factor_store_path = Path(factor_cfg["paths"].get("factors_store", "data/factors/factor_store.parquet"))
        if not factor_store_path.is_absolute():
            factor_store_path = project_root / factor_store_path
        
        if not factor_store_path.exists():
            return jsonify({
                "error": "因子数据不存在",
                "factors": [],
                "correlation_matrix": []
            }), 200
        
        factor_store = pd.read_parquet(factor_store_path)
        if not isinstance(factor_store.index, pd.MultiIndex):
            if "date" in factor_store.columns and "ticker" in factor_store.columns:
                factor_store["date"] = pd.to_datetime(factor_store["date"])
                factor_store = factor_store.set_index(["date", "ticker"]).sort_index()
        
        # 获取近12个月的数据（如果数据不足12个月，使用所有可用数据）
        latest_date = factor_store.index.get_level_values(0).max()
        start_date = latest_date - pd.DateOffset(months=12)
        recent_factors = factor_store.loc[factor_store.index.get_level_values(0) >= start_date]
        
        # 如果近12个月数据不足，使用所有可用数据
        if len(recent_factors) == 0:
            recent_factors = factor_store
        
        # 选择部分因子（避免矩阵过大）
        factors = list(recent_factors.columns)[:50]  # 限制为50个因子
        factor_subset = recent_factors[factors]
        
        # 计算相关性矩阵（按日期平均）
        # 方法：对每个日期计算相关性，然后取平均
        dates = factor_subset.index.get_level_values(0).unique()
        corr_list = []
        
        for date in dates:
            date_factors = factor_subset.loc[factor_subset.index.get_level_values(0) == date]
            if len(date_factors) > 10:  # 至少需要10个股票
                corr = date_factors.corr(method=method)
                corr_list.append(corr)
        
        if len(corr_list) == 0:
            return jsonify({
                "error": "没有足够的数据计算相关性",
                "factors": [],
                "correlation_matrix": []
            }), 200
        
        # 平均相关性矩阵
        mean_corr = pd.concat(corr_list).groupby(level=0).mean()
        mean_corr = mean_corr.fillna(0)
        
        return jsonify({
            "error": None,
            "factors": factors,
            "correlation_matrix": mean_corr.values.tolist()
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
    """多因子风险暴露（Barra-style）"""
    date = request.args.get('date')
    
    try:
        # 读取因子数据
        factor_cfg = load_factor_settings(str(SETTINGS))
        factor_store_path = Path(factor_cfg["paths"].get("factors_store", "data/factors/factor_store.parquet"))
        if not factor_store_path.is_absolute():
            factor_store_path = project_root / factor_store_path
        
        if not factor_store_path.exists():
            return jsonify({
                "error": "因子数据不存在",
                "factors": [],
                "exposures": [],
                "risk_contributions": []
            }), 200
        
        factor_store = pd.read_parquet(factor_store_path)
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
            except:
                date_obj = available_dates.max()
        else:
            date_obj = available_dates.max()
        
        # 获取该日期的因子数据
        date_factors = factor_store.loc[factor_store.index.get_level_values(0) == date_obj]
        
        # 计算因子暴露度（标准化后的因子值）
        exposures = {}
        for factor_name in date_factors.columns:
            factor_series = date_factors[factor_name].dropna()
            if len(factor_series) > 0:
                # 标准化（z-score）
                mean_val = factor_series.mean()
                std_val = factor_series.std()
                if std_val > 0:
                    exposures[factor_name] = (factor_series - mean_val) / std_val
                else:
                    exposures[factor_name] = pd.Series([0.0] * len(factor_series), index=factor_series.index)
        
        # 计算风险贡献（简化版：使用因子方差）
        risk_contributions = {}
        total_risk = 0
        for factor_name, exp_series in exposures.items():
            risk = exp_series.var()
            risk_contributions[factor_name] = risk
            total_risk += risk
        
        # 归一化风险贡献
        if total_risk > 0:
            for factor_name in risk_contributions:
                risk_contributions[factor_name] = risk_contributions[factor_name] / total_risk
        
        # 计算暴露度统计（使用绝对值的平均值，而不是平均值）
        # 因为标准化后的因子值平均值为0是正常的，应该显示绝对暴露度
        avg_exposures = {name: float(exp.abs().mean()) for name, exp in exposures.items()}
        
        # 排序（按风险贡献）
        sorted_factors = sorted(risk_contributions.items(), key=lambda x: x[1], reverse=True)[:30]  # Top 30
        
        return jsonify({
            "error": None,
            "date": date_obj.strftime("%Y-%m-%d"),
            "factors": [f[0] for f in sorted_factors],
            "exposures": [round(avg_exposures.get(f[0], 0.0), 4) for f in sorted_factors],  # 保留4位小数
            "risk_contributions": [round(f[1] * 100, 2) for f in sorted_factors]  # 转换为百分比
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
            factor_store_path = project_root / factor_store_path
        
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
            factor_store_path = project_root / factor_store_path
        
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
            factor_store_path = project_root / factor_store_path
        
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
    ibkr_positions_file = project_root / "outputs" / "ibkr_data" / "positions.json"
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
    weights_path = project_root / "outputs" / "portfolios" / "weights.parquet"
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
                cfg_copy["paths"]["prices_parquet"] = str(project_root / parquet_path)
        
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
                "description": "Use TopK Dropout Strategy: select 20 stocks with highest prediction scores, replace 3 stocks daily"
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
                        trades.append({
                            "time": exec_time,
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
                                trades.append({
                                    "time": exec_time,
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
    ibkr_trades_file = project_root / "outputs" / "ibkr_data" / "trades.json"
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
                    
                    trades.append({
                        "time": trade_time,
                        "symbol": symbol,
                        "side": "BUY" if order.get("target_weight", 0) > 0 else "SELL",
                        "qty": 0.0,
                        "price": 0.0,
                        "amount": 0.0
                    })
        except:
            continue
    
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
    """获取 IBKR 真实收益数据（PnL/Ledger）- 使用 src/ibkr_live_trader.py"""
    if not IBKR_CONFIG.get('enabled', False):
        return jsonify({
            "error": "IBKR 未启用",
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "daily_pnl": []
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
        
        return jsonify({
            "error": None,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
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
        monthly_ret = strategy_returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
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
            weights_path = project_root / "outputs" / "portfolios" / "weights.parquet"
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
                monthly_turnover = turnover_series.resample('M').mean()
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
    ibkr_pnl_file = project_root / "outputs" / "ibkr_data" / "pnl.json"
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
    daily_returns = load_parquet_safe(BACKTEST_DIR / "daily_returns.parquet")
    
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
        monthly_ret = strategy_returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
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

