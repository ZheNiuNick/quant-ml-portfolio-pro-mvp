#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json, collections
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

# 使用统一的路径管理
from src.config.path import SETTINGS_FILE, OUTPUT_REPORTS_DIR, OUTPUT_MODELS_DIR, get_path

SETTINGS = SETTINGS_FILE

# ----------------------------
# Config
# ----------------------------
def load_settings(path=SETTINGS_FILE):
    import yaml
    path = get_path(path) if isinstance(path, str) and not Path(path).is_absolute() else Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ----------------------------
# Utils
# ----------------------------
def forward_return(prices: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """
    通用前瞻收益： (close[t+horizon] / close[t]) - 1
    Args:
        prices: MultiIndex (date, ticker) DataFrame，需包含 'Adj Close'
        horizon: 预测窗口，单位=交易日
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    
    # 移除重复索引，避免 unstack 时出错
    if isinstance(prices.index, pd.MultiIndex):
        prices = prices[~prices.index.duplicated(keep='first')]
    
    px = prices["Adj Close"].unstack("ticker")
    fwd = (px.shift(-horizon) / px - 1.0).stack()
    fwd.name = f"fwd{horizon}d"
    
    # 移除结果中的重复索引
    if isinstance(fwd.index, pd.MultiIndex):
        fwd = fwd[~fwd.index.duplicated(keep='first')]
    
    return fwd


def forward_return_1d(prices: pd.DataFrame) -> pd.Series:
    """兼容旧接口：next-day return"""
    return forward_return(prices, horizon=1)

def qlib_label(prices: pd.DataFrame) -> pd.Series:
    """
    QLib 的标签定义：Ref($close, -2)/Ref($close, -1) - 1
    即：close[t+2] / close[t+1] - 1
    这是 t+2 到 t+1 的收益率（非标准的 next-day return）
    
    Args:
        prices: DataFrame with MultiIndex (date, ticker), must contain 'Adj Close'
    
    Returns:
        Series aligned to (date, ticker) with QLib-style label
    """
    px = prices["Adj Close"].unstack("ticker")
    # Ref($close, -2)/Ref($close, -1) - 1
    # = close[t+2] / close[t+1] - 1
    label = (px.shift(-2) / px.shift(-1) - 1.0).stack()
    label.name = "qlib_label"
    return label


def apply_feature_lag(X: pd.DataFrame, lag_days: int = 0) -> pd.DataFrame:
    """
    将特征整体向后滞后 lag_days 天，以避免信息泄露
    """
    if lag_days <= 0:
        return X
    shifted = X.groupby(level="ticker").shift(lag_days)
    return shifted


def build_label_series(prices: pd.DataFrame,
                       idx,
                       cfg: dict,
                       use_qlib_label: bool = False) -> pd.Series:
    """
    根据配置构建连续标签：
      - use_qlib_label=True 时使用 QLib LABEL0 定义
      - 否则根据 horizon/shift 配置计算 (close[t+h]/close[t]) - 1，并可再向未来平移
    """
    if use_qlib_label:
        return qlib_label(prices).reindex(idx)

    label_cfg = cfg.get("model", {}).get("label_options", {})
    horizon = max(1, int(label_cfg.get("horizon_days", 1)))
    shift_days = max(0, int(label_cfg.get("shift_days", 0)))

    y = forward_return(prices, horizon=horizon).reindex(idx)
    if shift_days > 0:
        y = y.groupby(level="ticker").shift(-shift_days)
    return y

def quantile_label(series: pd.Series, q: int = 5) -> pd.Series:
    """
    对 (date,ticker) 面板按“每天”把连续目标分到 0..q-1 的分位桶。
    - 自动清理 NaN/±inf
    - 当当日可用样本或唯一值不足 q 时，自动降阶到 k<=q
    - 仍然无法分箱的样本置 NaN（后续统一丢弃）
    返回：pandas 数值类型 Int64（可空）
    """
    s = series.replace([np.inf, -np.inf], np.nan)

    def one_day(x: pd.Series) -> pd.Series:
        xx = x.dropna()
        if xx.empty:
            return pd.Series(index=x.index, dtype="Int64")
        # 排序打散避免大量并列
        r = xx.rank(method="first")
        k = int(min(q, r.nunique()))
        if k < 2:
            # 样本太少或全相等 -> 无法形成有效排序标签
            bins = pd.Series(pd.NA, index=xx.index, dtype="Int64")
        else:
            # duplicates='drop' 允许边界重复时自动并箱
            bins = pd.qcut(r, q=k, labels=False, duplicates="drop")
            # qcut 产出为 Float，转可空 Int64；保持 index
            bins = pd.Series(bins.astype("Int64"), index=xx.index)
        # 还原到原 index，其它位置 NaN
        return bins.reindex(x.index).astype("Int64")

    lab = s.groupby(level="date", sort=False).apply(one_day)
    # groupby.apply 会产生额外 level，去掉它
    lab.index = lab.index.droplevel(0)
    return lab.astype("Int64")

def drop_bad_features(X: pd.DataFrame,
                      max_missing: float = 0.2,
                      near_const_std: float = 1e-9,
                      near_const_frac: float = 0.8) -> pd.DataFrame:
    """
    去掉：
      1) 全NaN/全局常数
      2) 缺失率 > max_missing
      3) 在 >near_const_frac 的交易日里横截面std < near_const_std 的“伪常数”
    """
    # (1) all-NaN / 全局常数
    keep = X.columns[X.notna().any(axis=0)]
    X1 = X[keep]
    keep = X1.columns[X1.nunique(dropna=True) > 1]
    X2 = X1[keep]

    # (2) 缺失率
    miss = X2.isna().mean()
    keep = miss.index[miss <= max_missing]
    X3 = X2[keep]

    # (3) 伪常数（横截面std几乎为0的交易日占比很高）
    bad = []
    for c in X3.columns:
        s = X3[c].unstack("ticker")
        cs_std = s.std(axis=1, ddof=0)
        if (cs_std < near_const_std).mean() > near_const_frac:
            bad.append(c)
    if bad:
        print(f"[Filter] dropped {len(bad)} near-constant features")
        X3 = X3.drop(columns=bad)

    return X3

def per_date_groups(index: pd.MultiIndex) -> list:
    """group sizes for LGBMRanker (group by date)."""
    # 更快的统计
    dates, counts = np.unique(index.get_level_values(0).values, return_counts=True)
    return counts.tolist()

def rank_ic_per_day(pred: pd.Series, y: pd.Series) -> pd.Series:
    df = pd.concat([pred.rename("pred"), y.rename("y")], axis=1).dropna()
    ric = df.groupby(level="date").apply(
        lambda x: x["pred"].rank().corr(x["y"].rank(), method="spearman")
    )
    ric.name = "rank_ic"
    return ric

def safe_rank_ic(pred: pd.Series, y: pd.Series) -> float:
    """
    安全计算 Rank IC（处理 NaN 和常数情况）
    返回：平均 Rank IC（跳过无效日期）
    """
    df = pd.DataFrame({"pred": pred, "y": y}).dropna()
    if len(df) < 2:
        return 0.0
    
    ric_daily = df.groupby(level="date").apply(
        lambda x: x["pred"].rank(method="first").corr(x["y"].rank(method="first"), method="spearman")
        if len(x) >= 2 and x["pred"].nunique() > 1 and x["y"].nunique() > 1
        else np.nan
    )
    ric_daily = ric_daily.dropna()
    return float(ric_daily.mean()) if len(ric_daily) > 0 else 0.0

def calculate_rank_ic_and_icir(pred: pd.Series, y: pd.Series) -> tuple:
    """
    计算 Rank IC 和 Rank ICIR（对齐 QLib 基准测试）
    
    Returns:
        (rank_ic_mean, rank_icir): 平均 Rank IC 和 Rank ICIR
    """
    df = pd.DataFrame({"pred": pred, "y": y}).dropna()
    if len(df) < 2:
        return 0.0, 0.0
    
    # 按日期计算每日 Rank IC
    ric_daily = df.groupby(level="date").apply(
        lambda x: x["pred"].rank(method="first").corr(x["y"].rank(method="first"), method="spearman")
        if len(x) >= 2 and x["pred"].nunique() > 1 and x["y"].nunique() > 1
        else np.nan
    )
    ric_daily = ric_daily.dropna()
    
    if len(ric_daily) == 0:
        return 0.0, 0.0
    
    # Rank IC = 平均相关性
    rank_ic_mean = float(ric_daily.mean())
    
    # Rank ICIR = mean(IC) / std(IC)
    rank_ic_std = float(ric_daily.std())
    rank_icir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0.0
    
    return rank_ic_mean, rank_icir

def diagnose_data_quality(cfg, use_qlib_label=False):
    """
    诊断数据质量和覆盖度（对齐 QLib 基准测试要求）
    """
    panel, dates = prepare_regression_data(cfg, use_qlib_label=use_qlib_label)
    
    print("\n" + "="*60)
    print("数据质量诊断（对齐 QLib 基准测试）")
    print("="*60)
    
    print(f"\n【总体统计】")
    print(f"  • 日期范围: {dates.min()} 到 {dates.max()}")
    print(f"  • 总日期数: {len(dates)}")
    print(f"  • 有效样本数: {len(panel)}")
    print(f"  • 特征数量: {len(panel.columns)-1}")
    print(f"  • 平均每日样本数: {len(panel) / len(dates):.1f}")
    
    # 标签统计
    y = panel["y"]
    print(f"\n【标签统计】")
    print(f"  • 标签缺失率: {(y.isna().sum() / len(y) * 100):.2f}%")
    print(f"  • 标签均值: {y.mean():.6f}")
    print(f"  • 标签标准差: {y.std():.6f}")
    print(f"  • 标签分位数 [0.1, 0.5, 0.9]: {y.quantile([0.1, 0.5, 0.9]).values}")
    
    # 每日样本分布
    daily_counts = panel.groupby(level="date").size()
    print(f"\n【每日样本分布】")
    print(f"  • 最小每日样本数: {daily_counts.min()}")
    print(f"  • 最大每日样本数: {daily_counts.max()}")
    print(f"  • 平均每日样本数: {daily_counts.mean():.1f}")
    print(f"  • 中位数每日样本数: {daily_counts.median():.1f}")
    
    # 特征缺失统计
    feature_missing = panel.drop(columns=["y"]).isnull().sum()
    if feature_missing.sum() > 0:
        print(f"\n【特征缺失统计】")
        print(f"  • 有缺失值的特征数: {(feature_missing > 0).sum()}")
        print(f"  • 最大缺失率: {(feature_missing.max() / len(panel) * 100):.2f}%")
    
    print("="*60 + "\n")
    
    return panel, dates

def prepare_regression_data(cfg, use_qlib_label=False):
    """
    共享的数据准备函数：为回归模型准备数据
    返回：(panel, dates) - panel包含特征和标签y，dates用于CV分割
    
    Args:
        cfg: 配置字典
        use_qlib_label: 是否使用 QLib 的标签定义
                        - 如果 factor_store 中包含 LABEL0（QLib 生成的），优先使用它
                        - 否则使用 Ref($close, -2)/Ref($close, -1) - 1 或 next-day return
    """
    # 准备数据（不转换为分位数）
    X = pd.read_parquet(cfg["paths"]["factors_store"])
    X.index = pd.MultiIndex.from_tuples(X.index, names=["date", "ticker"])
    X = X.sort_index()
    
    # 从价格数据计算标签
    prices = pd.read_parquet(cfg["paths"]["prices_parquet"])
    prices.index = pd.MultiIndex.from_tuples(prices.index, names=["date", "ticker"])
    prices = prices.sort_index()
    
    # 对齐索引
    idx = X.index.intersection(prices.index)
    X = X.reindex(idx)

    feature_lag_days = int(cfg.get("model", {}).get("feature_lag_days", 0))
    if feature_lag_days > 0:
        print(f"[INFO] Lagging features by {feature_lag_days} day(s) to avoid look-ahead bias")
        X = apply_feature_lag(X, feature_lag_days)
    
    # 根据配置选择标签定义
    y_cont = build_label_series(prices, idx, cfg, use_qlib_label=use_qlib_label)
    label_cfg = cfg.get("model", {}).get("label_options", {})
    if not use_qlib_label:
        print("[INFO] Label options -> horizon_days: "
              f"{label_cfg.get('horizon_days', 1)}, shift_days: {label_cfg.get('shift_days', 0)}")
    
    # 使用连续收益率作为标签
    panel = X.copy()
    panel["y"] = y_cont
    panel = panel.dropna(subset=["y"])
    
    # 粗筛特征
    panel = panel.loc[:, ~panel.isnull().all()]
    panel = panel.loc[:, (panel.nunique() > 1)]

    feature_filter_cfg = cfg.get("model", {}).get("feature_filter", {})
    if feature_filter_cfg.get("enabled", False):
        feats = panel.drop(columns=["y"])
        before = len(feats.columns)
        feats = drop_bad_features(
            feats,
            max_missing=feature_filter_cfg.get("max_missing", 0.2),
            near_const_std=feature_filter_cfg.get("near_const_std", 1e-9),
            near_const_frac=feature_filter_cfg.get("near_const_frac", 0.8),
        )
        removed = before - len(feats.columns)
        if removed > 0:
            print(f"[Filter] Removed {removed} features via feature_filter (max_missing / near-constant rules)")
        panel = feats.join(panel["y"], how="inner")
    
    # 可选：基于单因子 ICIR 过滤特征（如果 summary 文件存在）
    summary_path = OUTPUT_REPORTS_DIR / "single_factor_summary.json"
    if summary_path.exists() and cfg.get("model", {}).get("filter_by_icir", False):
        import json
        with open(summary_path, "r") as f:
            summary = json.load(f)
        icir_threshold = cfg["model"].get("min_icir", 0.05)
        # 修复：单因子分析结果中使用的是 "icir" 字段，不是 "icir_1d"
        feats = panel.drop(columns=["y"])
        good_features = [k for k, v in summary.items() 
                        if k in feats.columns and abs(v.get("icir", 0)) > icir_threshold]
        if good_features:
            print(f"[Filter] Keeping {len(good_features)}/{len(feats.columns)} features with |ICIR| > {icir_threshold}")
            panel = feats[good_features].join(panel["y"], how="inner")
        else:
            print(f"[Warn] No features with |ICIR| > {icir_threshold}, using all {len(feats.columns)} features")
    
    # PCA降维：如果使用PCA降维后的因子文件，这里不需要再处理
    # 使用 apply_pca_reduction.py 脚本预先降维并保存，然后修改 config/settings.yaml 中的 factors_store 路径即可
    
    # 再次检查特征数量
    if panel.empty or len(panel.columns) <= 1:  # 至少要有y列
        raise ValueError("No features remaining after filtering/reduction. Check configuration.")
    
    # 准备CV日期分割
    dates = np.array(sorted(panel.index.get_level_values("date").unique()))
    
    label_type = "QLib label" if use_qlib_label else "Next-day return"
    print(f"[Data] Regression mode ({label_type}): {len(panel)} rows, {len(panel.columns)-1} features")
    
    return panel, dates

def filter_small_groups(df: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """去掉样本数太少的交易日（对 lambdarank 不友好）"""
    cnt = df.groupby(level="date").size()
    good = cnt.index[cnt >= min_n]
    return df.loc[df.index.get_level_values("date").isin(good)]

# ----------------------------
# Panel build
# ----------------------------
def prepare_panel(cfg, q_bins: int = 20):
    # 载入因子
    X = pd.read_parquet(cfg["paths"]["factors_store"])
    X.index = pd.MultiIndex.from_tuples(X.index, names=["date", "ticker"])
    X = X.sort_index()

    # ---- 目标：强制使用 prices 计算的 forward_return_1d（标准 next-day return）
    # 注意：Qlib 的 LABEL0 = Ref($close, -2)/Ref($close, -1) - 1 是 t+2→t+1 的收益率，
    # 不是标准的 t+1→t 收益率，因此不使用 LABEL0 作为标签
    # 如果 LABEL0 在因子库中，需要先移除它（避免特征泄漏）
    if "LABEL0" in X.columns:
        print("[INFO] Removing LABEL0 from features (it's a label, not a feature)")
        X = X.drop(columns=["LABEL0"])
    
    prices = pd.read_parquet(cfg["paths"]["prices_parquet"])
    prices.index = pd.MultiIndex.from_tuples(prices.index, names=["date", "ticker"])
    prices = prices.sort_index()
    
    # 详细的对齐诊断
    X_dates = X.index.get_level_values("date").unique()
    X_tickers = X.index.get_level_values("ticker").unique()
    P_dates = prices.index.get_level_values("date").unique()
    P_tickers = prices.index.get_level_values("ticker").unique()
    
    print(f"[Data] Factor store: {len(X.index)} rows, {len(X_dates)} dates, {len(X_tickers)} tickers")
    print(f"[Data] Prices: {len(prices.index)} rows, {len(P_dates)} dates, {len(P_tickers)} tickers")
    
    idx = X.index.intersection(prices.index)
    if len(idx) < len(X.index):
        overlap_pct = len(idx) / len(X.index) * 100
        print(f"[Warn] prices 覆盖不足，截取交集：{len(idx)}/{len(X.index)} ({overlap_pct:.1f}%)")
        if overlap_pct < 10:
            print(f"[Error] Data overlap too low! This suggests factor_store and prices use different tickers or date ranges.")
            print(f"       Factor store date range: {X_dates.min()} to {X_dates.max()}")
            print(f"       Prices date range: {P_dates.min()} to {P_dates.max()}")
            print(f"       Factor store tickers (first 10): {list(X_tickers[:10])}")
            print(f"       Prices tickers (first 10): {list(P_tickers[:10])}")
            raise ValueError("Data alignment failure: factor_store and prices have insufficient overlap.")
    
    X = X.reindex(idx)
    prices = prices.reindex(idx)
    feature_lag_days = int(cfg.get("model", {}).get("feature_lag_days", 0))
    if feature_lag_days > 0:
        print(f"[INFO] Lagging features by {feature_lag_days} day(s) before ranking")
        X = apply_feature_lag(X, feature_lag_days)

    y_cont = build_label_series(prices, idx, cfg, use_qlib_label=False).reindex(idx)
    y_cont.name = "target"
    
    print(f"[Data] After alignment: {len(X.index)} rows, {len(X.columns)} features")

    # 先粗筛列（不依赖CV的过滤）
    X = drop_bad_features(X)
    
    # 检查过滤后是否有特征
    if X.empty or len(X.columns) == 0:
        raise ValueError("No features remaining after drop_bad_features. Check factor_store data quality.")
    
    # 可选：基于单因子 ICIR 过滤特征（如果 summary 文件存在）
    summary_path = OUTPUT_REPORTS_DIR / "single_factor_summary.json"
    if summary_path.exists() and cfg.get("model", {}).get("filter_by_icir", False):
        import json
        with open(summary_path, "r") as f:
            summary = json.load(f)
        icir_threshold = cfg["model"].get("min_icir", 0.05)
        good_features = [k for k, v in summary.items() 
                        if k in X.columns and abs(v.get("icir", 0)) > icir_threshold]
        if good_features:
            print(f"[Filter] Keeping {len(good_features)}/{len(X.columns)} features with |ICIR| > {icir_threshold}")
            X = X[good_features]
        else:
            print(f"[Warn] No features with |ICIR| > {icir_threshold}, using all {len(X.columns)} features")

    # PCA降维：如果使用PCA降维后的因子文件，这里不需要再处理
    # 使用 apply_pca_reduction.py 脚本预先降维并保存，然后修改 config/settings.yaml 中的 factors_store 路径即可

    # 再次检查特征数量
    if X.empty or len(X.columns) == 0:
        raise ValueError("No features remaining after filtering/reduction. Check configuration.")

    # 每日横截面用中位数填补残余缺失（大多数特征已zscore，接近0）
    # 处理可能存在的空组问题
    def safe_fillna(g):
        if g.empty or g.shape[0] == 0 or g.shape[1] == 0:
            return g
        # 计算每列的中位数（跳过全NaN的列）
        med = g.median()
        # 只对非全NaN的列填充
        return g.fillna(med)
    
    # 先检查每个日期是否有足够的样本
    date_counts = X.groupby(level="date").size()
    valid_dates = date_counts.index[date_counts > 0]
    if len(valid_dates) == 0:
        raise ValueError("No dates with valid data after filtering. Check data alignment between factor_store and prices.")
    
    X = X.loc[X.index.get_level_values("date").isin(valid_dates)]
    
    # 分组填充缺失值，使用 transform 方法
    # 如果某些日期组为空，transform 会返回空，我们需要处理这种情况
    filled_list = []
    for date in valid_dates:
        date_mask = X.index.get_level_values("date") == date
        date_group = X.loc[date_mask]
        if not date_group.empty and date_group.shape[1] > 0:
            filled = safe_fillna(date_group)
            if not filled.empty:
                filled_list.append(filled)
    
    if not filled_list:
        raise ValueError("All date groups are empty after processing. Check data quality.")
    
    X = pd.concat(filled_list).sort_index()

    # 生成分位标签（整数 0..q-1）
    # 只在 X 有值的索引上分箱，自动跳过缺失
# 生成分位标签（整数 0..q-1）
    y_cont = y_cont.reindex(X.index)
    y_rank = quantile_label(y_cont, q=q_bins)
    y_rank.name = "y_rank"          # ← 加这一行
    panel = X.join(y_rank, how="inner")
    panel = panel.dropna(subset=["y_rank"])   # 可选：提前过滤
    panel = filter_small_groups(panel, min_n=100)
    return panel

# ----------------------------
# Training & Evaluation
# ----------------------------
def train_ranker(cfg):
    import numpy as np
    import pandas as pd
    import lightgbm as lgb

    q_bins  = cfg["model"].get("rank_bins", 5)
    n_folds = cfg["model"]["cv"]["n_folds"]

    panel = prepare_panel(cfg, q_bins)  # 需确保 prepare_panel 里给 y_rank 命名
    # ——可选：如果你想更早过滤无标签样本，取消下一行注释
    # panel = panel.dropna(subset=["y_rank"])

    # 用日期做“扩增式”时间序列 CV（fold i 训练 <= i 的所有日期，测试第 i+1 个分片）
    dates = np.array(sorted(panel.index.get_level_values("date").unique()))
    folds = np.array_split(dates, n_folds)

    metrics  = {"folds": []}
    oof_pred = pd.Series(index=panel.index, dtype=float)

    for i in range(n_folds - 1):
        tr_dates = np.concatenate(folds[: i + 1])
        te_dates = folds[i + 1]

        tr = panel.loc[panel.index.get_level_values("date").isin(tr_dates)].copy()
        te = panel.loc[panel.index.get_level_values("date").isin(te_dates)].copy()

        # 1) 丢掉没有 y_rank 的样本（关键）
        tr = tr.dropna(subset=["y_rank"])
        te = te.dropna(subset=["y_rank"])

        # 2) 再拆 X / y，且只在这里做 astype
        Xtr = tr.drop(columns=["y_rank"])
        ytr = tr["y_rank"].astype("int64").to_numpy(copy=False)

        Xte = te.drop(columns=["y_rank"])
        yte = te["y_rank"].astype("int64").to_numpy(copy=False)

        # 3) 分组在过滤之后再算（按日期一组）
        def per_date_groups(idx):
            # idx 是 MultiIndex(date, symbol)
            # 返回 LightGBM 需要的 group 数组：每个日期对应的样本数
            dates = idx.get_level_values("date")
            _, counts = np.unique(dates, return_counts=True)
            return counts

        gtr = per_date_groups(Xtr.index)

        # 4) 建模
        params = dict(cfg["model"]["params"])
        params.setdefault("objective", "lambdarank")
        params.setdefault("metric",    "ndcg")
        params.setdefault("random_state", 42)
        # 添加早停以优化泛化能力
        params.setdefault("n_estimators", 2000)
        early_stop = cfg["model"].get("early_stopping_rounds", 100)
        
        # 准备验证集（使用训练集的最后 10% 日期作为验证，保持时间顺序）
        tr_dates_unique = sorted(tr.index.get_level_values("date").unique())
        n_val_dates = max(1, len(tr_dates_unique) // 10)
        val_dates = tr_dates_unique[-n_val_dates:]
        val_mask = tr.index.get_level_values("date").isin(val_dates)
        
        if val_mask.sum() > 0:
            Xval_tr = tr[val_mask].drop(columns=["y_rank"])
            yval_tr = tr[val_mask]["y_rank"].astype("int64").to_numpy(copy=False)
            gval_tr = per_date_groups(Xval_tr.index)
            
            # 从训练集中移除验证集
            tr_subset = tr[~val_mask]
            Xtr_subset = tr_subset.drop(columns=["y_rank"])
            ytr_subset = tr_subset["y_rank"].astype("int64").to_numpy(copy=False)
            gtr_subset = per_date_groups(Xtr_subset.index)
            
            model = lgb.LGBMRanker(**params)
            model.fit(
                Xtr_subset, ytr_subset, 
                group=gtr_subset,
                eval_set=[(Xval_tr, yval_tr)],
                eval_group=[gval_tr],
                callbacks=[lgb.early_stopping(early_stop, verbose=False)]
            )
        else:
            # 如果没有足够数据做验证集，不使用早停
            Xtr_subset = Xtr
            ytr_subset = ytr
            gtr_subset = gtr
            model = lgb.LGBMRanker(**params)
            model.fit(Xtr_subset, ytr_subset, group=gtr_subset)

        # 5) 预测与回填 OOF
        pred = model.predict(Xte)
        oof_pred.loc[Xte.index] = pred

        # 6) 每折指标：按日期的 Spearman（预测 vs. 真实标签的秩）
        dfte = Xte.copy()
        dfte["pred"]   = pred
        dfte["y_rank"] = yte
        ric_mean = dfte.groupby(level=0).apply(
            lambda x: x["pred"].rank().corr(x["y_rank"].rank(method="first"), method="spearman")
        ).mean()

        metrics["folds"].append({
            "fold": i + 1,
            "tr_dates": [str(tr_dates.min()), str(tr_dates.max())],
            "te_dates": [str(te_dates.min()), str(te_dates.max())],
            "mean_rank_ic": float(ric_mean),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_features": int(Xtr.shape[1]),
            "rank_bins": int(q_bins),
        })
        print(f"[Fold {i+1}] mean RankIC={ric_mean:.4f} | "
              f"train={len(tr)} test={len(te)} features={Xtr.shape[1]}")


    # final model（全部训练折，使用早停）
    full_dates = np.concatenate(folds[:-1])
    full = panel.loc[panel.index.get_level_values("date").isin(full_dates)]
    
    # 同样做验证集拆分用于早停（使用最后 10% 日期）
    full_dates_unique = sorted(full.index.get_level_values("date").unique())
    n_val_dates = max(1, len(full_dates_unique) // 10)
    val_dates = full_dates_unique[-n_val_dates:]
    val_mask = full.index.get_level_values("date").isin(val_dates)
    
    if val_mask.sum() > 0:
        Xval_full = full[val_mask].drop(columns=["y_rank"])
        yval_full = full[val_mask]["y_rank"].astype("int64").to_numpy(copy=False)
        gval_full = per_date_groups(Xval_full.index)
        
        full_subset = full[~val_mask]
        Xfull = full_subset.drop(columns=["y_rank"])
        yfull = full_subset["y_rank"].astype("int64").to_numpy(copy=False)
        gfull = per_date_groups(Xfull.index)
    else:
        # 如果没有足够数据，不使用验证集
        Xval_full, yval_full, gval_full = None, None, None
        Xfull = full.drop(columns=["y_rank"])
        yfull = full["y_rank"].astype("int64").to_numpy(copy=False)
        gfull = per_date_groups(Xfull.index)

    final_params = dict(cfg["model"]["params"])
    final_params.setdefault("objective", "lambdarank")
    final_params.setdefault("n_estimators", 2000)
    early_stop = cfg["model"].get("early_stopping_rounds", 100)
    
    final = lgb.LGBMRanker(**final_params)
    if Xval_full is not None:
        final.fit(
            Xfull, yfull, 
            group=gfull,
            eval_set=[(Xval_full, yval_full)],
            eval_group=[gval_full],
            callbacks=[lgb.early_stopping(early_stop, verbose=False)],
            feature_name=Xfull.columns.tolist()
        )
    else:
        final.fit(
            Xfull, yfull,
            group=gfull,
            feature_name=Xfull.columns.tolist()
        )

    # ----------------------------
    # Artifacts
    # ----------------------------
    out_dir = get_path(cfg["paths"].get("model_dir", "outputs/models"), OUTPUT_MODELS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存特征列表，供推理阶段对齐
    feature_list_path = out_dir / "feature_list_ranker.json"
    with open(feature_list_path, "w") as f:
        json.dump(Xfull.columns.tolist(), f, indent=2)
    
    final.booster_.save_model(str(out_dir / "lgbm_ranker.txt"))

    # SHAP Top-5（失败则回退为gain）
    top_feats = []
    try:
        import shap
        explainer = shap.TreeExplainer(final)
        ex = Xfull.tail(2000)
        sv = explainer.shap_values(ex)
        order = np.argsort(-np.mean(np.abs(sv), axis=0))[:5]
        top_feats = list(ex.columns[order])
    except Exception:
        gain = final.booster_.feature_importance(importance_type="gain")
        order = np.argsort(-gain)[:5]
        top_feats = list(Xfull.columns[order])

    shap_path = get_path(cfg["paths"].get("shap_top5_path", "outputs/shap_top5.json"))
    with open(shap_path, "w") as f:
        json.dump({"top5_features": top_feats}, f, indent=2)

    # OOF rolling RankIC
    oof = oof_pred.dropna()
    y_all = panel.loc[oof.index, "y_rank"].astype(int)
    ric_daily = rank_ic_per_day(oof, y_all)
    roll60 = ric_daily.rolling(60, min_periods=20).mean().rename("rank_ic_roll60")
    rric = pd.concat([ric_daily, roll60], axis=1)

    rric_path = get_path(cfg["paths"].get("rolling_rankic_path", "outputs/rolling_rankic.parquet"))
    rric.to_parquet(rric_path)

    metrics["oof_mean_rank_ic"] = float(ric_daily.mean())
    metrics_path = get_path(cfg["paths"].get("metrics_path", "outputs/reports/metrics.json"))
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[OK] Model trained.")
    print(f"  • Model: {out_dir/'lgbm_ranker.txt'}")
    print(f"  • SHAP top5: {shap_path}")
    print(f"  • Rolling RankIC: {rric_path}")
    print(f"  • Metrics: {metrics_path}")


# ----------------------------
# CLI
# ----------------------------
def train_regression(cfg):
    """
    使用 LightGBM 回归模型（MSE loss）训练，对齐 QLib 基准测试
    支持两种评估模式：
    1. walk_forward_cv: Walk-Forward 交叉验证（原有方式）
    2. qlib_segments: 固定 segments (train/valid/test)，对齐 QLib 基准测试
    """
    import numpy as np
    import pandas as pd
    import lightgbm as lgb

    # 数据质量诊断
    use_qlib_label = cfg.get("model", {}).get("use_qlib_label", False)
    diagnose_data_quality(cfg, use_qlib_label=use_qlib_label)
    
    # 获取评估模式
    eval_mode = cfg.get("model", {}).get("evaluation_mode", "walk_forward_cv")
    
    if eval_mode == "qlib_segments":
        # QLib 风格：固定 segments
        return _train_regression_qlib_segments(cfg)
    else:
        # 原有方式：Walk-Forward CV
        return _train_regression_walk_forward(cfg)

def _train_regression_qlib_segments(cfg):
    """
    QLib 风格的训练：使用固定 segments (train/valid/test)
    对齐 QLib workflow_config_lightgbm_Alpha158.yaml
    """
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    
    use_qlib_label = cfg.get("model", {}).get("use_qlib_label", False)
    panel, dates = prepare_regression_data(cfg, use_qlib_label=use_qlib_label)
    
    # 获取 segments 配置
    segments = cfg.get("model", {}).get("segments", {})
    # Use dynamic default end date (yesterday) to ensure we get latest data
    from datetime import datetime, timedelta
    default_test_end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    train_start = segments.get("train_start", "2010-01-01")
    train_end = segments.get("train_end", "2018-12-31")
    valid_start = segments.get("valid_start", "2019-01-01")
    valid_end = segments.get("valid_end", "2021-12-31")
    test_start = segments.get("test_start", "2022-01-01")
    test_end = segments.get("test_end", default_test_end)
    
    print(f"\n[QLib Segments Mode]")
    print(f"  Train: {train_start} to {train_end}")
    print(f"  Valid: {valid_start} to {valid_end}")
    print(f"  Test:  {test_start} to {test_end}")
    
    # 划分数据
    tr = panel.loc[(panel.index.get_level_values("date") >= train_start) & 
                   (panel.index.get_level_values("date") <= train_end)].copy()
    val = panel.loc[(panel.index.get_level_values("date") >= valid_start) & 
                   (panel.index.get_level_values("date") <= valid_end)].copy()
    te = panel.loc[(panel.index.get_level_values("date") >= test_start) & 
                   (panel.index.get_level_values("date") <= test_end)].copy()
    
    tr = tr.dropna(subset=["y"])
    val = val.dropna(subset=["y"])
    te = te.dropna(subset=["y"])
    
    print(f"\n  Data sizes: train={len(tr)}, valid={len(val)}, test={len(te)}")
    
    if len(tr) == 0:
        print("[ERROR] Train set is empty. Check segments configuration.")
        return
    
    if len(te) == 0:
        print("[WARN] Test set is empty. Using valid set as test set if available.")
        if len(val) > 0:
            te = val.copy()
            val = pd.DataFrame()  # 清空 valid，因为要用作 test
            print(f"  Using valid set as test: {len(te)} samples")
        else:
            print("[ERROR] Both valid and test sets are empty. Cannot proceed.")
            return
    
    Xtr, ytr = tr.drop(columns=["y"]), tr["y"]
    Xval, yval = val.drop(columns=["y"]), val["y"] if len(val) > 0 else pd.Series(dtype=float)
    Xte, yte = te.drop(columns=["y"]), te["y"]
    
    # 关键修复：填充缺失值的方式（优化：使用 groupby 替代双重循环）
    # 使用每日横截面中位数填充，而不是全局中位数
    # 这样可以避免跨日期污染，保持特征的区分度
    print(f"  [数据预处理] 使用每日横截面中位数填充缺失值（优化版）...")
    
    # 训练集：按日期分组，使用 transform 快速填充
    if isinstance(Xtr.index, pd.MultiIndex):
        # 计算每日横截面中位数
        train_medians_by_date = Xtr.groupby(level=0).transform('median')
        # 用每日中位数填充
        Xtr = Xtr.fillna(train_medians_by_date)
        # 对于某日期某特征全为NaN的情况，使用训练集全局中位数填充
        train_global_medians = Xtr.median()
        Xtr = Xtr.fillna(train_global_medians)
        # 如果全局中位数也是NaN，填充为0
        Xtr = Xtr.fillna(0.0)
    else:
        # 如果不是MultiIndex，使用全局中位数
        train_global_medians = Xtr.median()
        Xtr = Xtr.fillna(train_global_medians).fillna(0.0)
    
    # 验证集和测试集：先使用每日横截面中位数填充，再用训练集全局中位数填充剩余（对齐 QLib）
    # 这样保持特征的区分度，避免所有股票填充后特征相同
    if len(val) > 0:
        if isinstance(Xval.index, pd.MultiIndex):
            val_medians_by_date = Xval.groupby(level=0).transform('median')
            Xval = Xval.fillna(val_medians_by_date)
        Xval = Xval.fillna(train_global_medians).fillna(0.0)
    
    if isinstance(Xte.index, pd.MultiIndex):
        te_medians_by_date = Xte.groupby(level=0).transform('median')
        Xte = Xte.fillna(te_medians_by_date)
    Xte = Xte.fillna(train_global_medians).fillna(0.0)
    
    # 关键修复：保存训练集的全局中位数，供预测时使用
    import json
    model_dir = get_path(cfg["paths"]["model_dir"], OUTPUT_MODELS_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    train_medians_path = model_dir / "train_global_medians.json"
    train_medians_dict = train_global_medians.to_dict()
    with open(train_medians_path, "w") as f:
        json.dump(train_medians_dict, f, indent=2)
    print(f"  [保存] 训练集全局中位数已保存到 {train_medians_path}")
    
    # 准备 LightGBM 数据
    dtrain = lgb.Dataset(Xtr.values, label=ytr.values)
    # 如果验证集为空，不使用验证集（只用训练集监控）
    if len(Xval) > 0 and len(yval) > 0:
        dval = lgb.Dataset(Xval.values, label=yval.values, reference=dtrain)
    else:
        dval = None
        print(f"  [WARN] Valid set is empty, using train set for validation")
    
    # 回归参数（对齐 QLib）
    params = dict(cfg["model"]["params"])
    params.update({
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
    })
    params.pop("ndcg_eval_at", None)  # 移除排序相关参数
    
    # 确保 num_threads 设置（QLib 使用 20，但根据实际情况调整）
    if "num_threads" not in params:
        params["num_threads"] = 4  # 默认值，可根据 CPU 调整
    
    early_stop = cfg["model"].get("early_stopping_rounds", 100)
    
    print(f"\n[Training LightGBM]")
    print(f"  Params: learning_rate={params.get('learning_rate')}, num_leaves={params.get('num_leaves')}, max_depth={params.get('max_depth')}")
    
    # 训练（使用 valid 做早停）
    if dval is not None:
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=params.get("n_estimators", 2000),
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(early_stop, verbose=False),
                lgb.log_evaluation(100),
            ],
        )
    else:
        # 没有验证集时，只用训练集，不早停
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=params.get("n_estimators", 2000),
            valid_sets=[dtrain],
            valid_names=["train"],
            callbacks=[lgb.log_evaluation(100)],
        )
    
    # 在 test set 上预测和评估（对齐 QLib）
    # 关键修复：确保索引和预测值正确对应
    original_index = Xte.index.copy()
    pred_test = model.predict(Xte.values)
    
    # 验证长度匹配
    if len(pred_test) != len(original_index):
        raise ValueError(
            f"预测值数量 ({len(pred_test)}) 与索引数量 ({len(original_index)}) 不匹配！"
        )
    
    # 计算 Rank IC 和 Rank ICIR（对齐 QLib 基准测试）
    df_test = pd.DataFrame({"pred": pred_test, "y": yte.values}, index=original_index)
    rank_ic, rank_icir = calculate_rank_ic_and_icir(df_test["pred"], df_test["y"])
    
    metrics = {
        "mode": "regression_qlib_segments",
        "segments": {
            "train": {"start": train_start, "end": train_end, "n_samples": len(tr)},
            "valid": {"start": valid_start, "end": valid_end, "n_samples": len(val)},
            "test": {"start": test_start, "end": test_end, "n_samples": len(te)},
        },
        "test_rank_ic": float(rank_ic),
        "test_rank_icir": float(rank_icir),
        "n_features": len(Xtr.columns),
    }
    
    # 保存模型
    model_path = OUTPUT_MODELS_DIR / "lgbm_regression.txt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    
    # 保存 metrics
    metrics_path = Path(cfg["paths"].get("metrics_path", "outputs/reports/metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[OK] LightGBM Regression model trained (QLib Segments Mode)")
    print(f"  • Model: {model_path}")
    print(f"  • Metrics: {metrics_path}")
    print(f"  • Test Rank IC:  {rank_ic:.4f} (目标: 0.0469)")
    print(f"  • Test Rank ICIR: {rank_icir:.4f} (目标: 0.3877)")
    
    return model, metrics

def _train_regression_walk_forward(cfg):
    """
    Walk-Forward CV 模式（原有方式）
    """
    import numpy as np
    import pandas as pd
    import lightgbm as lgb

    n_folds = cfg["model"]["cv"]["n_folds"]
    
    use_qlib_label = cfg.get("model", {}).get("use_qlib_label", False)
    panel, dates = prepare_regression_data(cfg, use_qlib_label=use_qlib_label)
    folds = np.array_split(dates, n_folds)
    
    metrics = {"folds": [], "mode": "regression"}
    oof_pred = pd.Series(index=panel.index, dtype=float)
    
    for i in range(n_folds - 1):
        tr_dates = np.concatenate(folds[: i + 1])
        te_dates = folds[i + 1]
        
        tr = panel.loc[panel.index.get_level_values("date").isin(tr_dates)].copy()
        te = panel.loc[panel.index.get_level_values("date").isin(te_dates)].copy()
        
        tr = tr.dropna(subset=["y"])
        te = te.dropna(subset=["y"])
        
        Xtr, ytr = tr.drop(columns=["y"]), tr["y"]
        Xte, yte = te.drop(columns=["y"]), te["y"]
        
        # 关键修复：使用每日横截面中位数填充（优化版：使用 groupby）
        # 训练集：按日期分组，使用 transform 快速填充
        if isinstance(Xtr.index, pd.MultiIndex):
            # 计算每日横截面中位数
            train_medians_by_date = Xtr.groupby(level=0).transform('median')
            # 用每日中位数填充
            Xtr = Xtr.fillna(train_medians_by_date)
            # 对于某日期某特征全为NaN的情况，使用训练集全局中位数填充
            train_global_medians = Xtr.median()
            Xtr = Xtr.fillna(train_global_medians).fillna(0.0)
            
            # 测试集：使用训练集的全局中位数
            Xte = Xte.fillna(train_global_medians).fillna(0.0)
        else:
            # 如果不是MultiIndex，使用全局中位数
            train_global_medians = Xtr.median()
            Xtr = Xtr.fillna(train_global_medians).fillna(0.0)
            Xte = Xte.fillna(train_global_medians).fillna(0.0)
        
        # 准备 LightGBM 数据
        dtrain = lgb.Dataset(Xtr.values, label=ytr.values)
        dval = lgb.Dataset(Xte.values, label=yte.values, reference=dtrain)
        
        # 回归参数（参考 qlib）
        params = dict(cfg["model"]["params"])
        params.update({
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
        })
        params.pop("ndcg_eval_at", None)  # 移除排序相关参数
        
        early_stop = cfg["model"].get("early_stopping_rounds", 100)
        
        # 训练
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=params.get("n_estimators", 2000),
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(early_stop, verbose=False),
                lgb.log_evaluation(100),
            ],
        )
        
        # 预测
        pred = model.predict(Xte.values)
        oof_pred.loc[Xte.index] = pred
        
        # 计算 Rank IC（用 Spearman 相关性评估，安全处理 NaN）
        dfte = pd.DataFrame({"pred": pred, "y": yte.values}, index=Xte.index)
        ric_mean = safe_rank_ic(dfte["pred"], dfte["y"])
        
        # 调试信息
        pred_stats = f"pred: mean={pred.mean():.6f}, std={pred.std():.6f}, unique={len(np.unique(pred))}"
        y_stats = f"y: mean={yte.values.mean():.6f}, std={yte.values.std():.6f}, unique={len(np.unique(yte.values))}"
        print(f"[Fold {i+1}] RankIC={ric_mean:.4f} | train={len(Xtr)} test={len(Xte)} features={len(Xtr.columns)}")
        if ric_mean == 0.0 or abs(ric_mean) < 0.001:
            print(f"  [DEBUG] {pred_stats}, {y_stats}")
        
        metrics["folds"].append({
            "fold": i + 1,
            "tr_dates": [str(tr_dates.min()), str(tr_dates.max())],
            "te_dates": [str(te_dates.min()), str(te_dates.max())],
            "mean_rank_ic": float(ric_mean),
            "n_train": len(Xtr),
            "n_test": len(Xte),
            "n_features": len(Xtr.columns),
        })
    
    # OOF Rank IC（安全处理 NaN）
    oof = oof_pred.dropna()
    y_all = panel.loc[oof.index, "y"]
    ric_daily = oof.groupby(level=0).apply(
        lambda x: x.rank(method="first").corr(y_all.loc[x.index].rank(method="first"), method="spearman")
        if len(x) >= 2 and x.nunique() > 1 and y_all.loc[x.index].nunique() > 1
        else np.nan
    )
    ric_daily = ric_daily.dropna()
    metrics["oof_mean_rank_ic"] = float(ric_daily.mean()) if len(ric_daily) > 0 else 0.0
    
    # 保存最后一个模型（用于生产）
    if len(metrics["folds"]) > 0:
        # 训练最后一个 fold 的模型用于保存
        last_fold = metrics["folds"][-1]
        tr_dates = np.concatenate(folds[:-1])
        tr = panel.loc[panel.index.get_level_values("date").isin(tr_dates)].copy()
        tr = tr.dropna(subset=["y"])
        Xtr, ytr = tr.drop(columns=["y"]), tr["y"]
        for col in Xtr.columns:
            med = Xtr[col].median()
            Xtr[col] = Xtr[col].fillna(med)
        dtrain = lgb.Dataset(Xtr.values, label=ytr.values)
        params = dict(cfg["model"]["params"])
        params.update({"objective": "regression", "metric": "rmse", "verbosity": -1})
        params.pop("ndcg_eval_at", None)
        final_model = lgb.train(params, dtrain, num_boost_round=params.get("n_estimators", 2000))
        model_path = OUTPUT_MODELS_DIR / "lgbm_regression.txt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        final_model.save_model(str(model_path))
    else:
        model_path = OUTPUT_MODELS_DIR / "lgbm_regression.txt"
    
    metrics_path = Path(cfg["paths"].get("metrics_path", "outputs/reports/metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[OK] LightGBM Regression model trained (Walk-Forward CV)")
    print(f"  • Model: {model_path}")
    print(f"  • Metrics: {metrics_path}")
    print(f"  • OOF RankIC: {metrics['oof_mean_rank_ic']:.4f}")

def train_catboost(cfg):
    """
    使用 CatBoost 回归模型训练，对齐 QLib 基准测试
    支持两种评估模式：walk_forward_cv 或 qlib_segments
    """
    import numpy as np
    import pandas as pd
    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError:
        raise ImportError("CatBoost not installed. Please run: pip install catboost")

    # 数据质量诊断
    use_qlib_label = cfg.get("model", {}).get("use_qlib_label", False)
    diagnose_data_quality(cfg, use_qlib_label=use_qlib_label)
    
    # 获取评估模式
    eval_mode = cfg.get("model", {}).get("evaluation_mode", "walk_forward_cv")
    
    if eval_mode == "qlib_segments":
        return _train_catboost_qlib_segments(cfg)
    else:
        return _train_catboost_walk_forward(cfg)

def _train_catboost_qlib_segments(cfg):
    """CatBoost QLib segments 模式"""
    import numpy as np
    import pandas as pd
    from catboost import CatBoostRegressor
    
    use_qlib_label = cfg.get("model", {}).get("use_qlib_label", False)
    panel, dates = prepare_regression_data(cfg, use_qlib_label=use_qlib_label)
    
    segments = cfg.get("model", {}).get("segments", {})
    # Use dynamic default end date (yesterday) to ensure we get latest data
    from datetime import datetime, timedelta
    default_test_end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    train_start = segments.get("train_start", "2010-01-01")
    train_end = segments.get("train_end", "2018-12-31")
    valid_start = segments.get("valid_start", "2019-01-01")
    valid_end = segments.get("valid_end", "2021-12-31")
    test_start = segments.get("test_start", "2022-01-01")
    test_end = segments.get("test_end", default_test_end)
    
    tr = panel.loc[(panel.index.get_level_values("date") >= train_start) & 
                   (panel.index.get_level_values("date") <= train_end)].copy()
    val = panel.loc[(panel.index.get_level_values("date") >= valid_start) & 
                   (panel.index.get_level_values("date") <= valid_end)].copy()
    te = panel.loc[(panel.index.get_level_values("date") >= test_start) & 
                   (panel.index.get_level_values("date") <= test_end)].copy()
    
    tr = tr.dropna(subset=["y"])
    val = val.dropna(subset=["y"])
    te = te.dropna(subset=["y"])
    
    print(f"\n  Data sizes: train={len(tr)}, valid={len(val)}, test={len(te)}")
    
    if len(tr) == 0:
        print("[ERROR] Train set is empty. Check segments configuration.")
        return
    
    if len(te) == 0:
        print("[WARN] Test set is empty. Using valid set as test set if available.")
        if len(val) > 0:
            te = val.copy()
            val = pd.DataFrame()  # 清空 valid，因为要用作 test
            print(f"  Using valid set as test: {len(te)} samples")
        else:
            print("[ERROR] Both valid and test sets are empty. Cannot proceed.")
            return
    
    Xtr, ytr = tr.drop(columns=["y"]), tr["y"]
    Xval, yval = val.drop(columns=["y"]), val["y"] if len(val) > 0 else pd.Series(dtype=float)
    Xte, yte = te.drop(columns=["y"]), te["y"]
    
    # 关键修复：使用每日横截面中位数填充（优化版：使用 groupby）
    print(f"  [数据预处理] 使用每日横截面中位数填充缺失值（优化版）...")
    
    # 训练集：按日期分组，使用 transform 快速填充
    if isinstance(Xtr.index, pd.MultiIndex):
        # 计算每日横截面中位数
        train_medians_by_date = Xtr.groupby(level=0).transform('median')
        # 用每日中位数填充
        Xtr = Xtr.fillna(train_medians_by_date)
        # 对于某日期某特征全为NaN的情况，使用训练集全局中位数填充
        train_global_medians = Xtr.median()
        Xtr = Xtr.fillna(train_global_medians)
        # 如果全局中位数也是NaN，填充为0
        Xtr = Xtr.fillna(0.0)
    else:
        # 如果不是MultiIndex，使用全局中位数
        train_global_medians = Xtr.median()
        Xtr = Xtr.fillna(train_global_medians).fillna(0.0)
    
    # 验证集和测试集：先使用每日横截面中位数填充，再用训练集全局中位数填充剩余（对齐 QLib）
    # 这样保持特征的区分度，避免所有股票填充后特征相同
    if len(val) > 0:
        if isinstance(Xval.index, pd.MultiIndex):
            val_medians_by_date = Xval.groupby(level=0).transform('median')
            Xval = Xval.fillna(val_medians_by_date)
        Xval = Xval.fillna(train_global_medians).fillna(0.0)
    
    if isinstance(Xte.index, pd.MultiIndex):
        te_medians_by_date = Xte.groupby(level=0).transform('median')
        Xte = Xte.fillna(te_medians_by_date)
    Xte = Xte.fillna(train_global_medians).fillna(0.0)
    
    # 关键修复：保存训练集的全局中位数，供预测时使用
    import json
    model_dir = get_path(cfg["paths"]["model_dir"], OUTPUT_MODELS_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    train_medians_path = model_dir / "train_global_medians.json"
    train_medians_dict = train_global_medians.to_dict()
    with open(train_medians_path, "w") as f:
        json.dump(train_medians_dict, f, indent=2)
    print(f"  [保存] 训练集全局中位数已保存到 {train_medians_path}")
    
    cb_params = cfg["model"].get("catboost_params", {})
    default_params = {
        "loss_function": "RMSE",
        "iterations": 2000,
        "learning_rate": 0.0421,
        "depth": 6,
        "verbose": False,
        "early_stopping_rounds": cfg["model"].get("early_stopping_rounds", 100),
        "random_seed": 42,
    }
    default_params.update(cb_params)
    
    cb_params_final = default_params.copy()
    if cb_params_final.get("bootstrap_type") == "Poisson":
        cb_params_final["bootstrap_type"] = "Bayesian"
    if cb_params_final.get("bootstrap_type") == "Bayesian":
        cb_params_final.pop("subsample", None)
    
    model = CatBoostRegressor(**cb_params_final)
    # 如果验证集为空，不使用 eval_set
    if len(Xval) > 0 and len(yval) > 0:
        model.fit(Xtr.values, ytr.values, eval_set=(Xval.values, yval.values), use_best_model=True, verbose=False)
    else:
        print(f"  [WARN] Valid set is empty, training without early stopping")
        model.fit(Xtr.values, ytr.values, verbose=False)
    
    # 关键修复：确保索引和预测值正确对应
    original_index = Xte.index.copy()
    pred_test = model.predict(Xte.values)
    
    # 验证长度匹配
    if len(pred_test) != len(original_index):
        raise ValueError(
            f"预测值数量 ({len(pred_test)}) 与索引数量 ({len(original_index)}) 不匹配！"
        )
    
    rank_ic, rank_icir = calculate_rank_ic_and_icir(pd.Series(pred_test, index=original_index), yte)
    
    metrics = {
        "mode": "catboost_qlib_segments",
        "test_rank_ic": float(rank_ic),
        "test_rank_icir": float(rank_icir),
    }
    
    model_path = OUTPUT_MODELS_DIR / "catboost_regression.cbm"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    
    metrics_path = Path(cfg["paths"].get("metrics_path", "outputs/reports/metrics.json")).parent / "metrics_catboost.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[OK] CatBoost Regression model trained (QLib Segments Mode)")
    print(f"  • Test Rank IC:  {rank_ic:.4f} (目标: 0.0454)")
    print(f"  • Test Rank ICIR: {rank_icir:.4f} (目标: 0.3311)")
    
    return model, metrics

def _train_catboost_walk_forward(cfg):
    """CatBoost Walk-Forward CV 模式"""
    import numpy as np
    import pandas as pd
    from catboost import CatBoostRegressor

    n_folds = cfg["model"]["cv"]["n_folds"]
    
    use_qlib_label = cfg.get("model", {}).get("use_qlib_label", False)
    panel, dates = prepare_regression_data(cfg, use_qlib_label=use_qlib_label)
    folds = np.array_split(dates, n_folds)
    
    metrics = {"folds": [], "mode": "catboost"}
    oof_pred = pd.Series(index=panel.index, dtype=float)
    
    # CatBoost 参数（参考 qlib）
    cb_params = cfg["model"].get("catboost_params", {})
    default_params = {
        "loss_function": "RMSE",
        "iterations": 2000,
        "learning_rate": 0.0421,
        "depth": 6,
        "verbose": 100,
        "early_stopping_rounds": cfg["model"].get("early_stopping_rounds", 100),
        "random_seed": 42,
    }
    default_params.update(cb_params)
    
    for i in range(n_folds - 1):
        tr_dates = np.concatenate(folds[: i + 1])
        te_dates = folds[i + 1]
        
        tr = panel.loc[panel.index.get_level_values("date").isin(tr_dates)].copy()
        te = panel.loc[panel.index.get_level_values("date").isin(te_dates)].copy()
        
        tr = tr.dropna(subset=["y"])
        te = te.dropna(subset=["y"])
        
        Xtr, ytr = tr.drop(columns=["y"]), tr["y"]
        Xte, yte = te.drop(columns=["y"]), te["y"]
        
        # 关键修复：使用每日横截面中位数填充（优化版：使用 groupby）
        # 训练集：按日期分组，使用 transform 快速填充
        if isinstance(Xtr.index, pd.MultiIndex):
            # 计算每日横截面中位数
            train_medians_by_date = Xtr.groupby(level=0).transform('median')
            # 用每日中位数填充
            Xtr = Xtr.fillna(train_medians_by_date)
            # 对于某日期某特征全为NaN的情况，使用训练集全局中位数填充
            train_global_medians = Xtr.median()
            Xtr = Xtr.fillna(train_global_medians).fillna(0.0)
            
            # 测试集：使用训练集的全局中位数
            Xte = Xte.fillna(train_global_medians).fillna(0.0)
        else:
            # 如果不是MultiIndex，使用全局中位数
            train_global_medians = Xtr.median()
            Xtr = Xtr.fillna(train_global_medians).fillna(0.0)
            Xte = Xte.fillna(train_global_medians).fillna(0.0)
        
        # 训练 CatBoost（移除 CPU 不支持的参数）
        cb_params_final = default_params.copy()
        # 如果是 Poisson，改为 Bayesian（CPU 支持）
        if cb_params_final.get("bootstrap_type") == "Poisson":
            cb_params_final["bootstrap_type"] = "Bayesian"
        # Bayesian bootstrap 不支持 subsample 参数（无论从哪里来的）
        if cb_params_final.get("bootstrap_type") == "Bayesian":
            cb_params_final.pop("subsample", None)
        
        model = CatBoostRegressor(**cb_params_final)
        model.fit(
            Xtr.values, ytr.values,
            eval_set=(Xte.values, yte.values),
            use_best_model=True,
            verbose=100
        )
        
        # 预测
        pred = model.predict(Xte.values)
        oof_pred.loc[Xte.index] = pred
        
        # 计算 Rank IC（安全处理 NaN）
        dfte = pd.DataFrame({"pred": pred, "y": yte.values}, index=Xte.index)
        ric_mean = safe_rank_ic(dfte["pred"], dfte["y"])
        
        print(f"[Fold {i+1}] RankIC={ric_mean:.4f} | train={len(Xtr)} test={len(Xte)} features={len(Xtr.columns)}")
        
        metrics["folds"].append({
            "fold": i + 1,
            "tr_dates": [str(tr_dates.min()), str(tr_dates.max())],
            "te_dates": [str(te_dates.min()), str(te_dates.max())],
            "mean_rank_ic": float(ric_mean),
            "n_train": len(Xtr),
            "n_test": len(Xte),
            "n_features": len(Xtr.columns),
        })
    
    # OOF Rank IC（安全处理 NaN）
    oof = oof_pred.dropna()
    y_all = panel.loc[oof.index, "y"]
    ric_daily = oof.groupby(level=0).apply(
        lambda x: x.rank(method="first").corr(y_all.loc[x.index].rank(method="first"), method="spearman")
        if len(x) >= 2 and x.nunique() > 1 and y_all.loc[x.index].nunique() > 1
        else np.nan
    )
    ric_daily = ric_daily.dropna()
    metrics["oof_mean_rank_ic"] = float(ric_daily.mean()) if len(ric_daily) > 0 else 0.0
    
    # 保存模型和指标
    model_path = Path(cfg["paths"]["model_dir"]) / "catboost_regression.cbm"
    model.save_model(str(model_path))
    
    metrics_path = Path(cfg["paths"].get("metrics_path", "outputs/reports/metrics.json")).parent / "metrics_catboost.json"
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[OK] CatBoost Regression model trained.")
    print(f"  • Model: {model_path}")
    print(f"  • Metrics: {metrics_path}")
    print(f"  • OOF RankIC: {metrics['oof_mean_rank_ic']:.4f}")

def train_xgboost(cfg):
    """
    使用 XGBoost 回归模型训练，对齐 QLib 基准测试
    支持两种评估模式：walk_forward_cv 或 qlib_segments
    """
    import numpy as np
    import pandas as pd
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost not installed. Please run: pip install xgboost")

    # 数据质量诊断
    use_qlib_label = cfg.get("model", {}).get("use_qlib_label", False)
    diagnose_data_quality(cfg, use_qlib_label=use_qlib_label)
    
    # 获取评估模式
    eval_mode = cfg.get("model", {}).get("evaluation_mode", "walk_forward_cv")
    
    if eval_mode == "qlib_segments":
        return _train_xgboost_qlib_segments(cfg)
    else:
        return _train_xgboost_walk_forward(cfg)

def _train_xgboost_qlib_segments(cfg):
    """XGBoost QLib segments 模式"""
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    
    use_qlib_label = cfg.get("model", {}).get("use_qlib_label", False)
    panel, dates = prepare_regression_data(cfg, use_qlib_label=use_qlib_label)
    
    segments = cfg.get("model", {}).get("segments", {})
    # Use dynamic default end date (yesterday) to ensure we get latest data
    from datetime import datetime, timedelta
    default_test_end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    train_start = segments.get("train_start", "2010-01-01")
    train_end = segments.get("train_end", "2018-12-31")
    valid_start = segments.get("valid_start", "2019-01-01")
    valid_end = segments.get("valid_end", "2021-12-31")
    test_start = segments.get("test_start", "2022-01-01")
    test_end = segments.get("test_end", default_test_end)
    
    tr = panel.loc[(panel.index.get_level_values("date") >= train_start) & 
                   (panel.index.get_level_values("date") <= train_end)].copy()
    val = panel.loc[(panel.index.get_level_values("date") >= valid_start) & 
                   (panel.index.get_level_values("date") <= valid_end)].copy()
    te = panel.loc[(panel.index.get_level_values("date") >= test_start) & 
                   (panel.index.get_level_values("date") <= test_end)].copy()
    
    tr = tr.dropna(subset=["y"])
    val = val.dropna(subset=["y"])
    te = te.dropna(subset=["y"])
    
    print(f"\n  Data sizes: train={len(tr)}, valid={len(val)}, test={len(te)}")
    
    if len(tr) == 0:
        print("[ERROR] Train set is empty. Check segments configuration.")
        return
    
    if len(te) == 0:
        print("[WARN] Test set is empty. Using valid set as test set if available.")
        if len(val) > 0:
            te = val.copy()
            val = pd.DataFrame()  # 清空 valid，因为要用作 test
            print(f"  Using valid set as test: {len(te)} samples")
        else:
            print("[ERROR] Both valid and test sets are empty. Cannot proceed.")
            return
    
    Xtr, ytr = tr.drop(columns=["y"]), tr["y"]
    Xval, yval = val.drop(columns=["y"]), val["y"] if len(val) > 0 else pd.Series(dtype=float)
    Xte, yte = te.drop(columns=["y"]), te["y"]
    
    # 关键修复：使用每日横截面中位数填充（优化版：使用 groupby）
    print(f"  [数据预处理] 使用每日横截面中位数填充缺失值（优化版）...")
    
    # 训练集：按日期分组，使用 transform 快速填充
    if isinstance(Xtr.index, pd.MultiIndex):
        # 计算每日横截面中位数
        train_medians_by_date = Xtr.groupby(level=0).transform('median')
        # 用每日中位数填充
        Xtr = Xtr.fillna(train_medians_by_date)
        # 对于某日期某特征全为NaN的情况，使用训练集全局中位数填充
        train_global_medians = Xtr.median()
        Xtr = Xtr.fillna(train_global_medians)
        # 如果全局中位数也是NaN，填充为0
        Xtr = Xtr.fillna(0.0)
    else:
        # 如果不是MultiIndex，使用全局中位数
        train_global_medians = Xtr.median()
        Xtr = Xtr.fillna(train_global_medians).fillna(0.0)
    
    # 验证集和测试集：先使用每日横截面中位数填充，再用训练集全局中位数填充剩余（对齐 QLib）
    # 这样保持特征的区分度，避免所有股票填充后特征相同
    if len(val) > 0:
        if isinstance(Xval.index, pd.MultiIndex):
            val_medians_by_date = Xval.groupby(level=0).transform('median')
            Xval = Xval.fillna(val_medians_by_date)
        Xval = Xval.fillna(train_global_medians).fillna(0.0)
    
    if isinstance(Xte.index, pd.MultiIndex):
        te_medians_by_date = Xte.groupby(level=0).transform('median')
        Xte = Xte.fillna(te_medians_by_date)
    Xte = Xte.fillna(train_global_medians).fillna(0.0)
    
    # 关键修复：保存训练集的全局中位数，供预测时使用
    import json
    model_dir = get_path(cfg["paths"]["model_dir"], OUTPUT_MODELS_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    train_medians_path = model_dir / "train_global_medians.json"
    train_medians_dict = train_global_medians.to_dict()
    with open(train_medians_path, "w") as f:
        json.dump(train_medians_dict, f, indent=2)
    print(f"  [保存] 训练集全局中位数已保存到 {train_medians_path}")
    
    xgb_params = cfg["model"].get("xgboost_params", {})
    default_params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 2000,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "verbosity": 0,
    }
    default_params.update(xgb_params)
    
    # XGBoost early stopping - 对于旧版本，使用简化的早停策略
    early_stop = cfg["model"].get("early_stopping_rounds", 100)
    n_est = default_params.pop("n_estimators", 2000)  # 移除，后续手动设置
    
    print(f"\n[Training XGBoost]")
    print(f"  Params: learning_rate={default_params.get('learning_rate')}, max_depth={default_params.get('max_depth')}, n_estimators={n_est}")
    
    if len(Xval) > 0 and len(yval) > 0:
        # 使用分阶段训练和手动早停（兼容旧版本 XGBoost）
        print(f"  Using manual early stopping (checking every 50 iters, patience={early_stop})")
        
        best_score = float('inf')
        best_iter = 50  # 至少训练 50 轮
        no_improve = 0
        check_interval = 50
        
        # 从 50 轮开始，每 50 轮检查一次
        for check_iter in range(50, n_est + 1, check_interval):
            # 训练到当前检查点（复制参数并设置 n_estimators）
            model_params = default_params.copy()
            model_params["n_estimators"] = check_iter
            model = xgb.XGBRegressor(**model_params)
            model.fit(Xtr.values, ytr.values, verbose=False)
            
            # 评估验证集
            pred_val = model.predict(Xval.values)
            val_score = np.sqrt(np.mean((pred_val - yval.values) ** 2))  # RMSE
            
            print(f"    Iter {check_iter}: val_rmse={val_score:.6f}", end="")
            
            if val_score < best_score:
                best_score = val_score
                best_iter = check_iter
                no_improve = 0
                print(f" ✓ (new best)")
            else:
                no_improve += check_interval
                print(f" (best={best_score:.6f} @ iter {best_iter})")
            
            if no_improve >= early_stop:
                print(f"  Early stopping: best val_rmse={best_score:.6f} @ iter {best_iter}")
                # 重建最佳模型
                model_params = default_params.copy()
                model_params["n_estimators"] = best_iter
                model = xgb.XGBRegressor(**model_params)
                model.fit(Xtr.values, ytr.values, verbose=False)
                break
        
        # 如果没有早停，使用最终检查点的模型（已经是最后训练的模型）
        if no_improve < early_stop:
            print(f"  Training completed: {check_iter} iterations, best val_rmse={best_score:.6f} @ iter {best_iter}")
            if best_iter < check_iter:
                # 使用最佳模型
                model_params = default_params.copy()
                model_params["n_estimators"] = best_iter
                model = xgb.XGBRegressor(**model_params)
                model.fit(Xtr.values, ytr.values, verbose=False)
    else:
        print(f"  [WARN] Valid set is empty, training without early stopping")
        model_params = default_params.copy()
        model_params["n_estimators"] = n_est
        model = xgb.XGBRegressor(**model_params)
        model.fit(Xtr.values, ytr.values, verbose=False)
    
    # 关键修复：确保索引和预测值正确对应
    original_index = Xte.index.copy()
    pred_test = model.predict(Xte.values)
    
    # 验证长度匹配
    if len(pred_test) != len(original_index):
        raise ValueError(
            f"预测值数量 ({len(pred_test)}) 与索引数量 ({len(original_index)}) 不匹配！"
        )
    
    rank_ic, rank_icir = calculate_rank_ic_and_icir(pd.Series(pred_test, index=original_index), yte)
    
    metrics = {
        "mode": "xgboost_qlib_segments",
        "test_rank_ic": float(rank_ic),
        "test_rank_icir": float(rank_icir),
    }
    
    model_path = OUTPUT_MODELS_DIR / "xgboost_regression.model"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    
    metrics_path = Path(cfg["paths"].get("metrics_path", "outputs/reports/metrics.json")).parent / "metrics_xgboost.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[OK] XGBoost Regression model trained (QLib Segments Mode)")
    print(f"  • Test Rank IC:  {rank_ic:.4f} (目标: 0.0505)")
    print(f"  • Test Rank ICIR: {rank_icir:.4f} (目标: 0.4131)")
    
    return model, metrics

def _train_xgboost_walk_forward(cfg):
    """XGBoost Walk-Forward CV 模式"""
    import numpy as np
    import pandas as pd
    import xgboost as xgb

    n_folds = cfg["model"]["cv"]["n_folds"]
    
    use_qlib_label = cfg.get("model", {}).get("use_qlib_label", False)
    panel, dates = prepare_regression_data(cfg, use_qlib_label=use_qlib_label)
    folds = np.array_split(dates, n_folds)
    
    metrics = {"folds": [], "mode": "xgboost"}
    oof_pred = pd.Series(index=panel.index, dtype=float)
    
    # XGBoost 参数（参考 qlib）
    xgb_params = cfg["model"].get("xgboost_params", {})
    default_params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 2000,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "verbosity": 0,
    }
    default_params.update(xgb_params)
    
    early_stop = cfg["model"].get("early_stopping_rounds", 100)
    
    for i in range(n_folds - 1):
        tr_dates = np.concatenate(folds[: i + 1])
        te_dates = folds[i + 1]
        
        tr = panel.loc[panel.index.get_level_values("date").isin(tr_dates)].copy()
        te = panel.loc[panel.index.get_level_values("date").isin(te_dates)].copy()
        
        tr = tr.dropna(subset=["y"])
        te = te.dropna(subset=["y"])
        
        Xtr, ytr = tr.drop(columns=["y"]), tr["y"]
        Xte, yte = te.drop(columns=["y"]), te["y"]
        
        # 关键修复：使用每日横截面中位数填充（优化版：使用 groupby）
        # 训练集：按日期分组，使用 transform 快速填充
        if isinstance(Xtr.index, pd.MultiIndex):
            # 计算每日横截面中位数
            train_medians_by_date = Xtr.groupby(level=0).transform('median')
            # 用每日中位数填充
            Xtr = Xtr.fillna(train_medians_by_date)
            # 对于某日期某特征全为NaN的情况，使用训练集全局中位数填充
            train_global_medians = Xtr.median()
            Xtr = Xtr.fillna(train_global_medians).fillna(0.0)
            
            # 测试集：使用训练集的全局中位数
            Xte = Xte.fillna(train_global_medians).fillna(0.0)
        else:
            # 如果不是MultiIndex，使用全局中位数
            train_global_medians = Xtr.median()
            Xtr = Xtr.fillna(train_global_medians).fillna(0.0)
            Xte = Xte.fillna(train_global_medians).fillna(0.0)
        
        # 准备 XGBoost 数据
        dtrain = xgb.DMatrix(Xtr.values, label=ytr.values)
        dval = xgb.DMatrix(Xte.values, label=yte.values)
        
        # 训练
        model = xgb.train(
            default_params,
            dtrain,
            num_boost_round=default_params.get("n_estimators", 2000),
            evals=[(dtrain, "train"), (dval, "valid")],
            early_stopping_rounds=early_stop,
            verbose_eval=100,
        )
        
        # 预测
        pred = model.predict(dval)
        oof_pred.loc[Xte.index] = pred
        
        # 计算 Rank IC（安全处理 NaN）
        dfte = pd.DataFrame({"pred": pred, "y": yte.values}, index=Xte.index)
        ric_mean = safe_rank_ic(dfte["pred"], dfte["y"])
        
        print(f"[Fold {i+1}] RankIC={ric_mean:.4f} | train={len(Xtr)} test={len(Xte)} features={len(Xtr.columns)}")
        
        metrics["folds"].append({
            "fold": i + 1,
            "tr_dates": [str(tr_dates.min()), str(tr_dates.max())],
            "te_dates": [str(te_dates.min()), str(te_dates.max())],
            "mean_rank_ic": float(ric_mean),
            "n_train": len(Xtr),
            "n_test": len(Xte),
            "n_features": len(Xtr.columns),
        })
    
    # OOF Rank IC（安全处理 NaN）
    oof = oof_pred.dropna()
    y_all = panel.loc[oof.index, "y"]
    ric_daily = oof.groupby(level=0).apply(
        lambda x: x.rank(method="first").corr(y_all.loc[x.index].rank(method="first"), method="spearman")
        if len(x) >= 2 and x.nunique() > 1 and y_all.loc[x.index].nunique() > 1
        else np.nan
    )
    ric_daily = ric_daily.dropna()
    metrics["oof_mean_rank_ic"] = float(ric_daily.mean()) if len(ric_daily) > 0 else 0.0
    
    # 保存模型和指标
    model_path = Path(cfg["paths"]["model_dir"]) / "xgboost_regression.model"
    model.save_model(str(model_path))
    
    metrics_path = Path(cfg["paths"].get("metrics_path", "outputs/reports/metrics.json")).parent / "metrics_xgboost.json"
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[OK] XGBoost Regression model trained.")
    print(f"  • Model: {model_path}")
    print(f"  • Metrics: {metrics_path}")
    print(f"  • OOF RankIC: {metrics['oof_mean_rank_ic']:.4f}")

def train_ensemble(cfg):
    """
    模型集成：LightGBM + CatBoost + XGBoost + Transformer + GRU + LSTM 的加权平均
    使用等权重或基于验证集表现的权重（自适应权重）
    """
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    try:
        from catboost import CatBoostRegressor
        import xgboost as xgb
        HAS_CATBOOST = True
        HAS_XGBOOST = True
    except ImportError:
        print("[WARN] CatBoost or XGBoost not available. Using only LightGBM.")
        HAS_CATBOOST = False
        HAS_XGBOOST = False
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        HAS_TORCH = True
    except ImportError:
        print("[WARN] PyTorch not available. Deep learning models will be skipped.")
        HAS_TORCH = False

    n_folds = cfg["model"]["cv"]["n_folds"]
    
    # 使用共享的数据准备函数
    use_qlib_label = cfg.get("model", {}).get("use_qlib_label", False)
    panel, dates = prepare_regression_data(cfg, use_qlib_label=use_qlib_label)
    folds = np.array_split(dates, n_folds)
    
    metrics = {"folds": [], "mode": "ensemble_all"}
    oof_pred = pd.Series(index=panel.index, dtype=float)
    
    # 模型权重配置（包含所有模型）
    ensemble_weights = cfg["model"].get("ensemble_weights", {
        "lightgbm": 0.10,
        "catboost": 0.15,
        "xgboost": 0.15,
        "transformer": 0.20,
        "gru": 0.20,
        "lstm": 0.20
    })
    
    adaptive_weights = cfg["model"].get("adaptive_weights", True)
    min_weight = cfg["model"].get("min_weight", 0.05)
    
    for i in range(n_folds - 1):
        tr_dates = np.concatenate(folds[: i + 1])
        te_dates = folds[i + 1]
        
        tr = panel.loc[panel.index.get_level_values("date").isin(tr_dates)].copy()
        te = panel.loc[panel.index.get_level_values("date").isin(te_dates)].copy()
        
        tr = tr.dropna(subset=["y"])
        te = te.dropna(subset=["y"])
        
        Xtr, ytr = tr.drop(columns=["y"]), tr["y"]
        Xte, yte = te.drop(columns=["y"]), te["y"]
        
        # 关键修复：使用每日横截面中位数填充（优化版：使用 groupby）
        # 训练集：按日期分组，使用 transform 快速填充
        if isinstance(Xtr.index, pd.MultiIndex):
            # 计算每日横截面中位数
            train_medians_by_date = Xtr.groupby(level=0).transform('median')
            # 用每日中位数填充
            Xtr = Xtr.fillna(train_medians_by_date)
            # 对于某日期某特征全为NaN的情况，使用训练集全局中位数填充
            train_global_medians = Xtr.median()
            Xtr = Xtr.fillna(train_global_medians).fillna(0.0)
            
            # 测试集：使用训练集的全局中位数
            Xte = Xte.fillna(train_global_medians).fillna(0.0)
        else:
            # 如果不是MultiIndex，使用全局中位数
            train_global_medians = Xtr.median()
            Xtr = Xtr.fillna(train_global_medians).fillna(0.0)
            Xte = Xte.fillna(train_global_medians).fillna(0.0)
        
        predictions = []
        model_performances = {}
        
        # 1. LightGBM
        dtrain = lgb.Dataset(Xtr.values, label=ytr.values)
        dval = lgb.Dataset(Xte.values, label=yte.values, reference=dtrain)
        
        lgb_params = dict(cfg["model"]["params"])
        lgb_params.update({
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
        })
        lgb_params.pop("ndcg_eval_at", None)
        
        lgb_model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=lgb_params.get("n_estimators", 2000),
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(cfg["model"].get("early_stopping_rounds", 100)),
                lgb.log_evaluation(100),
            ],
        )
        pred_lgb = lgb_model.predict(Xte.values)
        predictions.append(pred_lgb)
        
        # 计算单个模型的 Rank IC（用于权重调整，安全处理 NaN）
        dfte_lgb = pd.DataFrame({"pred": pred_lgb, "y": yte.values}, index=Xte.index)
        ric_lgb = safe_rank_ic(dfte_lgb["pred"], dfte_lgb["y"])
        model_performances["lightgbm"] = abs(ric_lgb) if not np.isnan(ric_lgb) else 0.0
        
        # 2. CatBoost
        if HAS_CATBOOST:
            cb_params = cfg["model"].get("catboost_params", {})
            cb_default = {
                "loss_function": "RMSE",
                "iterations": 2000,
                "learning_rate": 0.0421,
                "depth": 6,
                "verbose": 100,
                "early_stopping_rounds": cfg["model"].get("early_stopping_rounds", 100),
                "random_seed": 42,
            }
            cb_default.update(cb_params)
            
            # 移除 CPU 不支持的参数
            cb_default_final = cb_default.copy()
            # 如果是 Poisson，改为 Bayesian（CPU 支持）
            if cb_default_final.get("bootstrap_type") == "Poisson":
                cb_default_final["bootstrap_type"] = "Bayesian"
            # Bayesian bootstrap 不支持 subsample 参数（无论从哪里来的）
            if cb_default_final.get("bootstrap_type") == "Bayesian":
                cb_default_final.pop("subsample", None)
            
            cb_model = CatBoostRegressor(**cb_default_final)
            cb_model.fit(
                Xtr.values, ytr.values,
                eval_set=(Xte.values, yte.values),
                use_best_model=True,
                verbose=False
            )
            pred_cb = cb_model.predict(Xte.values)
            predictions.append(pred_cb)
            
            dfte_cb = pd.DataFrame({"pred": pred_cb, "y": yte.values}, index=Xte.index)
            ric_cb = safe_rank_ic(dfte_cb["pred"], dfte_cb["y"])
            model_performances["catboost"] = abs(ric_cb) if not np.isnan(ric_cb) else 0.0
        
        # 3. XGBoost
        if HAS_XGBOOST:
            xgb_params = cfg["model"].get("xgboost_params", {})
            xgb_default = {
                "objective": "reg:squarederror",
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 2000,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": 42,
                "verbosity": 0,
            }
            xgb_default.update(xgb_params)
            
            dtrain_xgb = xgb.DMatrix(Xtr.values, label=ytr.values)
            dval_xgb = xgb.DMatrix(Xte.values, label=yte.values)
            
            xgb_model = xgb.train(
                xgb_default,
                dtrain_xgb,
                num_boost_round=xgb_default.get("n_estimators", 2000),
                evals=[(dtrain_xgb, "train"), (dval_xgb, "valid")],
                early_stopping_rounds=cfg["model"].get("early_stopping_rounds", 100),
                verbose_eval=False,
            )
            pred_xgb = xgb_model.predict(dval_xgb)
            predictions.append(pred_xgb)
            
            dfte_xgb = pd.DataFrame({"pred": pred_xgb, "y": yte.values}, index=Xte.index)
            ric_xgb = safe_rank_ic(dfte_xgb["pred"], dfte_xgb["y"])
            model_performances["xgboost"] = abs(ric_xgb) if not np.isnan(ric_xgb) else 0.0
        
        # 4. Transformer
        if HAS_TORCH:
            try:
                from torch.utils.data import DataLoader, TensorDataset
                device = torch.device(f"cuda:{cfg['model']['transformer_params']['GPU']}" 
                                    if torch.cuda.is_available() and cfg['model']['transformer_params']['GPU'] >= 0 
                                    else "cpu")
                
                class SimpleTransformer(nn.Module):
                    def __init__(self, d_feat, d_model=64, nhead=2, num_layers=2, dropout=0.0):
                        super().__init__()
                        self.input_layer = nn.Linear(d_feat, d_model)
                        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
                        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                        self.output_layer = nn.Linear(d_model, 1)
                    def forward(self, x):
                        x = self.input_layer(x)
                        x = x.unsqueeze(1)
                        x = self.transformer(x)
                        x = x.squeeze(1)
                        return self.output_layer(x).squeeze()
                
                tf_params = cfg["model"]["transformer_params"]
                d_feat = len(Xtr.columns)
                
                # 标准化输入特征（深度模型需要）
                from sklearn.preprocessing import StandardScaler
                scaler_tf = StandardScaler()
                Xtr_scaled_tf = scaler_tf.fit_transform(Xtr.values)
                Xte_scaled_tf = scaler_tf.transform(Xte.values)
                
                model_tf = SimpleTransformer(d_feat, tf_params["d_model"], tf_params["nhead"], tf_params["num_layers"], tf_params["dropout"]).to(device)
                optimizer_tf = optim.Adam(model_tf.parameters(), lr=tf_params["lr"], weight_decay=tf_params["reg"])
                criterion_tf = nn.MSELoss()
                
                # 使用 DataLoader 批训练（避免内存溢出）
                batch_size_tf = tf_params.get("batch_size", 2048)
                train_dataset_tf = TensorDataset(torch.FloatTensor(Xtr_scaled_tf), torch.FloatTensor(ytr.values))
                train_loader_tf = DataLoader(train_dataset_tf, batch_size=batch_size_tf, shuffle=True)
                
                Xte_tf = torch.FloatTensor(Xte_scaled_tf).to(device)
                yte_tf = torch.FloatTensor(yte.values).to(device)
                
                best_val_loss_tf = float('inf')
                patience_tf = 0
                best_model_state_tf = None
                
                for epoch in range(tf_params["n_epochs"]):
                    # 批训练
                    model_tf.train()
                    epoch_loss_tf = 0.0
                    n_batches_tf = 0
                    for batch_X, batch_y in train_loader_tf:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        
                        optimizer_tf.zero_grad()
                        pred_tr_tf = model_tf(batch_X)
                        loss_tf = criterion_tf(pred_tr_tf, batch_y)
                        loss_tf.backward()
                        optimizer_tf.step()
                        
                        epoch_loss_tf += loss_tf.item()
                        n_batches_tf += 1
                    
                    # 验证（每20个epoch）
                    if (epoch + 1) % 20 == 0:
                        model_tf.eval()
                        with torch.no_grad():
                            if len(Xte_scaled_tf) > batch_size_tf * 10:
                                pred_val_list = []
                                for i in range(0, len(Xte_scaled_tf), batch_size_tf * 10):
                                    end_idx = min(i + batch_size_tf * 10, len(Xte_scaled_tf))
                                    pred_val_list.append(model_tf(Xte_tf[i:end_idx]))
                                pred_val_tf = torch.cat(pred_val_list)
                            else:
                                pred_val_tf = model_tf(Xte_tf)
                            val_loss_tf = criterion_tf(pred_val_tf, yte_tf)
                            
                        if val_loss_tf.item() < best_val_loss_tf:
                            best_val_loss_tf = val_loss_tf.item()
                            patience_tf = 0
                            best_model_state_tf = collections.OrderedDict([(k, v.clone()) for k, v in model_tf.state_dict().items()])
                        else:
                            patience_tf += 1
                        if patience_tf >= tf_params["early_stop"]:
                            if best_model_state_tf is not None:
                                model_tf.load_state_dict(best_model_state_tf)
                            break
                
                # 恢复最佳模型
                if best_model_state_tf is not None:
                    model_tf.load_state_dict(best_model_state_tf)
                
                # 预测（使用批处理）
                model_tf.eval()
                with torch.no_grad():
                    if len(Xte_scaled_tf) > batch_size_tf * 10:
                        pred_list = []
                        for i in range(0, len(Xte_scaled_tf), batch_size_tf * 10):
                            end_idx = min(i + batch_size_tf * 10, len(Xte_scaled_tf))
                            pred_list.append(model_tf(Xte_tf[i:end_idx]))
                        pred_tf = torch.cat(pred_list).cpu().numpy()
                    else:
                        pred_tf = model_tf(Xte_tf).cpu().numpy()
                
                predictions.append(pred_tf)
                dfte_tf = pd.DataFrame({"pred": pred_tf, "y": yte.values}, index=Xte.index)
                ric_tf = safe_rank_ic(dfte_tf["pred"], dfte_tf["y"])
                model_performances["transformer"] = abs(ric_tf) if not np.isnan(ric_tf) else 0.0
            except Exception as e:
                print(f"[WARN] Transformer training failed: {e}")
        
        # 5. GRU
        if HAS_TORCH:
            try:
                class SimpleGRU(nn.Module):
                    def __init__(self, d_feat, hidden_size=64, num_layers=2, dropout=0.0):
                        super().__init__()
                        self.gru = nn.GRU(d_feat, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
                        self.fc = nn.Linear(hidden_size, 1)
                    def forward(self, x):
                        x = x.unsqueeze(1)
                        out, _ = self.gru(x)
                        return self.fc(out[:, -1, :]).squeeze()
                
                gru_params = cfg["model"]["gru_params"]
                d_feat = len(Xtr.columns)
                
                # 标准化输入特征（深度模型需要）
                from sklearn.preprocessing import StandardScaler
                scaler_gru = StandardScaler()
                Xtr_scaled_gru = scaler_gru.fit_transform(Xtr.values)
                Xte_scaled_gru = scaler_gru.transform(Xte.values)
                
                model_gru = SimpleGRU(d_feat, gru_params["hidden_size"], gru_params["num_layers"], gru_params["dropout"]).to(device)
                optimizer_gru = optim.Adam(model_gru.parameters(), lr=gru_params["lr"])
                criterion_gru = nn.MSELoss()
                
                # 使用 DataLoader 批训练（避免内存溢出）
                batch_size_gru = gru_params.get("batch_size", 800)
                train_dataset_gru = TensorDataset(torch.FloatTensor(Xtr_scaled_gru), torch.FloatTensor(ytr.values))
                train_loader_gru = DataLoader(train_dataset_gru, batch_size=batch_size_gru, shuffle=True)
                
                Xte_gru = torch.FloatTensor(Xte_scaled_gru).to(device)
                yte_gru = torch.FloatTensor(yte.values).to(device)
                
                best_val_loss_gru = float('inf')
                patience_gru = 0
                best_model_state_gru = None
                
                for epoch in range(gru_params["n_epochs"]):
                    # 批训练
                    model_gru.train()
                    epoch_loss_gru = 0.0
                    n_batches_gru = 0
                    for batch_X, batch_y in train_loader_gru:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        
                        optimizer_gru.zero_grad()
                        pred_tr_gru = model_gru(batch_X)
                        loss_gru = criterion_gru(pred_tr_gru, batch_y)
                        loss_gru.backward()
                        torch.nn.utils.clip_grad_value_(model_gru.parameters(), 3.0)
                        optimizer_gru.step()
                        
                        epoch_loss_gru += loss_gru.item()
                        n_batches_gru += 1
                    
                    # 验证（每20个epoch）
                    if (epoch + 1) % 20 == 0:
                        model_gru.eval()
                        with torch.no_grad():
                            if len(Xte_scaled_gru) > batch_size_gru * 10:
                                pred_val_list = []
                                for i in range(0, len(Xte_scaled_gru), batch_size_gru * 10):
                                    end_idx = min(i + batch_size_gru * 10, len(Xte_scaled_gru))
                                    pred_val_list.append(model_gru(Xte_gru[i:end_idx]))
                                pred_val_gru = torch.cat(pred_val_list)
                            else:
                                pred_val_gru = model_gru(Xte_gru)
                            val_loss_gru = criterion_gru(pred_val_gru, yte_gru)
                            
                        if val_loss_gru.item() < best_val_loss_gru:
                            best_val_loss_gru = val_loss_gru.item()
                            patience_gru = 0
                            best_model_state_gru = collections.OrderedDict([(k, v.clone()) for k, v in model_gru.state_dict().items()])
                        else:
                            patience_gru += 1
                        if patience_gru >= gru_params["early_stop"]:
                            if best_model_state_gru is not None:
                                model_gru.load_state_dict(best_model_state_gru)
                            break
                
                # 恢复最佳模型
                if best_model_state_gru is not None:
                    model_gru.load_state_dict(best_model_state_gru)
                
                # 预测（使用批处理）
                model_gru.eval()
                with torch.no_grad():
                    if len(Xte_scaled_gru) > batch_size_gru * 10:
                        pred_list = []
                        for i in range(0, len(Xte_scaled_gru), batch_size_gru * 10):
                            end_idx = min(i + batch_size_gru * 10, len(Xte_scaled_gru))
                            pred_list.append(model_gru(Xte_gru[i:end_idx]))
                        pred_gru = torch.cat(pred_list).cpu().numpy()
                    else:
                        pred_gru = model_gru(Xte_gru).cpu().numpy()
                
                predictions.append(pred_gru)
                dfte_gru = pd.DataFrame({"pred": pred_gru, "y": yte.values}, index=Xte.index)
                ric_gru = safe_rank_ic(dfte_gru["pred"], dfte_gru["y"])
                model_performances["gru"] = abs(ric_gru) if not np.isnan(ric_gru) else 0.0
            except Exception as e:
                print(f"[WARN] GRU training failed: {e}")
        
        # 6. LSTM
        if HAS_TORCH:
            try:
                class SimpleLSTM(nn.Module):
                    def __init__(self, d_feat, hidden_size=64, num_layers=2, dropout=0.0):
                        super().__init__()
                        self.lstm = nn.LSTM(d_feat, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
                        self.fc = nn.Linear(hidden_size, 1)
                    def forward(self, x):
                        x = x.unsqueeze(1)
                        out, _ = self.lstm(x)
                        return self.fc(out[:, -1, :]).squeeze()
                
                lstm_params = cfg["model"]["lstm_params"]
                d_feat = len(Xtr.columns)
                
                # 标准化输入特征（深度模型需要）
                from sklearn.preprocessing import StandardScaler
                scaler_lstm = StandardScaler()
                Xtr_scaled_lstm = scaler_lstm.fit_transform(Xtr.values)
                Xte_scaled_lstm = scaler_lstm.transform(Xte.values)
                
                model_lstm = SimpleLSTM(d_feat, lstm_params["hidden_size"], lstm_params["num_layers"], lstm_params["dropout"]).to(device)
                optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=lstm_params["lr"])
                criterion_lstm = nn.MSELoss()
                
                # 使用 DataLoader 批训练（避免内存溢出）
                batch_size_lstm = lstm_params.get("batch_size", 800)
                train_dataset_lstm = TensorDataset(torch.FloatTensor(Xtr_scaled_lstm), torch.FloatTensor(ytr.values))
                train_loader_lstm = DataLoader(train_dataset_lstm, batch_size=batch_size_lstm, shuffle=True)
                
                Xte_lstm = torch.FloatTensor(Xte_scaled_lstm).to(device)
                yte_lstm = torch.FloatTensor(yte.values).to(device)
                
                best_val_loss_lstm = float('inf')
                patience_lstm = 0
                best_model_state_lstm = None
                
                for epoch in range(lstm_params["n_epochs"]):
                    # 批训练
                    model_lstm.train()
                    epoch_loss_lstm = 0.0
                    n_batches_lstm = 0
                    for batch_X, batch_y in train_loader_lstm:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        
                        optimizer_lstm.zero_grad()
                        pred_tr_lstm = model_lstm(batch_X)
                        loss_lstm = criterion_lstm(pred_tr_lstm, batch_y)
                        loss_lstm.backward()
                        torch.nn.utils.clip_grad_value_(model_lstm.parameters(), 3.0)
                        optimizer_lstm.step()
                        
                        epoch_loss_lstm += loss_lstm.item()
                        n_batches_lstm += 1
                    
                    # 验证（每20个epoch）
                    if (epoch + 1) % 20 == 0:
                        model_lstm.eval()
                        with torch.no_grad():
                            if len(Xte_scaled_lstm) > batch_size_lstm * 10:
                                pred_val_list = []
                                for i in range(0, len(Xte_scaled_lstm), batch_size_lstm * 10):
                                    end_idx = min(i + batch_size_lstm * 10, len(Xte_scaled_lstm))
                                    pred_val_list.append(model_lstm(Xte_lstm[i:end_idx]))
                                pred_val_lstm = torch.cat(pred_val_list)
                            else:
                                pred_val_lstm = model_lstm(Xte_lstm)
                            val_loss_lstm = criterion_lstm(pred_val_lstm, yte_lstm)
                            
                        if val_loss_lstm.item() < best_val_loss_lstm:
                            best_val_loss_lstm = val_loss_lstm.item()
                            patience_lstm = 0
                            best_model_state_lstm = collections.OrderedDict([(k, v.clone()) for k, v in model_lstm.state_dict().items()])
                        else:
                            patience_lstm += 1
                        if patience_lstm >= lstm_params["early_stop"]:
                            if best_model_state_lstm is not None:
                                model_lstm.load_state_dict(best_model_state_lstm)
                            break
                
                # 恢复最佳模型
                if best_model_state_lstm is not None:
                    model_lstm.load_state_dict(best_model_state_lstm)
                
                # 预测（使用批处理）
                model_lstm.eval()
                with torch.no_grad():
                    if len(Xte_scaled_lstm) > batch_size_lstm * 10:
                        pred_list = []
                        for i in range(0, len(Xte_scaled_lstm), batch_size_lstm * 10):
                            end_idx = min(i + batch_size_lstm * 10, len(Xte_scaled_lstm))
                            pred_list.append(model_lstm(Xte_lstm[i:end_idx]))
                        pred_lstm = torch.cat(pred_list).cpu().numpy()
                    else:
                        pred_lstm = model_lstm(Xte_lstm).cpu().numpy()
                
                predictions.append(pred_lstm)
                dfte_lstm = pd.DataFrame({"pred": pred_lstm, "y": yte.values}, index=Xte.index)
                ric_lstm = safe_rank_ic(dfte_lstm["pred"], dfte_lstm["y"])
                model_performances["lstm"] = abs(ric_lstm) if not np.isnan(ric_lstm) else 0.0
            except Exception as e:
                print(f"[WARN] LSTM training failed: {e}")
        
        # 1. 投票机制（方向投票）：先统计每个模型预测的正负方向
        fusion_method = cfg["model"].get("fusion_method", "weighted_average")
        
        if fusion_method == "voting_weighted":
            # 方向投票 + 加权平均
            # 统计每个模型预测的正负方向
            pred_signs = np.array([np.sign(pred) for pred in predictions])  # [n_models, n_samples]
            votes = np.sum(pred_signs, axis=0)  # 每个样本的投票结果（正数 = 多数为正，负数 = 多数为负）
            # 多数投票的方向（-1, 0, 1），平票时使用加权平均的符号
            majority_direction = np.sign(votes)
            # 如果平票（votes == 0），使用加权平均预测的符号
            if np.any(votes == 0):
                # 先计算临时加权平均以获取方向
                temp_weights = np.array([ensemble_weights.get(name, 1.0/len(predictions)) 
                                       for name in ["lightgbm", "catboost", "xgboost", "transformer", "gru", "lstm"][:len(predictions)]])
                temp_weights = temp_weights / temp_weights.sum()
                temp_avg = np.average(predictions, axis=0, weights=temp_weights)
                majority_direction[votes == 0] = np.sign(temp_avg[votes == 0])
            
            # 计算加权平均的幅度
            model_names = []
            if "lightgbm" in model_performances:
                model_names.append("lightgbm")
            if "catboost" in model_performances:
                model_names.append("catboost")
            if "xgboost" in model_performances:
                model_names.append("xgboost")
            if "transformer" in model_performances:
                model_names.append("transformer")
            if "gru" in model_performances:
                model_names.append("gru")
            if "lstm" in model_performances:
                model_names.append("lstm")
            
            # 计算权重
            if adaptive_weights:
                perfs = [model_performances.get(name, 0.0) for name in model_names]
                total_perf = sum(perfs)
                if total_perf < 1e-6:
                    weights = [ensemble_weights.get(name, 1.0/len(predictions)) for name in model_names]
                else:
                    max_perf = max(perfs)
                    if max_perf > 0:
                        relative_perfs = [p / max_perf for p in perfs]
                        alpha = 0.5
                        scaled_perfs = [p ** alpha for p in relative_perfs]
                        total_scaled = sum(scaled_perfs)
                        weights = [p / total_scaled for p in scaled_perfs]
                        weights = [max(w, min_weight) for w in weights]
                        total_w = sum(weights)
                        weights = [w / total_w for w in weights]
                    else:
                        weights = [ensemble_weights.get(name, 1.0/len(predictions)) for name in model_names]
            else:
                weights = [ensemble_weights.get(name, 1.0/len(predictions)) for name in model_names]
            
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # 计算加权平均的幅度
            pred_magnitude = np.average(predictions, axis=0, weights=weights)
            
            # 应用投票方向：如果多数模型预测为正，则保持正号；否则应用负号
            pred_ensemble = np.abs(pred_magnitude) * majority_direction
            
            print(f"[Voting + Weighted] Majority direction agreement: {np.sum(majority_direction != 0) / len(majority_direction):.2%}, "
                  f"Weights={dict(zip(model_names, weights))}")
        else:
            # 2. 纯加权平均（原有逻辑）
            model_names = []
            if "lightgbm" in model_performances:
                model_names.append("lightgbm")
            if "catboost" in model_performances:
                model_names.append("catboost")
            if "xgboost" in model_performances:
                model_names.append("xgboost")
            if "transformer" in model_performances:
                model_names.append("transformer")
            if "gru" in model_performances:
                model_names.append("gru")
            if "lstm" in model_performances:
                model_names.append("lstm")
            
            # 可选：基于性能调整权重
            if adaptive_weights:
                perfs = [model_performances.get(name, 0.0) for name in model_names]
                total_perf = sum(perfs)
                if total_perf < 1e-6:
                    weights = [ensemble_weights.get(name, 1.0/len(predictions)) for name in model_names]
                else:
                    max_perf = max(perfs)
                    if max_perf > 0:
                        relative_perfs = [p / max_perf for p in perfs]
                        alpha = 0.5
                        scaled_perfs = [p ** alpha for p in relative_perfs]
                        total_scaled = sum(scaled_perfs)
                        weights = [p / total_scaled for p in scaled_perfs]
                        weights = [max(w, min_weight) for w in weights]
                        total_w = sum(weights)
                        weights = [w / total_w for w in weights]
                    else:
                        weights = [ensemble_weights.get(name, 1.0/len(predictions)) for name in model_names]
                print(f"[Adaptive Weights] Perfs={dict(zip(model_names, perfs))}, "
                      f"Weights={dict(zip(model_names, weights))}")
            else:
                weights = [ensemble_weights.get(name, 1.0/len(predictions)) for name in model_names]
            
            weights = np.array(weights)
            weights = weights / weights.sum()
            pred_ensemble = np.average(predictions, axis=0, weights=weights)
        
        oof_pred.loc[Xte.index] = pred_ensemble
        
        # 计算集成模型的 Rank IC（安全处理 NaN）
        dfte = pd.DataFrame({"pred": pred_ensemble, "y": yte.values}, index=Xte.index)
        ric_mean = safe_rank_ic(dfte["pred"], dfte["y"])
        
        model_info = ", ".join([f"{name}:{abs(perf):.4f}" for name, perf in model_performances.items()])
        weight_info = ", ".join([f"{w:.3f}" for w in weights])
        print(f"[Fold {i+1}] Ensemble RankIC={ric_mean:.4f} | weights=[{weight_info}] | models=[{model_info}]")
        
        metrics["folds"].append({
            "fold": i + 1,
            "tr_dates": [str(tr_dates.min()), str(tr_dates.max())],
            "te_dates": [str(te_dates.min()), str(te_dates.max())],
            "mean_rank_ic": float(ric_mean),
            "n_train": len(Xtr),
            "n_test": len(Xte),
            "n_features": len(Xtr.columns),
            "model_performances": {k: float(v) for k, v in model_performances.items()},
            "weights": [float(w) for w in weights],
        })
    
    # OOF Rank IC
    oof = oof_pred.dropna()
    y_all = panel.loc[oof.index, "y"]
    ric_daily = oof.groupby(level=0).apply(
        lambda x: x.rank().corr(y_all.loc[x.index].rank(method="first"), method="spearman")
    )
    metrics["oof_mean_rank_ic"] = float(ric_daily.mean())
    
    # 保存指标
    metrics_path = Path(cfg["paths"].get("metrics_path", "outputs/reports/metrics.json")).parent / "metrics_ensemble.json"
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[OK] Ensemble model trained.")
    print(f"  • Metrics: {metrics_path}")
    print(f"  • OOF RankIC: {metrics['oof_mean_rank_ic']:.4f}")

def train_transformer(cfg):
    """
    训练 Transformer 模型（对齐 QLib）
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        HAS_TORCH = True
    except ImportError:
        print("[ERROR] PyTorch not installed. Please install: pip install torch")
        return
    
    n_folds = cfg["model"]["cv"]["n_folds"]
    use_qlib_label = cfg.get("model", {}).get("use_qlib_label", False)
    panel, dates = prepare_regression_data(cfg, use_qlib_label=use_qlib_label)
    folds = np.array_split(dates, n_folds)
    
    params = cfg["model"]["transformer_params"]
    d_feat = len(panel.columns) - 1  # 特征数量
    params["d_feat"] = d_feat
    
    device = torch.device(f"cuda:{params['GPU']}" if torch.cuda.is_available() and params["GPU"] >= 0 else "cpu")
    
    # 简化的 Transformer 模型（用于面板数据）
    class SimpleTransformer(nn.Module):
        def __init__(self, d_feat, d_model=64, nhead=2, num_layers=2, dropout=0.0):
            super().__init__()
            self.input_layer = nn.Linear(d_feat, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_layer = nn.Linear(d_model, 1)
            
        def forward(self, x):
            # x: [batch, features]
            x = self.input_layer(x)  # [batch, d_model]
            x = x.unsqueeze(1)  # [batch, 1, d_model] - 添加序列维度
            x = self.transformer(x)  # [batch, 1, d_model]
            x = x.squeeze(1)  # [batch, d_model]
            return self.output_layer(x).squeeze()
    
    metrics = {"folds": []}
    oof_pred = pd.Series(index=panel.index, dtype=float)
    
    for i in range(n_folds):
        print(f"\n[Fold {i+1}/{n_folds}]")
        tr_dates = np.concatenate(folds[:i]) if i > 0 else np.array([])
        te_dates = folds[i]
        
        if len(tr_dates) == 0:
            print(f"  -> Skip (no training data)")
            continue
        
        tr_mask = panel.index.get_level_values("date").isin(tr_dates)
        te_mask = panel.index.get_level_values("date").isin(te_dates)
        
        Xtr, ytr = panel.loc[tr_mask].drop(columns=["y"]), panel.loc[tr_mask, "y"]
        Xte, yte = panel.loc[te_mask].drop(columns=["y"]), panel.loc[te_mask, "y"]
        
        if len(Xtr) == 0 or len(Xte) == 0:
            print(f"  -> Skip (empty data)")
            continue
        
        # 关键修复：使用每日横截面中位数填充（优化版：使用 groupby）
        # 训练集：按日期分组，使用 transform 快速填充
        if isinstance(Xtr.index, pd.MultiIndex):
            # 计算每日横截面中位数
            train_medians_by_date = Xtr.groupby(level=0).transform('median')
            # 用每日中位数填充
            Xtr = Xtr.fillna(train_medians_by_date)
            # 对于某日期某特征全为NaN的情况，使用训练集全局中位数填充
            train_global_medians = Xtr.median()
            Xtr = Xtr.fillna(train_global_medians).fillna(0.0)
            
            # 测试集：使用训练集的全局中位数
            Xte = Xte.fillna(train_global_medians).fillna(0.0)
        else:
            # 如果不是MultiIndex，使用全局中位数
            train_global_medians = Xtr.median()
            Xtr = Xtr.fillna(train_global_medians).fillna(0.0)
            Xte = Xte.fillna(train_global_medians).fillna(0.0)
        
        # 深度模型通常需要对输入特征做标准化（虽然 QLib 不做，但有助于训练）
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xtr_scaled = scaler.fit_transform(Xtr.values)
        Xte_scaled = scaler.transform(Xte.values)
        
        # 训练模型
        model = SimpleTransformer(d_feat, params["d_model"], params["nhead"], params["num_layers"], params["dropout"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["reg"])
        criterion = nn.MSELoss()
        
        # 使用 DataLoader 批训练（避免内存溢出）
        batch_size = params.get("batch_size", 2048)
        train_dataset = TensorDataset(torch.FloatTensor(Xtr_scaled), torch.FloatTensor(ytr.values))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        Xte_tensor = torch.FloatTensor(Xte_scaled).to(device)
        yte_tensor = torch.FloatTensor(yte.values).to(device)
        
        best_val_loss = float('inf')
        patience = 0
        best_model_state = None
        
        for epoch in range(params["n_epochs"]):
            # 批训练
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                pred_tr = model(batch_X)
                loss = criterion(pred_tr, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches if n_batches > 0 else float('inf')
            
            # 验证（每20个epoch或第一个epoch）
            if (epoch + 1) % 20 == 0 or epoch == 0:
                model.eval()
                with torch.no_grad():
                    # 验证集也使用批处理（如果数据太大）
                    if len(Xte_scaled) > batch_size * 10:
                        # 大批量验证，避免OOM
                        pred_val_list = []
                        for i in range(0, len(Xte_scaled), batch_size * 10):
                            end_idx = min(i + batch_size * 10, len(Xte_scaled))
                            batch_Xte = Xte_tensor[i:end_idx]
                            pred_val_list.append(model(batch_Xte))
                        pred_val = torch.cat(pred_val_list)
                    else:
                        pred_val = model(Xte_tensor)
                    val_loss = criterion(pred_val, yte_tensor)
                
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    patience = 0
                    # 保存最佳模型（深拷贝 OrderedDict）
                    best_model_state = collections.OrderedDict([(k, v.clone()) for k, v in model.state_dict().items()])
                else:
                    patience += 1
                    
                if patience >= params["early_stop"]:
                    # 恢复最佳模型
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    print(f"  -> Early stop at epoch {epoch+1}, best val_loss={best_val_loss:.6f}")
                    break
        
        # 训练完成后，恢复最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 预测（使用批处理）
        model.eval()
        with torch.no_grad():
            if len(Xte_scaled) > batch_size * 10:
                pred_list = []
                for i in range(0, len(Xte_scaled), batch_size * 10):
                    end_idx = min(i + batch_size * 10, len(Xte_scaled))
                    batch_Xte = Xte_tensor[i:end_idx]
                    pred_list.append(model(batch_Xte))
                pred = torch.cat(pred_list).cpu().numpy()
            else:
                pred = model(Xte_tensor).cpu().numpy()
        
        oof_pred.loc[Xte.index] = pred
        ric_mean = safe_rank_ic(pd.Series(pred, index=Xte.index), yte)
        
        metrics["folds"].append({
            "fold": i + 1,
            "mean_rank_ic": float(ric_mean),
            "n_train": len(Xtr),
            "n_test": len(Xte),
        })
        print(f"  -> RankIC: {ric_mean:.4f}")
    
    # OOF Rank IC
    oof = oof_pred.dropna()
    y_all = panel.loc[oof.index, "y"]
    ric_oof = safe_rank_ic(oof, y_all)
    metrics["oof_mean_rank_ic"] = float(ric_oof)
    
    # 保存
    metrics_path = Path(cfg["paths"].get("metrics_path", "outputs/reports/metrics.json")).parent / "metrics_transformer.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[OK] Transformer model trained. OOF RankIC: {ric_oof:.4f}")

def train_gru(cfg):
    """训练 GRU 模型（对齐 QLib）"""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        HAS_TORCH = True
    except ImportError:
        print("[ERROR] PyTorch not installed. Please install: pip install torch")
        return
    
    n_folds = cfg["model"]["cv"]["n_folds"]
    use_qlib_label = cfg.get("model", {}).get("use_qlib_label", False)
    panel, dates = prepare_regression_data(cfg, use_qlib_label=use_qlib_label)
    folds = np.array_split(dates, n_folds)
    
    params = cfg["model"]["gru_params"]
    d_feat = len(panel.columns) - 1
    params["d_feat"] = d_feat
    
    device = torch.device(f"cuda:{params['GPU']}" if torch.cuda.is_available() and params["GPU"] >= 0 else "cpu")
    
    class SimpleGRU(nn.Module):
        def __init__(self, d_feat, hidden_size=64, num_layers=2, dropout=0.0):
            super().__init__()
            self.gru = nn.GRU(d_feat, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            # x: [batch, features]
            x = x.unsqueeze(1)  # [batch, 1, features]
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :]).squeeze()
    
    metrics = {"folds": []}
    oof_pred = pd.Series(index=panel.index, dtype=float)
    
    for i in range(n_folds):
        print(f"\n[Fold {i+1}/{n_folds}]")
        tr_dates = np.concatenate(folds[:i]) if i > 0 else np.array([])
        te_dates = folds[i]
        
        if len(tr_dates) == 0:
            continue
        
        tr_mask = panel.index.get_level_values("date").isin(tr_dates)
        te_mask = panel.index.get_level_values("date").isin(te_dates)
        
        Xtr, ytr = panel.loc[tr_mask].drop(columns=["y"]), panel.loc[tr_mask, "y"]
        Xte, yte = panel.loc[te_mask].drop(columns=["y"]), panel.loc[te_mask, "y"]
        
        if len(Xtr) == 0 or len(Xte) == 0:
            continue
        
        # 关键修复：使用每日横截面中位数填充（优化版：使用 groupby）
        # 训练集：按日期分组，使用 transform 快速填充
        if isinstance(Xtr.index, pd.MultiIndex):
            # 计算每日横截面中位数
            train_medians_by_date = Xtr.groupby(level=0).transform('median')
            # 用每日中位数填充
            Xtr = Xtr.fillna(train_medians_by_date)
            # 对于某日期某特征全为NaN的情况，使用训练集全局中位数填充
            train_global_medians = Xtr.median()
            Xtr = Xtr.fillna(train_global_medians).fillna(0.0)
            
            # 测试集：使用训练集的全局中位数
            Xte = Xte.fillna(train_global_medians).fillna(0.0)
        else:
            # 如果不是MultiIndex，使用全局中位数
            train_global_medians = Xtr.median()
            Xtr = Xtr.fillna(train_global_medians).fillna(0.0)
            Xte = Xte.fillna(train_global_medians).fillna(0.0)
        
        # 深度模型通常需要对输入特征做标准化（虽然 QLib 不做，但有助于训练）
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xtr_scaled = scaler.fit_transform(Xtr.values)
        Xte_scaled = scaler.transform(Xte.values)
        
        model = SimpleGRU(d_feat, params["hidden_size"], params["num_layers"], params["dropout"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        criterion = nn.MSELoss()
        
        # 使用 DataLoader 批训练（避免内存溢出）
        batch_size = params.get("batch_size", 800)
        train_dataset = TensorDataset(torch.FloatTensor(Xtr_scaled), torch.FloatTensor(ytr.values))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        Xte_tensor = torch.FloatTensor(Xte_scaled).to(device)
        yte_tensor = torch.FloatTensor(yte.values).to(device)
        
        best_val_loss = float('inf')
        patience = 0
        best_model_state = None
        
        for epoch in range(params["n_epochs"]):
            # 批训练
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                pred_tr = model(batch_X)
                loss = criterion(pred_tr, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches if n_batches > 0 else float('inf')
            
            # 验证（每20个epoch）
            if (epoch + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    # 验证集也使用批处理（如果数据太大）
                    if len(Xte_scaled) > batch_size * 10:
                        pred_val_list = []
                        for i in range(0, len(Xte_scaled), batch_size * 10):
                            end_idx = min(i + batch_size * 10, len(Xte_scaled))
                            batch_Xte = Xte_tensor[i:end_idx]
                            pred_val_list.append(model(batch_Xte))
                        pred_val = torch.cat(pred_val_list)
                    else:
                        pred_val = model(Xte_tensor)
                    val_loss = criterion(pred_val, yte_tensor)
                
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    patience = 0
                    # 保存最佳模型
                    best_model_state = collections.OrderedDict([(k, v.clone()) for k, v in model.state_dict().items()])
                else:
                    patience += 1
                    
                if patience >= params["early_stop"]:
                    # 恢复最佳模型
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
        
        # 恢复最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 预测（使用批处理）
        model.eval()
        with torch.no_grad():
            if len(Xte_scaled) > batch_size * 10:
                pred_list = []
                for i in range(0, len(Xte_scaled), batch_size * 10):
                    end_idx = min(i + batch_size * 10, len(Xte_scaled))
                    batch_Xte = Xte_tensor[i:end_idx]
                    pred_list.append(model(batch_Xte))
                pred = torch.cat(pred_list).cpu().numpy()
            else:
                pred = model(Xte_tensor).cpu().numpy()
        
        oof_pred.loc[Xte.index] = pred
        ric_mean = safe_rank_ic(pd.Series(pred, index=Xte.index), yte)
        
        metrics["folds"].append({
            "fold": i + 1,
            "mean_rank_ic": float(ric_mean),
            "n_train": len(Xtr),
            "n_test": len(Xte),
        })
    
    oof = oof_pred.dropna()
    y_all = panel.loc[oof.index, "y"]
    ric_oof = safe_rank_ic(oof, y_all)
    metrics["oof_mean_rank_ic"] = float(ric_oof)
    
    metrics_path = Path(cfg["paths"].get("metrics_path", "outputs/reports/metrics.json")).parent / "metrics_gru.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[OK] GRU model trained. OOF RankIC: {ric_oof:.4f}")

def train_lstm(cfg):
    """训练 LSTM 模型（对齐 QLib）"""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        HAS_TORCH = True
    except ImportError:
        print("[ERROR] PyTorch not installed. Please install: pip install torch")
        return
    
    n_folds = cfg["model"]["cv"]["n_folds"]
    use_qlib_label = cfg.get("model", {}).get("use_qlib_label", False)
    panel, dates = prepare_regression_data(cfg, use_qlib_label=use_qlib_label)
    folds = np.array_split(dates, n_folds)
    
    params = cfg["model"]["lstm_params"]
    d_feat = len(panel.columns) - 1
    params["d_feat"] = d_feat
    
    device = torch.device(f"cuda:{params['GPU']}" if torch.cuda.is_available() and params["GPU"] >= 0 else "cpu")
    
    class SimpleLSTM(nn.Module):
        def __init__(self, d_feat, hidden_size=64, num_layers=2, dropout=0.0):
            super().__init__()
            self.lstm = nn.LSTM(d_feat, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            x = x.unsqueeze(1)  # [batch, 1, features]
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze()
    
    metrics = {"folds": []}
    oof_pred = pd.Series(index=panel.index, dtype=float)
    
    for i in range(n_folds):
        print(f"\n[Fold {i+1}/{n_folds}]")
        tr_dates = np.concatenate(folds[:i]) if i > 0 else np.array([])
        te_dates = folds[i]
        
        if len(tr_dates) == 0:
            continue
        
        tr_mask = panel.index.get_level_values("date").isin(tr_dates)
        te_mask = panel.index.get_level_values("date").isin(te_dates)
        
        Xtr, ytr = panel.loc[tr_mask].drop(columns=["y"]), panel.loc[tr_mask, "y"]
        Xte, yte = panel.loc[te_mask].drop(columns=["y"]), panel.loc[te_mask, "y"]
        
        if len(Xtr) == 0 or len(Xte) == 0:
            continue
        
        # 关键修复：使用每日横截面中位数填充（优化版：使用 groupby）
        # 训练集：按日期分组，使用 transform 快速填充
        if isinstance(Xtr.index, pd.MultiIndex):
            # 计算每日横截面中位数
            train_medians_by_date = Xtr.groupby(level=0).transform('median')
            # 用每日中位数填充
            Xtr = Xtr.fillna(train_medians_by_date)
            # 对于某日期某特征全为NaN的情况，使用训练集全局中位数填充
            train_global_medians = Xtr.median()
            Xtr = Xtr.fillna(train_global_medians).fillna(0.0)
            
            # 测试集：使用训练集的全局中位数
            Xte = Xte.fillna(train_global_medians).fillna(0.0)
        else:
            # 如果不是MultiIndex，使用全局中位数
            train_global_medians = Xtr.median()
            Xtr = Xtr.fillna(train_global_medians).fillna(0.0)
            Xte = Xte.fillna(train_global_medians).fillna(0.0)
        
        # 深度模型通常需要对输入特征做标准化（虽然 QLib 不做，但有助于训练）
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xtr_scaled = scaler.fit_transform(Xtr.values)
        Xte_scaled = scaler.transform(Xte.values)
        
        model = SimpleLSTM(d_feat, params["hidden_size"], params["num_layers"], params["dropout"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        criterion = nn.MSELoss()
        
        # 使用 DataLoader 批训练（避免内存溢出）
        batch_size = params.get("batch_size", 800)
        train_dataset = TensorDataset(torch.FloatTensor(Xtr_scaled), torch.FloatTensor(ytr.values))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        Xte_tensor = torch.FloatTensor(Xte_scaled).to(device)
        yte_tensor = torch.FloatTensor(yte.values).to(device)
        
        best_val_loss = float('inf')
        patience = 0
        best_model_state = None
        
        for epoch in range(params["n_epochs"]):
            # 批训练
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                pred_tr = model(batch_X)
                loss = criterion(pred_tr, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches if n_batches > 0 else float('inf')
            
            # 验证（每20个epoch）
            if (epoch + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    # 验证集也使用批处理（如果数据太大）
                    if len(Xte_scaled) > batch_size * 10:
                        pred_val_list = []
                        for i in range(0, len(Xte_scaled), batch_size * 10):
                            end_idx = min(i + batch_size * 10, len(Xte_scaled))
                            batch_Xte = Xte_tensor[i:end_idx]
                            pred_val_list.append(model(batch_Xte))
                        pred_val = torch.cat(pred_val_list)
                    else:
                        pred_val = model(Xte_tensor)
                    val_loss = criterion(pred_val, yte_tensor)
                
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    patience = 0
                    # 保存最佳模型
                    best_model_state = collections.OrderedDict([(k, v.clone()) for k, v in model.state_dict().items()])
                else:
                    patience += 1
                    
                if patience >= params["early_stop"]:
                    # 恢复最佳模型
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
        
        # 恢复最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 预测（使用批处理）
        model.eval()
        with torch.no_grad():
            if len(Xte_scaled) > batch_size * 10:
                pred_list = []
                for i in range(0, len(Xte_scaled), batch_size * 10):
                    end_idx = min(i + batch_size * 10, len(Xte_scaled))
                    batch_Xte = Xte_tensor[i:end_idx]
                    pred_list.append(model(batch_Xte))
                pred = torch.cat(pred_list).cpu().numpy()
            else:
                pred = model(Xte_tensor).cpu().numpy()
        
        oof_pred.loc[Xte.index] = pred
        ric_mean = safe_rank_ic(pd.Series(pred, index=Xte.index), yte)
        
        metrics["folds"].append({
            "fold": i + 1,
            "mean_rank_ic": float(ric_mean),
            "n_train": len(Xtr),
            "n_test": len(Xte),
        })
    
    oof = oof_pred.dropna()
    y_all = panel.loc[oof.index, "y"]
    ric_oof = safe_rank_ic(oof, y_all)
    metrics["oof_mean_rank_ic"] = float(ric_oof)
    
    metrics_path = Path(cfg["paths"].get("metrics_path", "outputs/reports/metrics.json")).parent / "metrics_lstm.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[OK] LSTM model trained. OOF RankIC: {ric_oof:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train ranking model (lambdarank)")
    parser.add_argument("--train-regression", action="store_true", help="Train LightGBM regression model")
    parser.add_argument("--train-catboost", action="store_true", help="Train CatBoost regression model")
    parser.add_argument("--train-xgboost", action="store_true", help="Train XGBoost regression model")
    parser.add_argument("--train-ensemble", action="store_true", help="Train ensemble model (LightGBM + CatBoost + XGBoost)")
    parser.add_argument("--train-transformer", action="store_true", help="Train Transformer model")
    parser.add_argument("--train-gru", action="store_true", help="Train GRU model")
    parser.add_argument("--train-lstm", action="store_true", help="Train LSTM model")
    args = parser.parse_args()
    cfg = load_settings()
    
    if args.train:
        train_ranker(cfg)
    elif args.train_regression:
        train_regression(cfg)
    elif args.train_catboost:
        train_catboost(cfg)
    elif args.train_xgboost:
        train_xgboost(cfg)
    # elif args.train_ensemble:
    #     train_ensemble(cfg)  # 暂时注释掉，避免内存溢出（已添加批训练，但待测试）
    # elif args.train_transformer:
    #     train_transformer(cfg)  # 暂时注释掉，避免内存溢出（已添加批训练，但待测试）
    # elif args.train_gru:
    #     train_gru(cfg)  # 暂时注释掉，先对齐 QLib 基准测试
    elif args.train_lstm:
        train_lstm(cfg)  # 暂时注释掉，先对齐 QLib 基准测试
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
