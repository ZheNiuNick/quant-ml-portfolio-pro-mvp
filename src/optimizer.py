#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Portfolio Optimizer - 对齐 QLib PortfolioOptimizer
支持所有策略类型和优化方法：
- TopkDropoutStrategy: Top-k 选股，每日替换 n_drop 只
- PortfolioOptimizer: inv, gmv, mvo, rp 四种优化方法

对齐 QLib 实现：
- qlib.contrib.strategy.optimizer.optimizer.PortfolioOptimizer
- qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional, Union, Dict

import numpy as np
import pandas as pd
import scipy.optimize as so
import yaml

# 使用统一的路径管理
from src.config.path import SETTINGS_FILE, OUTPUT_MODELS_DIR, OUTPUT_PORTFOLIOS_DIR, get_path

SETTINGS = SETTINGS_FILE


def load_settings(path=SETTINGS_FILE):
    path = get_path(path) if isinstance(path, str) and not Path(path).is_absolute() else Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================================
# PortfolioOptimizer (对齐 QLib)
# ============================================================================

class PortfolioOptimizer:
    """Portfolio Optimizer - 对齐 QLib qlib.contrib.strategy.optimizer.optimizer.PortfolioOptimizer
    
    支持的优化算法：
    - `inv`: Inverse Volatility（逆波动率加权，最简单，QLib 常用）
    - `gmv`: Global Minimum Variance Portfolio（全局最小方差组合）
    - `mvo`: Mean Variance Optimized Portfolio（均值-方差优化）
    - `rp`: Risk Parity（风险平价）
    
    注意：
        - 总是假设满仓投资且不做空（w >= 0, sum(w) == 1）
    """

    OPT_INV = "inv"
    OPT_GMV = "gmv"
    OPT_MVO = "mvo"
    OPT_RP = "rp"

    def __init__(
        self,
        method: str = "inv",
        lamb: float = 0,
        delta: float = 0,
        alpha: float = 0.0,
        scale_return: bool = True,
        tol: float = 1e-8,
    ):
        """
        Args:
            method (str): 组合优化方法 ("inv", "gmv", "mvo", "rp")
            lamb (float): 风险厌恶参数（越大越关注收益，仅用于 mvo）
            delta (float): 换手率限制（仅用于有 w0 的优化）
            alpha (float): L2 正则化项
            scale_return (bool): 是否缩放预期收益以匹配协方差矩阵的波动率
            tol (float): 优化终止容差
        """
        assert method in [self.OPT_INV, self.OPT_GMV, self.OPT_MVO, self.OPT_RP], \
            f"method `{method}` is not supported"
        self.method = method

        assert lamb >= 0, f"risk aversion parameter `lamb` should be non-negative"
        self.lamb = lamb

        assert delta >= 0, f"turnover limit `delta` should be non-negative"
        self.delta = delta

        assert alpha >= 0, f"l2 norm regularizer `alpha` should be non-negative"
        self.alpha = alpha

        self.tol = tol
        self.scale_return = scale_return

    def __call__(
        self,
        S: Union[np.ndarray, pd.DataFrame],
        r: Optional[Union[np.ndarray, pd.Series]] = None,
        w0: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Union[np.ndarray, pd.Series]:
        """
        Args:
            S (np.ndarray or pd.DataFrame): 协方差矩阵
            r (np.ndarray or pd.Series, optional): 预期收益
            w0 (np.ndarray or pd.Series, optional): 初始权重（用于换手控制）

        Returns:
            np.ndarray or pd.Series: 优化后的组合权重
        """
        # 转换 DataFrame 为 array
        index = None
        if isinstance(S, pd.DataFrame):
            index = S.index
            S = S.values

        # 转换收益
        if r is not None:
            assert len(r) == len(S), "`r` has mismatched shape"
            if isinstance(r, pd.Series):
                assert r.index.equals(index), "`r` has mismatched index"
                r = r.values

        # 转换初始权重
        if w0 is not None:
            assert len(w0) == len(S), "`w0` has mismatched shape"
            if isinstance(w0, pd.Series):
                assert w0.index.equals(index), "`w0` has mismatched index"
                w0 = w0.values

        # 缩放收益以匹配波动率
        if r is not None and self.scale_return:
            r = r / r.std()
            r *= np.sqrt(np.mean(np.diag(S)))

        # 优化
        w = self._optimize(S, r, w0)

        # 恢复索引
        if index is not None:
            w = pd.Series(w, index=index)

        return w

    def _optimize(self, S: np.ndarray, r: Optional[np.ndarray] = None, w0: Optional[np.ndarray] = None) -> np.ndarray:
        """内部优化方法"""
        # 逆波动率
        if self.method == self.OPT_INV:
            if r is not None:
                warnings.warn("`r` is set but will not be used for `inv` portfolio")
            if w0 is not None:
                warnings.warn("`w0` is set but will not be used for `inv` portfolio")
            return self._optimize_inv(S)

        # 全局最小方差
        if self.method == self.OPT_GMV:
            if r is not None:
                warnings.warn("`r` is set but will not be used for `gmv` portfolio")
            return self._optimize_gmv(S, w0)

        # 均值-方差优化
        if self.method == self.OPT_MVO:
            return self._optimize_mvo(S, r, w0)

        # 风险平价
        if self.method == self.OPT_RP:
            if r is not None:
                warnings.warn("`r` is set but will not be used for `rp` portfolio")
            return self._optimize_rp(S, w0)

    def _optimize_inv(self, S: np.ndarray) -> np.ndarray:
        """逆波动率加权"""
        vola = np.diag(S) ** 0.5
        w = 1 / vola
        w /= w.sum()
        return w

    def _optimize_gmv(self, S: np.ndarray, w0: Optional[np.ndarray] = None) -> np.ndarray:
        """全局最小方差组合
        min_w w' S w
        s.t. w >= 0, sum(w) == 1
        """
        return self._solve(len(S), self._get_objective_gmv(S), *self._get_constrains(w0))

    def _optimize_mvo(self, S: np.ndarray, r: Optional[np.ndarray] = None, w0: Optional[np.ndarray] = None) -> np.ndarray:
        """均值-方差优化
        min_w - w' r + lamb * w' S w
        s.t. w >= 0, sum(w) == 1
        """
        if r is None:
            raise ValueError("`r` is required for MVO optimization")
        return self._solve(len(S), self._get_objective_mvo(S, r), *self._get_constrains(w0))

    def _optimize_rp(self, S: np.ndarray, w0: Optional[np.ndarray] = None) -> np.ndarray:
        """风险平价组合
        min_w sum_i [w_i - (w' S w) / ((S w)_i * N)]**2
        s.t. w >= 0, sum(w) == 1
        """
        return self._solve(len(S), self._get_objective_rp(S), *self._get_constrains(w0))

    def _get_objective_gmv(self, S: np.ndarray):
        """全局最小方差目标函数"""
        def func(x):
            return x @ S @ x
        return func

    def _get_objective_mvo(self, S: np.ndarray, r: np.ndarray):
        """均值-方差优化目标函数"""
        def func(x):
            risk = x @ S @ x
            ret = x @ r
            return -ret + self.lamb * risk
        return func

    def _get_objective_rp(self, S: np.ndarray):
        """风险平价目标函数"""
        def func(x):
            N = len(x)
            Sx = S @ x
            xSx = x @ Sx
            return np.sum((x - xSx / Sx / N) ** 2)
        return func

    def _get_constrains(self, w0: Optional[np.ndarray] = None):
        """优化约束条件"""
        # 不做空且不加杠杆
        bounds = so.Bounds(0.0, 1.0)

        # 满仓约束
        cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

        # 换手约束
        if w0 is not None:
            cons.append({"type": "ineq", "fun": lambda x: self.delta - np.sum(np.abs(x - w0))})

        return bounds, cons

    def _solve(self, n: int, obj, bounds: so.Bounds, cons: list) -> np.ndarray:
        """求解优化问题"""
        # 添加 L2 正则化
        wrapped_obj = obj
        if self.alpha > 0:
            def opt_obj(x):
                return obj(x) + self.alpha * np.sum(np.square(x))
            wrapped_obj = opt_obj

        # 求解
        x0 = np.ones(n) / n  # 初始值（等权重）
        sol = so.minimize(wrapped_obj, x0, bounds=bounds, constraints=cons, tol=self.tol)
        if not sol.success:
            warnings.warn(f"optimization not success ({sol.status})")

        return sol.x


# ============================================================================
# TopkDropoutStrategy (对齐 QLib)
# ============================================================================

def topk_dropout_strategy(
    pred_score: pd.Series,
    current_positions: pd.Series,
    topk: int = 50,
    n_drop: int = 5,
    method_sell: str = "bottom",
    method_buy: str = "top",
) -> pd.Series:
    """
    TopkDropoutStrategy - 对齐 QLib qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy
    
    策略逻辑：
    1. 保持 topk 只股票
    2. 每日替换 n_drop 只
    3. 卖出：当前持仓中评分最低的 n_drop 只（method_sell="bottom"）
    4. 买入：未持仓中评分最高的 n_drop 只（method_buy="top"）
    
    Args:
        pred_score: 预测分数 Series (index=ticker)
        current_positions: 当前持仓权重 Series (index=ticker, value=weight)
        topk: 持仓股票数量
        n_drop: 每日替换数量
        method_sell: 卖出方法 ("bottom" 或 "random")
        method_buy: 买入方法 ("top" 或 "random")
    
    Returns:
        新的持仓权重 Series (index=ticker, value=weight)
    """
    pred_score = pred_score.dropna()
    if len(pred_score) == 0:
        return pd.Series(dtype=float)
    
    # 当前持仓的股票列表（按评分排序）
    current_stocks = current_positions[current_positions > 0].index
    if len(current_stocks) == 0:
        # 如果没有持仓，买入 topk 只
        target_stocks = pred_score.nlargest(topk).index
        weights = pd.Series(1.0 / len(target_stocks), index=target_stocks)
        return weights.reindex(pred_score.index, fill_value=0.0)
    
    # 关键修复：只保留在预测数据中的当前持仓股票
    # 如果当前持仓的股票不在预测数据中，这些股票会被视为已卖出
    current_stocks_in_pred = current_stocks.intersection(pred_score.index)
    
    # 当前持仓的评分排序（只包含在预测数据中的股票）
    last = pred_score.reindex(current_stocks_in_pred).dropna().sort_values(ascending=False).index
    
    # 关键修复：如果当前持仓数超过 topk，需要先卖出多余的
    # 确保最终持仓数始终等于 topk
    if len(last) > topk:
        # 如果当前持仓超过 topk，卖出评分最低的 (len(last) - topk) 只
        excess_count = len(last) - topk
        # 卖出评分最低的 excess_count 只（但至少卖出 n_drop 只）
        sell_count = max(excess_count, n_drop)
        last = last[:-sell_count]  # 保留评分最高的 topk 只（或更少）
        # 注意：如果当前持仓超过 topk，已经在上一步处理，这里 last 的长度应该 <= topk
    
    # 需要买入的候选股票（未持仓中评分最高的）
    if method_buy == "top":
        # 计算需要买入的数量：确保最终持仓数 = topk
        # 如果当前持仓 < topk，需要买入 (topk - len(last)) 只
        # 如果当前持仓 = topk，需要买入 n_drop 只（替换）
        n_buy = max(n_drop, topk - len(last))
        candidates = pred_score[~pred_score.index.isin(last)].nlargest(n_buy)
        today = candidates.index
    elif method_buy == "random":
        topk_candi = pred_score.nlargest(topk).index
        candi = [s for s in topk_candi if s not in last]
        n = max(n_drop, topk - len(last))
        today = pd.Index(np.random.choice(candi, min(n, len(candi)), replace=False))
    else:
        raise NotImplementedError(f"method_buy `{method_buy}` is not supported")
    
    # 合并：当前持仓 + 新候选
    comb = pred_score.reindex(last.union(today)).sort_values(ascending=False).index
    
    # 确定卖出的股票（持仓中评分最低的 n_drop 只）
    # 对齐 QLib：从合并列表的末尾（评分最低的 n_drop 只）中，选出在当前持仓中的股票
    if method_sell == "bottom":
        # 获取评分最低的 n_drop 只（从合并列表的末尾）
        bottom_n = comb[-n_drop:] if len(comb) >= n_drop else comb
        # 选出在当前持仓中的股票（对齐 QLib：last[last.isin(get_last_n(comb, self.n_drop))]）
        sell = last[last.isin(bottom_n)]
    elif method_sell == "random":
        candi = list(last)
        try:
            sell = pd.Index(np.random.choice(candi, n_drop, replace=False) if len(candi) >= n_drop else candi)
        except ValueError:  # No enough candidates
            sell = pd.Index(candi) if len(candi) > 0 else pd.Index([])
    else:
        raise NotImplementedError(f"method_sell `{method_sell}` is not supported")
    
    # 确定买入的股票（确保最终持仓数 = topk）
    # 修复：确保 buy 的数量使得 keep_stocks + new_stocks = topk
    keep_count = len(last) - len(sell)
    buy_count = topk - keep_count
    buy = today[:buy_count] if buy_count > 0 else pd.Index([])
    
    # 构建新持仓：保留的股票 + 新买入的股票
    keep_stocks = [s for s in last if s not in sell]
    new_stocks = list(buy)
    target_stocks = keep_stocks + new_stocks
    
    # 最终验证：确保持仓数 = topk
    if len(target_stocks) != topk:
        # 如果还是不对，强制调整到 topk
        if len(target_stocks) > topk:
            # 如果超过 topk，只保留评分最高的 topk 只
            all_candidates = pred_score.reindex(target_stocks).sort_values(ascending=False)
            target_stocks = all_candidates.head(topk).index.tolist()
        elif len(target_stocks) < topk:
            # 如果少于 topk，补充评分最高的股票
            remaining = topk - len(target_stocks)
            additional = pred_score[~pred_score.index.isin(target_stocks)].nlargest(remaining).index
            target_stocks = target_stocks + list(additional)
    
    if len(target_stocks) == 0:
        return pd.Series(dtype=float, index=pred_score.index)
    
    # 等权重分配
    weights = pd.Series(1.0 / len(target_stocks), index=target_stocks)
    
    return weights.reindex(pred_score.index, fill_value=0.0)


def full_rebalance_strategy(
    pred_score: pd.Series,
    current_positions: pd.Series = None,
    topk: int = 20,
) -> pd.Series:
    """
    Full Rebalance Strategy - 每日全量换仓策略
    
    策略逻辑：
    1. 忽略当前持仓
    2. 直接选择评分最高的 topk 只股票
    3. 等权重分配
    
    Args:
        pred_score: 预测分数 Series (index=ticker)
        current_positions: 当前持仓权重 Series（此策略中不使用，保留以兼容接口）
        topk: 持仓股票数量
    
    Returns:
        新的持仓权重 Series (index=ticker, value=weight)
    """
    pred_score = pred_score.dropna()
    if len(pred_score) == 0:
        return pd.Series(dtype=float)
    
    # 直接选择评分最高的 topk 只股票
    target_stocks = pred_score.nlargest(min(topk, len(pred_score))).index
    
    if len(target_stocks) == 0:
        return pd.Series(dtype=float, index=pred_score.index)
    
    # 等权重分配
    weights = pd.Series(1.0 / len(target_stocks), index=target_stocks)
    
    return weights.reindex(pred_score.index, fill_value=0.0)


# ============================================================================
# 辅助函数
# ============================================================================

def load_predictions(cfg, model_type: str = "lightgbm") -> pd.Series:
    """
    加载模型预测（对齐 QLib 工作流）
    
    优先顺序：
    1. 从已保存的预测文件加载（如果有）
    2. 从保存的模型重新预测（如果需要）
    3. 从 factor_store 加载（作为备用）
    
    Args:
        cfg: 配置字典
        model_type: 模型类型 ("lightgbm", "catboost", "xgboost")
    
    Returns:
        预测 Series (MultiIndex: date, ticker)
    """
    import numpy as np
    
    # 1. 尝试从模型输出目录加载已保存的预测
    model_dir = get_path(cfg["paths"]["model_dir"], OUTPUT_MODELS_DIR)
    specific_pred_file = model_dir / f"{model_type}_predictions.pkl"
    if specific_pred_file.exists():
        import pickle
        with open(specific_pred_file, "rb") as f:
            pred = pickle.load(f)
        if isinstance(pred, pd.Series):
            print(f"[Optimizer] Loaded cached predictions from {specific_pred_file}")
            return pred
        elif isinstance(pred, pd.DataFrame):
            print(f"[Optimizer] Loaded cached dataframe predictions from {specific_pred_file}, using first column")
            return pred.iloc[:, 0]
    
    # 兼容旧命名：任意 *_predictions.pkl
    pred_files = list(model_dir.glob("*_predictions.pkl"))
    if pred_files:
        import pickle
        with open(pred_files[0], "rb") as f:
            pred = pickle.load(f)
        if isinstance(pred, pd.Series):
            print(f"[Optimizer] Loaded cached predictions from {pred_files[0]}")
            return pred
        elif isinstance(pred, pd.DataFrame):
            print(f"[Optimizer] Loaded cached dataframe predictions from {pred_files[0]}, using first column")
            return pred.iloc[:, 0]  # 取第一列
    
    # 2. 从保存的模型重新预测（对齐 QLib 工作流）
    print(f"[Optimizer] Loading {model_type} model and generating predictions...")
    
    # 加载数据
    factor_store = pd.read_parquet(cfg["paths"]["factors_store"])
    factor_store.index = pd.MultiIndex.from_tuples(factor_store.index, names=["date", "ticker"])
    
    # 关键修复：通过类型检查识别并修复索引层级（不要相信名称）
    if isinstance(factor_store.index, pd.MultiIndex):
        level_0 = factor_store.index.get_level_values(0)
        level_1 = factor_store.index.get_level_values(1)
        # 通过类型检查识别日期层级，确保ticker是字符串
        if pd.api.types.is_datetime64_any_dtype(level_0):
            dates = pd.to_datetime(level_0, errors='coerce')
            if pd.api.types.is_datetime64_any_dtype(level_1):
                # 如果level_1也是datetime，说明层级顺序错了，交换
                dates = pd.to_datetime(level_1, errors='coerce')
                tickers = pd.Series(level_0).astype(str).values
            else:
                tickers = pd.Series(level_1).astype(str).values
        elif pd.api.types.is_datetime64_any_dtype(level_1):
            dates = pd.to_datetime(level_1, errors='coerce')
            tickers = pd.Series(level_0).astype(str).values
        else:
            # 都不是datetime，默认level_0是日期（QLib标准格式）
            dates = pd.to_datetime(level_0, errors='coerce')
            tickers = pd.Series(level_1).astype(str).values
        factor_store.index = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])
    
    # 移除 LABEL0（如果存在）
    if "LABEL0" in factor_store.columns:
        factor_store = factor_store.drop(columns=["LABEL0"])
    
    # 关键诊断：检查factor_store中的股票代码格式
    if isinstance(factor_store.index, pd.MultiIndex):
        factor_tickers = sorted(factor_store.index.get_level_values("ticker").unique().tolist())
        print(f"  [数据源诊断] factor_store中的股票代码:")
        print(f"    股票总数: {len(factor_tickers)}")
        print(f"    前20个股票代码: {factor_tickers[:20]}")
        print(f"    后20个股票代码: {factor_tickers[-20:]}")
        
        # 检查是否是US格式（字母代码）vs CN格式（数字代码）
        is_us_format = any(len(t) <= 5 and t.replace('.', '').isalpha() for t in factor_tickers[:100])
        is_cn_format = any(t.startswith(('SH', 'SZ', 'BJ')) or (len(t) == 6 and t.isdigit()) for t in factor_tickers[:100])
        
        if is_us_format:
            print(f"    ⚠️ 检测到US格式股票代码（如'A', 'AAPL'）")
        if is_cn_format:
            print(f"    ✓ 检测到CN格式股票代码（如'SH600000'或数字代码）")
    
    # 准备测试集数据（对齐 QLib segments）
    segments = cfg.get("model", {}).get("segments", {})
    test_start = segments.get("test_start", "2017-01-01")
    test_end = segments.get("test_end", "2020-08-01")
    
    test_mask = (factor_store.index.get_level_values("date") >= test_start) & \
                (factor_store.index.get_level_values("date") <= test_end)
    X_test = factor_store.loc[test_mask].copy()
    
    if len(X_test) == 0:
        raise ValueError(f"No test data found in range {test_start} to {test_end}")
    
    # 关键修复：只保留有足够数据的股票（缺失率<90%）
    # 这样可以避免预测时填充后特征完全相同的问题
    print(f"  [数据筛选] 过滤高缺失率股票（缺失率>90%）...")
    initial_count = len(X_test)
    
    if isinstance(X_test.index, pd.MultiIndex):
        # 检查每个样本（股票-日期对）的缺失率
        def has_sufficient_data(row):
            missing_rate = row.isna().sum() / len(row)
            return missing_rate < 0.9  # 保留缺失率<90%的样本
        
        valid_mask = X_test.apply(has_sufficient_data, axis=1)
        X_test = X_test.loc[valid_mask].copy()
        
        filtered_count = initial_count - len(X_test)
        if filtered_count > 0:
            print(f"    过滤掉 {filtered_count} 个高缺失率样本（缺失率>90%）")
            print(f"    保留 {len(X_test)} 个有效样本")
        
        # 诊断：检查过滤后的数据质量
        if len(X_test) > 0:
            sample_date = X_test.index.get_level_values(0)[0]
            sample_data = X_test.xs(sample_date, level=0)
            sample_missing_rates = sample_data.isna().sum(axis=1) / len(sample_data.columns)
            high_missing_after = (sample_missing_rates > 0.5).sum()
            remaining_stocks = len(sample_data)
            print(f"    过滤后，示例日期高缺失率（>50%）股票数: {high_missing_after} / {remaining_stocks}")
            
            if remaining_stocks == 0:
                raise ValueError(f"过滤后没有有效股票！可能factor_store数据质量有问题")
    
    # 关键修复：保存原始索引，确保预测值和索引正确对应
    original_index = X_test.index.copy()
    
    # 关键修复：填充缺失值的方式（对齐训练时的填充策略）
    # 问题：预测时必须使用训练集的全局中位数，而不是测试集的中位数
    # 这样才能保持与训练时一致的数据分布，避免数据泄露
    X_test_filled = X_test.copy()
    
    # 尝试加载训练集的全局中位数（训练时保存）
    import json
    train_medians_path = model_dir / "train_global_medians.json"
    train_global_medians = None
    
    if train_medians_path.exists():
        try:
            with open(train_medians_path, "r") as f:
                train_medians_dict = json.load(f)
            train_global_medians = pd.Series(train_medians_dict)
            print(f"  [填充策略] 使用训练集的全局中位数填充（对齐训练时策略）...")
        except Exception as e:
            print(f"  [警告] 无法加载训练集中位数: {e}，将使用测试集统计值")
    
    if train_global_medians is None:
        # 如果没有保存的训练集中位数，需要从训练集数据中计算
        # 加载训练集数据来计算全局中位数（对齐训练时的逻辑）
        print(f"  [填充策略] 计算训练集全局中位数（对齐训练时逻辑）...")
        segments = cfg.get("model", {}).get("segments", {})
        train_start = segments.get("train_start", "2008-01-01")
        train_end = segments.get("train_end", "2014-12-31")
        
        train_mask = (factor_store.index.get_level_values("date") >= train_start) & \
                    (factor_store.index.get_level_values("date") <= train_end)
        X_train = factor_store.loc[train_mask].copy()
        
        if len(X_train) > 0:
            # 训练集：先按日期填充（每日横截面中位数），然后计算全局中位数
            # 这与训练时的逻辑完全一致
            if isinstance(X_train.index, pd.MultiIndex):
                train_medians_by_date = X_train.groupby(level=0).transform('median')
                X_train = X_train.fillna(train_medians_by_date)
            train_global_medians = X_train.median()
            print(f"  [计算完成] 训练集全局中位数（基于 {len(X_train)} 个训练样本，已先按日填充）")
        else:
            print(f"  [警告] 无法加载训练集数据，使用测试集统计值（可能导致不一致）")
            # 即使是fallback，也先按日期填充
            if isinstance(X_test_filled.index, pd.MultiIndex):
                test_medians_by_date = X_test_filled.groupby(level=0).transform('median')
                X_test_temp = X_test_filled.fillna(test_medians_by_date)
                train_global_medians = X_test_temp.median()
            else:
                train_global_medians = X_test_filled.median()
    
    # 关键修复：使用每日横截面中位数填充（与训练时一致）
    # 这样保持特征的区分度，避免所有股票填充后特征相同
    if isinstance(X_test_filled.index, pd.MultiIndex):
        # 第一步：使用每日横截面中位数填充（与训练集处理一致）
        print(f"  [填充步骤1] 使用每日横截面中位数填充...")
        test_medians_by_date = X_test_filled.groupby(level=0).transform('median')
        X_test_filled = X_test_filled.fillna(test_medians_by_date)
        
        # 第二步：对于某日期某特征全为NaN的情况，使用训练集的全局中位数填充（避免数据泄露）
        print(f"  [填充步骤2] 对仍缺失的特征使用训练集全局中位数填充...")
        X_test_filled = X_test_filled.fillna(train_global_medians)
        
        # 第三步：如果全局中位数也是NaN，填充为0
        X_test_filled = X_test_filled.fillna(0.0)
    else:
        # 如果不是MultiIndex，直接使用训练集的全局中位数填充
        X_test_filled = X_test_filled.fillna(train_global_medians).fillna(0.0)
    
    # 关键诊断：检查特征缺失情况和填充效果
    print(f"  [诊断] 测试集统计:")
    print(f"    样本数: {len(X_test_filled)}")
    print(f"    特征数: {len(X_test_filled.columns)}")
    
    if isinstance(X_test_filled.index, pd.MultiIndex):
        sample_date = X_test_filled.index.get_level_values(0)[0]
        sample_data_filled = X_test_filled.xs(sample_date, level=0)
        sample_data_raw = X_test.xs(sample_date, level=0) if isinstance(X_test.index, pd.MultiIndex) else None
        
        print(f"    示例日期 {sample_date} 的样本数: {len(sample_data_filled)}")
        
        # 检查填充前的缺失率
        if sample_data_raw is not None:
            missing_rate = sample_data_raw.isna().sum() / len(sample_data_raw)
            high_missing_features = (missing_rate > 0.5).sum()
            print(f"    填充前高缺失率（>50%）特征数: {high_missing_features}")
            
            # 检查每个股票的缺失率
            stock_missing_rates = sample_data_raw.isna().sum(axis=1) / len(sample_data_raw.columns)
            high_missing_stocks = (stock_missing_rates > 0.5).sum()
            print(f"    填充前高缺失率（>50%）股票数: {high_missing_stocks} / {len(sample_data_raw)}")
            
            if high_missing_stocks > 0:
                print(f"    ⚠️ 警告：有 {high_missing_stocks} 个股票的特征缺失率超过50%")
                print(f"      这些股票填充后可能特征相似")
        
        # 检查填充后的特征区分度
        feature_stds = sample_data_filled.std()
        zero_std_features = (feature_stds < 1e-10).sum()
        print(f"    填充后零方差特征数: {zero_std_features} / {len(feature_stds)}")
        
        # 关键诊断：检查这些特征完全相同的股票在填充前的情况
        print(f"\n    [特征诊断] 检查填充后的特征区分度:")
        sample_stocks = sample_data_filled.index[:10].tolist()
        sample_stocks_features = sample_data_filled.loc[sample_stocks]
        
        feature_unique_counts = sample_stocks_features.nunique(axis=0)
        identical_features = (feature_unique_counts == 1).sum()
        print(f"      随机10个股票的特征分析:")
        print(f"        特征数: {len(feature_unique_counts)}")
        print(f"        完全相同的特征数: {identical_features}")
        print(f"        有变化的特征数: {(feature_unique_counts > 1).sum()}")
        
        if identical_features == len(feature_unique_counts):
            print(f"      ⚠️ [严重] 这10个股票的所有特征完全相同！")
            
            # 深入诊断：检查这些股票在填充前的特征情况
            if sample_data_raw is not None:
                sample_stocks_raw = sample_data_raw.loc[sample_stocks]
                raw_missing_count = sample_stocks_raw.isna().sum(axis=1)
                print(f"      [深入诊断] 这10个股票填充前的缺失情况:")
                print(f"        平均缺失特征数: {raw_missing_count.mean():.1f} / {len(sample_stocks_raw.columns)}")
                print(f"        全部缺失的股票数: {(raw_missing_count == len(sample_stocks_raw.columns)).sum()}")
                
                # 检查这些股票在填充前是否有任何非NaN值
                has_any_data = sample_stocks_raw.notna().any(axis=1).sum()
                print(f"        有至少一个非NaN值的股票数: {has_any_data} / {len(sample_stocks_raw)}")
                
                if has_any_data == 0:
                    print(f"      ⚠️ [根本问题] 这10个股票在填充前所有特征都是NaN！")
                    print(f"      这说明：这些股票的特征数据根本没有从数据源加载")
                    print(f"      可能原因：")
                    print(f"        1. QLib Alpha158 数据不完整（某些股票没有数据）")
                    print(f"        2. 数据日期/股票匹配问题")
                    print(f"        3. 特征计算失败（某些股票无法计算特征）")
                else:
                    print(f"      填充策略问题：即使有数据，填充后也完全相同")
    
    # 根据模型类型加载并预测
    # 关键：确保使用 X_test_filled.values，这会保持行的顺序，与 original_index 对应
    feature_names = None
    feature_list_path = None
    
    if model_type == "lightgbm":
        import lightgbm as lgb
        # 优先使用排序模型（lgbm_ranker），若不存在则回退到回归模型
        ranker_path = model_dir / "lgbm_ranker.txt"
        reg_path = model_dir / "lgbm_regression.txt"
        if ranker_path.exists():
            print("  [Model] Using LightGBM ranker (lgbm_ranker.txt) for predictions")
            model = lgb.Booster(model_file=str(ranker_path))
            feature_list_path = model_dir / "feature_list_ranker.json"
        elif reg_path.exists():
            print("  [Model] Using LightGBM regression model (lgbm_regression.txt) for predictions")
            model = lgb.Booster(model_file=str(reg_path))
            feature_list_path = model_dir / "feature_list_regression.json"
        else:
            raise ValueError(
                f"Model file not found: {ranker_path} or {reg_path}. "
                "Please run `python src/modeling.py --train` or `--train-regression` first."
            )
        feature_names = model.feature_name()
        # 如果有手动保存的特征列表，以它为准（可保持与训练阶段完全一致）
        if feature_list_path and feature_list_path.exists():
            try:
                with open(feature_list_path, "r") as f:
                    saved_feature_list = json.load(f)
                if isinstance(saved_feature_list, list) and saved_feature_list:
                    feature_names = saved_feature_list
                    print(f"  [Model] Loaded feature list from {feature_list_path}")
            except Exception as exc:
                print(f"  [Warn] Failed to load feature list from {feature_list_path}: {exc}")
    
    elif model_type == "catboost":
        from catboost import CatBoostRegressor
        model_path = model_dir / "catboost_regression.cbm"
        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}. Please run training first.")
        model = CatBoostRegressor()
        model.load_model(str(model_path))
        feature_names = model.feature_names_
    
    elif model_type == "xgboost":
        import xgboost as xgb
        model_path = model_dir / "xgboost_regression.model"
        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}. Please run training first.")
        model = xgb.Booster()
        model.load_model(str(model_path))
        feature_names = model.feature_names
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    # 对齐特征列顺序，与训练阶段保持一致
    if feature_names:
        missing_features = [feat for feat in feature_names if feat not in X_test_filled.columns]
        extra_features = [feat for feat in X_test_filled.columns if feat not in feature_names]
        
        if missing_features:
            print(f"  [Warn] {len(missing_features)} features missing in test data. Filling with 0: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
            for feat in missing_features:
                X_test_filled[feat] = 0.0
        if extra_features:
            print(f"  [Info] Dropping {len(extra_features)} extra features not used in training: {extra_features[:5]}{'...' if len(extra_features) > 5 else ''}")
            X_test_filled = X_test_filled.drop(columns=extra_features)
        
        X_test_filled = X_test_filled.reindex(columns=feature_names).fillna(0.0)
    
    if model_type == "lightgbm":
        pred_values = model.predict(X_test_filled.values)
    elif model_type == "catboost":
        pred_values = model.predict(X_test_filled.values)
    elif model_type == "xgboost":
        import xgboost as xgb
        dtest = xgb.DMatrix(X_test_filled.values)
        pred_values = model.predict(dtest)
        
    elif model_type == "catboost":
        from catboost import CatBoostRegressor
        model_path = model_dir / "catboost_regression.cbm"
        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}. Please run training first.")
        model = CatBoostRegressor()
        model.load_model(str(model_path))
        pred_values = model.predict(X_test_filled.values)
        
    elif model_type == "xgboost":
        import xgboost as xgb
        model_path = model_dir / "xgboost_regression.model"
        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}. Please run training first.")
        model = xgb.Booster()
        model.load_model(str(model_path))
        dtest = xgb.DMatrix(X_test_filled.values)
        pred_values = model.predict(dtest)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    # 关键修复：验证预测值和索引的长度匹配
    if len(pred_values) != len(original_index):
        raise ValueError(
            f"预测值数量 ({len(pred_values)}) 与索引数量 ({len(original_index)}) 不匹配！"
        )
    
    # 创建预测 Series，确保索引和值一一对应
    pred = pd.Series(pred_values, index=original_index, name="prediction")
    
    # 诊断：检查预测值的分布
    print(f"  [诊断] 预测值统计:")
    print(f"    预测值数量: {len(pred)}")
    print(f"    唯一值数: {pred.nunique()}")
    print(f"    标准差: {pred.std():.10f}")
    print(f"    值范围: [{pred.min():.10f}, {pred.max():.10f}]")
    
    # 检查某个日期的预测值分布
    if isinstance(pred.index, pd.MultiIndex):
        sample_date = pred.index.get_level_values(0)[0]
        sample_preds = pred.xs(sample_date, level=0)
        print(f"    示例日期 {sample_date} 的预测值统计:")
        print(f"      样本数: {len(sample_preds)}")
        print(f"      唯一值数: {sample_preds.nunique()}")
        print(f"      标准差: {sample_preds.std():.10f}")
        
        if sample_preds.nunique() == 1:
            print(f"      ⚠️ 警告：该日期所有股票的预测值相同！")
            print(f"      可能原因：特征数据有问题或模型失效")
    
    print(f"  Generated predictions: {len(pred)} samples")
    
    # 缓存预测结果，便于后续快速复用
    try:
        cache_path = model_dir / f"{model_type}_predictions.pkl"
        pred.to_pickle(cache_path)
        print(f"  [Cache] Saved predictions to {cache_path}")
    except Exception as exc:
        print(f"  [Warn] Failed to cache predictions: {exc}")
    
    return pred


def calculate_covariance(returns: pd.DataFrame, method: str = "ewma", lam: float = 0.94) -> pd.DataFrame:
    """
    计算协方差矩阵
    
    Args:
        returns: 收益率 DataFrame (date × ticker)
        method: 计算方法 ("ewma", "sample", "shrinkage")
        lam: EWMA 衰减因子
    
    Returns:
        协方差矩阵 DataFrame (ticker × ticker)
    """
    if method == "ewma":
        # 指数加权移动平均
        cov = returns.ewm(alpha=1-lam).cov().iloc[-len(returns.columns):]
        return cov
    elif method == "sample":
        # 样本协方差
        return returns.cov()
    elif method == "shrinkage":
        # Ledoit-Wolf 收缩
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(returns.fillna(0).values)
            cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
            return cov
        except ImportError:
            warnings.warn("sklearn not available, falling back to sample covariance")
            return returns.cov()
    else:
        raise ValueError(f"method `{method}` is not supported")


def run_optimize(cfg):
    """
    运行组合优化
    
    工作流程：
    1. 加载模型预测（作为信号）
    2. 加载历史收益率（用于计算协方差）
    3. 按日期循环：
       a. 选择策略类型（TopkDropout 或 PortfolioOptimizer）
       b. 生成目标权重
       c. 保存权重
    """
    print("[Optimizer] Loading data...")
    
    # 1. 加载预测（优先使用 LightGBM，如果不存在则尝试其他模型）
    model_priority = ["lightgbm", "catboost", "xgboost"]
    pred = None
    
    for model_type in model_priority:
        try:
            pred = load_predictions(cfg, model_type=model_type)
            print(f"  Loaded predictions from {model_type}: {len(pred)} samples")
            break
        except (ValueError, FileNotFoundError) as e:
            if model_type == model_priority[-1]:
                print(f"[ERROR] Failed to load predictions from all models: {e}")
                return
            continue
    
    if pred is None or len(pred) == 0:
        print("[ERROR] No predictions available")
        return
    
    # 2. 加载价格数据（用于计算收益率和协方差）
    prices = pd.read_parquet(cfg["paths"]["prices_parquet"])
    prices.index = pd.MultiIndex.from_tuples(prices.index, names=["date", "ticker"])
    prices = prices.sort_index()
    
    # 修复：去除重复索引（保留最后一个）
    if prices.index.duplicated().any():
        print(f"[Warning] Found {prices.index.duplicated().sum()} duplicate (date, ticker) entries, removing duplicates...")
        prices = prices[~prices.index.duplicated(keep='last')]
        print(f"[OK] Removed duplicates, remaining: {len(prices)} rows")
    
    returns = prices["Adj Close"].groupby("ticker").pct_change(fill_method=None).dropna()
    returns = returns.unstack("ticker")
    returns = returns.sort_index()
    
    # 3. 获取策略配置
    strategy_config = cfg.get("strategy", {})
    strategy_type = strategy_config.get("type", "topk_dropout")  # "topk_dropout" 或 "portfolio_optimizer"
    
    # 4. 初始化策略参数
    if strategy_type == "full_rebalance":
        topk = strategy_config.get("topk", 20)
        print(f"[Strategy] Full Rebalance: topk={topk} (每日全量换仓)")
    elif strategy_type == "topk_dropout":
        topk = strategy_config.get("topk", 50)
        n_drop = strategy_config.get("n_drop", 5)
        method_sell = strategy_config.get("method_sell", "bottom")
        method_buy = strategy_config.get("method_buy", "top")
        print(f"[Strategy] TopkDropoutStrategy: topk={topk}, n_drop={n_drop}")
    else:
        optimizer_method = strategy_config.get("optimizer_method", "inv")
        optimizer_params = strategy_config.get("optimizer_params", {})
        optimizer = PortfolioOptimizer(method=optimizer_method, **optimizer_params)
        print(f"[Strategy] PortfolioOptimizer: method={optimizer_method}")
    
    # 5. 检查是否使用QLib Label，如果是，需要shift预测以匹配交易日期
    # QLib Label: Ref($close, -2)/Ref($close, -1) - 1 = close[t+2] / close[t+1] - 1
    # 在日期T，模型预测的是T+2到T+1的收益率，应该用于T+1日的交易
    use_qlib_label = cfg.get("model", {}).get("use_qlib_label", False)
    if use_qlib_label:
        print("[警告] 使用QLib Label，需要将预测shift(-1)以匹配交易日期")
        print("  QLib Label在日期T预测的是T+2到T+1的收益，应该用于T+1的交易")
        print("  正在shift预测以修复时间对齐问题...")
        
        # 重新组织pred：T日的预测用于T+1的交易
        pred_dates = sorted(pred.index.get_level_values("date").unique())
        pred_shifted_list = []
        
        for i, date in enumerate(pred_dates):
            if i < len(pred_dates) - 1:
                # T日的预测用于T+1的交易
                next_date = pred_dates[i + 1]
                pred_day = pred.xs(date, level="date").copy()
                # 创建新的MultiIndex，将日期改为next_date
                new_index = pd.MultiIndex.from_tuples(
                    [(next_date, ticker) for ticker in pred_day.index],
                    names=["date", "ticker"]
                )
                pred_day.index = new_index
                pred_shifted_list.append(pred_day)
        
        if pred_shifted_list:
            pred = pd.concat(pred_shifted_list).sort_index()
            print(f"  Shift完成：原始预测日期数={len(pred_dates)}，shift后日期数={len(pred.index.get_level_values('date').unique())}")
        else:
            print("  [警告] 无法shift预测，保持原样（可能需要检查数据）")
    
    # 6. 按日期生成权重
    pred_by_date = pred.groupby(level="date")
    weights_dict = {}
    current_positions = pd.Series(dtype=float)
    
    print("[Optimizer] Generating portfolio weights...")
    diagnostic_stats = {"dates": 0, "pred_mean": [], "pred_std": [], "weights_sum": [], "positions": []}
    
    for date, pred_day in pred_by_date:
        pred_day = pred_day.droplevel("date")
        
        if len(pred_day) == 0:
            continue
        
        # 诊断：记录预测统计
        diagnostic_stats["dates"] += 1
        diagnostic_stats["pred_mean"].append(pred_day.mean())
        diagnostic_stats["pred_std"].append(pred_day.std())
        
        # 根据策略类型生成权重
        if strategy_type == "full_rebalance":
            # Full Rebalance Strategy - 每日全量换仓
            new_weights = full_rebalance_strategy(
                pred_day,
                current_positions,
                topk=topk,
            )
            
            # 诊断：记录权重统计（归一化前）
            diagnostic_stats["weights_sum"].append(new_weights.sum())
            diagnostic_stats["positions"].append((new_weights > 0).sum())
        elif strategy_type == "topk_dropout":
            # TopkDropoutStrategy
            new_weights = topk_dropout_strategy(
                pred_day,
                current_positions,
                topk=topk,
                n_drop=n_drop,
                method_sell=method_sell,
                method_buy=method_buy,
            )
            
            # 诊断：记录权重统计（归一化前）
            diagnostic_stats["weights_sum"].append(new_weights.sum())
            diagnostic_stats["positions"].append((new_weights > 0).sum())
        else:
            # PortfolioOptimizer
            # 计算协方差矩阵（使用历史窗口）
            lookback_days = strategy_config.get("lookback_days", 60)
            hist_returns = returns.loc[:date].tail(lookback_days)
            
            if len(hist_returns) < 30:
                # 数据不足，使用等权重
                new_weights = pd.Series(1.0 / len(pred_day), index=pred_day.index)
            else:
                # 计算协方差
                S = calculate_covariance(hist_returns, method=cfg.get("risk_model", {}).get("ewma_lambda", 0.94))
                
                # 对齐索引
                common_tickers = pred_day.index.intersection(S.index)
                if len(common_tickers) == 0:
                    new_weights = pd.Series(dtype=float)
                else:
                    S_aligned = S.loc[common_tickers, common_tickers]
                    r_aligned = pred_day.reindex(common_tickers)
                    w0_aligned = current_positions.reindex(common_tickers).fillna(0.0) if len(current_positions) > 0 else None
                    
                    # 优化
                    w_opt = optimizer(S_aligned.values, r=r_aligned.values, w0=w0_aligned.values if w0_aligned is not None else None)
                    new_weights = pd.Series(w_opt, index=common_tickers)
            
            # 扩展到所有股票
            new_weights = new_weights.reindex(pred_day.index, fill_value=0.0)
        
        # 归一化（对齐 QLib：确保权重和为1.0）
        if new_weights.sum() > 0:
            new_weights = new_weights / new_weights.sum()
        else:
            # 如果所有权重为0（不应该发生），使用等权重
            if len(pred_day) > 0:
                new_weights = pd.Series(1.0 / len(pred_day), index=pred_day.index)
                warnings.warn(f"No valid weights for {date}, using equal weights")
        
        weights_dict[str(date.date())] = new_weights.to_dict()
        current_positions = new_weights
        
        # 诊断：记录归一化后的统计（用于验证）
        if strategy_type == "topk_dropout" and diagnostic_stats["dates"] <= 5:
            print(f"  [Debug] {date}: positions={ (new_weights > 0).sum()}, weight_sum={new_weights.sum():.6f}, pred_range=[{pred_day.min():.4f}, {pred_day.max():.4f}]")
    
    # 6. 保存权重
    weights_df = pd.DataFrame(weights_dict).T.sort_index()
    weights_df.index = pd.to_datetime(weights_df.index)
    
    output_path = get_path(cfg["paths"]["portfolio_path"], OUTPUT_PORTFOLIOS_DIR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weights_df.to_parquet(output_path)
    
    print(f"[OK] Saved portfolio weights to {output_path}")
    print(f"  Date range: {weights_df.index.min()} to {weights_df.index.max()}")
    print(f"  Number of dates: {len(weights_df)}")
    print(f"  Number of stocks: {len(weights_df.columns)}")
    
    # 输出诊断统计
    if diagnostic_stats["dates"] > 0:
        print(f"\n[Diagnostic] Optimizer Statistics:")
        print(f"  Processed dates: {diagnostic_stats['dates']}")
        if diagnostic_stats["pred_mean"]:
            print(f"  Prediction mean (avg): {np.mean(diagnostic_stats['pred_mean']):.6f}")
            print(f"  Prediction std (avg): {np.mean(diagnostic_stats['pred_std']):.6f}")
        if diagnostic_stats["weights_sum"]:
            print(f"  Weights sum (avg): {np.mean(diagnostic_stats['weights_sum']):.6f} (should be ~1.0)")
            print(f"  Weights sum (min): {np.min(diagnostic_stats['weights_sum']):.6f}")
            print(f"  Weights sum (max): {np.max(diagnostic_stats['weights_sum']):.6f}")
        if diagnostic_stats["positions"]:
            print(f"  Positions count (avg): {np.mean(diagnostic_stats['positions']):.1f}")
            print(f"  Positions count (min): {np.min(diagnostic_stats['positions'])}")
            print(f"  Positions count (max): {np.max(diagnostic_stats['positions'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="Run portfolio optimization")
    args = parser.parse_args()
    
    cfg = load_settings()
    
    if args.optimize:
        run_optimize(cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()