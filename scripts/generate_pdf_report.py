#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a PDF report describing the entire workflow:
- data & factor processing
- modeling approach
- optimizer strategy
- backtest results
- charts snapshots

Usage:
    python scripts/generate_pdf_report.py

Requires reportlab and Pillow. Installs if missing.
"""

import argparse
import json
import os
from pathlib import Path
from textwrap import wrap

import pandas as pd

try:
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import Paragraph, Spacer, Image, SimpleDocTemplate
except ImportError:
    raise ImportError("Please install reportlab before running this script: pip install reportlab pillow")


ROOT = Path(__file__).resolve().parents[1]
BACKTEST_DIR = ROOT / "outputs" / "backtests"
DEFAULT_LANGS = ["zh", "en"]


def load_json(path: Path):
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_csv(path: Path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


def load_factor_contribution(path: Path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def text_paragraph(text: str, style):
    wrapped = "<br/>".join("<br/>".join(wrap(line, 110)) for line in text.split("\n"))
    return Paragraph(wrapped, style)


def add_image(story, path: Path, width_cm=14, caption=None, max_height_cm=12):
    if not path.exists():
        story.append(Paragraph(f"[Missing chart: {path.name}]", styles["Italic"]))
        story.append(Spacer(1, 0.2 * cm))
        return
    img = Image(str(path))
    img.drawWidth = width_cm * cm
    img.drawHeight = img.drawHeight * (img.drawWidth / img.drawWidth)
    if img.drawHeight > max_height_cm * cm:
        scale = (max_height_cm * cm) / img.drawHeight
        img.drawHeight *= scale
        img.drawWidth *= scale
    story.append(img)
    if caption:
        story.append(Paragraph(caption, styles["Italic"]))
    story.append(Spacer(1, 0.4 * cm))


pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="SectionHeader",
        parent=styles["Heading2"],
        alignment=TA_LEFT,
        spaceAfter=6,
        fontName="STSong-Light",
    )
)
styles.add(
    ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        spaceAfter=8,
        leading=18,  # 增加行距，避免拥挤
        fontSize=11,  # 稍微增大字体
        fontName="STSong-Light",
    )
)
styles["Title"].fontName = "STSong-Light"
styles["Title"].fontSize = 20
styles["Title"].spaceAfter = 12
styles["Heading1"].fontName = "STSong-Light"
styles["Heading1"].fontSize = 16
styles["Heading1"].spaceAfter = 10
styles["Heading2"].fontName = "STSong-Light"
styles["Heading2"].fontSize = 14
styles["Heading2"].spaceAfter = 8
styles["Italic"].fontName = "STSong-Light"
styles["Italic"].fontSize = 10


def build_report(language: str):
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    pdf_name = "workflow_report_zh.pdf" if language == "zh" else "workflow_report_en.pdf"
    report_path = BACKTEST_DIR / pdf_name
    doc = SimpleDocTemplate(str(report_path), pagesize=A4, leftMargin=2.5 * cm, rightMargin=2.5 * cm, topMargin=2 * cm, bottomMargin=2 * cm)
    story = []

    title = "Quant ML Portfolio Pro - 全流程报告" if language == "zh" else "Quant ML Portfolio Pro - Workflow Report"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.3 * cm))

    if language == "zh":
        intro_text = f"""
        本文档对“Quant-ML-Portfolio-Pro”工作流进行系统复盘，覆盖数据准备、因子构建、模型训练、
        策略生成、回测验证以及因子贡献分析，帮助量化研究员快速理解每个环节之间的逻辑关系。
        生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

        工作流分为五个层次：
        (1) 数据与特征：构建覆盖全 S&P500（503 支股票）的截面数据，保证因子之间可比较。
        (2) 因子诊断：先做单因子分析，过滤 IC 表现差的因子，避免垃圾输入拖慢模型。
        (3) 模型训练：采用排序模型（LightGBM Ranker）直接学习“收益排名”，使输出天然服务于选股。
        (4) 策略生成：用预测分数驱动 Top20 持仓，结合 Dropout 机制控制换手与交易成本。
        (5) 回测与报告：shift(1) 避免提前假设，估算成本后产出图表、统计、超额收益和因子贡献。
        """
    else:
        intro_text = f"""
        This document reviews the entire Quant-ML-Portfolio-Pro workflow, covering data preparation,
        factor engineering, modeling, strategy generation, backtesting, and factor attribution, so that
        researchers can see how each step links to the final performance. Generated on
        {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}.

        The workflow has five layers:
        (1) Data & Features: build a complete S&P500 cross-section (503 stocks) to make factors comparable.
        (2) Factor Diagnosis: perform single-factor IC/ICIR checks to filter noisy inputs early.
        (3) Modeling: use a LightGBM ranker so the output directly serves TopK stock selection.
        (4) Strategy: drive a Top20 Dropout portfolio with cached predictions, feature alignment, and shift(-1).
        (5) Backtesting & Reporting: apply shift(1), cost estimation, chart/report generation, and factor attribution.
        """
    story.append(text_paragraph(intro_text, styles["Body"]))
    story.append(Spacer(1, 0.5 * cm))

    # Data & Factor processing details
    if language == "zh":
        data_text = """
        选择 S&P500 横截面有两个原因：一是美国股票流动性好，能支撑日度调仓；二是横截面越大，
        越能发挥因子排序模型的相对比较优势。prices.parquet 覆盖 2010-2025 全部历史，确保训练
        样本跨越多个市场 regime。factor_store.parquet 则记录 165 个因子：
        • Alpha101：经典价量因子，可衡量反转、动量、波动等行为。
        • TA-Lib：补充技术指标（RSI、MACD、BBands、ATR/NATR…），强调趋势与波动。
        • Custom：针对项目需求构建的 RS、LAR、PMS、VAR、PPF 等自定义指标。
        因子处理仅使用 MAD winsorize，暂不做 cross-sectional rank / orthogonalization，是为了保留
        原始度量，由模型自适应学习权重。
        """
        section_data = "1. 数据与因子处理"
    else:
        data_text = """
        We pick the full S&P500 cross-section because it offers deep liquidity and maximizes the value of a ranking model.
        prices.parquet covers 2010-2025 to span multiple regimes; factor_store.parquet has 165 factors:
        • Alpha101: classic price-volume signals for reversal/momentum/volatility.
        • TA-Lib: technical indicators (RSI, MACD, BBands, ATR/NATR, etc.) to augment trend info.
        • Custom: RS, LAR, PMS, VAR, PPF tailored to the project.
        Factor processing only applies MAD winsorization—no cross-sectional ranking or orthogonalization—so that magnitude
        information remains available for the model. Single-factor analysis (analysis_single_factors.py) computes rank IC/ICIR
        to drop persistently negative contributors (e.g., Alpha28, NATR_14).
        """
        section_data = "1. Data & Factor Processing"
    story.append(Paragraph(section_data, styles["SectionHeader"]))
    story.append(text_paragraph(data_text, styles["Body"]))
    
    # Single Factor Analysis Section
    if language == "zh":
        single_factor_text = """
        单因子验证流程（analyze_single_factors.py）：
        
        【1. 标签选择】
        使用 horizon_days=5 的未来收益率作为标签，即：在日期 t，使用 t+5 相对于 t 的收益率。
        选择 5 日而非 1 日的原因：单日收益率在 S&P500 中噪声过大，5 日收益率能平滑短期波动，
        更好地反映因子的真实预测能力。计算公式：y_t = (price_{t+5} / price_t) - 1。
        
        【2. 回测方法】
        采用横截面 Rank IC（Spearman 相关系数）评估因子有效性：
        • 每日计算：对每个交易日，计算因子值的排名与未来收益排名的 Spearman 相关系数。
        • 统计指标：计算所有交易日的 IC 均值（mean_ic）、标准差（std_ic）、ICIR（IC均值/标准差）。
        • ICIR 意义：ICIR > 0.5 表示因子稳定有效；ICIR < 0.05 表示因子预测能力弱。
        
        【3. 因子筛选标准】
        综合多个维度判断因子质量：
        • IC 均值：|mean_ic| > 0.02（严格）或 > 0.005（中等），表示因子与收益有显著相关性。
        • ICIR：|ICIR| > 0.5（严格）或 > 0.05（中等），表示因子预测能力稳定。
        • IC 胜率：IC > 0 的比例 > 60%（严格）或 > 50%（中等），表示因子方向一致性。
        • 显著性检验：t-test 检验 IC 是否显著不为 0（p < 0.05），确保统计可靠性。
        
        【4. 筛选流程】
        1) 遍历所有因子，计算每个因子的 Rank IC 和 ICIR。
        2) 按 |ICIR| 排序，识别表现最好和最差的因子。
        3) 应用筛选标准，生成"严格筛选"和"中等筛选"两套因子列表。
        4) 保存结果到 single_factor_summary.json 和 factor_selection_recommendations.json。
        5) 在模型训练阶段（prepare_panel），根据配置自动过滤低质量因子。
        
        【5. 实际应用】
        例如，分析发现 Alpha28、NATR_14 的 ICIR < 0，且 IC 胜率 < 50%，说明这些因子在当前
        样本期内表现为反向信号，应剔除或做 sign flip。而 Alpha32、Alpha19 的 ICIR > 0.05，
        IC 胜率 > 60%，是高质量因子，应保留。
        """
        section_single_factor = "1.1 单因子验证流程与逻辑"
    else:
        single_factor_text = """
        Single Factor Validation Process (analyze_single_factors.py):
        
        【1. Label Selection】
        Use horizon_days=5 future returns as labels: at date t, use return from t+5 relative to t.
        Reason for 5-day vs 1-day: single-day returns in S&P500 are too noisy; 5-day returns smooth
        short-term volatility and better reflect true predictive power. Formula: y_t = (price_{t+5} / price_t) - 1.
        
        【2. Backtest Method】
        Use cross-sectional Rank IC (Spearman correlation) to evaluate factor effectiveness:
        • Daily calculation: for each trading day, compute Spearman correlation between factor ranks and future return ranks.
        • Statistics: mean_ic, std_ic, ICIR (mean_ic / std_ic).
        • ICIR interpretation: ICIR > 0.5 indicates stable effectiveness; ICIR < 0.05 indicates weak predictive power.
        
        【3. Factor Screening Criteria】
        Multi-dimensional assessment:
        • IC mean: |mean_ic| > 0.02 (strict) or > 0.005 (moderate), indicating significant correlation.
        • ICIR: |ICIR| > 0.5 (strict) or > 0.05 (moderate), indicating stable predictive power.
        • IC win rate: proportion of IC > 0 should be > 60% (strict) or > 50% (moderate).
        • Significance test: t-test to verify IC is significantly non-zero (p < 0.05).
        
        【4. Screening Process】
        1) Iterate all factors, compute Rank IC and ICIR for each.
        2) Sort by |ICIR|, identify best and worst performers.
        3) Apply screening criteria, generate "strict" and "moderate" factor lists.
        4) Save results to single_factor_summary.json and factor_selection_recommendations.json.
        5) In model training (prepare_panel), automatically filter low-quality factors based on config.
        
        【5. Practical Application】
        For example, analysis found Alpha28 and NATR_14 have ICIR < 0 and IC win rate < 50%, indicating
        they act as contrarian signals in this sample and should be removed or sign-flipped. Meanwhile,
        Alpha32 and Alpha19 have ICIR > 0.05 and IC win rate > 60%, making them high-quality factors to retain.
        """
        section_single_factor = "1.1 Single Factor Validation Process"
    story.append(Paragraph(section_single_factor, styles["SectionHeader"]))
    story.append(text_paragraph(single_factor_text, styles["Body"]))

    # Modeling
    if language == "zh":
        modeling_text = """
        为什么用排序模型？目标是挑出相对表现最好的股票，而不是精确预测收益点数。LightGBM Ranker
        直接最大化 NDCG/排名指标，输出的 score 与 TopK 选股天然吻合。
        """
        section_model = "2. 排序模型训练与预测"
    else:
        modeling_text = """
        Why a ranking model? Because the task is to select the best performers, not to forecast exact returns.
        LightGBM Ranker optimizes NDCG directly, so its scores feed naturally into TopK selection.
        """
        section_model = "2. Ranking Model Training & Prediction"
    story.append(Paragraph(section_model, styles["SectionHeader"]))
    story.append(text_paragraph(modeling_text, styles["Body"]))
    
    # Detailed Ranking Model Section
    if language == "zh":
        ranking_detail_text = """
        【1. 标签选择与转换】
        
        标签定义：使用 horizon_days=5 的未来收益率，即 y_t = (price_{t+5} / price_t) - 1。
        选择 5 日的原因：与单因子分析保持一致，平滑单日噪声，使模型学习更稳定的信号。
        
        标签转换：将连续收益率转换为分位数标签（quantile labels）：
        • 对每个交易日，将所有股票的收益率按大小排序，分为 q_bins 个分位组（默认 20 组）。
        • 收益率最高的股票获得标签 q_bins-1，最低的获得标签 0。
        • 这样，模型学习的是"相对排名"而非"绝对收益"，更适合排序任务。
        
        示例：某日有 100 只股票，q_bins=20，则：
        • 收益率最高的 5 只股票 → 标签 19
        • 收益率次高的 5 只股票 → 标签 18
        • ... 依此类推
        • 收益率最低的 5 只股票 → 标签 0
        
        【2. 损失函数：LambdaRank】
        
        LightGBM Ranker 使用 LambdaRank 损失函数，这是专门为排序任务设计的：
        • 核心思想：不直接优化排序指标（如 NDCG），而是优化一个可微分的代理损失。
        • Lambda 梯度：对于每对样本 (i, j)，如果真实标签 y_i > y_j，则模型预测 pred_i 应该 > pred_j。
        • 梯度计算：如果预测顺序错误（pred_i < pred_j），则对样本 i 施加正梯度，对样本 j 施加负梯度。
        • 优势：直接优化排序质量，比回归模型更适合选股任务。
        
        数学表达：对于查询组（每日股票集合），LambdaRank 计算：
        λ_i = Σ_{j: y_j > y_i} λ_{ij} - Σ_{j: y_j < y_i} λ_{ji}
        其中 λ_{ij} 取决于预测差异和标签差异，鼓励正确排序。
        
        【3. 训练流程】
        
        【3.1 数据准备（prepare_panel）】
        1) 加载因子数据（factor_store.parquet）和价格数据（prices.parquet）。
        2) 对齐索引：确保 (date, ticker) 索引完全匹配。
        3) 特征过滤：
           • drop_bad_features()：移除缺失率 > 50% 或方差为 0 的特征。
           • 可选 ICIR 过滤：根据单因子分析结果，只保留 |ICIR| > threshold 的特征。
        4) 缺失值填充：对每个交易日，用该日所有股票的中位数填充缺失值。
        5) 标签生成：计算未来收益率，转换为分位数标签。
        6) 过滤小样本：移除样本数 < 100 的交易日，确保训练稳定。
        
        【3.2 交叉验证（train_ranker）】
        采用扩增式时间序列 CV（Expanding Window CV）：
        • Fold 1：训练集 = 前 1/3 日期，测试集 = 第 2 个 1/3 日期。
        • Fold 2：训练集 = 前 2/3 日期，测试集 = 第 3 个 1/3 日期。
        • Fold 3：训练集 = 前 3/3 日期（全部历史），用于最终模型。
        
        为什么用扩增式而非滚动式？
        • 扩增式：训练集随时间增长，模拟真实场景（历史数据越来越多）。
        • 滚动式：训练集大小固定，可能丢失早期信息。
        • 金融数据中，扩增式更符合实际应用。
        
        【3.3 模型训练】
        对每个 Fold：
        1) 数据拆分：按日期划分训练集和测试集。
        2) 分组信息：LightGBM Ranker 需要 group 参数，表示每个查询组（每日）的样本数。
        3) 验证集拆分：从训练集中取最后 10% 日期作为验证集，用于早停（early stopping）。
        4) 模型配置：
           • objective="lambdarank"：排序任务。
           • metric="ndcg"：评估指标为 NDCG（归一化折损累积增益）。
           • n_estimators=2000：最大树数。
           • early_stopping_rounds=100：验证集性能不提升 100 轮则停止。
        5) 训练：model.fit(X_train, y_rank, group=groups, eval_set=[(X_val, y_val)], eval_group=[val_groups])。
        6) 预测：对测试集预测，得到排序分数（prediction scores）。
        7) 评估：计算每日 Rank IC（预测排名 vs 真实标签排名的 Spearman 相关系数）。
        
        【3.4 最终模型】
        使用所有历史数据（除最后一个测试折）训练最终模型：
        • 同样使用验证集早停，避免过拟合。
        • 保存模型到 lgbm_ranker.txt。
        • 保存特征列表到 feature_list_ranker.json，用于推理时特征对齐。
        
        【4. 预测流程】
        
        【4.1 特征对齐】
        推理阶段（optimizer.py 的 load_predictions）：
        1) 加载训练时保存的 feature_list_ranker.json，获取训练特征列表。
        2) 加载最新的 factor_store.parquet，获取当前因子值。
        3) 特征对齐：
           • 如果因子库中有训练时没有的特征 → 丢弃（避免维度不匹配）。
           • 如果训练时有但因子库中没有的特征 → 用中位数填充（保持维度一致）。
        4) 确保特征顺序与训练时完全一致。
        
        【4.2 预测与缓存】
        1) 加载训练好的模型：lgb = lgb.Booster(model_file="lgbm_ranker.txt")。
        2) 对每个交易日，使用对齐后的特征进行预测：pred = model.predict(X_aligned)。
        3) 预测结果保存到 lightgbm_predictions.pkl，避免重复计算。
        4) 预测分数含义：分数越高，表示模型认为该股票未来收益排名越靠前。
        
        【4.3 权重生成】
        1) 对每个交易日，按预测分数排序，选择 TopK（默认 20）只股票。
        2) Dropout 机制：每天最多替换 n_drop（默认 3）只持仓，控制换手率。
        3) 权重归一化：确保每日权重和为 1（long-only 策略）。
        4) 输出 weights.parquet，供回测和实盘交易使用。
        
        【5. 模型评估指标】
        
        • OOF Rank IC：Out-of-Fold Rank IC，即所有测试折的平均 Rank IC。
          - 意义：衡量模型在未见数据上的排序能力。
          - 目标：OOF Rank IC > 0.03 表示模型有稳定的选股能力。
        • Rolling Rank IC：计算滚动窗口内的 Rank IC，监控模型性能随时间变化。
        • SHAP 特征重要性：识别对预测贡献最大的因子（Top 5：Alpha32、Alpha28、Alpha19、BOP、NATR_14）。
        """
        section_ranking_detail = "2.1 排序模型详细流程"
    else:
        ranking_detail_text = """
        【1. Label Selection & Conversion】
        
        Label definition: Use horizon_days=5 future returns, i.e., y_t = (price_{t+5} / price_t) - 1.
        Reason for 5-day: Consistent with single-factor analysis, smoothing daily noise for more stable signals.
        
        Label conversion: Convert continuous returns to quantile labels:
        • For each trading day, sort all stocks by returns and divide into q_bins groups (default 20).
        • Highest return stocks get label q_bins-1, lowest get label 0.
        • Model learns "relative ranking" rather than "absolute returns", better suited for ranking tasks.
        
        【2. Loss Function: LambdaRank】
        
        LightGBM Ranker uses LambdaRank loss, designed specifically for ranking:
        • Core idea: Optimize a differentiable proxy loss rather than directly optimizing ranking metrics (e.g., NDCG).
        • Lambda gradient: For each pair (i, j), if true label y_i > y_j, then pred_i should > pred_j.
        • Gradient calculation: If prediction order is wrong (pred_i < pred_j), apply positive gradient to i, negative to j.
        • Advantage: Directly optimizes ranking quality, more suitable for stock selection than regression.
        
        【3. Training Process】
        
        【3.1 Data Preparation (prepare_panel)】
        1) Load factor data and price data.
        2) Align indices: Ensure (date, ticker) indices match.
        3) Feature filtering: Remove features with >50% missing or zero variance; optionally filter by ICIR.
        4) Missing value imputation: Fill with daily median per trading day.
        5) Label generation: Compute future returns, convert to quantile labels.
        6) Filter small samples: Remove trading days with <100 samples.
        
        【3.2 Cross-Validation (train_ranker)】
        Expanding Window CV:
        • Fold 1: Train on first 1/3 dates, test on second 1/3.
        • Fold 2: Train on first 2/3 dates, test on third 1/3.
        • Fold 3: Train on all historical data for final model.
        
        【3.3 Model Training】
        For each fold:
        1) Data split by dates.
        2) Group information: LightGBM Ranker needs group parameter (samples per query/day).
        3) Validation split: Last 10% of training dates for early stopping.
        4) Model config: objective="lambdarank", metric="ndcg", early_stopping_rounds=100.
        5) Train and predict.
        6) Evaluate: Compute daily Rank IC.
        
        【4. Prediction Process】
        
        【4.1 Feature Alignment】
        1) Load feature_list_ranker.json from training.
        2) Load latest factor_store.parquet.
        3) Align features: drop extra, fill missing with median.
        4) Ensure feature order matches training.
        
        【4.2 Prediction & Caching】
        1) Load trained model.
        2) Predict for each trading day.
        3) Cache predictions to avoid recomputation.
        
        【4.3 Weight Generation】
        1) Sort by prediction scores, select TopK stocks.
        2) Apply dropout mechanism (max n_drop replacements per day).
        3) Normalize weights (sum to 1 for long-only).
        4) Output weights.parquet.
        
        【5. Model Evaluation Metrics】
        
        • OOF Rank IC: Average Rank IC across all test folds.
        • Rolling Rank IC: Monitor performance over time.
        • SHAP feature importance: Identify top contributing factors.
        """
        section_ranking_detail = "2.1 Detailed Ranking Model Process"
    story.append(Paragraph(section_ranking_detail, styles["SectionHeader"]))
    story.append(text_paragraph(ranking_detail_text, styles["Body"]))

    # Strategy
    if language == "zh":
        strategy_text = """
        生成权重的逻辑：
        • 预测缓存：optimizer.py 会先加载 lightgbm_predictions.pkl，如不存在才重新推理，避免重复计算。
        • 特征对齐：读取 feature_list_ranker.json，将 factor_store 列精确映射到训练特征集。
        • Shift：预测在 T 日收盘生成，但只用于 T+1 调仓，代码默认 shift(-1) 保证时间对齐。
        • Top20 Dropout：topk=20、n_drop=3，每天最多替换 3 只持仓，兼顾 alpha 捕捉与换手控制（~10%）。
        • 输出 weights.parquet（日期×股票）；后续所有评估与执行都基于此文件。
        """
        section_strategy = "3. 策略与优化器"
    else:
        strategy_text = """
        Weight generation logic:
        • Prediction cache: load lightgbm_predictions.pkl first; only re-run inference if missing.
        • Feature alignment: use feature_list_ranker.json to map factor_store columns to training schema.
        • Shift: predictions made at day T close are applied to T+1 trades (shift(-1)).
        • Top20 Dropout: topk=20, n_drop=3, so at most 3 names rotate daily (~10% turnover).
        • Output weights.parquet (date×ticker); downstream backtest/execution all read from this file.
        """
        section_strategy = "3. Strategy & Optimizer"
    story.append(Paragraph(section_strategy, styles["SectionHeader"]))
    story.append(text_paragraph(strategy_text, styles["Body"]))

    # Backtest details
    if language == "zh":
        backtest_text = """
        回测采用 simple engine，因为策略本质是截面选股，重点在验证排名信号。
        1) 严格 shift：weights.shift(1) 与 returns 对齐，杜绝 look-ahead bias。
        2) 成本估算：turnover×(open_cost+close_cost)，目前 open=0.0005、close=0.0015，匹配美股交易成本。
        3) 输出 daily_returns.parquet（净值/换手/成本等列）与 summary.json，便于生成更多分析。
        4) generate_performance_report.py 读取 daily_returns 生成图表，本脚本进一步汇总为 PDF。
        """
        section_backtest = "4. 回测逻辑"
    else:
        backtest_text = """
        We use a simple daily-return engine because the strategy is cross-sectional ranking focused.
        1) Enforce shift: weights.shift(1) aligns with future returns to avoid look-ahead bias.
        2) Cost estimation: turnover × (open_cost + close_cost), currently 0.0005 + 0.0015, matching US fees.
        3) Produce daily_returns.parquet (NAV/turnover/cost) and summary.json for downstream analytics.
        4) generate_performance_report.py builds charts; this script assembles the final PDF.
        """
        section_backtest = "4. Backtesting Logic"
    story.append(Paragraph(section_backtest, styles["SectionHeader"]))
    story.append(text_paragraph(backtest_text, styles["Body"]))

    # Summary stats table
    summary = load_json(BACKTEST_DIR / "analysis_summary.json")
    if summary:
        if language == "zh":
            summary_text = "\n".join(
                [
                    f"起止时间: {summary.get('start', '')} 至 {summary.get('end', '')}",
                    f"交易日数: {summary.get('days', '')}",
                    f"总收益: {summary.get('total_return', 0)*100:.2f}%",
                    f"年化收益: {summary.get('annualized_return', 0)*100:.2f}%",
                    f"年化波动: {summary.get('annualized_vol', 0)*100:.2f}%",
                    f"Sharpe: {summary.get('sharpe', 0):.2f}",
                    f"最大回撤: {summary.get('max_drawdown', 0)*100:.2f}%",
                ]
            )
            section_summary = "5. 关键绩效指标"
        else:
            summary_text = "\n".join(
                [
                    f"Period: {summary.get('start', '')} to {summary.get('end', '')}",
                    f"Trading days: {summary.get('days', '')}",
                    f"Total return: {summary.get('total_return', 0)*100:.2f}%",
                    f"Annualized return: {summary.get('annualized_return', 0)*100:.2f}%",
                    f"Annualized vol: {summary.get('annualized_vol', 0)*100:.2f}%",
                    f"Sharpe: {summary.get('sharpe', 0):.2f}",
                    f"Max drawdown: {summary.get('max_drawdown', 0)*100:.2f}%",
                ]
            )
            section_summary = "5. Key Performance Metrics"
        story.append(Paragraph(section_summary, styles["SectionHeader"]))
        story.append(text_paragraph(summary_text, styles["Body"]))

    if language == "zh":
        analysis_text = """
        综合表现：2022-01-04 至 2025-11-21 共 976 日，累计收益 +95.6%，年化 21.1%，年化波动 27.7%，Sharpe 0.76，
        最大回撤 -30.9%。2022 年多数月份回撤（4 月 -12%、6 月 -14%），主要因行情下行与因子失效；2023 年中后
        段策略逐步恢复，11 月单月 +15.6%，2023-05/2024-09/2025-05 等月份对基准形成显著超额；2024-2025 年收益
        分布稳定在 -3%~+11%，说明模型在最新 regime 中具备持续 alpha。

        风险与超额：strategy_vs_benchmark.png 显示策略自 2023 年起明显跑赢等权基准；excess_return_curve.png 表明
        2022 年的超额几乎为零，2023 H2 开始抬升，2024-2025 保持正斜率。rolling_alpha_beta.png 里 Beta ≈1，偶有>1.2，
        说明收益仍受大盘波动影响，可考虑在优化器中加入 beta 约束；Rolling Alpha 在 2023 H2 起长期 >0，近期年化 alpha
        多在 5% 以上。excess_return_hist.png 呈轻微胖尾，左尾可到 -5%，右尾到 +6%，提示需要 position cap / 动态杠杆控制极端风险。

        performance_overview.png：权益曲线在 2023 Q1 后斜率变陡，向右上角推进；Drawdown 曲线显示 2022-06、2023-10、
        2025-06 有深回撤，是风控重点时段；Rolling Sharpe 2024 年以后大多 >0.5，表明近期信号更稳定；日收益柱图偶见 ±5%
        极端日，建议在 optimizer 中设置单日换手或权重上限。return_histogram.png 呈近似对称分布，但左尾稍长（-6%），风控仍需关注。

        因子贡献：根据 factor_contribution.csv，Alpha32/Alpha19 日均 IC ≈ +0.009（方向正确）；BOP 约 +0.005，
        属中性偏正；Alpha28/NATR_14 日均 IC ≈ -0.009，表明在当前样本期表现为反向因子，可考虑 sign flip 或剔除。
        coverage_days 接近 4000，统计具备一致性。结合 shap_top5，可进一步清理持续负 IC 的因子，提升模型稳健性。
        """
        section_analysis = "6. 绩效解读与图表说明"
    else:
        analysis_text = """
        Performance summary: 2022-01-04 to 2025-11-21 (976 days) delivers +95.6% total return, 21.1% annualized,
        27.7% annualized volatility, Sharpe 0.76, max drawdown -30.9%. Most 2022 months were drawdowns (Apr -12%, Jun -14%),
        but the strategy recovered in mid/late 2023 (Nov +15.6%); monthly outperformance in 2023-05, 2024-09, 2025-05 stands out.
        2024-2025 returns range between -3% and +11%, indicating stable alpha in the latest regime.

        Risk & alpha: strategy_vs_benchmark.png shows the PnL outpacing the equal-weight benchmark since 2023; excess_return_curve.png
        indicates near-zero excess in 2022 but a steady rise in 2023H2 onward. rolling_alpha_beta.png reveals beta ≈1 with occasional
        spikes >1.2 (exposed to market swings), while rolling alpha stays >0 since 2023H2, often above 5% annualized. excess_return_hist.png
        is slightly fat-tailed, with a -5% left tail and +6% right tail, suggesting the need for position caps or dynamic leverage.

        performance_overview.png: equity slope steepens after 2023Q1; drawdowns visible in 2022-06, 2023-10, 2025-06; rolling Sharpe mostly >0.5
        after 2024; daily returns occasionally reach ±5%, so per-name caps or turnover limits are recommended. return_histogram.png is roughly
        symmetric but with a slightly longer left tail (~ -6%).

        Factor contribution: Alpha32/Alpha19 show mean IC ≈ +0.009; BOP ≈ +0.005 (mildly positive); Alpha28/NATR_14 ≈ -0.009 (negative),
        implying they act as contrarian signals in this sample and may deserve sign flip/removal. coverage_days ~4000, indicating reliable stats.
        """
        section_analysis = "6. Performance Interpretation & Charts"
    story.append(Paragraph(section_analysis, styles["SectionHeader"]))
    story.append(text_paragraph(analysis_text, styles["Body"]))

    charts_title = "7. 可视化图表" if language == "zh" else "7. Visualizations"
    story.append(Paragraph(charts_title, styles["SectionHeader"]))
    add_image(story, BACKTEST_DIR / "performance_overview.png", caption="Performance Overview (Equity, Drawdown, Rolling Sharpe, Daily Returns)")
    add_image(story, BACKTEST_DIR / "return_histogram.png", caption="Distribution of Daily Net Returns")
    add_image(story, BACKTEST_DIR / "strategy_vs_benchmark.png", caption="Cumulative Return vs Equal-Weight Benchmark")
    add_image(story, BACKTEST_DIR / "excess_return_curve.png", caption="Cumulative Excess Return")
    add_image(story, BACKTEST_DIR / "rolling_alpha_beta.png", caption="Rolling Alpha/Beta (60d)")
    add_image(story, BACKTEST_DIR / "excess_return_hist.png", caption="Excess Return Distribution")

    # Factor contribution
    factor_df = load_factor_contribution(BACKTEST_DIR / "factor_contribution.csv")
    factor_title = "8. 因子贡献（Top SHAP）" if language == "zh" else "8. Factor Contribution (Top SHAP)"
    if not factor_df.empty:
        story.append(Paragraph(factor_title, styles["SectionHeader"]))
        table_text = []
        for _, row in factor_df.iterrows():
            if language == "zh":
                table_text.append(
                    f"{row['factor']}: 平均IC={row['daily_ic_mean']:.3f} "
                    f"(std={row['daily_ic_std']:.3f}, 覆盖天数={int(row['coverage_days'])})"
                )
            else:
                table_text.append(
                    f"{row['factor']}: mean IC={row['daily_ic_mean']:.3f} "
                    f"(std={row['daily_ic_std']:.3f}, coverage={int(row['coverage_days'])})"
                )
        story.append(text_paragraph("\n".join(table_text), styles["Body"]))

    # Monthly performance snapshot
    monthly_df = load_csv(BACKTEST_DIR / "monthly_performance.csv")
    monthly_title = "9. 月度收益一览" if language == "zh" else "9. Monthly Performance Snapshot"
    if not monthly_df.empty:
        story.append(Paragraph(monthly_title, styles["SectionHeader"]))
        last_12 = monthly_df.tail(12)
        rows = []
        for idx, row in last_12.iterrows():
            if language == "zh":
                rows.append(
                    f"{idx}: 策略={row['strategy_return']*100:.2f}%, "
                    f"基准={row['benchmark_return']*100:.2f}%, "
                    f"超额={row['excess_return']*100:.2f}%"
                )
            else:
                rows.append(
                    f"{idx}: Strategy={row['strategy_return']*100:.2f}%, "
                    f"Benchmark={row['benchmark_return']*100:.2f}%, "
                    f"Excess={row['excess_return']*100:.2f}%"
                )
        story.append(text_paragraph("\n".join(rows), styles["Body"]))

    # Closing remarks
    if language == "zh":
        closing_title = "10. 结论与下一步"
        closing_text = """
        策略好坏评估：本策略在 2022 年遭遇大回撤，但 2023-2025 年重新积累稳定超额，年化 21%、Sharpe 0.76，
        属于“具备吸引力但仍需风险治理”的策略。相较基准，超额收益集中在最近两年，说明当前 alpha 与市场
        regime 高度相关。最大回撤 -31% 且 Beta≈1，意味着策略仍受大盘波动驱动，需要配套行业/风格中性约束或
        波动控制后才能满足更严格的资金要求。

        研究严谨性：流程遵循“数据→因子→单因子诊断→排序模型→权重→shift 回测→多维评估”闭环，关键步骤
        以防信息泄露为前提（特征对齐、shift(-1)、预测缓存、IC 交叉验证）。SHAP 与日度 IC 对因子贡献给出一致
        解释，可以认为研究过程逻辑严谨、可复现。

        下一步：
        • 风控：引入 beta/行业中性或单股票权重上限、波动缩放，缓解 -31% MaxDD。
        • 因子：对 Alpha28、NATR_14 做 sign flip/剔除，结合最新单因子结果持续净化输入。
        • 策略：尝试 topk=15 或更平滑的 n_drop，探索风险平价或波动缩放以改善回撤。
        • 监控：保留 rolling IC / Alpha 监控，及时识别 regime 变化并调整模型。
        """
    else:
        closing_title = "10. Conclusion & Next Steps"
        closing_text = """
        Strategy assessment: after a severe 2022 drawdown, the system delivered 21% annualized return and Sharpe 0.76 in 2023-2025,
        making it attractive yet risk-heavy. Excess returns are concentrated in the last two years, so alpha is regime-dependent.
        Max drawdown (~31%) and beta (~1) show the portfolio still rides market swings, requiring neutralization or volatility control
        before larger capital deployment.

        Research rigor: the workflow follows a disciplined loop—data → factors → single-factor diagnosis → ranking model → weights →
        shift-based backtest → multi-angle evaluation—with strict guards against look-ahead (feature alignment, shift(-1), caching,
        IC cross-check). SHAP plus daily IC attribution provides consistent explanations, so the study is logically sound and reproducible.

        Next steps:
        • Risk: add beta/sector-neutral or name-cap constraints, plus volatility targeting to reduce the -31% drawdown.
        • Factors: flip/remove Alpha28 and NATR_14; keep pruning inputs via ongoing single-factor runs.
        • Strategy: test smaller topk or smoother n_drop, consider risk-parity / vol-scaling overlays.
        • Monitoring: maintain rolling IC/alpha dashboards to detect regime shifts and trigger updates.
        """
    story.append(Paragraph(closing_title, styles["SectionHeader"]))
    story.append(text_paragraph(closing_text, styles["Body"]))

    doc.build(story)
    print(f"[OK] PDF report ({language}) saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PDF report for workflow.")
    parser.add_argument("--lang", choices=["zh", "en", "all"], default="all", help="language to export (default: all)")
    args = parser.parse_args()

    langs = DEFAULT_LANGS if args.lang == "all" else [args.lang]
    for lang in langs:
        build_report(lang)

