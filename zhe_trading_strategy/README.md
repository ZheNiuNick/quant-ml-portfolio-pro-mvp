# 量化交易策略 Web Dashboard

这是一个完整的 Flask Web 应用，用于展示量化交易策略的完整工作流程、因子分析、持仓和收益。

## 功能特性

### 1. 项目介绍
- 项目目标和核心特点
- 策略表现概览
- 技术架构说明

### 2. 详细工作流程
- **7 步完整流程**，每一步都有详细说明：
  1. 数据获取与预处理
  2. 因子计算（160+ 个因子）
  3. 因子处理与增强
  4. 模型训练（LightGBM Ranker）
  5. 预测生成
  6. 组合优化（TopK Dropout Strategy）
  7. 实时交易执行（IBKR API）
- **交互式可视化**：
  - 策略 vs 基准收益曲线（Plotly 交互式图表）
  - 性能指标表格
  - 回测结果分析

### 3. 每日单因子看报
- 因子表现摘要（IC, ICIR, 胜率）
- Top 20 因子 IC 排名（交互式柱状图）
- 因子 IC 分布（直方图）
- 因子分类统计（Alpha101, TA-Lib, 自定义因子）

### 4. 每日真实持仓
- 当前持仓列表（股票代码、权重、价格）
- 持仓权重分布（饼图）
- 持仓统计信息
- **详细的持仓生成流程说明**（7 步流程）

### 5. 交易记录 (Blotter)
- 交易记录表格（时间、股票、方向、数量、价格、状态）
- 交易统计（总交易数、买入/卖出订单数、总交易额）
- 日期和股票代码筛选

### 6. 真实收益展示
- 关键指标卡片（累计收益、年化收益、Sharpe Ratio、最大回撤）
- 策略 vs 基准收益曲线（交互式）
- 回撤分析曲线
- 月度收益分析（柱状图）

## 安装和运行

### 1. 安装依赖

```bash
pip install flask pandas numpy plotly pyyaml
```

### 2. 运行应用

```bash
cd zhe_trading_strategy
python app.py
```

应用将在 `http://localhost:5000` 启动。

### 3. 访问页面

- 首页（项目介绍）：`http://localhost:5000/`
- 工作流程：`http://localhost:5000/workflow`
- 因子看报：`http://localhost:5000/factor-report`
- 持仓分析：`http://localhost:5000/positions`
- 交易记录：`http://localhost:5000/blotter`
- 收益展示：`http://localhost:5000/performance`

## 数据要求

应用需要以下数据文件（在项目根目录的 `outputs/` 目录下）：

- `backtests/summary.json` - 回测摘要
- `backtests/daily_returns.parquet` - 每日收益数据
- `backtests/monthly_performance.parquet` - 月度收益数据
- `portfolios/weights.parquet` - 持仓权重
- `reports/metrics.json` - 模型指标
- `reports/single_factor_summary.json` - 单因子摘要

如果某些文件不存在，相关页面会显示相应的错误信息。

## 技术栈

- **后端**: Flask (Python)
- **前端**: Bootstrap 5, Plotly.js
- **数据**: Pandas, NumPy
- **可视化**: Plotly (交互式图表)

## 项目结构

```
zhe_trading_strategy/
├── app.py                 # Flask 主应用
├── templates/             # HTML 模板
│   ├── base.html         # 基础模板
│   ├── index.html        # 首页
│   ├── workflow.html     # 工作流程
│   ├── factor_report.html # 因子看报
│   ├── positions.html    # 持仓分析
│   ├── blotter.html      # 交易记录
│   └── performance.html  # 收益展示
├── static/               # 静态资源
│   ├── css/
│   │   └── style.css    # 自定义样式
│   └── js/              # JavaScript 文件（如需要）
└── README.md            # 本文件
```

## 注意事项

1. **数据路径**: 应用假设数据文件在项目根目录的 `outputs/` 目录下。如果路径不同，需要修改 `app.py` 中的路径配置。

2. **IBKR 交易记录**: 目前交易记录页面从 API 读取，实际使用时需要从 IBKR 执行日志中读取真实交易数据。

3. **实时更新**: 如果需要实时更新数据，可以设置定时任务运行 `scripts/daily_update.py`，然后刷新页面。

## 开发建议

- 可以根据需要添加更多可视化图表
- 可以添加用户认证和权限管理
- 可以添加数据导出功能（CSV, PDF）
- 可以添加实时数据推送（WebSocket）

## 许可证

本项目仅供学习和研究使用。
