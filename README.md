# Quant-ML-Portfolio-Pro (MVP)

This is the **student-friendly, live-tradable** MVP that implements the modified blueprint you approved:
- Spec-bound LLM helper (lightweight code runner + daily explanations)
- Single-factor library with **correlation de-dup**
- LightGBM ranker with walk-forward CV
- Mean–Variance optimizer with beta/industry neutrality and turnover constraint
- IBKR paper-trading loop (daily rebalance)
- Streamlit dashboard & Prefect nightly flow

> Start by reading `config/settings.yaml` and `specs/*.yaml`, then run `python main_pipeline.py`.


<!-- 怎么跑（本地快捷步骤）

建环境并安装依赖

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt


拉历史日线数据（Yahoo）→ 写入 DuckDB & Parquet

python src/data_pipeline.py --fetch


构建与评估单因子库（含去重、IC/ICIR、Top-Bottom）

python src/factor_engine.py --build --evaluate


训练 LightGBM 排序模型（walk-forward CV）

python src/modeling.py --train


组合优化 + 回测（含换手成本）

python src/optimizer.py --optimize
python src/backtest.py --run


生成当日 LLM 风格简报（本地规则版，无需 API）

python src/llm_agent.py --explain-today


打开可视化仪表盘

streamlit run ui/app_streamlit.py


想一键夜间跑全流程：python main_pipeline.py（Prefect Flow）
想做 IBKR 纸上账户下单：编辑 exec/ibkr_paper.py 后运行 python exec/ibkr_paper.py -->