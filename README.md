# Quant-ML Portfolio Pro

<div align="center">

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

**A Production-Ready Machine Learning-Based Quantitative Trading System**

[Live Demo](https://zn22.blog) • [Documentation](#documentation) • [Features](#features) • [Quick Start](#quick-start)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Web Dashboard](#web-dashboard)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**Quant-ML Portfolio Pro** is a comprehensive, production-ready quantitative trading system that combines machine learning, factor engineering, and portfolio optimization to achieve stable excess returns in the S&P500 stock market.

### Key Highlights

- **160+ Quantitative Factors**: Alpha101, TA-Lib technical indicators, and custom factors
- **LightGBM Ranker**: Advanced ranking learning model with walk-forward cross-validation
- **TopK Dropout Strategy**: Intelligent portfolio optimization with turnover constraints
- **Real-Time Trading**: IBKR API integration for live paper trading
- **Interactive Web Dashboard**: Beautiful Flask-based dashboard with real-time analytics
- **Comprehensive Backtesting**: Full backtest engine with risk analysis and performance metrics

### Strategy Performance

The system has been tested on S&P500 stocks (2010-2025) with:
- **Annualized Return**: Competitive excess returns
- **Sharpe Ratio**: Risk-adjusted performance metrics
- **Maximum Drawdown**: Controlled risk exposure
- **Turnover Management**: Efficient portfolio rebalancing

---

## ✨ Features

### 🔬 Factor Engineering

- **Alpha101 Factors**: 101 quantitative factors based on academic research
- **TA-Lib Indicators**: 50+ technical analysis indicators (RSI, MACD, Bollinger Bands, etc.)
- **Custom Factors**: Proprietary factors including momentum, quality, volatility, and value factors
- **Factor Diagnostics**: IC/ICIR analysis, correlation matrix, long-short performance
- **Factor Enhancement**: Automatic factor selection and stability analysis

### 🤖 Machine Learning

- **LightGBM Ranker**: Ranking learning model for stock selection
- **Walk-Forward CV**: Time-series cross-validation to prevent look-ahead bias
- **Feature Engineering**: Automatic feature selection and importance analysis
- **Model Evaluation**: Comprehensive metrics including Rank IC, NDCG, and MAP

### 📊 Portfolio Optimization

- **TopK Dropout Strategy**: Intelligent stock selection with controlled turnover
- **Mean-Variance Optimization**: Risk-return optimization with constraints
- **Risk Management**: Beta neutrality, industry neutrality, and turnover limits
- **Transaction Costs**: Realistic modeling of fees and slippage

### 📈 Backtesting & Analysis

- **Full Backtest Engine**: Complete historical simulation with realistic constraints
- **Performance Metrics**: Annual return, Sharpe ratio, max drawdown, information ratio
- **Risk Analysis**: Rolling alpha/beta, factor contribution, excess return analysis
- **Visualization**: Interactive charts and comprehensive reports

### 💹 Live Trading

- **IBKR Integration**: Interactive Brokers API for real-time trading
- **Paper Trading**: Safe testing environment with paper account
- **Position Management**: Automatic position tracking and rebalancing
- **Trade Execution**: Limit orders with slippage protection

### 🌐 Web Dashboard

- **Real-Time Analytics**: Live performance monitoring and factor analysis
- **Interactive Visualizations**: Plotly-based charts with drill-down capabilities
- **Factor Diagnostics**: Comprehensive factor analysis and correlation matrices
- **Portfolio Tracking**: Current positions, trade history, and P&L analysis
- **Workflow Visualization**: Step-by-step strategy execution pipeline

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Dashboard (Flask)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Home    │  │ Workflow │  │ Factors  │  │ Positions│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Engine Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Factor Engine│  │   Modeling   │  │  Optimizer    │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Backtest   │  │ IBKR Trader  │  │ Data Pipeline│    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   DuckDB     │  │   Parquet    │  │   JSON       │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Backend**: Python 3.9+, Flask
- **Data Processing**: Pandas, NumPy, PyArrow
- **Database**: DuckDB
- **Machine Learning**: LightGBM, scikit-learn
- **Visualization**: Plotly, Matplotlib
- **Trading**: IBKR API (ib_insync)
- **Deployment**: Railway, Gunicorn

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/quant-ml.git
cd quant-ml
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r zhe_trading_strategy/requirements.txt
```

### 4. Configure Settings

Edit `config/settings.yaml` to configure:
- Data sources and date ranges
- Strategy parameters
- Risk model settings
- Optimizer constraints

### 5. Run the Web Dashboard

```bash
cd zhe_trading_strategy
python app.py
```

Visit `http://localhost:5000` in your browser.

---

## 📦 Installation

### Detailed Installation Steps

#### 1. System Requirements

- **OS**: Linux, macOS, or Windows
- **Python**: 3.9 or higher
- **Memory**: 8GB+ RAM recommended
- **Disk**: 10GB+ free space

#### 2. Install Python Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Web dashboard dependencies
pip install -r zhe_trading_strategy/requirements.txt

# Optional: IBKR trading (for live trading)
pip install ib_insync

# Optional: TA-Lib (for technical indicators)
# Note: Requires system-level TA-Lib library
# macOS: brew install ta-lib
# Linux: apt-get install ta-lib
pip install TA-Lib
```

#### 3. Data Setup

The system uses S&P500 stock data. Data will be automatically downloaded on first run, or you can manually fetch:

```bash
# Fetch historical data
python src/data_pipeline.py --fetch

# Generate factors
python src/factor_engine.py --build

# Generate precomputed diagnostics (for web dashboard)
python generate_factor_precomputed.py
```

---

## 💻 Usage

### Command-Line Interface

#### Data Pipeline

```bash
# Fetch and process market data
python src/data_pipeline.py --fetch

# Build factor library
python src/factor_engine.py --build --evaluate
```

#### Model Training

```bash
# Train LightGBM ranker model
python src/modeling.py --train

# Generate predictions
python src/modeling.py --predict
```

#### Portfolio Optimization

```bash
# Optimize portfolio
python src/optimizer.py --optimize

# Run backtest
python src/backtest.py --run
```

#### Live Trading

```bash
# Execute trades via IBKR (paper account)
python src/ibkr_live_trader.py \
    --weights outputs/portfolios/weights.parquet \
    --capital-usage-ratio 0.90 \
    --ib-host 127.0.0.1 \
    --ib-port 7497
```

### Daily Update Script

```bash
# Run complete daily update pipeline
python scripts/daily_update.py
```

This script:
1. Fetches latest market data
2. Calculates factors
3. Generates predictions
4. Optimizes portfolio
5. Updates web dashboard data

### Generate Precomputed Diagnostics

For optimal web dashboard performance, generate precomputed factor diagnostics:

```bash
# Generate precomputed JSON files for web dashboard
python generate_factor_precomputed.py
```

This creates:
- `outputs/factor_corr.json` - Factor correlation matrix
- `outputs/factor_long_short.json` - Long-short performance by factor
- `outputs/factor_exposure.json` - Risk exposure by date

These files are small (<4MB) and can be committed to Git for fast dashboard loading.

---

## 🌐 Web Dashboard

The web dashboard provides a comprehensive interface for monitoring and analyzing the trading strategy.

### Pages

#### 1. **Home** (`/`)
- Project overview and objectives
- Strategy performance summary
- Key metrics dashboard

#### 2. **Workflow** (`/workflow`)
- Interactive 7-step workflow visualization
- Strategy vs benchmark return curves
- Performance metrics table
- Backtest results analysis

#### 3. **Factor Diagnostics** (`/factor-diagnostics`)
- **Rolling IC/ICIR**: Factor predictive power over time
- **Factor Clusters**: Momentum, Quality, Volatility, Value, Size
- **Long-Short Performance**: Factor-based portfolio returns
- **Correlation Matrix**: Factor relationships
- **Risk Exposure**: Barra-style multi-factor risk analysis

#### 4. **Backtest Results** (`/backtest-results`)
- Total return, annual return, Sharpe ratio, max drawdown
- Strategy vs benchmark comparison
- Monthly performance analysis
- Real-time performance tracking

#### 5. **Positions** (`/positions`)
- Current portfolio holdings
- Position weights and prices
- Portfolio composition visualization
- Position generation workflow

#### 6. **Blotter** (`/blotter`)
- Trade history and execution logs
- Trade statistics and filters
- Order status tracking

### API Endpoints

All dashboard data is served via RESTful API:

- `GET /api/performance/summary` - Performance summary
- `GET /api/performance/monthly` - Monthly returns
- `GET /api/performance/real-time` - Real-time metrics
- `GET /api/factor-diagnostics/factors` - Factor list
- `GET /api/factor-diagnostics/rolling-ic` - Rolling IC
- `GET /api/factor-diagnostics/correlation` - Correlation matrix
- `GET /api/factor-diagnostics/long-short?factor=<name>` - Long-short performance
- `GET /api/factor-diagnostics/risk-exposure?date=<date>` - Risk exposure
- `GET /api/positions/current` - Current positions
- `GET /api/blotter/trades` - Trade history

---

## 🚢 Deployment

### Railway Deployment

The project is configured for easy deployment on Railway.

#### 1. Prerequisites

- Railway account
- GitHub repository

#### 2. Deploy Steps

1. **Connect Repository**: Link your GitHub repo to Railway
2. **Configure Build**: Railway will auto-detect the Python project
3. **Set Environment Variables** (if needed):
   ```
   PORT=8080
   API_BASE_URL=https://your-domain.com
   ```
4. **Deploy**: Railway will automatically build and deploy

#### 3. Custom Domain

1. In Railway dashboard, go to Settings → Domains
2. Add your custom domain (e.g., `zn22.blog`)
3. Configure DNS CNAME record pointing to Railway

### Configuration Files

- `railway.json` - Railway build and deploy configuration
- `nixpacks.toml` - Nixpacks build configuration
- `zhe_trading_strategy/start.sh` - Startup script
- `Procfile` - Process configuration

### Memory Optimization

The system uses precomputed JSON files to avoid memory issues on free-tier hosting:

- `outputs/factor_corr.json` - Factor correlation matrix
- `outputs/factor_long_short.json` - Long-short performance
- `outputs/factor_exposure.json` - Risk exposure data

These files are generated locally and committed to the repository.

---

## 📁 Project Structure

```
quant-ml/
├── config/
│   └── settings.yaml          # Main configuration file
├── src/
│   ├── alpha101.py            # Alpha101 factor library
│   ├── backtest.py            # Backtest engine
│   ├── custom_factors.py      # Custom factor definitions
│   ├── data_pipeline.py       # Data fetching and processing
│   ├── factor_engine.py       # Factor calculation engine
│   ├── factor_enhancement.py  # Factor selection and enhancement
│   ├── ibkr_live_trader.py   # IBKR trading integration
│   ├── modeling.py            # ML model training
│   ├── optimizer.py           # Portfolio optimization
│   └── talib_factors.py      # TA-Lib technical indicators
├── zhe_trading_strategy/
│   ├── app.py                 # Flask web application
│   ├── templates/             # HTML templates
│   ├── static/                # CSS, JS, images
│   └── requirements.txt       # Web app dependencies
├── scripts/
│   ├── daily_update.py       # Daily update pipeline
│   └── ...                   # Utility scripts
├── outputs/
│   ├── backtests/            # Backtest results
│   ├── models/               # Trained models
│   ├── portfolios/           # Portfolio weights
│   └── reports/              # Analysis reports
├── data/
│   ├── factors/              # Factor data
│   └── processed/            # Processed price data
├── generate_factor_precomputed.py  # Precompute diagnostics
└── README.md                 # This file
```

---

## ⚙️ Configuration

### Main Configuration File: `config/settings.yaml`

```yaml
# Data settings
data:
  start: "2010-01-01"
  end: "2025-11-22"
  instruments: "sp500"
  region: "us"

# Strategy settings
strategy:
  type: "full_rebalance"  # or "topk_dropout"
  topk: 20
  n_drop: 3

# Risk model
risk_model:
  ewma_lambda: 0.94
  shrinkage: "ledoit_wolf"

# Optimizer
optimizer:
  lambda_risk: 5e-3
  wmax: 0.10
  turnover_max: 0.30
  beta_neutral: true
```

### Web Dashboard Configuration: `zhe_trading_strategy/config.py`

```python
API_BASE_URL = ''  # Leave empty for relative URLs
IBKR_CONFIG = {
    'host': '127.0.0.1',
    'port': 7497,
    'client_id': 777,
    'enabled': False  # Set to True to enable IBKR features
}
```

---

## 📚 API Reference

### Factor Engine

```python
# imports
from src.factor_engine import read_prices, forward_return, daily_rank_ic
from src.factor_engine import load_settings as load_factor_settings

# Load configuration
factor_cfg = load_factor_settings("config/settings.yaml")

# Read price data
prices = read_prices(factor_cfg)

# Calculate forward returns
forward_ret = forward_return(prices, horizon=1)

# Calculate daily Rank IC
ic = daily_rank_ic(factor, forward_ret)
```

### Modeling

```python
from src.modeling import train_ranker, generate_predictions
from src.modeling import load_predictions

# Train model
model = train_ranker(features, labels)

# Generate predictions
predictions = generate_predictions(model, features)

# Load saved predictions
predictions = load_predictions("outputs/models/lightgbm_predictions.pkl")
```

### Portfolio Optimization

```python
from src.optimizer import PortfolioOptimizer
from src.optimizer import load_settings

# Load settings
settings = load_settings("config/settings.yaml")

# Create optimizer
optimizer = PortfolioOptimizer(settings)

# Optimize portfolio
weights = optimizer.optimize(predictions, prices)

# Save weights
weights.to_parquet("outputs/portfolios/weights.parquet")
```

### Backtesting

```python
from src.backtest import run_backtest, risk_analysis, load_settings

# Load settings
settings = load_settings("config/settings.yaml")

# Run backtest
results = run_backtest(weights, prices, benchmark)

# Risk analysis
returns = results['strategy_return']
metrics = risk_analysis(returns)

# Print metrics
print(f"Annual Return: {metrics['annualized_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
```

### Web Dashboard API

```python
import requests

# Get performance summary
response = requests.get("https://zn22.blog/api/performance/summary")
data = response.json()

# Get factor diagnostics
response = requests.get("https://zn22.blog/api/factor-diagnostics/correlation?method=pearson")
correlation_data = response.json()

# Get long-short performance
response = requests.get("https://zn22.blog/api/factor-diagnostics/long-short?factor=Alpha1")
long_short_data = response.json()
```

---

## 🔧 Development

### Running Tests

```bash
# Run unit tests (if available)
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Code Style

The project follows PEP 8 style guidelines. Use a linter:

```bash
# Install linter
pip install flake8 black

# Format code
black src/ zhe_trading_strategy/

# Check style
flake8 src/ zhe_trading_strategy/
```

### Adding New Factors

1. Add factor calculation in `src/custom_factors.py`
2. Register factor in `src/factor_engine.py`
3. Update factor list in configuration
4. Regenerate factors: `python src/factor_engine.py --build`

### Adding New API Endpoints

1. Add route in `zhe_trading_strategy/app.py`
2. Create template in `zhe_trading_strategy/templates/` (if needed)
3. Add API endpoint documentation

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guide
- Add docstrings to new functions
- Include unit tests for new features
- Update documentation as needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Alpha101 Factors**: Based on research by WorldQuant
- **TA-Lib**: Technical Analysis Library
- **LightGBM**: Microsoft's gradient boosting framework
- **Interactive Brokers**: Trading platform and API
- **Plotly**: Interactive visualization library

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/quant-ml/issues)
- **Documentation**: See inline code documentation
- **Email**: [Your Email]

---

## 🗺️ Roadmap

- [ ] Add more factor libraries
- [ ] Implement deep learning models
- [ ] Add real-time data streaming
- [ ] Enhance risk management features
- [ ] Add multi-asset support
- [ ] Improve backtest engine performance
- [ ] Add more visualization options
- [ ] Support for options and derivatives
- [ ] Multi-timeframe analysis
- [ ] Advanced risk models (Barra, Axioma)

---

## 📖 Examples

### Example 1: Complete Workflow

```bash
# 1. Fetch data
python src/data_pipeline.py --fetch

# 2. Build factors
python src/factor_engine.py --build

# 3. Train model
python src/modeling.py --train

# 4. Generate predictions
python src/modeling.py --predict

# 5. Optimize portfolio
python src/optimizer.py --optimize

# 6. Run backtest
python src/backtest.py --run

# 7. Generate precomputed diagnostics
python generate_factor_precomputed.py

# 8. Start web dashboard
cd zhe_trading_strategy && python app.py
```

### Example 2: Daily Update

```bash
# Run complete daily update
python scripts/daily_update.py

# Or manually:
python scripts/daily_factor_update.py
python src/modeling.py --predict
python src/optimizer.py --optimize
python generate_factor_precomputed.py
```

### Example 3: Custom Factor

```python
# In src/custom_factors.py
def my_custom_factor(prices: pd.DataFrame) -> pd.Series:
    """Calculate custom factor"""
    close = prices['Adj Close']
    volume = prices['Volume']
    return (close.pct_change() * volume).rolling(20).mean()

# Register in factor_engine.py
factors['my_custom_factor'] = my_custom_factor(prices)
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Memory Issues on Railway

**Problem**: Worker killed due to memory limits

**Solution**: Use precomputed JSON files instead of loading large Parquet files:
```bash
python generate_factor_precomputed.py
git add outputs/factor_*.json
git commit -m "Add precomputed diagnostics"
```

#### 2. TA-Lib Installation

**Problem**: `TA-Lib not available` warning

**Solution**: Install system library first:
```bash
# macOS
brew install ta-lib

# Linux
sudo apt-get install ta-lib

# Then install Python package
pip install TA-Lib
```

#### 3. IBKR Connection Issues

**Problem**: Cannot connect to IBKR

**Solution**: 
1. Ensure IB Gateway/TWS is running
2. Enable API in TWS: Configure → API → Settings
3. Check firewall settings
4. Verify host and port in configuration

#### 4. Data Not Loading

**Problem**: Factor diagnostics show "No data available"

**Solution**:
1. Check if precomputed files exist: `ls outputs/factor_*.json`
2. Regenerate if missing: `python generate_factor_precomputed.py`
3. Check file permissions and paths

---

## 📊 Performance Benchmarks

### Factor Performance

- **Top Factors**: Alpha1-101, RSI, MACD, Momentum factors
- **Average IC**: ~0.05-0.10
- **ICIR**: ~0.5-1.5 for top factors
- **Factor Count**: 160+ factors

### Model Performance

- **Rank IC**: ~0.08-0.12
- **NDCG@20**: ~0.65-0.75
- **Training Time**: ~5-10 minutes (full dataset)
- **Prediction Time**: <1 second per day

### Backtest Results

- **Period**: 2010-2025 (S&P500)
- **Universe**: ~500 stocks
- **Rebalance**: Daily
- **Transaction Costs**: 2 bps fees + slippage

---

## 🔒 Security & Best Practices

### Security

- **Paper Trading**: Always test with paper account first
- **API Keys**: Never commit API keys or tokens to Git
- **Environment Variables**: Use `.env` files for sensitive data
- **Rate Limiting**: Respect API rate limits

### Best Practices

1. **Version Control**: Commit frequently, use meaningful commit messages
2. **Testing**: Test strategies thoroughly before live trading
3. **Monitoring**: Monitor performance and risk metrics regularly
4. **Documentation**: Keep code and configuration documented
5. **Backup**: Regular backups of models and portfolios

---

<div align="center">

**Built with ❤️ for quantitative trading**

[⭐ Star this repo](https://github.com/yourusername/quant-ml) if you find it helpful!

[Live Demo](https://zn22.blog) • [Documentation](#documentation) • [Issues](https://github.com/yourusername/quant-ml/issues)

</div>
