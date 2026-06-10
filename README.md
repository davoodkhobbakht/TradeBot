# Testnet Trader V0.2

A sophisticated cryptocurrency trading bot designed for Binance testnet, featuring advanced multi-strategy trading, machine learning integration, comprehensive backtesting capabilities, and a **modern real-time web dashboard**.

![React](https://img.shields.io/badge/React-18.3-61dafb?style=flat-square&logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178c6?style=flat-square&logo=typescript)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38bdf8?style=flat-square&logo=tailwindcss)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python)

## 🚀 Overview

Testnet Trader V0.2 is an automated trading system that combines traditional technical analysis with machine learning models to generate trading signals. The bot supports multiple cryptocurrencies and includes risk management features like stop-loss and take-profit orders.

The system is designed for educational and testing purposes on Binance testnet before deploying to live markets. It now includes a **modern web dashboard** for real-time monitoring and control.

## ✨ Features

### 🌐 Web Dashboard (New!)
A modern, responsive web interface to monitor and control your trading bot in real-time:
- 🟢 **Bot Control**: Start/Stop the bot with one click across multiple modes (Live, Backtest, Train)
- 📊 **Live Monitoring**: Real-time performance metrics, equity curves, and open positions
- ⚙️ **Configuration Editor**: Adjust trading parameters, ML settings, and per-symbol risk on the fly
- 📜 **Live Logs**: Color-coded streaming logs via WebSocket
- 🧪 **Backtesting**: Trigger and view backtest results directly from the UI
- 📋 **Performance Reports**: Detailed analytics (Sharpe Ratio, Max Drawdown, Win Rate)

### Multi-Strategy Trading Engine
- **Trend Following**: Identifies and follows market trends using moving averages and ADX
- **Mean Reversion**: Capitalizes on price deviations from mean levels using Bollinger Bands and RSI
- **Breakout Trading**: Detects price breakouts from resistance/support levels with volume confirmation
- **Momentum Trading**: Analyzes price momentum using MACD and Stochastic indicators
- **Volatility Trading**: Adapts to market volatility conditions using ATR and Bollinger Bandwidth

### Machine Learning Integration
- Random Forest models for price prediction (3-day forward looking)
- Feature engineering with technical indicators and price ratios
- Model training pipeline with cross-validation
- Confidence-weighted signal combination

### Multi-Exchange Data Fetcher
- **5-exchange fallback** (Binance → Bybit → OKX → Kraken)
- **Auto-retry** with 3 attempts per symbol
- **Source tracking** - know which exchange provided each candle
- **Cloudflare bypass** with browser headers

### Advanced Signal Generation
- Market regime detection (trending bull/bear, ranging, high volatility)
- Strategy weight adjustment based on market conditions
- Multi-timeframe analysis (optional)
- Signal filtering to reduce over-trading

### Risk Management
- Configurable stop-loss and take-profit levels per symbol
- Position sizing based on volatility and account balance
- Trailing stops with dynamic adjustments
- Maximum trades per symbol limits

### Backtesting & Analysis
- Historical data backtesting with realistic fees and slippage
- Performance metrics (returns, drawdown, Sharpe ratio)
- Trade-by-trade analysis
- Comparative analysis vs. buy-and-hold strategy

### Supported Assets
- BTC/USDT
- ETH/USDT
- SOL/USDT

## 📋 Prerequisites

### Backend Requirements
- **Python 3.8** or higher
- **Binance account** with testnet API keys (for live testing)
- **Internet connection** for data fetching

### Frontend Requirements (for Web Dashboard)
- **Node.js 18+** and **npm** (LTS recommended)
- Modern web browser (Chrome, Firefox, Edge, Safari)

Verify installations:
```bash
python3 --version
node --version
npm --version
```

## 🛠️ Installation

### 1. Clone the repository
```bash
cd /path/to/your/projects
git clone <repository-url>
cd "testnet trader V0.2"
```

### 2. Install Backend Dependencies
```bash
pip install pandas numpy scikit-learn ccxt matplotlib shap imblearn tensorflow xgboost fastapi uvicorn
```

### 3. Install Frontend Dependencies (for Web Dashboard)
```bash
cd frontend
npm install
cd ..
```

### 4. Verify installation
```bash
python main.py
```

## 🚀 Running the Application

You can run the bot either through the **command line** or the **web dashboard** (or both!).

### Option A: Using the Web Dashboard (Recommended)

You need to run **both** the backend and frontend servers in **two separate terminals**.

**Terminal 1: Start the Backend (FastAPI)**
```bash
# Navigate to project root
cd path/to/TradeBot

# Activate virtual environment (if using one)
source path/to/your/venv/bin/activate

# Start the API server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
✅ *Success looks like:* `INFO: Application startup complete.`

**Terminal 2: Start the Frontend (React)**
```bash
# Navigate to frontend directory
cd path/to/TradeBot/frontend

# Start the Vite development server
npm run dev
```
✅ *Success looks like:* `Local: http://localhost:3000/`

**Open your browser** to: **http://localhost:3000**

You can now control the entire bot from the dashboard: start/stop, configure settings, run backtests, and monitor live positions.

### Option B: Using the Command Line

#### Simple System Test
Run a basic test to verify data fetching and indicators:
```bash
python main.py --simple
```

#### Train Machine Learning Models
Train ML models for each supported symbol:
```bash
python main.py --train
```

#### Enhanced Backtesting
Run comprehensive backtest with ML and multi-strategy signals:
```bash
python main.py --enhanced
```

#### Live Testnet Trading
**⚠️ WARNING: Only use testnet for safety!**
```bash
python main.py --live --api_key YOUR_BINANCE_TESTNET_API_KEY --api_secret YOUR_BINANCE_TESTNET_API_SECRET
```

#### Help
View all available options:
```bash
python main.py --help
```

## ⚙️ Configuration

The bot's behavior can be customized through `config.py` or directly via the web dashboard's Configuration page.

### Trade Settings
```python
TRADE_SETTINGS = {
    "initial_capital": 1000.0,      # Starting capital in USDT
    "trade_fee": 0.001,             # Trading fee (0.1%)
    "slippage": 0.002,              # Price slippage (0.2%)
    "max_trades_per_symbol": 50,    # Maximum trades per symbol
}
```

### Symbol-Specific Parameters
```python
SYMBOL_OPTIMIZED_PARAMS = {
    "BTC/USDT": {
        "stop_loss": 0.06,           # 6% stop loss
        "take_profit": 0.25,         # 25% take profit
        "position_size": 0.12,       # 12% of capital per trade
    },
    # ... other symbols
}
```

### ML Settings
```python
ML_SETTINGS = {
    "target_lookahead": 3,           # Predict 3 periods ahead
    "test_size": 0.3,                # 30% test data
    "min_positive_samples": 0.15,    # Minimum positive samples
}
```

## 📊 Backtesting

The backtesting engine provides detailed performance analysis:

### Running a Backtest
1. Ensure historical data is available (automatically fetched)
2. Run enhanced backtest: `python main.py --enhanced` (or use the dashboard)
3. Review performance metrics and trade logs

### Performance Metrics
- **Total Return**: Portfolio percentage gain/loss
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit divided by gross loss
- **Number of Trades**: Total trades executed

### Sample Output
```
🎉 نتایج بک‌تست پیشرفته:
==========================================
💰 سرمایه اولیه: 1,000 USDT
💰 سرمایه نهایی: 1,250 USDT
📈 بازدهی پرتفوی: +25.00%
🔢 کل معاملات: 45

🏆 رتبه‌بندی نمادها:
1. 🟢 BTC/USDT: +35.2% (معاملات: 18, Max DD: 12.5%)
2. 🟢 ETH/USDT: +28.7% (معاملات: 15, Max DD: 15.3%)
3. 🔴 SOL/USDT: -5.1% (معاملات: 12, Max DD: 22.1%)
```

## 🔌 API Endpoints

The web dashboard communicates with the backend via a FastAPI server. All endpoints are proxied through Vite during development.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/bot/start` | Start bot (payload: `{ mode: 'live' \| 'simple' \| 'enhanced' \| 'train' }`) |
| `POST` | `/bot/stop` | Stop the running bot |
| `GET` | `/bot/status` | Get current bot status, PID, and uptime |
| `GET` | `/config` | Fetch current trading and ML configuration |
| `PUT` | `/config` | Update configuration parameters |
| `GET` | `/positions` | Get list of currently open positions |
| `POST` | `/backtest/simple` | Trigger a simple backtest job |
| `POST` | `/backtest/enhanced` | Trigger an ML-enhanced backtest job |
| `GET` | `/backtest/result` | Fetch results of the last backtest job |
| `GET` | `/reports/latest` | Get latest performance report metrics |
| `GET` | `/chart/equity` | Get historical equity curve data |
| `WS` | `/ws/logs` | **WebSocket**: Stream live bot logs in real-time |

## 🏗️ Project Structure

```
testnet trader V0.2/
├── main.py                 # Main entry point
├── config.py               # Configuration settings
├── api/                    # FastAPI backend for web dashboard
│   ├── main.py             # API app entry point & routes
│   ├── bot_routes.py       # Bot control endpoints
│   ├── config_routes.py    # Configuration endpoints
│   └── ...
├── frontend/               # 🌐 REACT WEB DASHBOARD
│   ├── src/
│   │   ├── api/            # Axios API client
│   │   ├── components/     # Reusable UI components
│   │   │   ├── Layout/     # Sidebar, Header, Main Layout
│   │   │   ├── Dashboard/  # Status cards, Metrics, Equity Chart
│   │   │   ├── Bot/        # Start/Stop controls
│   │   │   ├── Config/     # Settings editor
│   │   │   ├── Positions/  # Live positions table
│   │   │   ├── Backtest/   # Backtest triggers & results
│   │   │   ├── Reports/    # Performance analytics
│   │   │   └── Logs/       # Real-time log viewer
│   │   ├── hooks/          # Custom React hooks
│   │   ├── types/          # TypeScript interfaces
│   │   ├── pages/          # Page components
│   │   ├── styles/         # Global CSS
│   │   ├── App.tsx         # Main routing
│   │   └── main.tsx        # React entry point
│   ├── package.json        # Node dependencies
│   ├── vite.config.ts      # Vite config (includes API proxy)
│   └── tailwind.config.js  # Tailwind CSS config
├── data/
│   ├── data_fetcher.py     # Binance data fetching
│   ├── data_processor.py   # Feature engineering
│   └── __init__.py
├── ml/
│   ├── base_ml.py          # ML model management
│   ├── simple_ml.py        # Random Forest trainer
│   ├── advanced_ml.py      # Advanced ML pipeline
│   ├── rl_trader.py        # Reinforcement learning
│   └── __init__.py
├── strategies/
│   ├── base_strategy.py    # Basic technical analysis
│   ├── multi_strategy.py   # Multi-strategy engine
│   ├── signal_generator.py # Signal combination
│   ├── gold_strategy.py    # Specialized strategies
│   └── __init__.py
├── backtest/
│   ├── backtester.py       # Backtesting engine
│   ├── performance.py      # Performance analysis
│   └── __init__.py
├── utils/
│   ├── indicators.py       # Technical indicators
│   ├── helpers.py          # Utility functions
│   ├── multi_timeframe.py  # Multi-timeframe analysis
│   └── __init__.py
└── models/                 # Trained ML models (generated)
```

## 🐛 Troubleshooting
#### "Cloudflare" or connection errors
**Fix:** Multi-exchange fallback enabled by default - auto-switches to Bybit/OKX if Binance fails.

### Backend Issues

#### `ModuleNotFoundError: No module named 'fastapi'`
**Cause:** The system is using the global Python installation instead of your virtual environment.
**Fix:** Always activate the venv and use `python -m`:
```bash
source path/to/your/venv/bin/activate
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Issues

#### Frontend shows "Unexpected token" or Babel parsing errors
**Cause:** Template literals (`` ` ``) or variables (`${}`) were accidentally escaped during file generation.
**Fix:** Ensure your `.tsx` files do not contain `\`` or `\${`. Run a find-and-replace in your IDE to remove the backslashes before backticks and dollar signs.

#### Frontend shows "No data" or Network Errors
**Cause:** The frontend cannot reach the backend.
**Fix:**
1. Verify the backend is running: `curl http://localhost:8000/bot/status`
2. Check the browser console (F12) for CORS or 404 errors.
3. Ensure `vite.config.ts` has the correct proxy settings pointing to `http://localhost:8000`.

#### Port 3000 or 8000 is already in use
**Fix:** Kill the process or use a different port.
```bash
# Kill process on port 3000 (Linux/Mac)
lsof -ti:3000 | xargs kill -9

# Or run frontend on a different port
npm run dev -- --port 3001
```

## 💻 Frontend Development

### Available NPM Scripts
```bash
cd frontend
npm run dev      # Start local development server with Hot Module Replacement (HMR)
npm run build    # Compile TypeScript and bundle for production (outputs to /dist)
npm run preview  # Locally preview the production build
```

### Adding a New Feature
1. **New Page:** Create `src/pages/NewPage.tsx`, add it to `src/App.tsx` routes, and add a link in `src/components/Layout/Sidebar.tsx`.
2. **New API Call:** Add the endpoint function in `src/api/client.ts` and define its return type in `src/types/index.ts`.

### Notes
- The Vite proxy (`vite.config.ts`) automatically handles CORS during development. Do not call `http://localhost:8000` directly from React; use `/api/...` so the proxy forwards it correctly.
- The dashboard polls `/bot/status` every 5 seconds and `/positions` every 10 seconds to keep the UI fresh.
- Logs are streamed via WebSocket for zero-latency updates.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-strategy`
3. Commit changes: `git commit -am 'Add new strategy'`
4. Push to branch: `git push origin feature/new-strategy`
5. Submit a pull request

### Adding New Strategies
1. Create new strategy class inheriting from base strategy
2. Implement `calculate_signal()` method
3. Add to multi-strategy engine with appropriate weights
4. Update configuration if needed

### Adding New Indicators
1. Add calculation to `utils/indicators.py`
2. Update feature lists in ML modules
3. Test with existing strategies

### Adding New Dashboard Features
1. Create a new component in `frontend/src/components/`
2. Add API endpoints in `api/` directory
3. Update `frontend/src/api/client.ts` with new endpoints
4. Add navigation link in `frontend/src/components/Layout/Sidebar.tsx`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and testing purposes only. Cryptocurrency trading involves significant risk of loss. Never trade with money you cannot afford to lose. Always test thoroughly on testnet before considering live trading.

The authors and contributors are not responsible for any financial losses incurred through the use of this software.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review Binance API documentation
3. Open an issue on GitHub with detailed error logs

---

**Version:** 0.2
**Last Updated:** June 2026
**Python Version:** 3.8+
**Node.js Version:** 18+
**Testnet Only:** Yes
**Web Dashboard:** ✅ Included
- 🔄 **Multi-Exchange Data Fetching**: Automatic fallback between 5 exchanges (Binance Futures, Binance Spot, Bybit, OKX, Kraken) with Cloudflare bypass for reliable data collection
