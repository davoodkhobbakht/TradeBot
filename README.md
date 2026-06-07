# Testnet Trader V0.2

A sophisticated cryptocurrency trading bot designed for Binance testnet, featuring advanced multi-strategy trading, machine learning integration, and comprehensive backtesting capabilities.

## 🚀 Overview

Testnet Trader V0.2 is an automated trading system that combines traditional technical analysis with machine learning models to generate trading signals. The bot supports multiple cryptocurrencies and includes risk management features like stop-loss and take-profit orders.

The system is designed for educational and testing purposes on Binance testnet before deploying to live markets.

## ✨ Features

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

- Python 3.8 or higher
- Binance account with testnet API keys (for live testing)
- Internet connection for data fetching

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   cd /path/to/your/projects
   git clone <repository-url>
   cd "testnet trader V0.2"
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn ccxt matplotlib shap imblearn tensorflow xgboost
   ```

3. **Verify installation:**
   ```bash
   python main.py
   ```

## ⚙️ Configuration

The bot's behavior can be customized through `config.py`:

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

## 🚀 Usage

### Simple System Test
Run a basic test to verify data fetching and indicators:
```bash
python main.py --simple
```

### Train Machine Learning Models
Train ML models for each supported symbol:
```bash
python main.py --train
```

### Enhanced Backtesting
Run comprehensive backtest with ML and multi-strategy signals:
```bash
python main.py --enhanced
```

### Live Testnet Trading
**⚠️ WARNING: Only use testnet for safety!**
```bash
python main.py --live --api_key YOUR_BINANCE_TESTNET_API_KEY --api_secret YOUR_BINANCE_TESTNET_API_SECRET
```

### Help
View all available options:
```bash
python main.py --help
```

## 📊 Backtesting

The backtesting engine provides detailed performance analysis:

### Running a Backtest
1. Ensure historical data is available (automatically fetched)
2. Run enhanced backtest: `python main.py --enhanced`
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

## 🏗️ Project Structure

```
testnet trader V0.2/
├── main.py                 # Main entry point
├── config.py              # Configuration settings
├── data/
│   ├── data_fetcher.py    # Binance data fetching
│   ├── data_processor.py  # Feature engineering
│   └── __init__.py
├── ml/
│   ├── base_ml.py         # ML model management
│   ├── simple_ml.py       # Random Forest trainer
│   ├── advanced_ml.py     # Advanced ML pipeline
│   ├── rl_trader.py       # Reinforcement learning
│   └── __init__.py
├── strategies/
│   ├── base_strategy.py   # Basic technical analysis
│   ├── multi_strategy.py  # Multi-strategy engine
│   ├── signal_generator.py # Signal combination
│   ├── gold_strategy.py   # Specialized strategies
│   └── __init__.py
├── backtest/
│   ├── backtester.py      # Backtesting engine
│   ├── performance.py     # Performance analysis
│   └── __init__.py
├── utils/
│   ├── indicators.py      # Technical indicators
│   ├── helpers.py         # Utility functions
│   ├── multi_timeframe.py # Multi-timeframe analysis
│   └── __init__.py
└── models/                # Trained ML models (generated)
```

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
**Last Updated:** October 2025  
**Python Version:** 3.8+  
**Testnet Only:** Yes
