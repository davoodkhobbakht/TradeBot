# -*- coding: utf-8 -*-
# config.py

# پارامترهای بهینه‌شده برای هر نماد
SYMBOL_OPTIMIZED_PARAMS = {
    "BTC/USDT": {
        "stop_loss": 0.015,  # 1.5% - wider for bull markets
        "take_profit": 0.10,  # 10% - more aggressive TP
        "position_size": 0.10,  # 10% - larger position for better returns
    },
    "ETH/USDT": {
        "stop_loss": 0.02,  # 2%
        "take_profit": 0.12,  # 12%
        "position_size": 0.08,  # 8%
    },
    "SOL/USDT": {
        "stop_loss": 0.025,  # 2.5% - adjusted for volatility
        "take_profit": 0.15,  # 15%
        "position_size": 0.08,  # 8%
    },
}

# تنظیمات عمومی
TRADE_SETTINGS = {
    "initial_capital": 1000.0,
    "trade_fee": 0.001,
    "slippage": 0.005,  # Increased from 0.002 for more realistic crypto slippage
    "bid_ask_spread": 0.0005,  # Added bid-ask spread for realism
    "max_trades_per_symbol": 20,
    "min_distance_between_trades": 30,  # Minimum candles between trades to align with max_trades limit
}

# تنظیمات ML - بهبود یافته
ML_SETTINGS = {
    "target_lookahead": 3,  # کاهش از 5 به 3
    "test_size": 0.3,
    "random_state": 42,
    "min_positive_samples": 0.15,  # حداقل درصد داده‌های مثبت
}

# تنظیمات RL
RL_SETTINGS = {
    "state_size": 20,  # کاهش اندازه state
    "episodes": 50,  # کاهش episodes برای سریع‌تر شدن
    "batch_size": 32,
}
