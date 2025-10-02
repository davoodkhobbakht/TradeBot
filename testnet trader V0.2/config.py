# -*- coding: utf-8 -*-
# config.py

# پارامترهای بهینه‌شده برای هر نماد
SYMBOL_OPTIMIZED_PARAMS = {
    "BTC/USDT": {
        "stop_loss": 0.06,
        "take_profit": 0.25,
        "position_size": 0.12,
    },
    "ETH/USDT": {
        "stop_loss": 0.07,
        "take_profit": 0.22,
        "position_size": 0.12,
    },
    "SOL/USDT": {
        "stop_loss": 0.10,
        "take_profit": 0.30,
        "position_size": 0.08,
    },
}

# تنظیمات عمومی
TRADE_SETTINGS = {
    "initial_capital": 1000.0,
    "trade_fee": 0.001,
    "slippage": 0.002,
    "max_trades_per_symbol": 50,
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
