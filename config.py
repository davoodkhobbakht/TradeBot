# -*- coding: utf-8 -*-
# config.py

SYMBOL_OPTIMIZED_PARAMS = {
    "BTC/USDT": {
        "stop_loss": 0.025,
        "take_profit": 0.075,
        "position_size": 0.01,
    },  # 1:3 RR, 1% risk
    "ETH/USDT": {"stop_loss": 0.03, "take_profit": 0.09, "position_size": 0.01},
    "SOL/USDT": {"stop_loss": 0.03, "take_profit": 0.09, "position_size": 0.01},
}


# تنظیمات عمومی
TRADE_SETTINGS = {
    "initial_capital": 1000.0,
    "trade_fee": 0.001,
    "slippage": 0.005,  # Increased from 0.002 for more realistic crypto slippage
    "bid_ask_spread": 0.0005,  # Added bid-ask spread for realism
    "max_trades_per_symbol": 100,
    "min_distance_between_trades": 10,  # Minimum candles between trades to align with max_trades limit
}

# تنظیمات ML - بهبود یافته
ML_SETTINGS = {
    "target_lookahead": 1,  # کاهش از 5 به 3
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
