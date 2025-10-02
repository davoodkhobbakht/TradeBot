# -*- coding: utf-8 -*-
# utils/multi_timeframe.py

import pandas as pd
import numpy as np
from data.data_fetcher import fetch_ohlcv
import ccxt


def get_multi_timeframe_analysis(symbol, current_df, timeframe="1d"):
    """تحلیل چند تایم‌فریمی سریع برای تایید سیگنال‌ها"""

    print(f"⏰ تحلیل چند تایم‌فریمی برای {symbol}...")

    # تایم‌فریم‌های محدود برای سرعت
    timeframes = {
        "4h": "4h",  # فقط 4 ساعته و روزانه
        "1d": "1d",
    }

    analysis = {}
    exchange = ccxt.binance()

    for tf_name, tf_value in timeframes.items():
        try:
            # دریافت داده محدود برای سرعت
            since = exchange.parse8601("2024-06-01T00:00:00Z")  # فقط 4 ماه گذشته
            ohlcv = fetch_ohlcv(symbol, tf_value, since, limit=100)  # فقط 100 کندل

            if ohlcv and len(ohlcv) > 20:
                df_tf = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df_tf["timestamp"] = pd.to_datetime(df_tf["timestamp"], unit="ms")
                df_tf.set_index("timestamp", inplace=True)

                # محاسبه اندیکاتورهای سریع
                df_tf["SMA_20"] = df_tf["close"].rolling(20, min_periods=1).mean()
                df_tf["RSI"] = calculate_rsi(df_tf["close"])

                # تحلیل روند
                if len(df_tf) > 0:
                    current_price = df_tf["close"].iloc[-1]
                    sma_20 = df_tf["SMA_20"].iloc[-1]
                    rsi = df_tf["RSI"].iloc[-1]

                    trend = "صعودی" if current_price > sma_20 else "نزولی"
                    rsi_status = (
                        "اشباع خرید"
                        if rsi > 70
                        else "اشباع فروش" if rsi < 30 else "عادی"
                    )

                    analysis[tf_name] = {
                        "trend": trend,
                        "rsi": rsi,
                        "rsi_status": rsi_status,
                        "price_vs_sma": (current_price / sma_20 - 1) * 100,
                    }

                    print(
                        f"   📊 {tf_name}: روند {trend}, RSI: {rsi:.1f} ({rsi_status})"
                    )

        except Exception as e:
            print(f"⚠️ خطا در تحلیل {tf_name} برای {symbol}: {e}")

    return analysis


def calculate_rsi(prices, period=14):
    """محاسبه RSI سریع"""
    if len(prices) < period:
        return 50  # مقدار پیش‌فرض

    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if len(rsi) > 0 else 50


def confirm_signal_with_timeframes(signal, symbol, current_df):
    """تایید سیگنال با تحلیل چند تایم‌فریمی - منطق ساده‌تر"""

    if signal == 0:
        return 0, "سیگنال خنثی"

    analysis = get_multi_timeframe_analysis(symbol, current_df)

    if not analysis:
        return signal, "تحلیل تایم‌فریمی در دسترس نیست"

    # منطق تایید ساده‌تر
    confirmations = 0
    total_timeframes = len(analysis)

    for tf, data in analysis.items():
        if signal == 1:  # سیگنال خرید
            if data["trend"] == "صعودی" and data["rsi"] < 75:  # آستانه RSI بالاتر
                confirmations += 1
        else:  # سیگنال فروش
            if data["trend"] == "نزولی" and data["rsi"] > 25:  # آستانه RSI پایین‌تر
                confirmations += 1

    confirmation_ratio = confirmations / total_timeframes

    # فقط سیگنال‌های کاملاً تایید شده
    if confirmation_ratio >= 0.7:  # 70% تایید
        return signal, f"تایید شد ({confirmations}/{total_timeframes})"
    else:
        return 0, f"تایید نشد ({confirmations}/{total_timeframes})"
