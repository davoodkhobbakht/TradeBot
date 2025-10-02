# -*- coding: utf-8 -*-
# strategies/base_strategy.py

import pandas as pd
import numpy as np


def generate_signals(df):
    """تولید سیگنال‌های معاملاتی با فیلترهای متعادل"""
    # ایجاد کپی از دیتافریم و حفظ نام نماد
    df = df.copy()
    symbol = getattr(df, "name", "UNKNOWN")

    print(f"\n🔍 تولید سیگنال برای {symbol}...")

    # ۱. محاسبه اندیکاتورهای پایه
    df["avg_volume"] = df["volume"].rolling(window=20, min_periods=1).mean()
    df["primary_trend"] = np.where(df["SMA_50"] > df["SMA_200"], 1, -1)

    # ۲. سیستم امتیازدهی متعادل‌تر
    buy_score = (
        ((df["SMA_20"] > df["SMA_50"]) & (df["primary_trend"] == 1)) * 2
        + (df["close"] > df["SMA_20"]) * 2
        + ((df["RSI"] > 40) & (df["RSI"] < 70)) * 2
        + (df["MACD"] > df["Signal_Line"]) * 2
        + (df["ADX"] > 20) * 1  # کاهش آستانه ADX
        + (df["volume"] > df["avg_volume"] * 1.0) * 1  # کاهش آستانه حجم
        + ((df["close"] > df["BB_Lower"]) & (df["close"] < df["BB_Middle"])) * 1
    )

    sell_score = (
        ((df["SMA_20"] < df["SMA_50"]) & (df["primary_trend"] == -1)) * 2
        + (df["close"] < df["SMA_20"]) * 2
        + (df["RSI"] > 70) * 2
        + (df["MACD"] < df["Signal_Line"]) * 2
        + (df["close"] > df["BB_Upper"]) * 1
        + (df["ADX"] > 20) * 1
        + (df["volume"] > df["avg_volume"] * 1.0) * 1
    )

    # ۳. تولید سیگنال با آستانه‌های متعادل
    df["signal"] = 0
    buy_threshold = 6  # آستانه متعادل
    sell_threshold = 6  # آستانه متعادل

    for i in range(len(df)):
        if buy_score.iloc[i] >= buy_threshold and df["RSI"].iloc[i] < 70:
            df.iloc[i, df.columns.get_loc("signal")] = 1
        if sell_score.iloc[i] >= sell_threshold and df["RSI"].iloc[i] > 30:
            df.iloc[i, df.columns.get_loc("signal")] = -1

    print(f"📊 میانگین امتیاز خرید {symbol}: {buy_score.mean():.2f}")
    print(f"📊 میانگین امتیاز فروش {symbol}: {sell_score.mean():.2f}")
    print(f"🎯 آستانه سیگنال: {buy_threshold}")

    # ۴. ایجاد موقعیت‌های واقعی با فیلتر نویز
    df["position"] = 0
    last_position = 0
    min_hold_days = 3  # حداقل نگهداری 3 روز

    for i in range(1, len(df)):
        current_signal = df["signal"].iloc[i]

        # فقط اگر سیگنال تغییر کرده و فاصله کافی داره
        if current_signal != last_position:
            df.iloc[i, df.columns.get_loc("position")] = current_signal
            last_position = current_signal
        else:
            df.iloc[i, df.columns.get_loc("position")] = last_position

    # ۵. آمار نهایی
    buy_positions = len(df[df["position"] == 1])
    sell_positions = len(df[df["position"] == -1])
    total_positions = buy_positions + sell_positions

    print(f"✅ سیگنال‌های نهایی {symbol}: {buy_positions} خرید, {sell_positions} فروش")
    print(f"📈 درصد موقعیت‌ها: {(total_positions/len(df)*100):.1f}%")

    return df
