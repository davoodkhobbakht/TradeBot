# -*- coding: utf-8 -*-
# utils/indicators.py

import pandas as pd
import numpy as np


def calculate_indicators(df):
    """محاسبه تمام اندیکاتورهای مورد نیاز"""
    df = df.copy()

    # میانگین‌های متحرک
    df["SMA_20"] = df["close"].rolling(window=20, min_periods=1).mean()
    df["SMA_50"] = df["close"].rolling(window=50, min_periods=1).mean()
    df["SMA_200"] = df["close"].rolling(window=200, min_periods=1).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False, min_periods=1).mean()
    ema26 = df["close"].ewm(span=26, adjust=False, min_periods=1).mean()
    df["MACD"] = ema12 - ema26
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False, min_periods=1).mean()
    df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]

    # Bollinger Bands
    df["BB_Middle"] = df["close"].rolling(window=20, min_periods=1).mean()
    std = df["close"].rolling(window=20, min_periods=1).std()
    df["BB_Upper"] = df["BB_Middle"] + (std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (std * 2)
    df["BB_Bandwidth"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]

    # ATR
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df["ATR"] = true_range.rolling(window=14, min_periods=1).mean()

    # ADX
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where(plus_dm > 0, 0)
    minus_dm = minus_dm.where(minus_dm > 0, 0)
    tr = true_range
    atr = tr.rolling(window=14, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=1).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=1).mean() / (atr + 1e-10))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df["ADX"] = dx.rolling(window=14, min_periods=1).mean()

    # Stochastic Oscillator
    low_min = df["low"].rolling(window=14, min_periods=1).min()
    high_max = df["high"].rolling(window=14, min_periods=1).max()
    df["stoch_k"] = ((df["close"] - low_min) / (high_max - low_min + 1e-10)) * 100
    df["stoch_d"] = df["stoch_k"].rolling(window=3, min_periods=1).mean()

    # ویژگی‌های اضافی
    df["volatility_ratio"] = df["BB_Bandwidth"] / df["BB_Bandwidth"].rolling(50).mean()
    df["volume_spike"] = df["volume"] / df["volume"].rolling(20).mean()

    return df


def dynamic_trailing_stop(current_price, entry_price, trailing_high, volatility):
    """ترییلینگ استاپ پویا بر اساس نوسان"""
    base_trailing = 0.06  # 6% پایه

    if volatility > 0.02:  # نوسان بالا
        trailing_pct = base_trailing * 1.5
    else:
        trailing_pct = base_trailing

    return trailing_high * (1 - trailing_pct)
