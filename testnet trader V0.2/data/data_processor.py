# -*- coding: utf-8 -*-
# data/data_processor.py

import pandas as pd
import numpy as np
from utils.indicators import calculate_indicators


def extract_advanced_features(df, cross_symbol_data=None, symbol=None):
    """استخراج ویژگی‌های پیشرفته برای یادگیری ماشین"""
    df = df.copy()

    try:
        current_symbol = symbol if symbol else getattr(df, "name", "UNKNOWN")

        # 1. ویژگی‌های مبتنی بر قیمت
        df["price_momentum"] = df["close"].pct_change(5)
        df["price_acceleration"] = df["close"].pct_change(5) - df["close"].pct_change(
            10
        )
        df["volatility"] = df["close"].rolling(20).std() / (
            df["close"].rolling(20).mean() + 1e-10
        )

        # 2. ویژگی‌های مبتنی بر حجم
        df["volume_trend"] = df["volume"].pct_change(10)
        df["volume_volatility"] = df["volume"].rolling(20).std() / (
            df["volume"].rolling(20).mean() + 1e-10
        )

        # 3. ویژگی‌های الگوهای کندلی
        df["body_size"] = (df["close"] - df["open"]).abs() / (
            df["high"] - df["low"] + 1e-10
        )
        df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (
            df["high"] - df["low"] + 1e-10
        )
        df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (
            df["high"] - df["low"] + 1e-10
        )

        # 4. ویژگی‌های روند و مومنتوم
        df["trend_strength"] = (df["close"] - df["close"].rolling(50).mean()) / (
            df["close"].rolling(50).std() + 1e-10
        )

        # 5. ویژگی‌های نوسان
        df["atr_ratio"] = df["ATR"] / (df["close"] + 1e-10)
        df["bb_squeeze"] = (df["BB_Upper"] - df["BB_Lower"]) / (df["BB_Middle"] + 1e-10)

        # 6. Lagged features (کمتر برای جلوگیری از overfitting)
        for lag in [1, 2]:
            df[f"RSI_lag{lag}"] = df["RSI"].shift(lag)
            df[f"MACD_lag{lag}"] = df["MACD"].shift(lag)

        # 7. Interactions
        df["RSI_MACD_interact"] = df["RSI"] * df["MACD"]

        # 8. New features
        df["price_vs_sma20"] = df["close"] / df["SMA_20"]
        df["price_vs_sma50"] = df["close"] / df["SMA_50"]
        df["sma_ratio"] = df["SMA_20"] / df["SMA_50"]

        print(f"✅ استخراج ویژگی‌های ML برای {current_symbol} موفق بود")

    except Exception as e:
        print(f"❌ خطا در استخراج ویژگی‌های ML برای {current_symbol}: {e}")
        # Fallback features
        df["price_momentum"] = df["close"].pct_change(5)
        df["volatility"] = df["close"].rolling(20).std() / (
            df["close"].rolling(20).mean() + 1e-10
        )
        df["volume_trend"] = df["volume"].pct_change(5)

    return df


def create_ml_features(df, target_lookahead=3):  # کاهش lookahead به 3
    """ایجاد ویژگی‌ها و تارگت برای مدل ML با آستانه منطقی"""
    from config import ML_SETTINGS

    target_lookahead = ML_SETTINGS["target_lookahead"]
    symbol = getattr(df, "name", "UNKNOWN")
    df = extract_advanced_features(df, cross_symbol_data=None, symbol=symbol)

    # تارگت با آستانه پویا و منطقی
    future_returns = df["close"].shift(-target_lookahead) / df["close"] - 1
    df["future_return"] = future_returns

    # آستانه هوشمند بر اساس volatility بازار
    market_volatility = df["close"].pct_change().std()
    return_threshold = max(
        0.02, market_volatility * 2
    )  # حداقل 2%، حداکثر 2 برابر نوسان

    # تارگت سه کلاسی: -1 (فروش), 0 (خنثی), 1 (خرید)
    df["target"] = 0  # پیش‌فرض خنثی
    df.loc[future_returns > return_threshold, "target"] = 1  # خرید
    df.loc[future_returns < -return_threshold, "target"] = -1  # فروش

    print(
        f"📌 آستانه تارگت برای {symbol}: {return_threshold:.4f} ({return_threshold*100:.2f}%)"
    )

    # ویژگی‌های اصلی (کاهش تعداد برای جلوگیری از overfitting)
    base_features = [
        "price_momentum",
        "volatility",
        "volume_trend",
        "body_size",
        "trend_strength",
        "RSI",
        "MACD",
        "ADX",
        "BB_Bandwidth",
        "atr_ratio",
        "price_vs_sma20",
        "sma_ratio",
    ]

    available_features = [f for f in base_features if f in df.columns]

    # اضافه کردن lagged features اگر موجود هستند
    for lag in [1, 2]:
        if f"RSI_lag{lag}" in df.columns:
            available_features.append(f"RSI_lag{lag}")
        if f"MACD_lag{lag}" in df.columns:
            available_features.append(f"MACD_lag{lag}")

    df_clean = df[available_features + ["future_return", "target"]].dropna()

    print(f"📊 تعداد ویژگی‌های ML: {len(available_features)}")
    print(f"📈 تعداد نمونه‌های قابل استفاده: {len(df_clean)}")

    # نمایش توزیع تارگت
    target_dist = df_clean["target"].value_counts().sort_index()
    print(
        f"📊 توزیع تارگت: فروش={target_dist.get(-1, 0)}, خنثی={target_dist.get(0, 0)}, خرید={target_dist.get(1, 0)}"
    )

    return df_clean, available_features
