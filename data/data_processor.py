# -*- coding: utf-8 -*-
# data/data_processor.py - FIXED VERSION

import pandas as pd
import numpy as np
from utils.indicators import calculate_indicators


def extract_advanced_features(df, cross_symbol_data=None, symbol=None):
    """استخراج ویژگی‌های پیشرفته - فقط از داده‌های گذشته"""
    df = df.copy()

    try:
        current_symbol = symbol if symbol else getattr(df, "name", "UNKNOWN")

        # 1. ویژگی‌های مبتنی بر قیمت (کاهش تعداد برای جلوگیری از overfitting)
        df["price_momentum_5"] = df["close"].pct_change(5)
        df["price_acceleration"] = df["close"].pct_change(5) - df["close"].pct_change(
            10
        )

        # 2. نوسانات تاریخی (فقط یکی)
        df["volatility_20"] = df["close"].rolling(20).std() / (
            df["close"].rolling(20).mean() + 1e-10
        )

        # 3. ویژگی‌های حجم (کاهش)
        df["volume_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)

        # 4. الگوهای کندلی (فقط body_size)
        df["body_size"] = (df["close"] - df["open"]).abs() / (
            df["high"] - df["low"] + 1e-10
        )

        # 5. نسبت‌های قیمتی (کاهش)
        df["price_to_sma20"] = df["close"] / (df["SMA_20"] + 1e-10)

        # 6. قدرت روند
        df["trend_strength"] = (df["close"] - df["close"].rolling(20).mean()) / (
            df["close"].rolling(20).std() + 1e-10
        )

        # 7. ویژگی‌های نوسان
        df["bb_position"] = (df["close"] - df["BB_Lower"]) / (
            df["BB_Upper"] - df["BB_Lower"] + 1e-10
        )

        # 8. Lagged features محدود (فقط یکی برای جلوگیری از overfitting)
        df["rsi_lag1"] = df["RSI"].shift(1)

        print(f"✅ استخراج ویژگی‌های ML برای {current_symbol} موفق بود")

    except Exception as e:
        print(f"❌ خطا در استخراج ویژگی‌های ML برای {current_symbol}: {e}")
        # Fallback به ویژگی‌های ساده
        df["price_momentum_5"] = df["close"].pct_change(5)
        df["volatility_20"] = df["close"].rolling(20).std() / (
            df["close"].rolling(20).mean() + 1e-10
        )

    return df


def create_ml_features(df, target_lookahead=3):
    """ایجاد ویژگی‌ها و تارگت - بدون Data Leakage"""
    from config import ML_SETTINGS

    target_lookahead = ML_SETTINGS["target_lookahead"]
    symbol = getattr(df, "name", "UNKNOWN")

    # استخراج ویژگی‌ها (فقط از گذشته)
    df = extract_advanced_features(df, cross_symbol_data=None, symbol=symbol)

    # === محاسبه تارگت (جداگانه، فقط برای training) ===
    # این بخش فقط موقع training استفاده میشه، نه prediction
    future_price = df["close"].shift(-target_lookahead)
    future_return = (future_price - df["close"]) / df["close"]

    # آستانه پویا بر اساس volatility
    market_volatility = df["close"].pct_change().std()
    return_threshold = max(0.02, market_volatility * 2)

    # تارگت سه کلاسی
    target = pd.Series(0, index=df.index)
    target[future_return > return_threshold] = 1  # خرید
    target[future_return < -return_threshold] = -1  # فروش

    # CRITICAL: target رو جدا نگه می‌داریم، نه توی df
    df["target"] = target

    print(f"📌 آستانه تارگت {symbol}: ±{return_threshold*100:.2f}%")

    # لیست ویژگی‌های اصلی (کاهش یافته برای جلوگیری از overfitting)
    feature_columns = [
        # Momentum
        "price_momentum_5",
        "price_acceleration",
        # Volatility
        "volatility_20",
        # Volume
        "volume_ratio",
        # Candle patterns
        "body_size",
        # Technical indicators
        "RSI",
        "MACD",
        "ADX",
        # Price ratios
        "price_to_sma20",
        # Trend
        "trend_strength",
        # Bollinger
        "bb_position",
        # Lagged
        "rsi_lag1",
    ]

    # فقط ویژگی‌هایی که وجود دارن
    available_features = [f for f in feature_columns if f in df.columns]

    # حذف NaN ها
    required_cols = available_features + ["target"]
    df_clean = df[required_cols].dropna()

    print(f"📊 تعداد ویژگی‌ها: {len(available_features)}")
    print(f"📈 تعداد نمونه‌ها: {len(df_clean)}")

    # نمایش توزیع تارگت
    target_dist = df_clean["target"].value_counts().sort_index()
    print(
        f"📊 توزیع: Sell={target_dist.get(-1, 0)}, Hold={target_dist.get(0, 0)}, Buy={target_dist.get(1, 0)}"
    )

    return df_clean, available_features


def prepare_train_test_split(df, feature_columns, test_size=0.3):
    """Split بدون Leakage برای Time Series"""
    from sklearn.preprocessing import StandardScaler

    # مرتب‌سازی
    df = df.sort_index()

    # Split نقطه
    split_idx = int(len(df) * (1 - test_size))

    # جداسازی
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # X و y
    X_train = train_df[feature_columns]
    y_train = train_df["target"]
    X_test = test_df[feature_columns]
    y_test = test_df["target"]

    # استانداردسازی (فقط روی train fit میشه)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # فقط transform

    # تبدیل به DataFrame
    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=feature_columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=feature_columns, index=X_test.index
    )

    print(f"📊 Train: {len(X_train)} ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"📊 Test: {len(X_test)} ({test_df.index[0]} to {test_df.index[-1]})")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def walk_forward_validation(df, feature_columns, n_splits=5):
    """Walk-Forward Validation"""
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    df = df.sort_index()
    tscv = TimeSeriesSplit(n_splits=n_splits)

    X = df[feature_columns]
    y = df["target"]

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        # Split
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale (هر fold جدا)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # تبدیل به DataFrame
        X_train_scaled = pd.DataFrame(
            X_train_scaled, columns=feature_columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled, columns=feature_columns, index=X_test.index
        )

        print(f"📊 Fold {fold}: Train={len(X_train)}, Test={len(X_test)}")

        fold_results.append(
            {
                "X_train": X_train_scaled,
                "X_test": X_test_scaled,
                "y_train": y_train,
                "y_test": y_test,
                "scaler": scaler,
            }
        )

    return fold_results
