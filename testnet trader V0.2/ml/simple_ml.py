# -*- coding: utf-8 -*-
# ml/simple_ml.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data.data_processor import extract_advanced_features


class SimpleMLTrainer:
    """آموزش مدل‌های ML ساده و مؤثر"""

    def __init__(self):
        self.models = {}

    def create_simple_features(self, df):
        """ایجاد ویژگی‌های ساده و مؤثر"""
        df = df.copy()

        # ویژگی‌های اصلی
        features = [
            "RSI",
            "MACD",
            "ADX",
            "BB_Bandwidth",
            "volatility_ratio",
            "volume_spike",
            "SMA_20",
            "SMA_50",
            "close",
        ]

        # نسبت‌های قیمت
        df["price_vs_sma20"] = df["close"] / df["SMA_20"]
        df["price_vs_sma50"] = df["close"] / df["SMA_50"]
        df["sma_ratio"] = df["SMA_20"] / df["SMA_50"]

        # مومنتوم
        df["momentum_5"] = df["close"].pct_change(5)
        df["momentum_10"] = df["close"].pct_change(10)

        return df

    def train_simple_model(self, df, symbol):
        """آموزش یک مدل ساده و مؤثر"""

        # ایجاد ویژگی‌ها
        df_enhanced = self.create_simple_features(df)

        # تارگت: آیا قیمت در 3 روز آینده 2% افزایش می‌یابد؟
        future_return = df_enhanced["close"].shift(-3) / df_enhanced["close"] - 1
        target = (future_return > 0.02).astype(int)

        # ویژگی‌های نهایی
        feature_columns = [
            "RSI",
            "MACD",
            "ADX",
            "BB_Bandwidth",
            "volatility_ratio",
            "price_vs_sma20",
            "price_vs_sma50",
            "sma_ratio",
            "momentum_5",
        ]

        # اطمینان از وجود تمام ویژگی‌ها
        available_features = [f for f in feature_columns if f in df_enhanced.columns]
        X = df_enhanced[available_features]
        y = target

        # حذف مقادیر NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 100:
            print(f"⚠️ داده کافی برای {symbol} وجود ندارد")
            return None, None, None

        # تقسیم داده
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"📊 آموزش مدل ساده برای {symbol}: {len(X_train)} نمونه آموزش")

        # مدل
        model = RandomForestClassifier(
            n_estimators=50, max_depth=8, min_samples_split=20, random_state=42
        )

        model.fit(X_train, y_train)

        # ارزیابی
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"✅ مدل ساده {symbol}: دقت = {accuracy:.3f}")

        if accuracy > 0.55:  # بهتر از تصادفی
            return model, None, available_features
        else:
            return None, None, None

    def train_all_models(self, symbols_data):
        """آموزش مدل برای تمام نمادها"""
        successful_models = 0

        for symbol, df in symbols_data.items():
            if len(df) > 200:
                print(f"🎯 آموزش مدل ساده برای {symbol}...")
                model, scaler, features = self.train_simple_model(df, symbol)

                if model is not None:
                    self.models[symbol] = (model, scaler, features, "sklearn")
                    successful_models += 1
                    print(f"✅ مدل ساده {symbol} آموزش داده شد")
                else:
                    print(f"❌ آموزش مدل ساده {symbol} ناموفق بود")

        print(f"\n🎉 آموزش کامل: {successful_models} مدل ساده آموزش داده شدند")
        return self.models
