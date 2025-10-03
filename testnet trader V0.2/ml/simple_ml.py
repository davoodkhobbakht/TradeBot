# -*- coding: utf-8 -*-
# ml/simple_ml.py - FIXED VERSION

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from data.data_processor import create_ml_features, prepare_train_test_split


class SimpleMLTrainer:
    """آموزش مدل‌های ML ساده و صحیح"""

    def __init__(self):
        self.models = {}

    def train_simple_model(self, df, symbol):
        """آموزش مدل بدون Data Leakage"""

        print(f"\n{'='*60}")
        print(f"🎯 آموزش مدل برای {symbol}")
        print(f"{'='*60}")

        # ایجاد features و target (بدون leakage)
        df_features, feature_columns = create_ml_features(df)

        if len(df_features) < 200:
            print(f"⚠️ داده ناکافی برای {symbol}: {len(df_features)} نمونه")
            return None, None, None

        # بررسی تعادل کلاس‌ها
        target_counts = df_features["target"].value_counts()
        total = len(df_features)

        print(f"\n📊 توزیع اولیه کلاس‌ها:")
        for cls in [-1, 0, 1]:
            count = target_counts.get(cls, 0)
            pct = count / total * 100
            class_name = {-1: "Sell", 0: "Hold", 1: "Buy"}[cls]
            print(f"   {class_name}: {count} ({pct:.1f}%)")

        # اگر کلاس‌های Buy/Sell خیلی کم هستند، از binary target استفاده کن
        buy_count = target_counts.get(1, 0)
        sell_count = target_counts.get(-1, 0)

        if buy_count < 30 or sell_count < 30:
            print(f"⚠️ کلاس‌های Buy/Sell کم هستند. تبدیل به binary...")
            # فقط Buy (1) و Not Buy (0) رو در نظر بگیر
            df_features["target"] = (df_features["target"] == 1).astype(int)

        # Split بدون leakage
        X_train, X_test, y_train, y_test, scaler = prepare_train_test_split(
            df_features, feature_columns, test_size=0.3
        )

        # بررسی توزیع در train/test
        print(f"\n📊 توزیع Train: {y_train.value_counts().to_dict()}")
        print(f"📊 توزیع Test: {y_test.value_counts().to_dict()}")

        # مدل ساده
        print(f"\n🤖 آموزش Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        # آموزش
        model.fit(X_train, y_train)

        # ارزیابی
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_f1 = f1_score(y_train, y_train_pred, average="weighted")
        test_f1 = f1_score(y_test, y_test_pred, average="weighted")

        print(f"\n📈 نتایج:")
        print(f"   Train Accuracy: {train_acc:.3f}")
        print(f"   Test Accuracy: {test_acc:.3f}")
        print(f"   Train F1: {train_f1:.3f}")
        print(f"   Test F1: {test_f1:.3f}")

        # بررسی overfitting
        overfit_gap = train_acc - test_acc
        if overfit_gap > 0.15:
            print(f"⚠️ احتمال Overfitting: Gap = {overfit_gap:.3f}")

        # نمایش Classification Report
        print(f"\n📊 Classification Report (Test):")
        print(classification_report(y_test, y_test_pred))

        # Feature Importance
        feature_importance = pd.DataFrame(
            {"feature": feature_columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        print(f"\n🔝 Top 5 Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")

        # قبول مدل اگر:
        # 1. Test accuracy بهتر از baseline (0.5 برای binary, 0.33 برای 3-class)
        # 2. Test F1 معقول باشه
        # 3. Overfitting زیاد نباشه

        baseline = 0.5 if len(y_test.unique()) == 2 else 0.33

        if test_acc > baseline + 0.05 and test_f1 > 0.4 and overfit_gap < 0.20:
            print(f"\n✅ مدل قابل قبول است")
            return model, scaler, feature_columns
        else:
            print(f"\n❌ مدل قابل قبول نیست:")
            if test_acc <= baseline + 0.05:
                print(
                    f"   - Test Accuracy پایین: {test_acc:.3f} vs baseline {baseline:.3f}"
                )
            if test_f1 <= 0.4:
                print(f"   - Test F1 پایین: {test_f1:.3f}")
            if overfit_gap >= 0.20:
                print(f"   - Overfitting زیاد: {overfit_gap:.3f}")
            return None, None, None

    def train_all_models(self, symbols_data):
        """آموزش مدل برای همه symbols"""
        successful_models = 0

        for symbol, df in symbols_data.items():
            if len(df) > 300:
                model, scaler, features = self.train_simple_model(df, symbol)

                if model is not None:
                    self.models[symbol] = (model, scaler, features, "sklearn")
                    successful_models += 1
                    print(f"\n✅ مدل {symbol} با موفقیت آموزش داده شد\n")
                else:
                    print(f"\n❌ مدل {symbol} رد شد\n")

        print(f"\n{'='*60}")
        print(f"🎉 آموزش کامل: {successful_models}/{len(symbols_data)} مدل موفق")
        print(f"{'='*60}\n")

        return self.models
