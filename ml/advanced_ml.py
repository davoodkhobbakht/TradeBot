# -*- coding: utf-8 -*-
# ml/advanced_ml.py

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.multioutput import ClassifierChain
import warnings

warnings.filterwarnings("ignore")

from data.data_processor import create_ml_features


class AdvancedMLTrainer:
    """آموزش پیشرفته مدل‌های ML با طبقه‌بندی سه کلاسی"""

    def __init__(self):
        self.best_models = {}

    def train_multiclass_model(self, X, y, feature_columns, symbol):
        """آموزش مدل برای طبقه‌بندی سه کلاسی"""
        from sklearn.multioutput import ClassifierChain
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score

        # تقسیم داده
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"📊 تقسیم داده {symbol}: آموزش={len(X_train)}, تست={len(X_test)}")

        # مدل‌های مختلف برای تست
        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                class_weight="balanced",
                random_state=42,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            ),
            "LogisticRegression": LogisticRegression(
                class_weight="balanced", random_state=42, max_iter=1000
            ),
        }

        best_model = None
        best_score = 0
        best_name = ""

        for name, model in models.items():
            try:
                print(f"🔍 آموزش {name} برای {symbol}...")

                # آموزش مدل
                model.fit(X_train, y_train)

                # پیش‌بینی و ارزیابی
                y_pred = model.predict(X_test)

                # محاسبه F1 برای هر کلاس
                f1_scores = f1_score(y_test, y_pred, average=None, labels=[-1, 0, 1])
                macro_f1 = f1_score(y_test, y_pred, average="macro")

                print(
                    f"   📊 F1 Scores: فروش={f1_scores[0]:.3f}, خنثی={f1_scores[1]:.3f}, خرید={f1_scores[2]:.3f}"
                )
                print(f"   📈 Macro F1: {macro_f1:.3f}")

                # وزن بیشتر به کلاس‌های خرید و فروش
                weighted_score = (
                    (f1_scores[0] * 0.4) + (f1_scores[2] * 0.4) + (macro_f1 * 0.2)
                )

                if weighted_score > best_score and macro_f1 > 0.3:
                    best_score = weighted_score
                    best_model = model
                    best_name = name

            except Exception as e:
                print(f"❌ خطا در آموزش {name} برای {symbol}: {e}")

        if best_model is not None:
            print(
                f"✅ بهترین مدل برای {symbol}: {best_name} با امتیاز: {best_score:.3f}"
            )

            # آنالیز اهمیت ویژگی‌ها
            if hasattr(best_model, "feature_importances_"):
                feature_importance = pd.DataFrame(
                    {
                        "feature": feature_columns,
                        "importance": best_model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)

                print(f"🔝 5 ویژگی مهم برای {symbol}:")
                for _, row in feature_importance.head().iterrows():
                    print(f"   {row['feature']}: {row['importance']:.4f}")

            return best_model, None, feature_columns, "sklearn"

        return None, None, None, None

    def train_binary_model(self, X, y, feature_columns, symbol):
        """آموزش مدل باینری ساده‌تر"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score

        # فقط کلاس‌های خرید و فروش را در نظر بگیریم (خنثی را حذف کنیم)
        binary_mask = y != 0
        X_binary = X[binary_mask]
        y_binary = y[binary_mask]

        if len(X_binary) < 100:
            print(f"⚠️ داده باینری کافی برای {symbol} وجود ندارد")
            return None, None, None, None

        # تقسیم داده
        split_idx = int(len(X_binary) * 0.7)
        X_train, X_test = X_binary.iloc[:split_idx], X_binary.iloc[split_idx:]
        y_train, y_test = y_binary.iloc[:split_idx], y_binary.iloc[split_idx:]

        print(f"🎯 آموزش مدل باینری برای {symbol}...")
        print(f"📊 داده‌های باینری: {len(X_binary)} نمونه")

        # مدل ساده
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            min_samples_split=15,
            class_weight="balanced",
            random_state=42,
        )

        model.fit(X_train, y_train)

        # ارزیابی
        y_pred = model.predict(X_test)
        test_f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"📊 مدل باینری {symbol}: F1 = {test_f1:.3f}")

        if test_f1 > 0.4:  # آستانه منطقی
            return model, None, feature_columns, "sklearn"
        else:
            return None, None, None, None

    def train_advanced_model(self, df, symbol):
        """آموزش مدل پیشرفته"""

        # ایجاد ویژگی‌ها
        df_features, feature_columns = create_ml_features(df)

        if len(df_features) < 200:
            print(f"⚠️ داده ناکافی برای آموزش پیشرفته {symbol}")
            return None, None, None, None

        X = df_features[feature_columns]
        y = df_features["target"]

        # نمایش اطلاعات داده
        target_counts = y.value_counts().sort_index()
        print(f"📊 توزیع تارگت {symbol}: {target_counts.to_dict()}")

        # اگر داده کافی داریم، مدل چندکلاسی آموزش بده
        if (
            len(y) > 300
            and target_counts.get(1, 0) > 50
            and target_counts.get(-1, 0) > 50
        ):
            print("🎯 آموزش مدل چندکلاسی...")
            return self.train_multiclass_model(X, y, feature_columns, symbol)
        else:
            print("🎯 آموزش مدل باینری (خرید/فروش)...")
            return self.train_binary_model(X, y, feature_columns, symbol)

    def create_advanced_features(self, df, symbol):
        """ایجاد ویژگی‌های پیشرفته‌تر"""
        df = df.copy()

        # ویژگی‌های زمانی ساده
        df["day_of_week"] = df.index.dayofweek

        # ویژگی‌های قیمتی
        df["hl_ratio"] = (df["high"] - df["low"]) / (df["close"] + 1e-10)
        df["price_range"] = (df["high"] - df["low"]) / df["close"]

        return df


def enhanced_ml_training_pipeline(symbols_data, ml_manager):
    """پایپلاین آموزش پیشرفته ML"""
    print("🚀 شروع پایپلاین آموزش پیشرفته ML...")

    advanced_trainer = AdvancedMLTrainer()

    successful_models = 0
    for symbol, df in symbols_data.items():
        if len(df) < 300:
            continue

        print(f"\n🎯 آموزش پیشرفته برای {symbol}...")
        model, scaler, features, model_type = advanced_trainer.train_advanced_model(
            df, symbol
        )

        if model is not None:
            ml_manager.models[symbol] = (model, scaler, features, model_type)
            successful_models += 1
            print(f"✅ مدل پیشرفته {symbol} آموزش داده شد (نوع: {model_type})")
        else:
            print(f"❌ آموزش پیشرفته {symbol} ناموفق بود")

    print(
        f"\n🎉 آموزش کامل: {successful_models} از {len(symbols_data)} مدل با موفقیت آموزش داده شدند"
    )
    return ml_manager
