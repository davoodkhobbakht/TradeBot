# -*- coding: utf-8 -*-
# ml/base_ml.py

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import shap
from data.data_processor import create_ml_features


class MLModelManager:
    """مدیریت مدل‌های یادگیری ماشین برای ارزهای مختلف"""

    def __init__(self):
        self.models = {}  # {symbol: (model, scaler, features, model_type)}

    def save_models(self, path="models/"):
        """ذخیره مدل‌ها با پشتیبانی از انواع مختلف مدل"""
        import os

        os.makedirs(path, exist_ok=True)

        for symbol, (model, scaler, features, model_type) in self.models.items():
            symbol_filename = symbol.replace("/", "_")

            try:
                # ذخیره مدل بر اساس نوع
                if model_type == "keras":
                    model.save(f"{path}{symbol_filename}_model.h5")
                else:  # scikit-learn models
                    joblib.dump(model, f"{path}{symbol_filename}_model.pkl")

                # ذخیره scaler و features
                joblib.dump(scaler, f"{path}{symbol_filename}_scaler.pkl")
                joblib.dump(features, f"{path}{symbol_filename}_features.pkl")
                joblib.dump(model_type, f"{path}{symbol_filename}_type.pkl")

                print(f"💾 مدل {symbol} ذخیره شد (نوع: {model_type})")

            except Exception as e:
                print(f"❌ خطا در ذخیره مدل {symbol}: {e}")

        print(f"✅ تمام مدل‌ها در {path} ذخیره شدند")

    def load_models(self, path="models/"):
        """بارگذاری مدل‌های ذخیره شده"""
        import os
        from tensorflow.keras.models import load_model

        if not os.path.exists(path):
            print("📂 پوشه مدل‌ها یافت نشد")
            return False

        loaded_count = 0
        for filename in os.listdir(path):
            if filename.endswith("_model.h5") or filename.endswith("_model.pkl"):
                try:
                    if filename.endswith("_model.h5"):
                        symbol = filename.replace("_model.h5", "").replace("_", "/")
                        model_type = "keras"
                    else:
                        symbol = filename.replace("_model.pkl", "").replace("_", "/")
                        model_type = "sklearn"

                    model_path = f"{path}{filename}"
                    scaler_path = f'{path}{symbol.replace("/", "_")}_scaler.pkl'
                    features_path = f'{path}{symbol.replace("/", "_")}_features.pkl'
                    type_path = f'{path}{symbol.replace("/", "_")}_type.pkl'

                    if all(os.path.exists(p) for p in [scaler_path, features_path]):
                        # بارگذاری مدل
                        if model_type == "keras":
                            model = load_model(model_path)
                        else:
                            model = joblib.load(model_path)

                        scaler = joblib.load(scaler_path)
                        features = joblib.load(features_path)

                        # اگر فایل نوع موجود باشد استفاده کن، در غیر این صورت از پسوند فایل
                        if os.path.exists(type_path):
                            model_type = joblib.load(type_path)

                        self.models[symbol] = (model, scaler, features, model_type)
                        loaded_count += 1
                        print(f"📂 مدل {symbol} بارگذاری شد (نوع: {model_type})")
                    else:
                        print(f"⚠️ فایل‌های کامل برای {symbol} یافت نشد")
                except Exception as e:
                    print(f"❌ خطا در بارگذاری مدل {filename}: {e}")

        if loaded_count == 0:
            print("⚠️ هیچ مدلی بارگذاری نشد")
            return False
        else:
            print(f"✅ تعداد {loaded_count} مدل بارگذاری شد")
            return True

    def get_model(self, symbol):
        """دریافت مدل مربوط به یک ارز"""
        return self.models.get(symbol, (None, None, None, None))

    def train_models_for_symbols(self, symbols_data):
        """آموزش مدل برای تمام ارزها"""
        print("🤖 شروع آموزش مدل‌های یادگیری ماشین...")
        successful_models = 0

        for symbol, df in symbols_data.items():
            if len(df) > 300:
                print(f"📚 آموزش مدل برای {symbol}...")
                try:
                    df.name = symbol
                    model, scaler, features, model_type = self.train_ml_model(
                        df, symbol
                    )
                    if model is not None:
                        self.models[symbol] = (model, scaler, features, model_type)
                        successful_models += 1
                        print(f"✅ مدل {symbol} آموزش داده شد")
                    else:
                        print(f"❌ آموزش مدل {symbol} ناموفق بود")
                except Exception as e:
                    print(f"❌ خطا در آموزش مدل {symbol}: {e}")
            else:
                print(f"⚠️ داده کافی برای {symbol} وجود ندارد (فقط {len(df)} کندل)")

        print(
            f"🎯 آموزش کامل شد: {successful_models} از {len(symbols_data)} مدل با موفقیت آموزش دیدند"
        )

    def train_ml_model(self, df, symbol):
        """آموزش مدل LSTM برای یک نماد"""
        try:
            df_features, feature_columns = create_ml_features(df)

            if len(df_features) < 100:
                print(f"⚠️ داده کافی برای آموزش مدل {symbol} وجود ندارد")
                return None, None, None, None

            X = df_features[feature_columns]
            y = df_features["target"]

            # بررسی تعادل داده‌ها
            target_counts = y.value_counts()
            print(f"📊 توزیع تارگت {symbol}: {dict(target_counts)}")

            # اگر داده‌های مثبت خیلی کم هستند، آستانه را کاهش می‌دهیم
            if target_counts.get(1, 0) < len(y) * 0.15:
                print(f"⚠️ داده‌های مثبت کم هستند. تنظیم آستانه...")
                # استفاده از مدل ساده‌تر
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import f1_score

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, shuffle=False
                )

                model = RandomForestClassifier(
                    n_estimators=50, max_depth=10, min_samples_split=10, random_state=42
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                test_f1 = f1_score(y_test, y_pred)

                print(f"🔍 مدل ساده {symbol}: F1 = {test_f1:.3f}")

                if test_f1 > 0.1:  # آستانه پایین‌تر برای مدل‌های ساده
                    return model, None, feature_columns, "sklearn"
                else:
                    return None, None, None, None

            # تقسیم داده
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, shuffle=False
            )

            if len(X_train) < 50:
                print(f"⚠️ داده آموزش کافی برای {symbol} وجود ندارد")
                return None, None, None, None

            # استانداردسازی
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # SMOTE برای مقابله با عدم تعادل داده‌ها
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(
                X_train_scaled, y_train
            )

            # مدل LSTM
            X_train_lstm = X_train_resampled.reshape(
                (X_train_resampled.shape[0], 1, X_train_resampled.shape[1])
            )
            X_test_lstm = X_test_scaled.reshape(
                (X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
            )

            model = Sequential(
                [
                    LSTM(
                        32,
                        return_sequences=True,
                        input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]),
                    ),
                    Dropout(0.2),
                    LSTM(16, return_sequences=False),
                    Dropout(0.2),
                    Dense(8, activation="relu"),
                    Dense(1, activation="sigmoid"),
                ]
            )

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            print(f"⏳ در حال آموزش مدل LSTM برای {symbol}...")
            history = model.fit(
                X_train_lstm,
                y_train_resampled,
                epochs=30,
                batch_size=16,
                validation_data=(X_test_lstm, y_test),
                verbose=0,
            )

            # ارزیابی مدل
            train_accuracy = history.history["accuracy"][-1]
            val_accuracy = history.history["val_accuracy"][-1]
            print(
                f"✅ مدل {symbol} آموزش داده شد - دقت آموزش: {train_accuracy:.3f}, دقت اعتبارسنجی: {val_accuracy:.3f}"
            )

            return model, scaler, feature_columns, "keras"

        except Exception as e:
            print(f"❌ خطا در آموزش مدل {symbol}: {e}")
            return None, None, None, None


def predict_with_ml(model, scaler, feature_columns, df_current, model_type="keras"):
    """پیش‌بینی با مدل ML روی داده‌های جاری"""
    available_features = [f for f in feature_columns if f in df_current.columns]

    if len(available_features) != len(feature_columns):
        missing_features = [f for f in feature_columns if f not in df_current.columns]
        print(f"⚠️ ویژگی‌های غایب: {len(missing_features)} از {len(feature_columns)}")
        return 0.5

    X_current = df_current[available_features].iloc[[-1]]

    if scaler is not None:
        X_scaled = scaler.transform(X_current)
    else:
        X_scaled = X_current.values

    if model_type == "keras":
        X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        probability = model.predict(X_lstm)[0][0]
    else:  # sklearn models
        probability = model.predict_proba(X_scaled)[0][1]

    return probability
