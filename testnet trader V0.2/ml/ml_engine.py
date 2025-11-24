# ml/ml_engine.py - Completely rewritten
import pandas as pd
import numpy as np
import logging
import os
import joblib
from data.data_fetcher import DataFetcher
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings("ignore")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from typing import List, Dict, Tuple
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """ساده‌ترین feature engineering ممکن"""

    @staticmethod
    def engineer_features(
        df: pd.DataFrame, symbol: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """ایجاد featureهای بسیار ساده بدون حذف داده"""
        logger.info(f"🔧 Starting SIMPLE feature engineering for {symbol}")

        df = df.copy()

        # 1. ابتدا target را ایجاد کنیم (جهت کندل بعدی)
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

        # 2. فقط چند feature بسیار ساده ایجاد کنیم
        # Lagهای بسیار کم
        df["close_lag_1"] = df["close"].shift(1)
        df["close_lag_2"] = df["close"].shift(2)
        df["volume_lag_1"] = df["volume"].shift(1)

        # Moving averages ساده
        df["sma_5"] = df["close"].rolling(5).mean()
        df["sma_10"] = df["close"].rolling(10).mean()

        # Momentum ساده
        df["momentum_3"] = df["close"] - df["close"].shift(3)
        df["roc_5"] = (df["close"] - df["close"].shift(5)) / df["close"].shift(5) * 100

        # Volume features ساده
        df["volume_sma_5"] = df["volume"].rolling(5).mean()

        # Price range
        df["price_range"] = df["high"] - df["low"]
        df["price_range_pct"] = df["price_range"] / df["close"] * 100

        # 3. حذف سطرهای با مقادیر NaN - اما فقط برای ستون‌های ضروری
        essential_cols = ["close_lag_1", "close_lag_2", "sma_5", "target"]
        initial_len = len(df)
        df_clean = df.dropna(subset=essential_cols)
        final_len = len(df_clean)

        logger.info(
            f"📊 Data reduction: {initial_len} → {final_len} rows ({final_len/initial_len*100:.1f}% remaining)"
        )

        if final_len == 0:
            logger.error("❌ CRITICAL: All data lost after basic feature engineering!")
            # بازگشت به داده اصلی با فقط target
            df_fallback = df[["open", "high", "low", "close", "volume"]].copy()
            df_fallback["target"] = (
                df_fallback["close"].shift(-1) > df_fallback["close"]
            ).astype(int)
            df_fallback = df_fallback.dropna()
            logger.info(f"🔄 Using ULTRA fallback data: {len(df_fallback)} rows")

            feature_columns = ["open", "high", "low", "close", "volume"]
            return df_fallback, feature_columns

        # لیست featureها (همه ستون‌ها به جز OHLCV اصلی و target)
        feature_columns = [
            col
            for col in df_clean.columns
            if col not in ["target", "open", "high", "low", "close", "volume"]
        ]

        # اگر featureای نداریم، از OHLCV اصلی استفاده کنیم
        if not feature_columns:
            feature_columns = ["open", "high", "low", "close", "volume"]
            logger.info("🔄 Using basic OHLCV as features")

        logger.info(f"✅ {len(feature_columns)} features for {symbol}")
        logger.info(f"   Features: {feature_columns}")

        return df_clean, feature_columns


class DataCleaner:
    """پاکسازی و پیش‌پردازش داده‌های مالی"""

    @staticmethod
    def clean_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """پاکسازی جامع داده‌های مالی"""
        logger.info(f"🧹 Cleaning data for {symbol}")

        df_clean = df.copy()
        initial_len = len(df_clean)

        # 1. حذف سطرهای با مقادیر NaN در OHLCV
        df_clean = df_clean.dropna(subset=["open", "high", "low", "close", "volume"])

        # 2. حذف کندل‌های با volume صفر یا منفی
        df_clean = df_clean[df_clean["volume"] > 0]

        # 3. حذف outliers بر اساس range قیمت
        price_range_pct = (df_clean["high"] - df_clean["low"]) / df_clean["close"] * 100
        valid_range_mask = price_range_pct < 20  # حذف کندل‌های با range بیش از 20%
        df_clean = df_clean[valid_range_mask]

        # 4. حذف کندل‌های با قیمت غیرمعقول
        valid_price_mask = (
            (df_clean["close"] > 0)
            & (df_clean["high"] >= df_clean["low"])
            & (df_clean["high"] >= df_clean["open"])
            & (df_clean["high"] >= df_clean["close"])
            & (df_clean["low"] <= df_clean["open"])
            & (df_clean["low"] <= df_clean["close"])
        )
        df_clean = df_clean[valid_price_mask]

        # 5. Smooth کردن قیمت برای کاهش noise
        df_clean["close_smooth"] = df_clean["close"].ewm(span=3).mean()
        df_clean["volume_smooth"] = df_clean["volume"].ewm(span=5).mean()

        final_len = len(df_clean)
        logger.info(
            f"✅ Data cleaning: {initial_len} → {final_len} rows ({final_len/initial_len*100:.1f}% remaining)"
        )

        if final_len < initial_len * 0.5:
            logger.warning(f"⚠️ Over 50% data removed during cleaning for {symbol}")

        return df_clean


class DataSplitter:
    """تقسیم داده به آموزش و تست"""

    @staticmethod
    def split_data(df: pd.DataFrame, test_size=0.2):
        """تقسیم ساده داده"""
        if len(df) < 100:
            test_size = min(0.1, 10 / len(df))  # حداقل 10 نمونه برای تست

        split_idx = int(len(df) * (1 - test_size))
        train_data = df.iloc[:split_idx]
        test_data = df.iloc[split_idx:]

        logger.info(f"📋 Data split - Train: {len(train_data)}, Test: {len(test_data)}")
        return train_data, test_data


class TabularTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}

    def prepare_data(self, df: pd.DataFrame, feature_columns: List[str]):
        """آماده‌سازی داده"""
        X = df[feature_columns]
        y = df["target"]

        # حذف سطرهای با مقادیر NaN
        mask = y.notna()
        X = X[mask]
        y = y[mask]

        return X, y

    def train_tabular(
        self, df: pd.DataFrame, symbol: str, timeframe: str, feature_columns: List[str]
    ):
        """آموزش مدل‌های tabular"""
        try:
            logger.info(f"🤖 Training tabular models for {symbol}_{timeframe}")

            # تقسیم داده
            train_data, test_data = DataSplitter.split_data(df)

            if len(train_data) == 0 or len(test_data) == 0:
                logger.warning(f"❌ Not enough data for {symbol}_{timeframe}")
                return

            X_train, y_train = self.prepare_data(train_data, feature_columns)
            X_test, y_test = self.prepare_data(test_data, feature_columns)

            if len(X_train) == 0 or len(X_test) == 0:
                logger.warning(
                    f"❌ No data after preprocessing for {symbol}_{timeframe}"
                )
                return

            logger.info(f"   Training data: {X_train.shape}, Test data: {X_test.shape}")

            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # مدل‌های ساده
            models = {
                "RandomForest": RandomForestClassifier(
                    n_estimators=50, random_state=42
                ),
                "LogisticRegression": LogisticRegression(
                    random_state=42, max_iter=1000
                ),
            }

            best_model = None
            best_accuracy = 0
            model_results = {}

            for name, model in models.items():
                try:
                    logger.info(f"   Training {name}...")
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)

                    model_results[name] = {
                        "accuracy": accuracy,
                        "f1": f1,
                        "predictions": y_pred,
                    }

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model

                    logger.info(f"   ✅ {name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")

                    # تحلیل پیش‌بینی‌ها
                    actual_up = y_test.mean()
                    predicted_up = y_pred.mean()
                    correct_predictions = (y_pred == y_test).mean()

                    logger.info(
                        f"      Actual UP: {actual_up:.1%}, Predicted UP: {predicted_up:.1%}"
                    )
                    logger.info(f"      Correct predictions: {correct_predictions:.1%}")

                except Exception as e:
                    logger.warning(f"   ❌ {name} failed: {str(e)}")

            if best_model:
                key = f"{symbol}_{timeframe}"
                self.models[key] = best_model
                self.scalers[key] = scaler
                self.results[key] = model_results
                logger.info(f"🎯 Best model for {key}: {best_accuracy:.3f} accuracy")
            else:
                logger.warning(f"❌ No valid model for {symbol}_{timeframe}")

        except Exception as e:
            logger.error(
                f"❌ Tabular training failed for {symbol}_{timeframe}: {str(e)}"
            )

    def save_all(self, path="models/"):
        """ذخیره مدل‌ها"""
        os.makedirs(path, exist_ok=True)

        for key, model in self.models.items():
            model_path = f"{path}/tabular_{key.replace('/', '_')}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"💾 Saved tabular model: {model_path}")

        for key, scaler in self.scalers.items():
            scaler_path = f"{path}/scaler_{key.replace('/', '_')}.joblib"
            joblib.dump(scaler, scaler_path)


class TimeSeriesTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}  # ✅ این خط را اضافه کنید
        self.results = {}

    def create_sequences(
        self,
        feature_data: np.ndarray,
        target_data: np.ndarray,
        sequence_length: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ایجاد sequences با مدیریت صحیح memory"""
        X, y = [], []

        for i in range(len(feature_data) - sequence_length):
            X.append(feature_data[i : i + sequence_length])
            y.append(target_data[i + sequence_length])

        return np.array(X), np.array(y)

    def train_time_series(
        self, df: pd.DataFrame, symbol: str, timeframe: str, feature_columns: List[str]
    ):
        """آموزش بهبود یافته LSTM"""
        try:
            logger.info(f"🧠 Training IMPROVED LSTM for {symbol}_{timeframe}")

            # تقسیم داده
            train_data, test_data = DataSplitter.split_data(df)

            if len(train_data) < 50 or len(test_data) < 20:
                logger.warning(f"❌ Insufficient data for LSTM: {symbol}_{timeframe}")
                return

            # آماده‌سازی داده‌ها
            X_train_raw = train_data[feature_columns]
            X_test_raw = test_data[feature_columns]
            y_train = train_data["target"].values
            y_test = test_data["target"].values

            # Scaling داده‌ها
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)

            # ایجاد sequences
            seq_length = min(
                20, len(X_train_scaled) // 10
            )  # تطبیق طول sequence با حجم داده
            X_train_seq, y_train_seq = self.create_sequences(
                X_train_scaled, y_train, seq_length
            )
            X_test_seq, y_test_seq = self.create_sequences(
                X_test_scaled, y_test, seq_length
            )

            if len(X_train_seq) == 0:
                logger.warning(f"❌ No sequences created for {symbol}_{timeframe}")
                return

            logger.info(
                f"   Sequences - Train: {X_train_seq.shape}, Test: {X_test_seq.shape}"
            )

            # ساخت مدل LSTM بهبود یافته
            model = Sequential(
                [
                    LSTM(
                        64,
                        return_sequences=True,
                        input_shape=(seq_length, len(feature_columns)),
                    ),
                    Dropout(0.3),
                    LSTM(32, return_sequences=False),
                    Dropout(0.3),
                    Dense(16, activation="relu"),
                    Dropout(0.2),
                    Dense(1, activation="sigmoid"),
                ]
            )

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            # آموزش با callbacks بیشتر
            callbacks = [
                EarlyStopping(
                    patience=5, restore_best_weights=True, monitor="val_loss"
                ),
            ]

            history = model.fit(
                X_train_seq,
                y_train_seq,
                epochs=30,
                batch_size=32,
                validation_split=0.2,
                verbose=0,
                callbacks=callbacks,
                shuffle=False,  # مهم برای داده‌های زمانی
            )

            # ارزیابی
            y_pred_proba = model.predict(X_test_seq)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()

            accuracy = accuracy_score(y_test_seq, y_pred)
            f1 = f1_score(y_test_seq, y_pred)

            logger.info(f"   ✅ IMPROVED LSTM: Accuracy={accuracy:.3f}, F1={f1:.3f}")

            # ذخیره مدل و scaler
            key = f"{symbol}_{timeframe}"
            self.models[key] = model
            self.scalers[key] = scaler  # ✅ حالا self.scalers وجود دارد
            self.results[key] = {
                "accuracy": accuracy,
                "f1": f1,
                "predictions": y_pred,
                "actual": y_test_seq,
            }

        except Exception as e:
            logger.error(
                f"❌ Improved LSTM training failed for {symbol}_{timeframe}: {str(e)}"
            )

    def save_all(self, path="models/"):
        """ذخیره مدل‌های LSTM و scalerها"""
        os.makedirs(path, exist_ok=True)

        for key, model in self.models.items():
            model_path = f"{path}/lstm_{key.replace('/', '_')}.keras"
            model.save(model_path)
            logger.info(f"💾 Saved LSTM model: {model_path}")

        # ✅ ذخیره scalerهای LSTM
        for key, scaler in self.scalers.items():
            scaler_path = f"{path}/lstm_scaler_{key.replace('/', '_')}.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"💾 Saved LSTM scaler: {scaler_path}")


class MLManager:
    def __init__(self):
        self.data_cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()
        self.tabular = TabularTrainer()
        self.timeseries = TimeSeriesTrainer()

    def train_all(self, symbols_data: Dict[Tuple[str, str], pd.DataFrame]):
        """آموزش بهبود یافته تمام مدل‌ها"""
        results = {}

        for (symbol, timeframe), df in symbols_data.items():
            if df.empty or len(df) < 100:  # افزایش حداقل داده
                logger.warning(
                    f"⚠️ Skipping {symbol}_{timeframe} - insufficient data ({len(df)} rows)"
                )
                continue

            logger.info(f"\n{'='*50}")
            logger.info(
                f"🚀 IMPROVED TRAINING {symbol} {timeframe} ({len(df)} samples)"
            )
            logger.info(f"{'='*50}")

            try:
                # 1. پاکسازی داده
                df_clean = self.data_cleaner.clean_data(df, symbol)

                if len(df_clean) < 50:
                    logger.warning(
                        f"❌ Not enough data after cleaning for {symbol}_{timeframe}"
                    )
                    continue

                # 2. Feature Engineering
                df_engineered, features = self.feature_engineer.engineer_features(
                    df_clean, symbol
                )

                if len(df_engineered) == 0:
                    logger.error(
                        f"❌ No data after feature engineering for {symbol}_{timeframe}"
                    )
                    continue

                logger.info(f"📈 Final data shape: {df_engineered.shape}")
                logger.info(
                    f"🎯 Target distribution: {df_engineered['target'].mean():.3f}"
                )

                # 3. آموزش مدل‌ها
                self.tabular.train_tabular(df_engineered, symbol, timeframe, features)
                self.timeseries.train_time_series(
                    df_engineered, symbol, timeframe, features
                )

                # ذخیره نتایج
                key = f"{symbol}_{timeframe}"
                results[key] = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "initial_data_points": len(df),
                    "cleaned_data_points": len(df_clean),
                    "final_data_points": len(df_engineered),
                    "features": len(features),
                    "tabular_results": self.tabular.results.get(key, {}),
                    "timeseries_results": self.timeseries.results.get(key, {}),
                }

            except Exception as e:
                logger.error(
                    f"❌ Improved training failed for {symbol}_{timeframe}: {str(e)}"
                )
                continue

        self._print_detailed_summary(results)
        return results

    def _print_detailed_summary(self, results):
        """نمایش خلاصه دقیق آموزش"""
        logger.info(f"\n{'='*60}")
        logger.info("🎯 DETAILED TRAINING SUMMARY")
        logger.info(f"{'='*60}")

        if not results:
            logger.info("❌ No models were trained!")
            return

        for key, result in results.items():
            logger.info(f"\n📊 {key}:")
            logger.info(
                f"   Data: {result['initial_data_points']} → {result['cleaned_data_points']} → {result['final_data_points']} points"
            )
            logger.info(f"   Features: {result['features']}")

            # نتایج tabular
            tabular = result.get("tabular_results", {})
            if tabular:
                for model_name, metrics in tabular.items():
                    if isinstance(metrics, dict):
                        acc = metrics.get("accuracy", 0)
                        f1 = metrics.get("f1", 0)
                        logger.info(f"   {model_name}: Accuracy={acc:.3f}, F1={f1:.3f}")

            # نتایج time series
            ts = result.get("timeseries_results", {})
            if ts:
                acc = ts.get("accuracy", 0)
                f1 = ts.get("f1", 0)
                logger.info(f"   LSTM: Accuracy={acc:.3f}, F1={f1:.3f}")

    def save_all(self, path="models/"):
        """ذخیره تمام مدل‌ها"""
        self.tabular.save_all(path)
        self.timeseries.save_all(path)
        logger.info("💾 All models saved successfully!")


if __name__ == "__main__":
    logger.info("🚀 Starting ML Engine...")

    fetcher = DataFetcher()

    symbols = ["BTC/USDT", "ETH/USDT"]
    timeframes = ["1h", "4h"]

    symbols_data = {}

    for symbol in symbols:
        for timeframe in timeframes:
            try:
                logger.info(f"📥 Loading {symbol} {timeframe}...")
                df = fetcher.get_stored_data(symbol, timeframe)

                if not df.empty:
                    symbols_data[(symbol, timeframe)] = df
                    logger.info(f"   ✅ Loaded {len(df)} records")
                    logger.info(f"   📅 Date range: {df.index[0]} to {df.index[-1]}")
                    logger.info(
                        f"   💰 Price range: {df['close'].min():.2f} - {df['close'].max():.2f}"
                    )
                else:
                    logger.warning(f"   ⚠️ No data for {symbol} {timeframe}")

            except Exception as e:
                logger.error(f"   ❌ Error loading {symbol} {timeframe}: {str(e)}")

    if symbols_data:
        manager = MLManager()
        results = manager.train_all(symbols_data)
        manager.save_all()
        logger.info("✅ All processes completed successfully!")
    else:
        logger.error("❌ No data available for training!")
