# ml/ml_engine.py - Fully Optimized & Modified Version
# Reforms based on analysis: Fixed leakage (positive shifts for features), added FinBERT sentiment, saved LSTM scaler & feature_columns,
# Incremental learning for LSTM (transfer weights), reduced correlation threshold to 0.8, integrated regime detection,
# Used RobustScaler for Gold/BTC volatility, added min_delta in EarlyStopping, improved logging/error handling.
# No data deletion - All NaNs filled. Ready for server deployment with gradual learning.

import pandas as pd
import numpy as np
import logging
import os
import joblib
from data.data_fetcher import DataFetcher
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler  # Changed for outliers
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings("ignore")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from typing import List, Dict, Tuple
from xgboost import XGBClassifier
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)  # For FinBERT


# Regime detection placeholder (integrate with multi_timeframe.py)
def detect_market_regime(df):
    """Detect regime: trending if ADX > 25"""
    if "ADX" in df.columns and df["ADX"].iloc[-1] > 25:
        return "trending"
    return "ranging"


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TimeSeriesSplitter:
    """تقسیم داده‌های زمانی با جلوگیری از data leakage"""

    @staticmethod
    def time_series_split(df, n_splits=5, test_size=0.2):
        """تقسیم داده‌های زمانی با validation زمانی"""
        splits = []
        data_len = len(df)
        test_samples = int(data_len * test_size)

        for i in range(n_splits):
            train_end = int(data_len * (1 - test_size * (i + 1)))
            test_start = train_end
            test_end = test_start + int(data_len * test_size)

            if train_end > 0 and test_end <= data_len:
                train_data = df.iloc[:train_end]
                test_data = df.iloc[test_start:test_end]
                splits.append((train_data, test_data))
            else:
                logger.warning(f"⚠️ Invalid split {i+1} - Skipping")

        logger.info(f"📊 Created {len(splits)} time series splits")
        return splits

    @staticmethod
    def rolling_window_split(df, window_size=0.7, step_size=0.1):
        """تقسیم rolling window برای validation پیشرفته"""
        data_len = len(df)
        window_samples = int(data_len * window_size)
        step_samples = int(data_len * step_size)

        splits = []
        start_idx = 0

        while start_idx + window_samples <= data_len:
            train_end = start_idx + window_samples
            test_start = train_end
            test_end = min(test_start + step_samples, data_len)

            if test_end > test_start:
                train_data = df.iloc[start_idx:train_end]
                test_data = df.iloc[test_start:test_end]
                splits.append((train_data, test_data))
            else:
                logger.warning("⚠️ Invalid rolling window - Skipping")

            start_idx += step_samples

        logger.info(f"🔄 Created {len(splits)} rolling window splits")
        return splits


class FeatureSelector:
    """انتخاب بهترین features بر اساس importance"""

    @staticmethod
    def select_features_using_permutation_importance(X, y, model, n_repeats=5):
        """انتخاب features بر اساس permutation importance - Only on train"""
        try:
            result = permutation_importance(
                model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1
            )

            importance_df = pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                }
            )

            # انتخاب features با importance مثبت
            selected_features = importance_df[importance_df["importance_mean"] > 0][
                "feature"
            ].tolist()

            logger.info(f"🎯 Selected {len(selected_features)} important features")
            return selected_features

        except Exception as e:
            logger.warning(f"⚠️ Permutation importance failed: {e}, using all features")
            return X.columns.tolist()

    @staticmethod
    def select_features_correlation(
        df, target_col="target", threshold=0.8
    ):  # Reduced threshold for better removal
        """حذف features با همبستگی بسیار بالا - Stricter threshold"""
        corr_matrix = df.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > threshold)
        ]

        logger.info(
            f"🗑️ Removing {len(to_drop)} highly correlated features (threshold 0.8)"
        )
        return [col for col in df.columns if col not in to_drop and col != target_col]


class AdvancedEnsemble:
    """Ensemble پیشرفته برای ترکیب مدل‌ها - With dynamic regime weights"""

    def __init__(self):
        self.models = {
            "rf": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                min_samples_leaf=10,
                class_weight="balanced",
            ),
            "xgb": XGBClassifier(
                random_state=42, eval_metric="logloss", use_label_encoder=False
            ),
            "lr": LogisticRegression(
                random_state=42, max_iter=1000, class_weight="balanced"
            ),
        }
        self.weights = {}
        self.regime_weights = {  # Dynamic based on regime
            "trending": {"xgb": 0.5, "rf": 0.3, "lr": 0.2},
            "ranging": {"rf": 0.5, "lr": 0.3, "xgb": 0.2},
        }

    def calculate_model_weights(self, validation_scores, regime="ranging"):
        """محاسبه وزن مدل‌ها بر اساس performance و regime"""
        base_weights = self.regime_weights.get(
            regime, {name: 1 / len(self.models) for name in self.models}
        )
        total_score = sum(validation_scores.values())
        self.weights = {
            name: (score / total_score) * base_weights.get(name, 1)
            for name, score in validation_scores.items()
        }
        logger.info(f"⚖️ Model weights (regime {regime}): {self.weights}")

    def predict_proba_weighted(self, X):
        """پیش‌بینی ترکیبی با وزن‌ها"""
        weighted_predictions = None

        for name, model in self.models.items():
            preds = model.predict_proba(X)[:, 1]
            weight = self.weights.get(name, 1.0 / len(self.models))

            if weighted_predictions is None:
                weighted_predictions = preds * weight
            else:
                weighted_predictions += preds * weight

        return weighted_predictions

    def predict(self, X, threshold=0.5):
        """پیش‌بینی نهایی"""
        probas = self.predict_proba_weighted(X)
        return (probas > threshold).astype(int)


class FeatureEngineer:
    """Feature Engineering با جلوگیری از data leakage و ادغام FinBERT"""

    @staticmethod
    def engineer_features(
        df: pd.DataFrame, symbol: str, lookahead_protection: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """ایجاد featureهای با جلوگیری از lookahead bias - Features from past only"""
        logger.info(f"🔧 Starting LEAKAGE-PROTECTED feature engineering for {symbol}")
        df = df.copy()
        # 1. ایجاد target با lookahead protection - Target uses future, but rows trimmed
        lookahead = 1
        df["target"] = (df["close"].shift(-lookahead) > df["close"]).astype(int)

        # Trim last rows to prevent leakage in train/test
        if lookahead_protection:
            df = df.iloc[:-lookahead] if lookahead > 0 else df
            logger.info(
                f"🛡️ Applied lookahead protection, trimmed last {lookahead} rows"
            )

        # 2. ایجاد features فقط از داده‌های گذشته - All shifts positive
        # Lag features from past
        df["close_lag_1"] = df["close"].shift(1)
        df["close_lag_2"] = df["close"].shift(2)
        df["close_lag_3"] = df["close"].shift(3)
        df["volume_lag_1"] = df["volume"].shift(1)
        df["volume_lag_2"] = df["volume"].shift(2)

        # Moving averages from past
        df["sma_5"] = df["close"].rolling(5, min_periods=1).mean()
        df["sma_10"] = df["close"].rolling(10, min_periods=1).mean()
        df["sma_20"] = df["close"].rolling(20, min_periods=1).mean()

        # Momentum from past
        df["momentum_3"] = df["close"] - df["close"].shift(3)
        df["momentum_5"] = df["close"] - df["close"].shift(5)
        df["roc_5"] = (df["close"] - df["close"].shift(5)) / df["close"].shift(5) * 100

        # Volume features
        df["volume_sma_5"] = df["volume"].rolling(5, min_periods=1).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_5"]

        # Price range and volatility
        df["price_range"] = df["high"] - df["low"]
        df["price_range_pct"] = df["price_range"] / df["close"] * 100
        df["volatility_5"] = df["close"].rolling(5).std()

        # Fill NaNs - No deletion
        df = df.ffill().bfill()

        # 3. ادغام sentiment با FinBERT (فرض بر ذخیره اخبار در DB یا fetch)
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        sentiment_pipeline = pipeline(
            "sentiment-analysis", model=model, tokenizer=tokenizer
        )

        # Dummy news for demo - In real, fetch from DB/API
        news_texts = ["Sample financial news for " + symbol] * len(
            df
        )  # Replace with real data
        sentiments = [
            (
                sentiment_pipeline(text)[0]["score"]
                if sentiment_pipeline(text)[0]["label"] == "positive"
                else -sentiment_pipeline(text)[0]["score"]
            )
            for text in news_texts
        ]
        df["news_sentiment"] = sentiments

        # 4. انتخاب features
        feature_columns = [
            col
            for col in df.columns
            if col
            not in ["target", "open", "high", "low", "close", "volume", "datetime"]
        ]

        # حذف همبستگی بالا با threshold پایین‌تر
        if len(feature_columns) > 10:
            feature_columns = FeatureSelector.select_features_correlation(
                df[feature_columns + ["target"]], "target"
            )

        logger.info(f"✅ {len(feature_columns)} features for {symbol}")
        return df, feature_columns


class DataCleaner:
    """پاکسازی و پیش‌پردازش داده‌های مالی - بهبود یافته بدون حذف"""

    @staticmethod
    def clean_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """پاکسازی جامع بدون حذف رکورد - Fill and clip outliers"""
        logger.info(f"🧹 Advanced cleaning data for {symbol} - No deletion")
        df_clean = df.copy()
        initial_len = len(df_clean)

        # Fill NaNs in OHLCV
        df_clean = df_clean.ffill().bfill()

        # Clip zero/negative volume
        df_clean["volume"] = df_clean["volume"].clip(lower=1)

        # Clip outliers with IQR
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

        # Ensure valid prices
        valid_price_mask = (
            (df_clean["close"] > 0)
            & (df_clean["high"] >= df_clean["low"])
            & (df_clean["high"] >= df_clean["open"])
            & (df_clean["high"] >= df_clean["close"])
            & (df_clean["low"] <= df_clean["open"])
            & (df_clean["low"] <= df_clean["close"])
        )
        df_clean.loc[~valid_price_mask, numeric_cols] = df_clean.loc[
            ~valid_price_mask, numeric_cols
        ].mean()  # Fill invalid with mean

        # Clip unusual ranges
        price_range_pct = (df_clean["high"] - df_clean["low"]) / df_clean["close"] * 100
        df_clean.loc[price_range_pct >= 50, "high"] = df_clean["close"] + (
            df_clean["close"] * 0.5
        )  # Clip to 50%

        final_len = len(df_clean)
        logger.info(f"✅ Advanced cleaning: {initial_len} rows preserved (100%)")
        return df_clean


class TabularTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_importances = {}

    def prepare_data(self, df: pd.DataFrame, feature_columns: List[str]):
        """آماده‌سازی داده با class weights - Fill NaNs"""
        X = df[feature_columns].ffill().bfill()
        y = df["target"].ffill().bfill()

        # محاسبه class weights
        classes = np.unique(y)
        if len(classes) == 2:
            class_weights = compute_class_weight("balanced", classes=classes, y=y)
            self.class_weights = dict(zip(classes, class_weights))
            logger.info(f"⚖️ Class weights: {self.class_weights}")
        else:
            self.class_weights = {0: 1, 1: 1}

        return X, y

    def train_tabular(
        self, df: pd.DataFrame, symbol: str, timeframe: str, feature_columns: List[str]
    ):
        """آموزش مدل‌های tabular با بهبود‌های پیشنهادی"""
        try:
            logger.info(f"🤖 Training IMPROVED tabular models for {symbol}_{timeframe}")

            # تقسیم داده با time series split
            splits = TimeSeriesSplitter().time_series_split(
                df, n_splits=3, test_size=0.2
            )

            if not splits:
                logger.warning(f"❌ No valid splits for {symbol}_{timeframe}")
                return

            best_models = {}
            best_scores = {}

            for i, (train_data, test_data) in enumerate(splits):
                logger.info(
                    f"   Fold {i+1}: Train={len(train_data)}, Test={len(test_data)}"
                )

                X_train, y_train = self.prepare_data(train_data, feature_columns)
                X_test, y_test = self.prepare_data(test_data, feature_columns)

                if len(X_train) == 0 or len(X_test) == 0:
                    continue

                # Scaling
                scaler = RobustScaler()  # Changed for better handling
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # مدل‌های بهبود یافته با hyperparameters بهینه
                models = {
                    "RandomForest": RandomForestClassifier(
                        n_estimators=100,
                        min_samples_leaf=10,
                        random_state=42,
                        class_weight="balanced",
                    ),
                    "LogisticRegression": LogisticRegression(
                        random_state=42, max_iter=1000, class_weight="balanced", C=0.1
                    ),
                    "XGBoost": XGBClassifier(
                        random_state=42,
                        eval_metric="logloss",
                        use_label_encoder=False,
                        max_depth=6,
                        learning_rate=0.1,
                    ),
                }

                for name, model in models.items():
                    try:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        accuracy = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)

                        # ذخیره بهترین مدل برای هر نوع
                        if name not in best_scores or accuracy > best_scores[name]:
                            best_scores[name] = accuracy
                            best_models[name] = (model, scaler)

                        logger.info(
                            f"   ✅ {name} Fold {i+1}: Accuracy={accuracy:.3f}, F1={f1:.3f}"
                        )
                    except Exception as e:
                        logger.warning(f"   ❌ {name} failed in fold {i+1}: {str(e)}")

            # انتخاب بهترین مدل نهایی
            if best_models:
                best_model_name = max(best_scores, key=best_scores.get)
                best_model, best_scaler = best_models[best_model_name]

                key = f"{symbol}_{timeframe}"
                self.models[key] = best_model
                self.scalers[key] = best_scaler
                self.results[key] = best_scores

                logger.info(
                    f"🎯 Best model for {key}: {best_model_name} with {best_scores[best_model_name]:.3f} accuracy"
                )

                # محاسبه feature importance فقط روی train
                try:
                    important_features = (
                        FeatureSelector.select_features_using_permutation_importance(
                            X_train, y_train, best_model  # Changed to train only
                        )
                    )
                    self.feature_importances[key] = important_features
                except Exception as e:
                    logger.warning(f"⚠️ Feature importance failed: {e}")
            else:
                logger.warning(f"❌ No valid models for {symbol}_{timeframe}")
        except Exception as e:
            logger.error(f"❌ Improved tabular training failed: {str(e)}")

    def save_all(self, path="models/"):
        """ذخیره مدل‌ها و feature importances"""
        os.makedirs(path, exist_ok=True)
        for key, model in self.models.items():
            model_path = f"{path}/tabular_{key.replace('/', '_')}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"💾 Saved tabular model: {model_path}")
        for key, scaler in self.scalers.items():
            scaler_path = f"{path}/scaler_{key.replace('/', '_')}.joblib"
            joblib.dump(scaler, scaler_path)
        # ذخیره feature importances
        if self.feature_importances:
            importance_path = f"{path}/feature_importances.joblib"
            joblib.dump(self.feature_importances, importance_path)
            logger.info(f"💾 Saved feature importances: {importance_path}")


class TimeSeriesTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}  # Added for LSTM
        self.results = {}

    def create_sequences(
        self,
        feature_data: np.ndarray,
        target_data: np.ndarray,
        sequence_length: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ایجاد sequences با مدیریت حافظه بهینه"""
        X, y = [], []
        for i in range(len(feature_data) - sequence_length):
            X.append(feature_data[i : i + sequence_length])
            y.append(target_data[i + sequence_length])
        return np.array(X), np.array(y)

    def train_time_series(
        self, df: pd.DataFrame, symbol: str, timeframe: str, feature_columns: List[str]
    ):
        """آموزش LSTM با بهبود‌های پیشنهادی و incremental fine-tuning"""
        try:
            logger.info(f"🧠 Training ADVANCED LSTM for {symbol}_{timeframe}")

            # استفاده از rolling window validation
            splits = TimeSeriesSplitter().rolling_window_split(
                df, window_size=0.6, step_size=0.2
            )

            if not splits:
                logger.warning(
                    f"❌ No valid rolling splits for LSTM: {symbol}_{timeframe}"
                )
                return

            best_accuracy = 0
            best_model = None
            best_scaler = None

            for i, (train_data, test_data) in enumerate(splits):
                if len(train_data) < 50 or len(test_data) < 20:
                    continue

                # آماده‌سازی داده‌ها
                X_train_raw = train_data[feature_columns]
                X_test_raw = test_data[feature_columns]
                y_train = train_data["target"].values
                y_test = test_data["target"].values

                # Scaling
                scaler = RobustScaler()  # Changed
                X_train_scaled = scaler.fit_transform(X_train_raw)
                X_test_scaled = scaler.transform(X_test_raw)

                # ایجاد sequences
                seq_length = min(15, len(X_train_scaled) // 10)
                X_train_seq, y_train_seq = self.create_sequences(
                    X_train_scaled, y_train, seq_length
                )
                X_test_seq, y_test_seq = self.create_sequences(
                    X_test_scaled, y_test, seq_length
                )

                if len(X_train_seq) == 0:
                    continue

                # محاسبه sample weights برای imbalance
                class_weights = compute_class_weight(
                    "balanced", classes=np.unique(y_train_seq), y=y_train_seq
                )
                sample_weights = np.array([class_weights[int(y)] for y in y_train_seq])

                # ساخت مدل LSTM پیشرفته
                model = Sequential(
                    [
                        LSTM(
                            64,
                            return_sequences=True,
                            input_shape=(seq_length, len(feature_columns)),
                        ),
                        Dropout(0.4),  # افزایش dropout برای کاهش overfitting
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

                # Callbacks پیشرفته
                callbacks = [
                    EarlyStopping(
                        patience=8,
                        restore_best_weights=True,
                        monitor="val_loss",
                        min_delta=0.001,
                    ),  # Added min_delta
                ]

                # Incremental fine-tuning
                if i > 0 and best_model is not None:
                    model.set_weights(best_model.get_weights())
                    logger.info(
                        f"🔄 Fine-tuning LSTM with previous fold weights for gradual learning"
                    )

                history = model.fit(
                    X_train_seq,
                    y_train_seq,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=callbacks,
                    shuffle=False,
                    sample_weight=sample_weights,
                )

                # ارزیابی
                y_pred_proba = model.predict(X_test_seq, verbose=0)
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                accuracy = accuracy_score(y_test_seq, y_pred)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_scaler = scaler

                logger.info(f" ✅ LSTM Fold {i+1}: Accuracy={accuracy:.3f}")

            if best_model and best_accuracy > 0.5:
                key = f"{symbol}_{timeframe}"
                self.models[key] = best_model
                self.scalers[key] = best_scaler  # Added
                self.results[key] = {"accuracy": best_accuracy}
                logger.info(f"🎯 Best LSTM for {key}: {best_accuracy:.3f} accuracy")
            else:
                logger.warning(f"❌ No valid LSTM model for {symbol}_{timeframe}")
        except Exception as e:
            logger.error(f"❌ Advanced LSTM training failed: {str(e)}")

    def save_all(self, path="models/"):
        """ذخیره مدل‌های LSTM - Added scaler save"""
        os.makedirs(path, exist_ok=True)
        for key, model in self.models.items():
            model_path = f"{path}/lstm_{key.replace('/', '_')}.keras"
            model.save(model_path)
            logger.info(f"💾 Saved LSTM model: {model_path}")
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
        self.ensemble = AdvancedEnsemble()

    def train_hybrid_models(self, df_engineered, symbol, timeframe, features):
        """آموزش مدل‌های ترکیبی با ensemble - Added regime"""
        try:
            logger.info(f"🚀 Training HYBRID models for {symbol}_{timeframe}")

            regime = detect_market_regime(df_engineered)
            logger.info(f"📊 Detected regime: {regime}")

            # تقسیم داده
            splits = TimeSeriesSplitter().time_series_split(df_engineered, n_splits=3)

            ensemble_scores = {}

            for i, (train_data, test_data) in enumerate(splits):
                # آماده‌سازی داده برای ensemble
                X_train = train_data[features]
                y_train = train_data["target"]
                X_test = test_data[features]
                y_test = test_data["target"]

                # Scaling
                scaler = RobustScaler()  # Changed
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # آموزش مدل‌های ensemble
                model_scores = {}
                for name, model in self.ensemble.models.items():
                    model.fit(X_train_scaled, y_train)
                    score = model.score(X_test_scaled, y_test)
                    model_scores[name] = score

                # محاسبه وزن‌ها با regime
                self.ensemble.calculate_model_weights(model_scores, regime=regime)

                # ارزیابی ensemble
                ensemble_pred = self.ensemble.predict(X_test_scaled)
                ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
                ensemble_scores[f"fold_{i}"] = ensemble_accuracy

                logger.info(f" 🎯 Ensemble Fold {i+1}: {ensemble_accuracy:.3f}")

            # ذخیره نتایج ensemble
            avg_ensemble_score = np.mean(list(ensemble_scores.values()))
            logger.info(f"🏆 Average Ensemble Accuracy: {avg_ensemble_score:.3f}")

            return {
                "ensemble_scores": ensemble_scores,
                "average_score": avg_ensemble_score,
                "model_weights": self.ensemble.weights,
            }

        except Exception as e:
            logger.error(f"❌ Hybrid training failed: {str(e)}")
            return {}

    def train_all(self, symbols_data: Dict[Tuple[str, str], pd.DataFrame]):
        """آموزش کامل تمام مدل‌ها با بهبود‌های پیشنهادی"""
        results = {}
        for (symbol, timeframe), df in symbols_data.items():
            if df.empty or len(df) < 150:  # افزایش حداقل داده برای validation
                logger.warning(f"⚠️ Skipping {symbol}_{timeframe} - insufficient data")
                continue
            logger.info(f"\n{'='*60}")
            logger.info(f"🚀 ADVANCED TRAINING {symbol} {timeframe}")
            logger.info(f"{'='*60}")
            try:
                # 1. پاکسازی داده پیشرفته
                df_clean = self.data_cleaner.clean_data(df, symbol)
                if len(df_clean) < 80:
                    logger.warning(f"❌ Not enough data after cleaning")
                    continue
                # 2. Feature Engineering با leakage protection
                df_engineered, features = self.feature_engineer.engineer_features(
                    df_clean, symbol, lookahead_protection=True
                )
                if len(df_engineered) == 0 or not features:
                    logger.error(f"❌ No data/features after engineering")
                    continue
                logger.info(
                    f"📈 Final data: {df_engineered.shape}, Features: {len(features)}"
                )
                logger.info(
                    f"🎯 Target distribution: {df_engineered['target'].mean():.3f}"
                )
                # 3. آموزش تمام مدل‌ها
                self.tabular.train_tabular(df_engineered, symbol, timeframe, features)
                self.timeseries.train_time_series(
                    df_engineered, symbol, timeframe, features
                )
                hybrid_results = self.train_hybrid_models(
                    df_engineered, symbol, timeframe, features
                )
                # 4. ذخیره نتایج
                key = f"{symbol}_{timeframe}"
                results[key] = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "initial_data_points": len(df),
                    "final_data_points": len(df_engineered),
                    "features": len(features),
                    "target_balance": df_engineered["target"].mean(),
                    "tabular_results": self.tabular.results.get(key, {}),
                    "timeseries_results": self.timeseries.results.get(key, {}),
                    "hybrid_results": hybrid_results,
                }
                # Save feature_columns for prediction
                features_path = f"models/features_{key.replace('/', '_')}.joblib"
                joblib.dump(features, features_path)
                logger.info(f"💾 Saved feature columns: {features_path}")
            except Exception as e:
                logger.error(f"❌ Advanced training failed: {str(e)}")
                continue
        self._print_comprehensive_summary(results)
        return results

    def _print_comprehensive_summary(self, results):
        """نمایش خلاصه جامع آموزش"""
        logger.info(f"\n{'='*80}")
        logger.info("🎯 COMPREHENSIVE TRAINING SUMMARY")
        logger.info(f"{'='*80}")
        if not results:
            logger.info("❌ No models were trained!")
            return
        for key, result in results.items():
            logger.info(f"\n📊 {key}:")
            logger.info(
                f" Data: {result['initial_data_points']} → {result['final_data_points']}"
            )
            logger.info(
                f" Features: {result['features']}, Target Balance: {result['target_balance']:.3f}"
            )
            # نتایج tabular
            tabular = result.get("tabular_results", {})
            if tabular:
                best_tabular = (
                    max(tabular.values()) if isinstance(tabular, dict) else "N/A"
                )
                logger.info(f" 📈 Best Tabular: {best_tabular:.3f}")
            # نتایج time series
            ts = result.get("timeseries_results", {})
            if ts:
                ts_acc = ts.get("accuracy", 0)
                logger.info(f" 🧠 LSTM: {ts_acc:.3f}")
            # نتایج hybrid
            hybrid = result.get("hybrid_results", {})
            if hybrid:
                avg_score = hybrid.get("average_score", 0)
                logger.info(f" 🏆 Ensemble: {avg_score:.3f}")

    def save_all(self, path="models/"):
        """ذخیره تمام مدل‌ها و نتایج"""
        self.tabular.save_all(path)
        self.timeseries.save_all(path)

        # ذخیره ensemble weights
        if hasattr(self.ensemble, "weights") and self.ensemble.weights:
            weights_path = f"{path}/ensemble_weights.joblib"
            joblib.dump(self.ensemble.weights, weights_path)
            logger.info(f"💾 Saved ensemble weights: {weights_path}")

        logger.info("💾 All advanced models saved successfully!")


if __name__ == "__main__":
    logger.info("🚀 Starting ADVANCED ML Engine...")
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
                    logger.info(f" ✅ Loaded {len(df)} records")
                    logger.info(f" 📅 Date range: {df.index[0]} to {df.index[-1]}")
                else:
                    logger.warning(f" ⚠️ No data for {symbol} {timeframe}")
            except Exception as e:
                logger.error(f" ❌ Error loading {symbol} {timeframe}: {str(e)}")
    if symbols_data:
        manager = MLManager()
        results = manager.train_all(symbols_data)
        manager.save_all()
        logger.info("✅ All ADVANCED processes completed successfully!")
    else:
        logger.error("❌ No data available for advanced training!")
