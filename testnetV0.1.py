# -*- coding: utf-8 -*-
import pandas as pd
import ccxt
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, make_scorer
from imblearn.over_sampling import SMOTE
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import shap  # For feature importance analysis
import argparse  # For command-line arguments like --live

# تنظیم نمایش بهتر نمودارها
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

# پارامترهای بهینه‌شده برای هر نماد
SYMBOL_OPTIMIZED_PARAMS = {
    "BTC/USDT": {
        "stop_loss": 0.06,  # کاهش استاپ لاس به 6%
        "take_profit": 0.25,  # افزایش تیک پروفیت به 25%
        "position_size": 0.12,  # افزایش سایز پوزیشن به 12%
    },
    "ETH/USDT": {"stop_loss": 0.07, "take_profit": 0.22, "position_size": 0.12},
    "SOL/USDT": {
        "stop_loss": 0.10,  # استاپ لاس بزرگتر برای SOL
        "take_profit": 0.30,  # تیک پروفیت بزرگتر
        "position_size": 0.08,  # سایز پوزیشن کوچک‌تر
    },
}


class MLModelManager:
    """مدیریت مدل‌های یادگیری ماشین برای ارزهای مختلف"""

    def __init__(self):
        self.models = {}  # {symbol: (model, scaler, features)}

    def train_models_for_symbols(self, symbols_data):
        """آموزش مدل برای تمام ارزها با Walk-Forward Optimization"""
        print("🤖 شروع آموزش مدل‌های یادگیری ماشین با Walk-Forward Optimization...")
        successful_models = 0
        for symbol, df in symbols_data.items():
            if len(df) > 300:
                print(f"📚 آموزش مدل برای {symbol}...")
                try:
                    df.name = symbol  # Ensure name is set
                    model, scaler, features = train_ml_models_with_wfo(df, symbol)
                    if model is not None:
                        self.models[symbol] = (model, scaler, features)
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

    def get_model(self, symbol):
        """دریافت مدل مربوط به یک ارز"""
        return self.models.get(symbol, (None, None, None))

    def save_models(self, path="ml_models/"):
        """ذخیره مدل‌ها"""
        import os

        os.makedirs(path, exist_ok=True)

        for symbol, (model, scaler, features) in self.models.items():
            model.save(f'{path}{symbol.replace("/", "_")}_model.h5')  # For Keras model
            joblib.dump(scaler, f'{path}{symbol.replace("/", "_")}_scaler.pkl')
            joblib.dump(features, f'{path}{symbol.replace("/", "_")}_features.pkl')

        print(f"💾 مدل‌ها در {path} ذخیره شدند")

    def load_models(self, path="ml_models/"):
        """بارگذاری مدل‌های ذخیره شده"""
        import os
        from tensorflow.keras.models import load_model

        if not os.path.exists(path):
            print("📂 پوشه مدل‌ها یافت نشد")
            raise FileNotFoundError("پوشه مدل‌ها وجود ندارد")

        loaded_count = 0
        for filename in os.listdir(path):
            if filename.endswith("_model.h5"):
                try:
                    symbol = filename.replace("_model.h5", "").replace("_", "/")
                    model_path = f"{path}{filename}"
                    scaler_path = f'{path}{symbol.replace("/", "_")}_scaler.pkl'
                    features_path = f'{path}{symbol.replace("/", "_")}_features.pkl'

                    if os.path.exists(scaler_path) and os.path.exists(features_path):
                        model = load_model(model_path)
                        scaler = joblib.load(scaler_path)
                        features = joblib.load(features_path)

                        self.models[symbol] = (model, scaler, features)
                        loaded_count += 1
                        print(f"📂 مدل {symbol} بارگذاری شد")
                    else:
                        print(f"⚠️ فایل‌های کامل برای {symbol} یافت نشد")
                except Exception as e:
                    print(f"❌ خطا در بارگذاری مدل {filename}: {e}")

        if loaded_count == 0:
            raise Exception("هیچ مدلی بارگذاری نشد")
        else:
            print(f"✅ تعداد {loaded_count} مدل بارگذاری شد")


# تابع برای دریافت داده‌های تاریخی از صرافی Binance
def fetch_ohlcv(symbol, timeframe, since, limit=1000):
    """دریافت داده‌های تاریخی از صرافی"""
    exchange = ccxt.binance()
    data = []
    retries = 0
    max_retries = 5
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if len(ohlcv) == 0:
                break
            since = ohlcv[-1][0] + 1
            data.extend(ohlcv)
            print(f"تعداد داده‌های دریافت شده: {len(data)}")
            if len(ohlcv) < limit:
                break
            time.sleep(0.5)
        except Exception as e:
            print(f"خطا در دریافت داده: {e}")
            retries += 1
            if retries >= max_retries:
                break
            time.sleep(3 * retries)
    return data


# تابع محاسبه اندیکاتورها
def calculate_indicators(df):
    """محاسبه تمام اندیکاتورهای مورد نیاز با ADX و Bollinger Bandwidth"""
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

    # افزایش ترییلینگ استاپ در بازارهای پرنوسان
    if volatility > 0.02:  # نوسان بالا
        trailing_pct = base_trailing * 1.5
    else:
        trailing_pct = base_trailing

    return trailing_high * (1 - trailing_pct)


def market_regime_filter(df):
    """فیلتر شرایط بازار"""
    # تشخیص روند بازار
    market_trend = "bullish" if df["SMA_50"] > df["SMA_200"] else "bearish"

    # فقط در بازارهای رونددار معامله کنید
    if market_trend == "bullish":
        return df[df["ADX"] > 25]  # فقط روندهای قوی
    else:
        return df[df["ADX"] > 30]  # شرایط سخت‌گیرانه‌تر در روند نزولی


def generate_signals(df):
    """تولید سیگنال‌های معاملاتی با فیلترهای پیشرفته و آستانه بالاتر برای سود پیش‌بینی‌شده"""
    df = df.copy()

    # محاسبه میانگین حجم
    df["avg_volume"] = df["volume"].rolling(window=20, min_periods=1).mean()

    # فیلتر روند اصلی
    df["primary_trend"] = np.where(df["SMA_50"] > df["SMA_200"], 1, -1)

    # شرایط سخت‌گیرانه‌تر برای نمادهای پرنوسان مثل SOL
    if hasattr(df, "name") and "SOL" in getattr(df, "name", ""):
        # فیلترهای ویژه برای SOL
        buy_conditions = (
            (df["SMA_20"] > df["SMA_50"])
            & (df["SMA_50"] > df["SMA_200"])  # روند صعودی قوی
            & (df["RSI"] > 45)
            & (df["RSI"] < 65)
            & (df["ADX"] > 25)  # روند قوی
            & (df["volatility_ratio"] < 1.5)  # نوسان کنترل شده
            & (df["volume_spike"] > 1.2)  # تایید حجم
        )
    else:
        # شرایط معمول برای BTC و ETH
        buy_conditions = (
            (df["SMA_20"] > df["SMA_50"])
            & (df["primary_trend"] == 1)
            & (df["RSI"] > 40)
            & (df["RSI"] < 70)
            & (df["ADX"] > 15)  # آستانه پایین‌تر برای افزایش سیگنال‌ها
            & (df["volume"] > df["avg_volume"])
        )

    # سیستم امتیازدهی بهبود یافته برای خرید
    buy_score = (
        ((df["SMA_20"] > df["SMA_50"]) & (df["primary_trend"] == 1)) * 2
        + (df["close"] > df["SMA_20"]) * 2
        + ((df["RSI"] > 35) & (df["RSI"] < 65)) * 2  # محدوده RSI وسیع‌تر
        + (df["MACD"] > df["Signal_Line"]) * 2
        + ((df["close"] > df["BB_Lower"]) & (df["close"] < df["BB_Middle"])) * 1
        + (df["ADX"] > 20) * 1  # آستانه ADX پایین‌تر
        + (df["volume"] > df["avg_volume"] * 0.8) * 1  # حجم کمتر سخت‌گیرانه
        + ((df["stoch_k"] > 20) & (df["stoch_k"] > df["stoch_d"]))
        * 1  # Stochastic crossover bonus
    )

    # سیستم امتیازدهی برای فروش
    sell_score = (
        ((df["SMA_20"] < df["SMA_50"]) & (df["primary_trend"] == -1)) * 2
        + (df["close"] < df["SMA_20"]) * 2
        + (df["RSI"] > 75) * 2  # آستانه بالاتر برای فروش
        + (df["MACD"] < df["Signal_Line"]) * 2
        + (df["close"] > df["BB_Upper"]) * 1
        + (df["ADX"] > 20) * 1
        + (df["volume"] > df["avg_volume"] * 0.8) * 1
        + ((df["stoch_k"] < 80) & (df["stoch_k"] < df["stoch_d"]))
        * 1  # Stochastic sell bonus
    )

    # فیلتر بر اساس قدرت روند
    strong_trend = df["ADX"] > 30
    weak_trend = df["ADX"] < 20

    # در روند قوی، آستانه پایین‌تر
    # در روند ضعیف، آستانه بالاتر
    df["signal"] = 0
    for i in range(len(df)):
        if strong_trend.iloc[i]:
            buy_threshold = 6
            sell_threshold = 6
        elif weak_trend.iloc[i]:
            buy_threshold = 8
            sell_threshold = 8
        else:
            buy_threshold = 7
            sell_threshold = 7

        if buy_score.iloc[i] >= buy_threshold and df["RSI"].iloc[i] < 70:
            df.iloc[i, df.columns.get_loc("signal")] = 1
        if sell_score.iloc[i] >= sell_threshold and df["RSI"].iloc[i] > 30:
            df.iloc[i, df.columns.get_loc("signal")] = -1

    print("\n🔍 شرایط سیگنال (نسخه پیشرفته):")
    print(f"امتیاز خرید میانگین: {buy_score.mean():.2f}")
    print(f"امتیاز فروش میانگین: {sell_score.mean():.2f}")

    # فیلتر نویز - فقط تغییرات واقعی با آستانه حداقل 3% سود پیش‌بینی‌شده
    df["signal_changed"] = df["signal"].diff().fillna(0)
    df["expected_profit"] = (
        df["close"].pct_change().shift(-1)
    )  # تخمین ساده، در لایو از مدل استفاده شود
    df.loc[df["expected_profit"] < 0.03, "signal"] = 0  # فیلتر سیگنال‌های با سود کم

    # شناسایی موقعیت‌های واقعی
    df["position"] = 0

    # فقط اولین سیگنال پس از تغییر را در نظر بگیرید
    for i in range(1, len(df)):
        if df["signal"].iloc[i] != 0 and df["signal_changed"].iloc[i] != 0:
            # بررسی کنید که آیا این یک سیگنال جدید است
            if df["signal"].iloc[i] != df["signal"].iloc[i - 1]:
                df.iloc[i, df.columns.get_loc("position")] = df["signal"].iloc[i]

    # حذف سیگنال‌های پشت سر هم
    position_changes = df["position"].diff().fillna(0)
    df.loc[position_changes == 0, "position"] = 0

    # Min distance filter (e.g., 3 days min between positions)
    last_position_idx = -np.inf
    for i in range(len(df)):
        if df["position"].iloc[i] != 0:
            if i - last_position_idx < 3:  # Too close
                df.iloc[i, df.columns.get_loc("position")] = 0
            else:
                last_position_idx = i

    return df


# تابع دیباگ سیگنال‌ها
def debug_signals(df):
    """بررسی و نمایش اطلاعات سیگنال‌ها"""
    print("\n" + "=" * 60)
    print("🔍 دیباگ سیگنال‌ها")
    print("=" * 60)

    print(f"تعداد کل داده‌ها: {len(df)}")
    print(f"سیگنال‌های خرید (signal=1): {len(df[df['signal'] == 1])}")
    print(f"سیگنال‌های فروش (signal=-1): {len(df[df['signal'] == -1])}")
    print(f"موقعیت‌های خرید (position=1): {len(df[df['position'] == 1])}")
    print(f"موقعیت‌های فروش (position=-1): {len(df[df['position'] == -1])}")

    if len(df[df["position"] != 0]) > 0:
        print("\n📅 نقاط معاملاتی شناسایی شده:")
        trade_points = df[df["position"] != 0]
        for i, (idx, row) in enumerate(trade_points.iterrows()):
            action = "خرید" if row["position"] == 1 else "فروش"
            print(
                f"  {i+1}. {idx.strftime('%Y-%m-%d')}: {action} - قیمت: {row['close']:.0f}"
            )
    else:
        print("\n⚠️ هیچ نقطه معاملاتی شناسایی نشد!")


def run_backtest(
    df,
    initial_capital=1000.0,
    trade_fee=0.001,
    slippage=0.002,  # بهبود: اضافه کردن slippage 0.2%
    trailing_stop_pct=0.06,
    risk_per_trade=0.02,
):
    """اجرای بک‌تست با مدیریت سرمایه کاملاً اصلاح شده و هزینه‌های واقعی"""
    capital = initial_capital
    position = 0
    trades = []
    entry_price = 0
    entry_date = None
    trailing_high = 0
    in_position = False

    # ایجاد equity_curve با طول برابر داده‌ها
    equity_curve = [initial_capital] * len(df)

    # استفاده از پارامترهای بهینه‌شده برای هر نماد
    current_symbol = getattr(df, "name", "BTC/USDT")
    if current_symbol in SYMBOL_OPTIMIZED_PARAMS:
        params = SYMBOL_OPTIMIZED_PARAMS[current_symbol]
        stop_loss_pct = params["stop_loss"]
        take_profit_pct = params["take_profit"]
        position_size_percent = params["position_size"]
    else:
        # پارامترهای پیش‌فرض
        stop_loss_pct = 0.08
        take_profit_pct = 0.20
        position_size_percent = 0.10

    print(
        f"⚙️ پارامترهای معاملاتی {current_symbol}: "
        f"استاپ لاس {stop_loss_pct*100}%، "
        f"تیک پروفیت {take_profit_pct*100}%، "
        f"سایز پوزیشن {position_size_percent*100}%"
    )

    # شروع از کندل 200 برای اطمینان از محاسبات اندیکاتورها
    start_idx = 200
    trade_count = 0
    max_trades = 50

    for i in range(start_idx, len(df)):
        current_price = df["close"].iloc[i] * (
            1 + slippage if in_position else 1 - slippage
        )  # بهبود: اضافه کردن slippage
        current_date = df.index[i]
        volatility = df["BB_Bandwidth"].iloc[i] if i < len(df) else 0.02

        if in_position:
            pnl_pct = (current_price - entry_price) / entry_price
            trailing_high = max(trailing_high, current_price)

            # ترییلینگ استاپ پویا
            trailing_stop_price = dynamic_trailing_stop(
                current_price, entry_price, trailing_high, volatility
            )
            stop_loss_price = entry_price * (1 - stop_loss_pct)

            # شرایط خروج
            exit_condition = False
            exit_type = ""

            # اولویت‌بندی شرایط خروج
            if pnl_pct >= take_profit_pct:
                exit_condition = True
                exit_type = "TAKE_PROFIT"
            elif current_price <= min(trailing_stop_price, stop_loss_price):
                exit_condition = True
                exit_type = "STOP_LOSS"
            elif df["position"].iloc[i] == -1:
                exit_condition = True
                exit_type = "STRATEGY"

            if exit_condition:
                # محاسبه سود/ضرر
                exit_value = position * current_price * (1 - trade_fee)
                profit = exit_value - (position * entry_price)
                profit_pct = (
                    (profit / (position * entry_price)) * 100
                    if position * entry_price > 0
                    else 0
                )

                capital += exit_value
                trades.append(
                    (
                        "SELL",
                        exit_type,
                        current_date,
                        current_price,
                        exit_value,
                        profit,
                        profit_pct,
                        (current_date - entry_date).days,
                    )
                )
                position = 0
                in_position = False

                profit_icon = "🟢" if profit > 0 else "🔴"
                print(
                    f"{profit_icon} خروج {exit_type}: {current_date.strftime('%Y-%m-%d')} - "
                    f"قیمت: {current_price:.0f} - سود: {profit:+.0f} USDT ({profit_pct:+.1f}%) - "
                    f"سرمایه جدید: {capital:.0f} USDT"
                )

        # شرایط ورود
        if (
            df["position"].iloc[i] == 1
            and not in_position
            and capital > 10
            and trade_count < max_trades
        ):

            # مدیریت اندازه پوزیشن بر اساس پارامترهای بهینه
            investment_amount = capital * position_size_percent

            # حداقل و حداکثر سرمایه‌گذاری
            min_investment = initial_capital * 0.01
            max_investment = initial_capital * 0.20

            investment_amount = max(
                min_investment, min(investment_amount, max_investment)
            )

            if investment_amount > capital:
                continue

            position = investment_amount / current_price
            entry_price = current_price
            entry_date = current_date
            entry_capital = investment_amount
            capital -= investment_amount
            trailing_high = entry_price
            in_position = True
            trade_count += 1

            trades.append(
                ("BUY", "ENTRY", current_date, entry_price, entry_capital, 0, 0, 0)
            )
            print(
                f"🟢 خرید #{trade_count}: {current_date.strftime('%Y-%m-%d')} - "
                f"قیمت: {entry_price:.0f} - سرمایه: {entry_capital:.0f} USDT "
                f"({(entry_capital/initial_capital*100):.1f}% از سرمایه اولیه) - "
                f"سرمایه باقیمانده: {capital:.0f} USDT"
            )

        # محاسبه ارزش فعلی سبد
        if in_position:
            current_equity = capital + (position * current_price)
        else:
            current_equity = capital

        equity_curve[i] = current_equity

    # بستن پوزیشن باز در پایان دوره
    if in_position and len(df) > 0:
        current_price = df["close"].iloc[-1]
        current_date = df.index[-1]
        exit_value = position * current_price * (1 - trade_fee)
        profit = exit_value - (position * entry_price)
        profit_pct = (
            (profit / (position * entry_price)) * 100
            if position * entry_price > 0
            else 0
        )

        capital += exit_value

        trades.append(
            (
                "SELL",
                "END_OF_PERIOD",
                current_date,
                current_price,
                exit_value,
                profit,
                profit_pct,
                (current_date - entry_date).days,
            )
        )
        profit_icon = "🟢" if profit > 0 else "🔴"
        print(
            f"{profit_icon} بستن پوزیشن در پایان دوره: سود {profit:+.0f} USDT ({profit_pct:+.1f}%)"
        )

    # محاسبه ارزش نهایی
    final_value = capital
    if in_position and len(df) > 0:
        final_value = capital + (position * df["close"].iloc[-1])

    if len(trades) == 0:
        print("⚠️ هشدار: هیچ معامله‌ای انجام نشد!")
    else:
        total_profit = final_value - initial_capital
        print(
            f"📊 نتیجه نهایی: سرمایه اولیه {initial_capital:.0f} → سرمایه نهایی {final_value:.0f}"
        )
        print(
            f"📈 سود/ضرر کل: {total_profit:+.0f} USDT ({(total_profit/initial_capital*100):+.1f}%)"
        )

    return trades, equity_curve, capital, position


def multi_symbol_backtest_with_ml(
    symbols, ml_models, timeframe="1d", initial_capital=1000.0
):
    """نسخه بهبود یافته بک‌تست با قابلیت ML"""
    results = {}
    exchange = ccxt.binance()
    since = exchange.parse8601("2023-01-01T00:00:00Z")
    capital_per_symbol = initial_capital / len(symbols)

    print(f"🎯 شروع بک‌تست چند ارزی با قابلیت ML")
    print(f"💰 سرمایه کل: {initial_capital:,.0f} USDT")
    print(f"💰 سرمایه هر نماد: {capital_per_symbol:,.0f} USDT")
    print(f"🔢 تعداد نمادها: {len(symbols)}")
    print(f"🤖 وضعیت ML: {'فعال' if ml_models else 'غیرفعال'}")

    # Fetch data for all symbols to enable cross-symbol correlations
    symbols_data = {}
    for symbol in symbols:
        print(f"📥 دریافت داده‌های تاریخی برای {symbol}...")
        ohlcv = fetch_ohlcv(symbol, timeframe, since)
        if not ohlcv or len(ohlcv) < 300:
            print(f"❌ داده‌های ناکافی برای {symbol}")
            continue
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df.name = symbol  # Explicitly set name
        df = calculate_indicators(df)
        symbols_data[symbol] = df

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"📈 تحلیل نماد: {symbol}")
        print(f"{'='*60}")

        if symbol not in symbols_data:
            print(f"❌ داده‌های ناکافی برای {symbol}")
            continue

        df = symbols_data[symbol]
        # Create cross-symbol data excluding the current symbol
        cross_data = {s: df for s, df in symbols_data.items() if s != symbol}

        # === استفاده از ML برای تولید سیگنال ===
        if ml_models:
            print(f"🤖 استفاده از مدل ML برای {symbol}...")
            df = generate_signals_with_ml(df, ml_models, cross_data)
        else:
            print(f"⚡ استفاده از تحلیل تکنیکال خالص برای {symbol}...")
            df = generate_signals(df)

        # نمایش خلاصه سیگنال‌ها
        buy_signals = len(df[df["position"] == 1])
        sell_signals = len(df[df["position"] == -1])
        print(f"📊 سیگنال‌های شناسایی شده: {buy_signals} خرید, {sell_signals} فروش")

        if buy_signals > 0:
            print(f"🔧 اجرای بک‌تست برای {symbol}")
            trades, equity_curve, final_capital, final_position = run_backtest(
                df, capital_per_symbol
            )

            # محاسبه ارزش نهایی
            if final_position > 0 and len(df) > 0:
                final_value = final_capital + (final_position * df["close"].iloc[-1])
            else:
                final_value = final_capital

            total_return, max_drawdown = analyze_performance(
                trades, equity_curve, capital_per_symbol, final_value, df
            )

            results[symbol] = {
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "trades": trades,
                "equity_curve": equity_curve,
                "final_value": final_value,
                "num_trades": len([t for t in trades if t[0] == "BUY"]),
                "num_signals": buy_signals + sell_signals,
            }

            # نمایش نمودار برای نمادهای با معاملات قابل توجه
            if len(trades) >= 3:
                plot_results(df, equity_curve, trades, capital_per_symbol)
        else:
            print(f"❌ هیچ سیگنال خرید معتبری برای {symbol} شناسایی نشد")

    # تحلیل نهایی
    if results:
        print(f"\n{'='*70}")
        print("📊 نتیجه‌گیری نهایی پرتفوی")
        print(f"{'='*70}")

        total_final_value = sum(res["final_value"] for res in results.values())
        portfolio_return = (
            (total_final_value - initial_capital) / initial_capital
        ) * 100
        total_trades = sum(res["num_trades"] for res in results.values())

        print(f"💰 سرمایه اولیه: {initial_capital:,.0f} USDT")
        print(f"💰 ارزش نهایی: {total_final_value:,.0f} USDT")
        print(f"📈 بازدهی پرتفوی: {portfolio_return:+.2f}%")
        print(f"🔢 کل معاملات: {total_trades}")
        print(f"🎯 نمادهای موفق: {len(results)} از {len(symbols)}")

        # رتبه‌بندی
        print(f"\n🏆 عملکرد نمادها:")
        sorted_results = sorted(
            results.items(), key=lambda x: x[1]["total_return"], reverse=True
        )

        for i, (symbol, res) in enumerate(sorted_results, 1):
            icon = "🟢" if res["total_return"] > 0 else "🔴"
            win_trades = len([t for t in res["trades"] if t[0] == "SELL" and t[5] > 0])
            total_sell = len([t for t in res["trades"] if t[0] == "SELL"])
            win_rate = (win_trades / total_sell * 100) if total_sell > 0 else 0

            print(
                f"{i}. {icon} {symbol}: {res['total_return']:+.1f}% "
                f"(Win Rate: {win_rate:.0f}%, معاملات: {res['num_trades']})"
            )

    return results


# تابع تحلیل پیشرفته عملکرد
def analyze_performance(trades, equity_curve, initial_capital, final_value, df):
    """تحلیل جامع عملکرد استراتژی با معیارهای پیشرفته"""
    print("\n" + "=" * 80)
    print("📊 تحلیل پیشرفته عملکرد")
    print("=" * 80)

    # محاسبه بازدهی کلی
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    buy_hold_return = (
        (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]
    ) * 100

    # اطلاعات پایه
    print(f"💰 سرمایه اولیه: {initial_capital:,.0f} USDT")
    print(f"💰 ارزش نهایی: {final_value:,.0f} USDT")
    print(f"📈 سود/ضرر استراتژی: {total_return:+.2f}%")
    print(f"📊 سود/ضرر Buy & Hold: {buy_hold_return:+.2f}%")
    print(
        f"📅 دوره معاملاتی: {df.index[0].strftime('%Y-%m-%d')} تا {df.index[-1].strftime('%Y-%m-%d')}"
    )
    print(f"📆 مدت دوره: {(df.index[-1] - df.index[0]).days} روز")

    # تحلیل معاملات
    buy_trades = [t for t in trades if t[0] == "BUY"]
    sell_trades = [t for t in trades if t[0] == "SELL"]
    total_trades = len(buy_trades)

    print(f"\n🔢 آمار معاملات:")
    print(f"   • تعداد کل معاملات: {total_trades}")
    print(f"   • معاملات خرید: {len(buy_trades)}")
    print(f"   • معاملات فروش: {len(sell_trades)}")

    if total_trades == 0:
        print("⚠️ هیچ معامله‌ای برای تحلیل وجود ندارد!")
        return total_return, 0

    # تحلیل سود و ضرر
    profits = []
    profits_pct = []
    durations = []
    win_types = {"TAKE_PROFIT": 0, "STOP_LOSS": 0, "STRATEGY": 0, "END_OF_PERIOD": 0}

    for sell_trade in sell_trades:
        if len(sell_trade) > 5:
            profit = sell_trade[5]
            profits.append(profit)

            if len(sell_trade) > 6:
                profit_pct = sell_trade[6]
                profits_pct.append(profit_pct)

            if len(sell_trade) > 7:
                duration = sell_trade[7]
                durations.append(duration)

            if len(sell_trade) > 1:
                exit_type = sell_trade[1]
                if exit_type in win_types:
                    win_types[exit_type] += 1

    # محاسبات پیشرفته
    winning_trades = [p for p in profits if p > 0]
    losing_trades = [p for p in profits if p < 0]
    breakeven_trades = [p for p in profits if p == 0]

    winning_trades_pct = [p for p in profits_pct if p > 0] if profits_pct else []
    losing_trades_pct = [p for p in profits_pct if p < 0] if profits_pct else []

    # آمار سود و ضرر
    print(f"\n💵 آمار مالی:")
    print(f"   • کل سود حاصل: {sum(profits):+.0f} USDT")
    print(f"   • مجموع سود معاملات برنده: {sum(winning_trades):+.0f} USDT")
    print(f"   • مجموع ضرر معاملات بازنده: {sum(losing_trades):.0f} USDT")

    if profits:
        avg_profit = np.mean(profits)
        print(f"   • میانگین سود/ضرر هر معامله: {avg_profit:+.0f} USDT")

    # Win Rate و نسبت‌ها
    win_rate = (len(winning_trades) / len(profits)) * 100 if profits else 0

    print(f"\n🎯 کارایی معاملاتی:")
    print(f"   • Win Rate: {win_rate:.1f}% ({len(winning_trades)}/{len(profits)})")
    print(f"   • معاملات برنده: {len(winning_trades)}")
    print(f"   • معاملات بازنده: {len(losing_trades)}")
    print(f"   • معاملات بدون سود/ضرر: {len(breakeven_trades)}")

    if winning_trades and losing_trades:
        avg_win = np.mean(winning_trades)
        avg_loss = np.mean(losing_trades)
        profit_factor = (
            abs(sum(winning_trades)) / abs(sum(losing_trades))
            if losing_trades
            else float("inf")
        )

        print(f"   • میانگین سود معاملات برنده: {avg_win:+.0f} USDT")
        print(f"   • میانگین ضرر معاملات بازنده: {avg_loss:.0f} USDT")
        print(f"   • نسبت سود به ضرر (Profit Factor): {profit_factor:.2f}")

    if winning_trades_pct and losing_trades_pct:
        avg_win_pct = np.mean(winning_trades_pct)
        avg_loss_pct = np.mean(losing_trades_pct)
        risk_reward_ratio = abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else 0

        print(f"   • میانگین سود درصدی: {avg_win_pct:+.1f}%")
        print(f"   • میانگین ضرر درصدی: {avg_loss_pct:.1f}%")
        print(f"   • نسبت پاداش به ریسک: {risk_reward_ratio:.2f}")

    # تحلیل مدت زمان معاملات
    if durations:
        avg_duration = np.mean(durations)
        max_duration = max(durations)
        min_duration = min(durations)

        print(f"\n⏱ تحلیل زمانی:")
        print(f"   • میانگین مدت معامله: {avg_duration:.1f} روز")
        print(f"   • کوتاه‌ترین معامله: {min_duration} روز")
        print(f"   • طولانی‌ترین معامله: {max_duration} روز")

        short_trades = len([d for d in durations if d <= 7])
        medium_trades = len([d for d in durations if 7 < d <= 30])
        long_trades = len([d for d in durations if d > 30])

        print(f"   • معاملات کوتاه‌مدت (≤7 روز): {short_trades}")
        print(f"   • معاملات میان‌مدت (8-30 روز): {medium_trades}")
        print(f"   • معاملات بلندمدت (>30 روز): {long_trades}")

    # تحلیل انواع خروج
    print(f"\n🚪 انواع خروج از معاملات:")
    total_exits = sum(win_types.values())
    for exit_type, count in win_types.items():
        if count > 0:
            percentage = (count / total_exits) * 100
            icon = (
                "🟢"
                if exit_type == "TAKE_PROFIT"
                else "🔴" if exit_type == "STOP_LOSS" else "🟡"
            )
            print(f"   • {icon} {exit_type}: {count} معامله ({percentage:.1f}%)")

    # تحلیل drawdown
    if equity_curve:
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdowns = (rolling_max - equity_series) / rolling_max
        max_drawdown = drawdowns.max() * 100
        avg_drawdown = drawdowns.mean() * 100

        max_dd_idx = drawdowns.idxmax()
        max_dd_date = (
            df.index[max_dd_idx] if max_dd_idx < len(df.index) else df.index[-1]
        )

        print(f"\n📉 تحلیل کاهش سرمایه:")
        print(f"   • حداکثر کاهش سرمایه (Max Drawdown): {max_drawdown:.2f}%")
        print(f"   • میانگین کاهش سرمایه: {avg_drawdown:.2f}%")
        print(f"   • تاریخ بدترین کاهش: {max_dd_date.strftime('%Y-%m-%d')}")

        drawdown_periods = (drawdowns > 0.01).sum()
        print(f"   • تعداد دوره‌های کاهش سرمایه: {drawdown_periods}")

        # محاسبه نسبت شارپ و سورتینو
        daily_returns = equity_series.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() * 252) / (
                daily_returns.std() * np.sqrt(252)
            )

            negative_returns = daily_returns[daily_returns < 0]
            downside_std = negative_returns.std() if len(negative_returns) > 0 else 0
            sortino_ratio = (
                (daily_returns.mean() * 252) / (downside_std * np.sqrt(252))
                if downside_std > 0
                else 0
            )

            calmar_ratio = (
                (total_return / 100) / (max_drawdown / 100) if max_drawdown > 0 else 0
            )

            print(f"\n📈 معیارهای ریسک-بازده:")
            print(f"   • نسبت شارپ: {sharpe_ratio:.3f}")
            print(f"   • نسبت سورتینو: {sortino_ratio:.3f}")
            print(f"   • نسبت کالمار: {calmar_ratio:.3f}")

    # تحلیل توزیع سود/ضرر
    if profits:
        print(f"\n📊 توزیع سود/ضرر:")
        profit_ranges = {
            "بسیار خوب (>+20%)": len([p for p in profits_pct if p > 20]),
            "خوب (+10% تا +20%)": len([p for p in profits_pct if 10 < p <= 20]),
            "متوسط (+5% تا +10%)": len([p for p in profits_pct if 5 < p <= 10]),
            "کم (+1% تا +5%)": len([p for p in profits_pct if 1 < p <= 5]),
            "نزدیک صفر (±1%)": len([p for p in profits_pct if -1 <= p <= 1]),
            "کم (-5% تا -1%)": len([p for p in profits_pct if -5 <= p < -1]),
            "متوسط (-10% تا -5%)": len([p for p in profits_pct if -10 <= p < -5]),
            "بد (-20% تا -10%)": len([p for p in profits_pct if -20 <= p < -10]),
            "بسیار بد (<-20%)": len([p for p in profits_pct if p < -20]),
        }

        for range_name, count in profit_ranges.items():
            if count > 0:
                percentage = (count / len(profits_pct)) * 100
                print(f"   • {range_name}: {count} معامله ({percentage:.1f}%)")

    # تحلیل عملکرد در شرایط مختلف بازار
    print(f"\n🌡 عملکرد در شرایط بازار:")
    market_conditions = {
        "روند صعودی قوی": (df["ADX"] > 30) & (df["SMA_50"] > df["SMA_200"]),
        "روند نزولی قوی": (df["ADX"] > 30) & (df["SMA_50"] < df["SMA_200"]),
        "روند خنثی": (df["ADX"] < 20),
        "نوسان بالا": (df["BB_Bandwidth"] > df["BB_Bandwidth"].quantile(0.7)),
        "نوسان پایین": (df["BB_Bandwidth"] < df["BB_Bandwidth"].quantile(0.3)),
    }

    for condition_name, condition in market_conditions.items():
        condition_days = condition.sum()
        if condition_days > 0:
            percentage = (condition_days / len(df)) * 100
            print(f"   • {condition_name}: {condition_days} روز ({percentage:.1f}%)")

    # خلاصه نهایی
    print(f"\n🏆 خلاصه نهایی عملکرد:")
    performance_rating = (
        "عالی"
        if total_return > 50 and win_rate > 60
        else (
            "خوب"
            if total_return > 20 and win_rate > 50
            else "متوسط" if total_return > 0 else "ضعیف"
        )
    )

    risk_rating = (
        "کم"
        if max_drawdown < 10
        else (
            "متوسط"
            if max_drawdown < 20
            else "بالا" if max_drawdown < 30 else "بسیار بالا"
        )
    )

    print(f"   • رتبه عملکرد: {performance_rating}")
    print(f"   • سطح ریسک: {risk_rating}")
    print(
        f"   • کارایی نسبت به Buy & Hold: {'برتر' if total_return > buy_hold_return else 'ضعیف‌تر'}"
    )

    if total_trades > 0:
        trades_per_month = total_trades / ((df.index[-1] - df.index[0]).days / 30)
        print(f"   • میانگین معاملات در ماه: {trades_per_month:.1f}")

    return total_return, max_drawdown if "max_drawdown" in locals() else 0


# تابع رسم نمودارها
def plot_results(df, equity_curve, trades, initial_capital):
    """رسم نمودارهای بهبود یافته"""
    if len(equity_curve) != len(df):
        extended_equity = [initial_capital] * len(df)
        for i in range(len(equity_curve)):
            if i < len(df):
                extended_equity[i] = equity_curve[i]
        equity_curve = extended_equity

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

    # نمودار قیمت و معاملات
    ax1.plot(df.index, df["close"], label="قیمت", linewidth=1, color="black")
    ax1.plot(df.index, df["SMA_50"], label="SMA 50", alpha=0.7, color="blue")
    ax1.plot(df.index, df["SMA_200"], label="SMA 200", alpha=0.7, color="red")

    # سیگنال‌های خرید و فروش
    buy_signals = df[df["position"] == 1]
    sell_signals = df[df["position"] == -1]

    if not buy_signals.empty:
        ax1.scatter(
            buy_signals.index,
            buy_signals["close"],
            color="green",
            marker="^",
            s=100,
            label="سیگنال خرید",
            zorder=5,
        )

    if not sell_signals.empty:
        ax1.scatter(
            sell_signals.index,
            sell_signals["close"],
            color="red",
            marker="v",
            s=100,
            label="سیگنال فروش",
            zorder=5,
        )

    ax1.set_title("نمودار قیمت و سیگنال‌های معاملاتی")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    # نمودار RSI
    ax2.plot(df.index, df["RSI"], label="RSI", color="purple", linewidth=1)
    ax2.axhline(70, color="red", linestyle="--", alpha=0.5, label="اشباع خرید")
    ax2.axhline(30, color="green", linestyle="--", alpha=0.5, label="اشباع فروش")
    ax2.axhline(50, color="gray", linestyle="--", alpha=0.3)
    ax2.set_ylim(0, 100)
    ax2.set_title("اندیکاتور RSI")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    # نمودار MACD
    ax3.plot(df.index, df["MACD"], label="MACD", color="blue", linewidth=1)
    ax3.plot(
        df.index, df["Signal_Line"], label="Signal Line", color="orange", linewidth=1
    )
    ax3.bar(
        df.index,
        df["MACD_Histogram"],
        label="Histogram",
        color="gray",
        alpha=0.3,
        width=1,
    )
    ax3.axhline(0, color="black", linestyle="-", alpha=0.5)
    ax3.set_title("اندیکاتور MACD")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis="x", rotation=45)

    # نمودار سرمایه
    ax4.plot(df.index, equity_curve, label="سرمایه", linewidth=2, color="green")
    ax4.axhline(initial_capital, color="red", linestyle="--", label="سرمایه اولیه")

    for trade in trades:
        if trade[0] == "BUY":
            ax4.scatter(trade[2], trade[4], color="blue", marker="^", s=80, zorder=5)
        elif trade[0] == "SELL":
            color = "green" if trade[5] > 0 else "red"
            ax4.scatter(trade[2], trade[4], color=color, marker="v", s=80, zorder=5)

    ax4.set_title("منحنی رشد سرمایه")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    # نمودار اضافی برای ADX و Bollinger Bands
    fig, (ax5, ax6) = plt.subplots(2, 1, figsize=(15, 8))

    # ADX
    ax5.plot(df.index, df["ADX"], label="ADX", color="red", linewidth=2)
    ax5.axhline(25, color="orange", linestyle="--", label="روند قوی")
    ax5.set_title("شاخص میانگین حرکت جهت‌دار (ADX)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis="x", rotation=45)

    # Bollinger Bands
    ax6.plot(df.index, df["close"], label="قیمت", color="black", linewidth=1)
    ax6.plot(df.index, df["BB_Upper"], label="باند بالایی", color="red", alpha=0.7)
    ax6.plot(df.index, df["BB_Lower"], label="باند پایینی", color="green", alpha=0.7)
    ax6.plot(df.index, df["BB_Middle"], label="باند میانی", color="blue", alpha=0.7)
    ax6.fill_between(df.index, df["BB_Upper"], df["BB_Lower"], alpha=0.1)
    ax6.set_title("باندهای بولینگر")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def extract_advanced_features(df, cross_symbol_data=None, symbol=None):
    """استخراج ویژگی‌های پیشرفته برای یادگیری ماشین - محاسبه یک بار"""
    df = df.copy()
    try:
        # Use provided symbol or df.name as fallback
        current_symbol = symbol if symbol else getattr(df, "name", "UNKNOWN")

        # Ensure required indicators are present
        required_indicators = [
            "RSI",
            "MACD",
            "BB_Upper",
            "BB_Lower",
            "BB_Middle",
            "ATR",
            "stoch_k",
            "stoch_d",
        ]
        missing_indicators = [
            ind for ind in required_indicators if ind not in df.columns
        ]
        if missing_indicators:
            print(f"⚠️ اندیکاتورهای غایب برای {current_symbol}: {missing_indicators}")
            df = calculate_indicators(df)

        # 1. ویژگی‌های مبتنی بر قیمت
        df["price_momentum"] = df["close"].pct_change(5)
        df["price_acceleration"] = df["close"].pct_change(5) - df["close"].pct_change(
            10
        )
        df["volatility"] = df["close"].rolling(20).std() / (
            df["close"].rolling(20).mean() + 1e-10
        )

        # 2. ویژگی‌های مبتنی بر حجم
        df["volume_trend"] = df["volume"].pct_change(10).rolling(10).mean()
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
        df["momentum_oscillator"] = (
            df["close"].pct_change(5) - df["close"].pct_change(20)
        ) / (df["close"].pct_change(20).std() + 1e-10)

        # 5. ویژگی‌های نوسان
        df["atr_ratio"] = df["ATR"] / (df["close"] + 1e-10)
        df["bb_squeeze"] = (df["BB_Upper"] - df["BB_Lower"]) / (df["BB_Middle"] + 1e-10)

        # 6. Lagged features (کمتر برای جلوگیری از overfitting)
        for lag in [1, 3]:
            df[f"RSI_lag{lag}"] = df["RSI"].shift(lag)
            df[f"MACD_lag{lag}"] = df["MACD"].shift(lag)

        # 7. Interactions
        df["RSI_MACD_interact"] = df["RSI"] * df["MACD"]
        df["vol_mom_interact"] = df["volatility"] * df["price_momentum"]

        # 8. New features
        df["vol_adjusted_return"] = df["close"].pct_change(5) / (
            df["volatility"] + 1e-10
        )
        df["rsi_trend"] = df["RSI"] - df["RSI"].rolling(20).mean()
        df["stoch_trend"] = df["stoch_k"] - df["stoch_d"]

        # 9. ویژگی‌های پیشرفته‌تر
        df["price_range"] = (df["high"] - df["low"]) / df["close"]
        df["volume_price_trend"] = df["volume"] * df["close"].pct_change()
        df["rsi_volatility"] = df["RSI"].rolling(10).std()
        df["macd_momentum"] = df["MACD"].diff()
        df["adx_trend"] = df["ADX"].diff()

        # ویژگی‌های میانگین متحرک
        df["sma_ratio"] = df["SMA_20"] / df["SMA_50"]
        df["price_to_sma20"] = df["close"] / df["SMA_20"]
        df["price_to_sma50"] = df["close"] / df["SMA_50"]

        # Cross-asset correlation - بهبود: چک برای وجود نمادها
        if cross_symbol_data and isinstance(cross_symbol_data, dict):
            for cross_symbol, cross_df in cross_symbol_data.items():
                if cross_symbol != current_symbol and "close" in cross_df.columns:
                    common_index = df.index.intersection(cross_df.index)
                    if len(common_index) > 20:
                        corr_values = []
                        for idx in df.index:
                            if idx in common_index:
                                window_data = df.loc[:idx, "close"].tail(20)
                                cross_window_data = cross_df.loc[:idx, "close"].tail(20)
                                if (
                                    len(window_data) == 20
                                    and len(cross_window_data) == 20
                                ):
                                    corr_values.append(
                                        window_data.corr(cross_window_data)
                                    )
                                else:
                                    corr_values.append(0)
                            else:
                                corr_values.append(0)
                        df[f"corr_{cross_symbol}"] = corr_values
                    else:
                        df[f"corr_{cross_symbol}"] = 0

        feature_count = len(
            [
                col
                for col in df.columns
                if col.startswith("price_")
                or col.startswith("RSI_")
                or col.startswith("corr_")
            ]
        )
        print(f"✅ استخراج {feature_count} ویژگی ML برای {current_symbol} موفق بود")

    except Exception as e:
        print(f"❌ خطا در استخراج ویژگی‌های ML برای {current_symbol}: {e}")
        # Fallback features
        df["price_momentum"] = df["close"].pct_change(5)
        df["volatility"] = df["close"].rolling(20).std() / (
            df["close"].rolling(20).mean() + 1e-10
        )
        df["volume_trend"] = df["volume"].pct_change(5)

    return df


def create_ml_features(df, target_lookahead=5):
    """ایجاد ویژگی‌ها و تارگت برای مدل ML - بهبود: آستانه پویا بر اساس ATR"""
    df = calculate_indicators(df)
    symbol = getattr(df, "name", "UNKNOWN")
    df = extract_advanced_features(df, cross_symbol_data=None, symbol=symbol)

    # تارگت با آستانه پویا بر اساس ATR
    future_returns = df["close"].shift(-target_lookahead) / df["close"] - 1
    df["future_return"] = future_returns

    # محاسبه آستانه بهتر
    atr_mean = df["ATR"].mean() / df["close"].mean()  # normalize ATR
    return_threshold = max(0.03, atr_mean * 2)  # حداقل 3%، پویا بر اساس volatility

    df["target"] = (df["future_return"] > return_threshold).astype(int)

    print(
        f"📌 آستانه تارگت برای {symbol}: {return_threshold:.4f} ({return_threshold*100:.2f}%)"
    )

    # ویژگی‌های کامل (حذف lagged برای جلوگیری از overfitting)
    base_features = [
        "price_momentum",
        "price_acceleration",
        "volatility",
        "volume_trend",
        "volume_volatility",
        "body_size",
        "upper_shadow",
        "lower_shadow",
        "trend_strength",
        "momentum_oscillator",
        "RSI",
        "MACD",
        "ADX",
        "BB_Bandwidth",
        "RSI_MACD_interact",
        "vol_mom_interact",
        "vol_adjusted_return",
        "rsi_trend",
        "stoch_trend",
        "price_range",
        "volume_price_trend",
        "rsi_volatility",
        "macd_momentum",
        "adx_trend",
        "sma_ratio",
        "price_to_sma20",
        "price_to_sma50",
    ]

    optional_features = [
        "atr_ratio",
        "bb_squeeze",
        "corr_BTC/USDT",
        "corr_ETH/USDT",
        "corr_SOL/USDT",
    ]

    available_features = [f for f in base_features if f in df.columns]
    for feature in optional_features:
        if feature in df.columns:
            available_features.append(feature)

    df_clean = df[available_features + ["future_return", "target"]].dropna()

    print(f"📊 تعداد ویژگی‌های ML: {len(available_features)}")
    print(f"📈 تعداد نمونه‌های قابل استفاده: {len(df_clean)}")

    # نمایش توزیع تارگت
    target_dist = df_clean["target"].value_counts()
    print(
        f"📊 توزیع تارگت: Negative={target_dist.get(0, 0)}, Positive={target_dist.get(1, 0)}"
    )

    return df_clean, available_features


def train_ml_models_with_wfo(df, symbol):
    """آموزش مدل LSTM با Walk-Forward Optimization و SHAP"""
    try:
        # Make sure df has name attribute
        if not hasattr(df, "name"):
            df.name = symbol

        df_features, feature_columns = create_ml_features(df)

        if len(df_features) < 100:
            print(
                f"⚠️ داده کافی برای آموزش مدل {symbol} وجود ندارد (فقط {len(df_features)} نمونه)"
            )
            return None, None, None

        target_counts = df_features["target"].value_counts()
        print(f"📊 توزیع تارگت برای {symbol}: {dict(target_counts)}")

        # اصلاح: تبدیل به لیست پایتون
        target_values = list(target_counts.values)

        if len(target_counts) < 2 or min(target_values) < 10:
            print(f"⚠️ داده‌های تارگت نامتعادل برای {symbol}")
            return None, None, None

        # Walk-Forward Optimization: تقسیم داده به 5 window
        n_windows = 5
        window_size = len(df_features) // n_windows
        best_model = None
        best_score = 0
        best_scaler = None
        best_X_train = None

        for w in range(1, n_windows):
            train_end = w * window_size
            test_start = train_end
            test_end = min((w + 1) * window_size, len(df_features))

            X_train = df_features.iloc[:train_end][feature_columns]
            y_train = df_features.iloc[:train_end]["target"]
            X_test = df_features.iloc[test_start:test_end][feature_columns]
            y_test = df_features.iloc[test_start:test_end]["target"]

            if len(X_train) < 50 or len(X_test) < 10:
                continue

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # SMOTE
            min_class_count = min(y_train.value_counts().values)
            k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_resampled, y_train_resampled = smote.fit_resample(
                X_train_scaled, y_train
            )

            # LSTM Model - Fix the input shape warning
            X_train_lstm = X_train_resampled.reshape(
                (X_train_resampled.shape[0], 1, X_train_resampled.shape[1])
            )
            X_test_lstm = X_test_scaled.reshape(
                (X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
            )

            # Fix LSTM model construction to remove warning
            model = Sequential(
                [
                    LSTM(
                        50,
                        return_sequences=True,
                        input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]),
                    ),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(25, activation="relu"),
                    Dense(1, activation="sigmoid"),
                ]
            )

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            print(f"⏳ در حال آموزش مدل LSTM {symbol} در window {w}...")
            model.fit(
                X_train_lstm, y_train_resampled, epochs=50, batch_size=32, verbose=0
            )

            y_pred_prob = model.predict(X_test_lstm)
            y_pred = (y_pred_prob > 0.5).astype(int)
            score = f1_score(y_test, y_pred.flatten())

            print(f"🎯 امتیاز F1 در window {w} برای {symbol}: {score:.3f}")

            if score > best_score:
                best_score = score
                best_model = model
                best_scaler = scaler
                best_X_train = X_train_scaled

        if best_model is None:
            print(f"❌ هیچ مدلی برای {symbol} آموزش داده نشد")
            return None, None, None

        # SHAP with TreeExplainer as surrogate model
        print("🔍 در حال محاسبه اهمیت ویژگی‌ها با SHAP...")

        # Train a tree-based model as surrogate for SHAP
        from sklearn.ensemble import RandomForestClassifier

        # Use the best training data
        X_train_flat = best_X_train
        y_pred_proba = best_model.predict(
            best_X_train.reshape((best_X_train.shape[0], 1, best_X_train.shape[1]))
        )
        y_pred_train = (y_pred_proba > 0.5).astype(int).flatten()

        # Train surrogate model
        surrogate_model = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10
        )
        surrogate_model.fit(X_train_flat, y_pred_train)

        # Calculate SHAP values
        explainer = shap.TreeExplainer(surrogate_model)

        # Use a subset for SHAP calculation
        sample_size = min(100, len(X_train_flat))
        X_sample = X_train_flat[:sample_size]

        shap_values = explainer.shap_values(X_sample)

        # Plot results - Handle different SHAP output formats
        plt.figure(figsize=(10, 8))

        if isinstance(shap_values, list) and len(shap_values) == 2:
            # Binary classification case
            shap.summary_plot(
                shap_values[1], X_sample, feature_names=feature_columns, show=False
            )
            feature_importance = np.abs(shap_values[1]).mean(0)
        else:
            # Single array case
            shap.summary_plot(
                shap_values, X_sample, feature_names=feature_columns, show=False
            )
            feature_importance = np.abs(shap_values).mean(0)

        plt.savefig(
            f"shap_summary_{symbol.replace('/', '_')}.png", bbox_inches="tight", dpi=300
        )
        plt.close()
        print(f"✅ نمودار SHAP ذخیره شد: shap_summary_{symbol.replace('/', '_')}.png")

        # نمایش 5 ویژگی مهم - FIXED: Ensure feature_importance is 1D
        if len(feature_importance.shape) > 1:
            feature_importance = feature_importance.mean(
                axis=0
            )  # Average across samples if needed

        # Create feature importance dictionary safely
        importance_dict = {}
        for i, feature in enumerate(feature_columns):
            if i < len(feature_importance):
                importance_dict[feature] = float(feature_importance[i])

        # Sort and get top features
        top_features = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )[:5]

        print(f"🔝 5 ویژگی مهم برای {symbol}:")
        for feature, importance in top_features:
            print(f"   {feature}: {importance:.4f}")

        return best_model, best_scaler, feature_columns

    except Exception as e:
        print(f"❌ خطا در آموزش مدل {symbol}: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None


def predict_with_ml(model, scaler, feature_columns, df_current, cross_data=None):
    """پیش‌بینی با مدل ML روی داده‌های جاری - بهبود: بدون تکرار استخراج"""
    # استخراج ویژگی‌ها فقط یک بار در caller انجام می‌شود
    # فرض می‌کنیم df_current ویژگی‌ها رو داره

    # اطمینان از وجود تمام ویژگی‌ها
    available_features = [f for f in feature_columns if f in df_current.columns]
    if len(available_features) != len(feature_columns):
        missing_features = [f for f in feature_columns if f not in df_current.columns]
        print(f"⚠️ ویژگی‌های غایب: {missing_features}")
        return 0.5  # مقدار خنثی

    X_current = df_current[available_features].iloc[[-1]]  # آخرین کندل
    X_scaled = scaler.transform(X_current)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # پیش‌بینی احتمال کلاس مثبت
    probability = model.predict(X_lstm)[0][0]

    return probability


def generate_signals_with_ml(df, ml_models=None, cross_data=None):
    """تولید سیگنال‌ها با ترکیب تحلیل تکنیکال و ML - نسخه بهبود یافته بدون loop تکراری"""
    df = df.copy()
    symbol = (
        df.name if hasattr(df, "name") else "BTC/USDT"
    )  # Fallback to default symbol

    # تحلیل تکنیکال اصلی
    df = generate_signals(df)

    # اگر مدل ML موجود باشد
    if ml_models and len(df) > 200 and symbol in ml_models:
        model, scaler, features = ml_models[symbol]

        # استخراج ویژگی‌ها یک بار برای کل دیتافریم
        df = extract_advanced_features(df, cross_data, symbol=symbol)

        # بررسی وجود تمام ویژگی‌های مورد نیاز
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"⚠️ ویژگی‌های گمشده برای {symbol}: {missing_features}")
            # استفاده فقط از ویژگی‌های موجود
            available_features = [f for f in features if f in df.columns]

            if (
                len(available_features) < len(features) * 0.8
            ):  # اگر کمتر از 80% ویژگی‌ها موجود باشد
                print(f"❌ ویژگی‌های کافی برای پیش‌بینی ML موجود نیست")
                return df
            else:
                features = available_features
                print(
                    f"✅ استفاده از {len(features)} ویژگی از {len(ml_models[symbol][2])} ویژگی اصلی"
                )

        # پیش‌بینی ML برای تمام کندل‌ها از 200 به بعد
        try:
            X = df.iloc[200:][features]

            # بررسی وجود داده کافی
            if len(X) == 0:
                print(f"⚠️ داده کافی برای پیش‌بینی ML وجود ندارد")
                return df

            X_scaled = scaler.transform(X)
            X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            ml_predictions = model.predict(X_lstm).flatten()

            # اضافه کردن پیش‌بینی‌های ML به دیتافریم
            df_ml = df.iloc[200:].copy()
            df_ml["ml_confidence"] = ml_predictions

            # منطق ترکیبی بهبود یافته
            for i in df_ml.index:
                idx = df.index.get_loc(i)
                ml_conf = df_ml.loc[i, "ml_confidence"]
                tech_signal = df.loc[i, "signal"]
                adx = df.loc[i, "ADX"]

                # Stricter ML requirements in neutral markets
                buy_ml_threshold = 0.65 if adx < 20 else 0.55
                sell_ml_threshold = 0.35 if adx < 20 else 0.45

                if tech_signal == 1:
                    if ml_conf >= buy_ml_threshold:
                        df.loc[i, "signal"] = 1
                        df.loc[i, "position"] = 1 if df.loc[i, "position"] == 1 else 0
                    elif ml_conf < 0.35:
                        df.loc[i, "signal"] = 0
                        df.loc[i, "position"] = 0
                elif tech_signal == -1:
                    if ml_conf <= sell_ml_threshold:
                        df.loc[i, "signal"] = -1
                        df.loc[i, "position"] = -1 if df.loc[i, "position"] == -1 else 0
                    elif ml_conf > 0.65:
                        df.loc[i, "signal"] = 0
                        df.loc[i, "position"] = 0

            print(f"🤖 ML Integration: {len(ml_predictions)} پیش‌بینی انجام شد")
            print(f"   میانگین اعتماد ML: {np.mean(ml_predictions):.3f}")
            print(
                f"   دامنه اعتماد ML: [{np.min(ml_predictions):.3f}, {np.max(ml_predictions):.3f}]"
            )

            # تحلیل توزیع اعتماد ML
            high_confidence = len([x for x in ml_predictions if x > 0.6])
            low_confidence = len([x for x in ml_predictions if x < 0.4])
            print(f"   پیش‌بینی‌های با اعتماد بالا (>0.6): {high_confidence}")
            print(f"   پیش‌بینی‌های با اعتماد پایین (<0.4): {low_confidence}")

        except Exception as e:
            print(f"❌ خطا در پیش‌بینی ML برای {symbol}: {e}")
            import traceback

            traceback.print_exc()

    return df


def live_trading(symbols, ml_models, api_key, api_secret):
    """حالت لایو تریدینگ با Binance Testnet"""
    from binance.client import Client

    client = Client(api_key, api_secret, testnet=True)  # استفاده از Testnet

    print("🟢 حالت لایو (Testnet) فعال شد. نظارت بر بازار...")

    while True:
        for symbol in symbols:
            # Fetch live data (last 200 candles)
            bars = client.get_klines(
                symbol=symbol.replace("/", ""),
                interval=Client.KLINE_INTERVAL_1DAY,
                limit=200,
            )
            df = pd.DataFrame(
                bars,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_av",
                    "trades",
                    "tb_base_av",
                    "tb_quote_av",
                    "ignore",
                ],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df["close"] = pd.to_numeric(df["close"])
            df.name = symbol
            df = calculate_indicators(df)
            df = extract_advanced_features(df)

            model, scaler, features = ml_models.get(symbol, (None, None, None))
            if model:
                df = generate_signals_with_ml(df, ml_models)

                # چک برای سیگنال جدید
                last_position = df["position"].iloc[-1]
                if last_position == 1:
                    # Place buy order
                    order = client.order_market_buy(
                        symbol=symbol.replace("/", ""), quantity=0.001
                    )  # نمونه کوچک
                    print(f"🟢 خرید لایو برای {symbol}: {order}")
                elif last_position == -1:
                    # Place sell order
                    order = client.order_market_sell(
                        symbol=symbol.replace("/", ""), quantity=0.001
                    )
                    print(f"🔴 فروش لایو برای {symbol}: {order}")

        time.sleep(3600)  # چک هر ساعت


# اجرای اصلی
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run in live testnet mode")
    parser.add_argument("--api_key", type=str, help="Binance API Key for live")
    parser.add_argument("--api_secret", type=str, help="Binance API Secret for live")
    args = parser.parse_args()

    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    print("🤖 در حال راه‌اندازی سیستم یادگیری ماشین...")
    ml_manager = MLModelManager()
    try:
        ml_manager.load_models()
        print("✅ مدل‌های ML از حافظه بارگذاری شدند")
        print(f"📊 تعداد مدل‌های بارگذاری شده: {len(ml_manager.models)}")
    except Exception as e:
        print(f"📚 مدل‌های ذخیره شده یافت نشد ({e})، در حال آموزش مدل‌های جدید...")
        symbols_data = {}
        exchange = ccxt.binance()
        for symbol in symbols:
            print(f"📥 دریافت داده‌های تاریخی برای آموزش ML {symbol}...")
            since = exchange.parse8601("2020-01-01T00:00:00Z")
            ohlcv = fetch_ohlcv(symbol, "1d", since)
            if ohlcv and len(ohlcv) > 500:
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                df.name = symbol
                df = calculate_indicators(df)
                symbols_data[symbol] = df
                print(f"✅ داده‌های {symbol} برای آموزش آماده شد ({len(ohlcv)} کندل)")
            else:
                print(f"❌ داده کافی برای {symbol} دریافت نشد")
        if symbols_data:
            # Align indices across symbols
            common_index = None
            for symbol, df in symbols_data.items():
                if common_index is None:
                    common_index = df.index
                else:
                    common_index = common_index.intersection(df.index)
            for symbol in symbols_data:
                symbols_data[symbol] = symbols_data[symbol].loc[common_index]
                cross_data = {s: df for s, df in symbols_data.items() if s != symbol}
                symbols_data[symbol] = extract_advanced_features(
                    symbols_data[symbol], cross_data, symbol
                )
            ml_manager.train_models_for_symbols(symbols_data)
            ml_manager.save_models()
            print(f"🎉 آموزش مدل‌های ML کامل شد. {len(ml_manager.models)} مدل ذخیره شد.")
        else:
            print("❌ هیچ داده‌ای برای آموزش ML دریافت نشد")
            ml_manager.models = {}

    # === پایان بخش ML ===

    # نمایش وضعیت نهایی ML
    if ml_manager.models:
        print(f"🤖 وضعیت ML: فعال ({len(ml_manager.models)} مدل آماده)")
        for symbol in ml_manager.models.keys():
            print(f"   ✅ {symbol}: مدل ML آماده")
    else:
        print("🤖 وضعیت ML: غیرفعال (استفاده از تحلیل تکنیکال خالص)")

    if args.live:
        if not args.api_key or not args.api_secret:
            print("❌ API Key و Secret لازم است برای حالت لایو.")
        else:
            live_trading(symbols, ml_manager.models, args.api_key, args.api_secret)
    else:
        # اجرای بک‌تست اصلی با ML
        results = multi_symbol_backtest_with_ml(symbols, ml_manager.models)
