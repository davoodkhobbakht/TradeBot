# -*- coding: utf-8 -*-
# strategies/gold_strategy.py

import pandas as pd
import numpy as np
from datetime import datetime, time


class GoldStrategy:
    """استراتژی طلایی کامل با الهام از کد TradingView"""

    def __init__(self):
        self.name = "Gold Strategy"
        self.weight = 0.3  # وزن در ترکیب با سایر استراتژی‌ها

    def calculate_indicators(self, df):
        """محاسبه تمام اندیکاتورهای مورد نیاز"""
        df = df.copy()

        # ۱. میانگین‌های متحرک
        df["ema_8"] = df["close"].ewm(span=8, adjust=False).mean()
        df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()

        # ۲. RSI
        df["rsi_14"] = self.calculate_rsi(df["close"], 14)

        # ۳. ATR (نوسان)
        df["atr_14"] = self.calculate_atr(df, 14)
        df["atr_percent"] = (df["atr_14"] / df["close"]) * 100

        # ۴. حجم
        df["volume_ma_10"] = df["volume"].rolling(10).mean()

        # ۵. MACD (اگر موجود نیست محاسبه کن)
        if "MACD" not in df.columns:
            ema_12 = df["close"].ewm(span=12, adjust=False).mean()
            ema_26 = df["close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = ema_12 - ema_26
            df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # ۶. ADX (اگر موجود نیست محاسبه کن)
        if "ADX" not in df.columns:
            df["ADX"] = self.calculate_adx(df)

        return df

    def calculate_rsi(self, prices, period=14):
        """محاسبه RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_atr(self, df, period=14):
        """محاسبه ATR"""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return atr

    def calculate_adx(self, df, period=14):
        """محاسبه ADX ساده‌شده"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # محاسبه +DM و -DM
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # محاسجه TR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        # محاسبه +DI و -DI
        plus_di = 100 * (
            pd.Series(plus_dm).rolling(period).mean() / tr.rolling(period).mean()
        )
        minus_di = 100 * (
            pd.Series(minus_dm).rolling(period).mean() / tr.rolling(period).mean()
        )

        # محاسبه DX و ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()

        return adx

    def is_good_trading_session(self, timestamp):
        """بررسی ساعات معاملاتی مناسب (لندن و نیویورک)"""
        try:
            hour = timestamp.hour

            # ساعات لندن: ۷ صبح تا ۴ عصر
            london_session = 7 <= hour < 16

            # ساعات نیویورک: ۱ ظهر تا ۱۰ شب
            new_york_session = 13 <= hour < 22

            return london_session or new_york_session
        except:
            return True  # اگر تاریخ مشکل داشت، همیشه True برگردون

    def analyze_conditions(self, df):
        """آنالیز تمام شرایط بازار"""

        # ۱. روندها
        conditions = {
            "trend_up": (df["ema_8"] > df["ema_21"]) & (df["close"] > df["sma_20"]),
            "trend_down": (df["ema_8"] < df["ema_21"]) & (df["close"] < df["sma_20"]),
            "primary_trend_up": df["sma_50"] > df["sma_50"].shift(20),
            "primary_trend_down": df["sma_50"] < df["sma_50"].shift(20),
        }

        # ۲. RSI
        conditions.update(
            {
                "rsi_ob": df["rsi_14"] > 65,  # اشباع خرید
                "rsi_os": df["rsi_14"] < 35,  # اشباع فروش
                "rsi_neutral": (df["rsi_14"] >= 40) & (df["rsi_14"] <= 60),
                "rsi_bullish": (df["rsi_14"] > 45) & (df["rsi_14"] < 70),
                "rsi_bearish": (df["rsi_14"] < 55) & (df["rsi_14"] > 30),
            }
        )

        # ۳. نوسانات
        conditions.update(
            {
                "low_volatility": df["atr_percent"] < 0.1,
                "high_volatility": df["atr_percent"] > 0.3,
                "normal_volatility": (df["atr_percent"] >= 0.1)
                & (df["atr_percent"] <= 0.3),
            }
        )

        # ۴. حجم
        conditions.update(
            {
                "volume_spike": df["volume"] > df["volume_ma_10"] * 1.5,
                "good_volume": df["volume"] > df["volume_ma_10"],
                "low_volume": df["volume"] < df["volume_ma_10"],
            }
        )

        # ۵. مومنتوم
        conditions.update(
            {
                "macd_bullish": df["MACD"] > df["Signal_Line"],
                "macd_bearish": df["MACD"] < df["Signal_Line"],
                "strong_trend": df["ADX"] > 25,
                "weak_trend": df["ADX"] < 20,
            }
        )

        # ۶. کندل استیک
        body_size = abs(df["close"] - df["open"])
        avg_body = body_size.rolling(20).mean()
        conditions.update(
            {
                "strong_bull_candle": (df["close"] > df["open"])
                & (body_size > avg_body * 1.2)
                & (df["close"] > df["high"].shift(1)),
                "strong_bear_candle": (df["close"] < df["open"])
                & (body_size > avg_body * 1.2)
                & (df["close"] < df["low"].shift(1)),
                "weak_candle": body_size < avg_body * 0.8,
            }
        )

        # ۷. ساعات معاملاتی
        session_conditions = []
        for idx in df.index:
            session_conditions.append(self.is_good_trading_session(idx))
        conditions["good_session"] = pd.Series(session_conditions, index=df.index)

        return conditions

    def calculate_score(self, df, conditions):
        """محاسبه امتیاز استراتژی"""

        # سیستم امتیازدهی برای خرید
        buy_score = (
            (conditions["trend_up"]) * 3
            + (conditions["primary_trend_up"]) * 2
            + (conditions["rsi_bullish"]) * 2
            + (~conditions["rsi_ob"]) * 2
            + (conditions["normal_volatility"]) * 2
            + (conditions["good_volume"]) * 2
            + (conditions["strong_bull_candle"]) * 2
            + (conditions["macd_bullish"]) * 1
            + (conditions["strong_trend"]) * 1
            + (conditions["good_session"]) * 1
        )

        # سیستم امتیازدهی برای فروش
        sell_score = (
            (conditions["trend_down"]) * 3
            + (conditions["primary_trend_down"]) * 2
            + (conditions["rsi_bearish"]) * 2
            + (~conditions["rsi_os"]) * 2
            + (conditions["normal_volatility"]) * 2
            + (conditions["good_volume"]) * 2
            + (conditions["strong_bear_candle"]) * 2
            + (conditions["macd_bearish"]) * 1
            + (conditions["strong_trend"]) * 1
            + (conditions["good_session"]) * 1
        )

        return buy_score, sell_score

    def generate_signals(self, df):
        """تولید سیگنال‌های استراتژی طلایی"""
        df = df.copy()
        symbol = getattr(df, "name", "UNKNOWN")

        print(f"\n🏆 اجرای استراتژی طلایی برای {symbol}...")

        # ۱. محاسبه اندیکاتورها
        df = self.calculate_indicators(df)

        # ۲. آنالیز شرایط
        conditions = self.analyze_conditions(df)

        # ۳. محاسبه امتیاز
        buy_score, sell_score = self.calculate_score(df, conditions)

        # ۴. تولید سیگنال با آستانه‌های پویا
        df["gold_signal"] = 0

        # آستانه‌های پویا
        buy_threshold = 12
        sell_threshold = 12

        for i in range(len(df)):
            # فیلترهای نهایی
            volume_ok = (
                conditions["good_volume"].iloc[i] or conditions["volume_spike"].iloc[i]
            )
            volatility_ok = conditions["normal_volatility"].iloc[i]
            session_ok = conditions["good_session"].iloc[i]

            if (
                buy_score.iloc[i] >= buy_threshold
                and volume_ok
                and volatility_ok
                and session_ok
                and not conditions["rsi_ob"].iloc[i]
            ):
                df.iloc[i, df.columns.get_loc("gold_signal")] = 1

            if (
                sell_score.iloc[i] >= sell_threshold
                and volume_ok
                and volatility_ok
                and session_ok
                and not conditions["rsi_os"].iloc[i]
            ):
                df.iloc[i, df.columns.get_loc("gold_signal")] = -1

        # ۵. ایجاد موقعیت‌های واقعی
        df["gold_position"] = 0
        last_position = 0
        consecutive_signals = 0

        for i in range(1, len(df)):
            current_signal = df["gold_signal"].iloc[i]

            if current_signal != 0 and current_signal != last_position:
                # فیلتر نویز: حداقل ۲ سیگنال پشت سر هم
                if consecutive_signals >= 1:
                    df.iloc[i, df.columns.get_loc("gold_position")] = current_signal
                    last_position = current_signal
                    consecutive_signals = 0
                else:
                    consecutive_signals += 1
            else:
                df.iloc[i, df.columns.get_loc("gold_position")] = last_position
                consecutive_signals = 0

        # ۶. آمار و گزارش
        buy_signals = len(df[df["gold_signal"] == 1])
        sell_signals = len(df[df["gold_signal"] == -1])
        buy_positions = len(df[df["gold_position"] == 1])
        sell_positions = len(df[df["gold_position"] == -1])

        print(f"📊 سیگنال‌های شناسایی شده: {buy_signals} خرید, {sell_signals} فروش")
        print(f"📈 موقعیت‌های واقعی: {buy_positions} خرید, {sell_positions} فروش")
        print(f"🎯 میانگین امتیاز خرید: {buy_score.mean():.2f}")
        print(f"🎯 میانگین امتیاز فروش: {sell_score.mean():.2f}")
        print(
            f"🕒 استفاده از فیلتر زمانی: {'✅' if any(conditions['good_session']) else '❌'}"
        )

        return df

    def get_strategy_score(self, df):
        """دریافت امتیاز استراتژی برای ترکیب با سایر استراتژی‌ها"""
        df_with_signals = self.generate_signals(df)

        # محاسبه امتیاز بر اساس آخرین سیگنال
        if len(df_with_signals) > 0:
            last_signal = df_with_signals["gold_signal"].iloc[-1]
            last_position = df_with_signals["gold_position"].iloc[-1]

            # اگر موقعیت فعال داریم، امتیاز کامل
            if last_position != 0:
                return last_position * 1.0
            # اگر سیگنال داریم اما موقعیت نه (ممکن است فیلتر شده باشد)
            elif last_signal != 0:
                return last_signal * 0.7
            else:
                return 0.0
        else:
            return 0.0
