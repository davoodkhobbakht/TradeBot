# -*- coding: utf-8 -*-
# strategies/gold_strategy.py

import pandas as pd
import numpy as np
from datetime import datetime, time


class GoldStrategy:
    """Further Optimized Gold Strategy for enhanced signal generation in crypto"""

    def __init__(self):
        self.name = "Gold Strategy"
        self.weight = 0.35  # Weight in multi-strategy

    def calculate_indicators(self, df):
        """Calculate indicators with optimized periods"""
        df = df.copy()

        # Moving Averages (optimized to 9/21 for short-term responsiveness)
        df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["sma_200"] = df["close"].rolling(200).mean()

        # RSI
        df["rsi_14"] = self.calculate_rsi(df["close"], 14)

        # ATR
        df["atr_14"] = self.calculate_atr(df, 14)
        df["atr_percent"] = (df["atr_14"] / df["close"]) * 100

        # Volume
        df["volume_ma_10"] = df["volume"].rolling(10).mean()

        # MACD
        if "MACD" not in df.columns:
            ema_12 = df["close"].ewm(span=12, adjust=False).mean()
            ema_26 = df["close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = ema_12 - ema_26
            df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # ADX (14-period for balance, as per searches)
        if "adx_14" not in df.columns:
            df["adx_14"] = self.calculate_adx(df, period=14)

        return df

    def calculate_rsi(self, prices, period=14):
        """RSI calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_atr(self, df, period=14):
        """ATR calculation"""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return atr

    def calculate_adx(self, df, period=14):
        """ADX calculation"""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        plus_di = 100 * (
            pd.Series(plus_dm).rolling(period).mean() / tr.rolling(period).mean()
        )
        minus_di = 100 * (
            pd.Series(minus_dm).rolling(period).mean() / tr.rolling(period).mean()
        )
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        return adx

    def is_good_trading_session(self, timestamp):
        """Check trading sessions"""
        try:
            hour = timestamp.hour
            london_session = 7 <= hour < 16
            new_york_session = 13 <= hour < 22
            return london_session or new_york_session
        except:
            return True

    def analyze_conditions(self, df):
        """Analyze conditions with optimized RSI"""
        conditions = {
            "trend_up": (df["ema_fast"] > df["ema_slow"])
            & (df["adx_14"] > 20),  # Optimized ADX >20
            "trend_down": (df["ema_fast"] < df["ema_slow"]) & (df["adx_14"] > 20),
            "primary_trend_up": df["sma_50"] > df["sma_200"],
            "primary_trend_down": df["sma_50"] < df["sma_200"],
        }

        conditions.update(
            {
                "rsi_ob": df["rsi_14"] > 70,  # Overbought
                "rsi_os": df["rsi_14"] < 30,  # Oversold
                "rsi_neutral": (df["rsi_14"] >= 40) & (df["rsi_14"] <= 60),
                "rsi_bullish": (df["rsi_14"] > 40)
                & (df["rsi_14"] < 65),  # Optimized for more buys
                "rsi_bearish": (df["rsi_14"] < 60)
                & (df["rsi_14"] > 35),  # Adjusted for balance
            }
        )

        conditions.update(
            {
                "low_volatility": df["atr_percent"] < 0.1,
                "high_volatility": df["atr_percent"] > 0.3,
                "normal_volatility": (df["atr_percent"] >= 0.1)
                & (df["atr_percent"] <= 0.3),
            }
        )

        conditions.update(
            {
                "volume_spike": df["volume"] > df["volume_ma_10"] * 1.5,
                "good_volume": df["volume"] > df["volume_ma_10"],
                "low_volume": df["volume"] < df["volume_ma_10"],
            }
        )

        conditions.update(
            {
                "macd_bullish": df["MACD"] > df["Signal_Line"],
                "macd_bearish": df["MACD"] < df["Signal_Line"],
                "strong_trend": df["adx_14"] > 20,  # Optimized threshold
                "weak_trend": df["adx_14"] < 15,
            }
        )

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

        session_conditions = [self.is_good_trading_session(idx) for idx in df.index]
        conditions["good_session"] = pd.Series(session_conditions, index=df.index)

        return conditions

    def calculate_score(self, df, conditions):
        """Scores with adjusted weights for more signals"""
        buy_score = (
            (conditions["trend_up"]) * 4
            + (conditions["primary_trend_up"]) * 2
            + (conditions["rsi_bullish"]) * 3  # Boost RSI weight
            + (~conditions["rsi_ob"]) * 2
            + (conditions["normal_volatility"]) * 2
            + (conditions["volume_spike"]) * 3
            + (conditions["strong_bull_candle"]) * 2
            + (conditions["macd_bullish"]) * 1
            + (conditions["strong_trend"]) * 2
            + (conditions["good_session"]) * 1
        )

        sell_score = (
            (conditions["trend_down"]) * 4
            + (conditions["primary_trend_down"]) * 2
            + (conditions["rsi_bearish"]) * 3
            + (~conditions["rsi_os"]) * 2
            + (conditions["normal_volatility"]) * 2
            + (conditions["volume_spike"]) * 3
            + (conditions["strong_bear_candle"]) * 2
            + (conditions["macd_bearish"]) * 1
            + (conditions["strong_trend"]) * 2
            + (conditions["good_session"]) * 1
        )

        return buy_score, sell_score

    def generate_signals(self, df):
        """Generate signals with lower thresholds"""
        df = df.copy()
        symbol = getattr(df, "name", "UNKNOWN")

        print(f"\n🏆 Running further optimized Gold Strategy for {symbol}...")

        df = self.calculate_indicators(df)
        conditions = self.analyze_conditions(df)
        buy_score, sell_score = self.calculate_score(df, conditions)

        df["gold_signal"] = 0
        buy_threshold = 8  # Lowered for more signals
        sell_threshold = 8

        for i in range(len(df)):
            volume_ok = (
                conditions["volume_spike"].iloc[i] or conditions["good_volume"].iloc[i]
            )
            volatility_ok = conditions["normal_volatility"].iloc[i]
            session_ok = conditions["good_session"].iloc[i]
            trend_ok = conditions["strong_trend"].iloc[i]

            atr_stop = df["atr_14"].iloc[i] * 3.5
            df.loc[df.index[i], "stop_loss"] = (
                df["close"].iloc[i] - atr_stop
                if conditions["trend_up"].iloc[i]
                else df["close"].iloc[i] + atr_stop
            )

            if (
                buy_score.iloc[i] >= buy_threshold
                and volume_ok
                and volatility_ok
                and session_ok
                and trend_ok
                and not conditions["rsi_ob"].iloc[i]
            ):
                df.iloc[i, df.columns.get_loc("gold_signal")] = 1

            if (
                sell_score.iloc[i] >= sell_threshold
                and volume_ok
                and volatility_ok
                and session_ok
                and trend_ok
                and not conditions["rsi_os"].iloc[i]
            ):
                df.iloc[i, df.columns.get_loc("gold_signal")] = -1

        df["gold_position"] = 0
        last_position = 0
        consecutive_signals = 0
        min_consecutive = 3

        for i in range(1, len(df)):
            current_signal = df["gold_signal"].iloc[i]
            if current_signal != 0 and current_signal != last_position:
                if consecutive_signals >= min_consecutive - 1:
                    df.iloc[i, df.columns.get_loc("gold_position")] = current_signal
                    last_position = current_signal
                    consecutive_signals = 0
                else:
                    consecutive_signals += 1
            else:
                df.iloc[i, df.columns.get_loc("gold_position")] = last_position
                consecutive_signals = 0

        buy_signals = len(df[df["gold_signal"] == 1])
        sell_signals = len(df[df["gold_signal"] == -1])
        buy_positions = len(df[df["gold_position"] == 1])
        sell_positions = len(df[df["gold_position"] == -1])

        print(f"📊 Signals detected: {buy_signals} buys, {sell_signals} sells")
        print(f"📈 Actual positions: {buy_positions} buys, {sell_positions} sells")
        print(f"🎯 Average buy score: {buy_score.mean():.2f}")
        print(f"🎯 Average sell score: {sell_score.mean():.2f}")
        print(
            f"🕒 Time filter used: {'✅' if any(conditions['good_session']) else '❌'}"
        )

        return df

    def get_strategy_score(self, df):
        """Get score for integration"""
        df_with_signals = self.generate_signals(df)
        if len(df_with_signals) > 0:
            last_signal = df_with_signals["gold_signal"].iloc[-1]
            last_position = df_with_signals["gold_position"].iloc[-1]
            if last_position != 0:
                return last_position * 1.0
            elif last_signal != 0:
                return last_signal * 0.7
            else:
                return 0.0
        else:
            return 0.0
