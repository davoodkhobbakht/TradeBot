# -*- coding: utf-8 -*-
# strategies/multi_strategy.py

import pandas as pd
import numpy as np
from strategies.gold_strategy import GoldStrategy  # Import optimized GoldStrategy


class MultiStrategyEngine:
    """Multi-strategy trading engine with GoldStrategy integration"""

    def __init__(self):
        self.strategies = {
            "trend_following": self.trend_following_strategy,
            "mean_reversion": self.mean_reversion_strategy,
            "breakout": self.breakout_strategy,
            "momentum": self.momentum_strategy,
            "volatility": self.volatility_strategy,
            "gold": self.gold_strategy,  # New: Add GoldStrategy
        }
        self.strategy_weights = {
            "trend_following": 0.25,
            "mean_reversion": 0.15,
            "breakout": 0.20,
            "momentum": 0.15,
            "volatility": 0.15,
            "gold": 0.25,  # Balanced weight for Gold
        }

    def trend_following_strategy(self, df):
        """Trend-following strategy"""
        score = 0
        if df["SMA_20"].iloc[-1] > df["SMA_50"].iloc[-1] and df["ADX"].iloc[-1] > 14:
            score += 2
        else:
            score -= 2
        if df["SMA_50"].iloc[-1] > df["SMA_200"].iloc[-1]:
            score += 1
        else:
            score -= 1
        if df["ADX"].iloc[-1] > 25:
            score += 1
        elif df["ADX"].iloc[-1] < 10:
            score -= 1
        return score / 4.0

    def mean_reversion_strategy(self, df):
        """Mean reversion strategy"""
        current_price = df["close"].iloc[-1]
        bb_middle = df["BB_Middle"].iloc[-1]
        bb_upper = df["BB_Upper"].iloc[-1]
        bb_lower = df["BB_Lower"].iloc[-1]
        rsi = df["RSI"].iloc[-1]
        score = 0
        if current_price < bb_lower:
            score += 2
        elif current_price > bb_upper:
            score -= 2
        else:
            distance_from_mean = (current_price - bb_middle) / (bb_upper - bb_lower)
            score -= distance_from_mean * 2
        if rsi < 30:
            score += 1
        elif rsi > 70:
            score -= 1
        return score / 3.0

    def breakout_strategy(self, df):
        """Breakout strategy"""
        score = 0
        current_close = df["close"].iloc[-1]
        current_volume = df["volume"].iloc[-1]
        avg_volume = df["volume"].rolling(20).mean().iloc[-1]
        resistance = df["high"].rolling(20).max().iloc[-2]
        support = df["low"].rolling(20).min().iloc[-2]
        if current_close > resistance and current_volume > avg_volume * 1.2:
            score += 3
        if current_close < support and current_volume > avg_volume * 1.2:
            score -= 3
        if current_close > df["BB_Upper"].iloc[-1]:
            score += 1
        elif current_close < df["BB_Lower"].iloc[-1]:
            score -= 1
        return score / 4.0

    def momentum_strategy(self, df):
        """Momentum strategy"""
        score = 0
        momentum_5 = (df["close"].iloc[-1] / df["close"].iloc[-6] - 1) * 100
        momentum_10 = (df["close"].iloc[-1] / df["close"].iloc[-11] - 1) * 100
        if momentum_5 > 2:
            score += 1
        elif momentum_5 < -2:
            score -= 1
        if momentum_10 > 5:
            score += 1
        elif momentum_10 < -5:
            score -= 1
        if df["MACD"].iloc[-1] > df["Signal_Line"].iloc[-1]:
            score += 1
        else:
            score -= 1
        if df["stoch_k"].iloc[-1] > df["stoch_d"].iloc[-1]:
            score += 0.5
        else:
            score -= 0.5
        return score / 3.5

    def volatility_strategy(self, df):
        """Volatility-based strategy"""
        current_volatility = df["BB_Bandwidth"].iloc[-1]
        avg_volatility = df["BB_Bandwidth"].rolling(50).mean().iloc[-1]
        atr_ratio = df["ATR"].iloc[-1] / df["close"].iloc[-1]
        score = 0
        if current_volatility > avg_volatility * 1.5:
            score += 1
        elif current_volatility < avg_volatility * 0.7:
            score -= 1
        if atr_ratio > 0.02:
            score += 0.5
        elif atr_ratio < 0.005:
            score -= 0.5
        return score / 1.5

    def gold_strategy(self, df):
        """Gold Strategy integration"""
        gold = GoldStrategy()
        return gold.get_strategy_score(df)

    def calculate_combined_signal(self, df, market_regime):
        """Calculate combined signal with GoldStrategy"""
        strategy_scores = {}
        total_score = 0
        for name, strategy_func in self.strategies.items():
            score = strategy_func(df)
            strategy_scores[name] = score
            total_score += score * self.strategy_weights[name]
        adjusted_weights = self.adjust_weights_for_regime(market_regime)
        final_score = sum(
            score * adjusted_weights[name] for name, score in strategy_scores.items()
        )
        if final_score > 0.25:  # Stricter threshold
            return 1, strategy_scores
        elif final_score < -0.25:
            return -1, strategy_scores
        return 0, strategy_scores

    def adjust_weights_for_regime(self, market_regime):
        """Adjust strategy weights based on market regime"""
        base_weights = self.strategy_weights.copy()
        if market_regime == "trending_bull":
            base_weights["trend_following"] *= 1.8  # Boost for bull
            base_weights["gold"] *= 1.5  # Gold aligns with trends
            base_weights["breakout"] *= 1.2
            base_weights["mean_reversion"] *= 0.5
        elif market_regime == "trending_bear":
            base_weights["trend_following"] *= 1.3
            base_weights["gold"] *= 1.2
            base_weights["breakout"] *= 1.1
            base_weights["mean_reversion"] *= 0.7
        elif market_regime == "ranging":
            base_weights["mean_reversion"] *= 2.0
            base_weights["volatility"] *= 1.5
            base_weights["trend_following"] *= 0.3
            base_weights["gold"] *= 0.5
        elif market_regime == "high_volatility":
            base_weights["volatility"] *= 2.0
            base_weights["breakout"] *= 1.3
            base_weights["momentum"] *= 0.8
            base_weights["gold"] *= 1.0
        total = sum(base_weights.values())
        return {k: v / total for k, v in base_weights.items()}

    def detect_market_regime(self, df):
        """Detect current market regime"""
        adx = df["ADX"].iloc[-1]
        bb_bandwidth = df["BB_Bandwidth"].iloc[-1]
        avg_bandwidth = df["BB_Bandwidth"].rolling(50).mean().iloc[-1]
        trend = df["SMA_50"].iloc[-1] > df["SMA_200"].iloc[-1]
        atr_ratio = df["ATR"].iloc[-1] / df["close"].iloc[-1]
        if adx > 25 or (adx > 14 and trend):  # Adjusted for GoldStrategy
            return "trending_bull" if trend else "trending_bear"
        elif bb_bandwidth > avg_bandwidth * 1.5 or atr_ratio > 0.02:
            return "high_volatility"
        return "ranging"
