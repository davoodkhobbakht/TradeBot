# -*- coding: utf-8 -*-
# strategies/signal_generator.py

import pandas as pd
import numpy as np
from strategies.multi_strategy import MultiStrategyEngine
from strategies.gold_strategy import GoldStrategy  # New: Import optimized GoldStrategy
from data.data_processor import extract_advanced_features
from ml.base_ml import predict_with_ml
from strategies.base_strategy import generate_signals
from config import TRADE_SETTINGS

# Add multi-timeframe if available
try:
    from utils.multi_timeframe import confirm_signal_with_timeframes

    MULTI_TIMEFRAME_AVAILABLE = True
except ImportError:
    print("⚠️ Multi-timeframe analysis unavailable")
    MULTI_TIMEFRAME_AVAILABLE = False


def enhanced_signal_generation(
    df, symbol="BTC/USDT", ml_models=None, rl_integration=None, verbose=True
):
    """Optimized signal generation with GoldStrategy integration"""
    df = df.copy()

    display_symbol = str(symbol).split("/")[0] if "/" in str(symbol) else str(symbol)

    if verbose:
        print(f"\n🔍 Generating signals for {display_symbol}...")
    from utils.indicators import calculate_indicators
    df = calculate_indicators(df)

    # Base signals first
    df = generate_signals(df)

    # Multi-strategy engine with Gold added
    strategy_engine = MultiStrategyEngine()
    strategy_engine.strategies["gold"] = strategy_engine.gold_strategy  # New: Add Gold
    strategy_engine.strategy_weights["gold"] = 0.25  # Balanced weight

    # Detect regime
    market_regime = strategy_engine.detect_market_regime(df)
    print(f"📊 Market regime for {display_symbol}: {market_regime}")

    # Combined signal (now includes gold)
    strategy_signal, strategy_scores = strategy_engine.calculate_combined_signal(
        df, market_regime
    )

    print(f"🎯 Strategy scores for {display_symbol}:")
    for strategy, score in strategy_scores.items():
        print(f"   {strategy}: {score:.3f}")

    # ML signal (optimized thresholds)
    ml_signal = 0
    ml_confidence = 0.5

    if ml_models and symbol in ml_models:
        model, scaler, features, model_type = ml_models[symbol]
        try:
            df_enhanced = extract_advanced_features(df, None, symbol)
            available_features = [f for f in features if f in df_enhanced.columns]
            if len(available_features) > len(features) * 0.7:
                ml_confidence = predict_with_ml(
                    model, scaler, features, df_enhanced, model_type
                )
                ml_signal = (
                    1 if ml_confidence > 0.65 else (-1 if ml_confidence < 0.35 else 0)
                )  # Stricter
                print(
                    f"🤖 ML signal {display_symbol}: {ml_signal} (confidence: {ml_confidence:.3f})"
                )
            else:
                print(f"⚠️ Insufficient features for ML in {display_symbol}")
        except Exception as e:
            print(f"⚠️ ML prediction error for {display_symbol}: {e}")

    # Optimized final combination (stronger bull bias, conflict fix)
    final_signal = 0
    signal_source = "None"
    bullish_bias = market_regime in ["trending_bull", "high_volatility"]
    ml_threshold = 0.65 if bullish_bias else 0.7  # Higher for precision
    strategy_threshold = 0.25 if bullish_bias else 0.35  # Stricter

    ml_strength = abs(ml_confidence) if ml_signal != 0 else 0
    strategy_strength = abs(strategy_signal)

    if (
        ml_signal != 0
        and strategy_signal != 0
        and ml_signal != np.sign(strategy_signal)
    ):
        # Conflict: Favor stronger (now biases strategy in bulls)
        if bullish_bias and strategy_strength > ml_strength:
            final_signal = np.sign(strategy_signal)
            signal_source = "Strategy (bull bias win)"
        elif ml_strength > 0.75:
            final_signal = ml_signal
            signal_source = "ML (conflict win)"
        else:
            final_signal = 0
            signal_source = "Conflict - rejected"
    elif ml_signal != 0 and ml_strength > ml_threshold:
        final_signal = ml_signal
        signal_source = "ML"
    elif strategy_signal != 0 and strategy_strength > strategy_threshold:
        final_signal = np.sign(strategy_signal)
        signal_source = "Strategy"
    else:
        if bullish_bias and strategy_strength > 0.15:  # Lower for bull entry
            final_signal = 1
            signal_source = "Strategy (bull bias)"
        else:
            final_signal = 0

    print(
        f"🎯 Initial signal {display_symbol}: {final_signal} (source: {signal_source})"
    )

    # Multi-timeframe confirmation (if available)
    if MULTI_TIMEFRAME_AVAILABLE and final_signal != 0 and symbol != "UNKNOWN":
        try:
            confirmed_signal, confirmation_message = confirm_signal_with_timeframes(
                final_signal, symbol, df
            )
            print(f"✅ Multi-timeframe: {confirmation_message}")
            if abs(confirmed_signal) < 0.9:  # Stricter
                final_signal = 0
                signal_source = "Not confirmed"
            else:
                final_signal = np.sign(confirmed_signal)
                signal_source += " + Multi-timeframe"
        except Exception as e:
            print(f"⚠️ Multi-timeframe error: {e}")

    print(f"🎯 Final signal {display_symbol}: {final_signal} (source: {signal_source})")

    # Optimized positions (min_distance=10, ATR stops)
    df["final_signal"] = final_signal
    df["new_position"] = 0
    min_distance = 10  # Increased to reduce over-trading
    last_position_change = -min_distance

    for i in range(1, len(df)):
        current_signal = df["final_signal"].iloc[i]
        prev_position = df["new_position"].iloc[i - 1]
        if i - last_position_change >= min_distance:
            if current_signal == 1 and prev_position != 1:
                df.iloc[i, df.columns.get_loc("new_position")] = 1
                last_position_change = i
            elif current_signal == -1 and prev_position != -1:
                df.iloc[i, df.columns.get_loc("new_position")] = -1
                last_position_change = i
            else:
                df.iloc[i, df.columns.get_loc("new_position")] = prev_position
        else:
            df.iloc[i, df.columns.get_loc("new_position")] = prev_position

    # ATR-based dynamic SL (3.5x)
    if "atr_14" not in df.columns:
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        df["atr_14"] = tr.rolling(14).mean()
    df["stop_loss"] = (
        df["close"] - (df["atr_14"] * 3.5)
        if final_signal > 0
        else df["close"] + (df["atr_14"] * 3.5)
    )

    df["position"] = df["new_position"]
    df.drop("new_position", axis=1, inplace=True)

    df["strategy_signal"] = strategy_signal
    df["ml_signal"] = ml_signal
    df["ml_confidence"] = ml_confidence

    # Stats
    position_changes = (df["position"] != df["position"].shift(1)).sum()
    buy_positions = len(df[df["position"] == 1])
    sell_positions = len(df[df["position"] == -1])
    print(
        f"📊 Real positions {display_symbol}: {buy_positions} buys, {sell_positions} sells"
    )
    print(f"🔄 Position changes: {position_changes}")

    df.name = symbol
    return df


# Backward-compatible func
def generate_signals_with_ml(df, ml_models=None, cross_data=None):
    return enhanced_signal_generation(df, ml_models, None)
