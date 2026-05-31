# trading_engine.py
# -*- coding: utf-8 -*-
"""
Pure trading logic module — importable by CLI or API.
No __main__ block, no sys.argv parsing, no print() side effects.
"""

import os
import sys
import logging
from typing import List, Optional, Dict, Callable, Any
import numpy as np
import pandas as pd

# Ensure project path is available (safe for imports from anywhere)
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import TRADE_SETTINGS
from data.data_fetcher import DataFetcher
from ml.base_ml import MLModelManager
from ml.simple_ml import SimpleMLTrainer
from ml.advanced_ml import enhanced_ml_training_pipeline, AdvancedMLTrainer
from ml.rl_trader import RLIntegration
from backtest.validation import validate_strategy

# Default logger (can be overridden per call)
logger = logging.getLogger(__name__)


def simple_backtest(
    symbols: List[str],
    initial_capital: float = 1000.0,
    logger_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Pure simple backtest — returns data, no side effects.
    
    Args:
        symbols: List of trading symbols
        initial_capital: Starting capital per portfolio
        logger_callback: Optional function(log_message) for progress updates
    
    Returns:
        Dict with backtest results per symbol
    """
    def _log(msg: str):
        if logger_callback:
            logger_callback(msg)
        else:
            logger.info(msg)

    _log("🎯 Starting simple backtest...")
    results = {}
    capital_per_symbol = initial_capital / len(symbols)

    # Fetch data
    _log("📥 Fetching historical data...")
    fetcher = DataFetcher()
    fetcher.fetch_multiple_symbols_data(symbols)
    
    symbols_data = {}
    for symbol in symbols:
        df = fetcher.get_stored_data(symbol, "1d")
        if not df.empty:
            symbols_data[symbol] = df

    if not symbols_data:
        _log("❌ No data received!")
        return results

    for symbol, df in symbols_data.items():
        _log(f"\n{'='*50}\n📊 Analyzing: {symbol}\n{'='*50}")
        _log(f"📅 Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        _log(f"📈 Current Price: {df['close'].iloc[-1]:.2f}")
        _log(f"📊 RSI: {df['RSI'].iloc[-1]:.2f} | MACD: {df['MACD'].iloc[-1]:.4f} | ADX: {df['ADX'].iloc[-1]:.2f}")

        # Buy & Hold calculation
        initial_price = df["close"].iloc[0]
        final_price = df["close"].iloc[-1]
        buy_hold_return = ((final_price - initial_price) / initial_price) * 100
        _log(f"💰 Buy & Hold Return: {buy_hold_return:+.2f}%")

        results[symbol] = {
            "buy_hold_return": buy_hold_return,
            "initial_price": float(initial_price),
            "final_price": float(final_price),
            "data_points": len(df),
            "symbol": symbol,
        }

    _log("✅ Simple backtest completed")
    return results


def train_ml_models(
    symbols: List[str],
    logger_callback: Optional[Callable[[str], None]] = None,
) -> Optional[MLModelManager]:
    """
    Train ML models with robust error handling.
    
    Args:
        symbols: List of symbols to train on
        logger_callback: Optional progress logger
    
    Returns:
        MLModelManager instance with trained models, or None on failure
    """
    def _log(msg: str, level: str = "info"):
        if logger_callback:
            logger_callback(msg)
        else:
            getattr(logger, level)(msg)

    _log("🤖 Starting advanced ML model training...")
    ml_manager = MLModelManager()

    try:
        # Fetch multi-timeframe data
        _log("📥 Fetching multi-timeframe historical data...")
        fetcher = DataFetcher()
        timeframes = ["5m", "15m", "30m", "1h", "4h"]
        
        for attempt in range(3):
            try:
                fetcher.fetch_multiple_symbols_data(
                    symbols, timeframes=timeframes, start_date="2020-01-01T00:00:00Z"
                )
                break
            except Exception as e:
                _log(f"Attempt {attempt+1} failed: {e}. Retrying...", "warning")
                if attempt == 2:
                    raise ValueError("❌ Failed to fetch data after 3 attempts!")

        symbols_data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                df_main = fetcher.get_stored_data(symbol, "1d")
                if len(df_main) < 200:
                    _log(f"⚠️ Insufficient data for {symbol} (<200 rows) - skipping", "warning")
                    continue

                df = df_main.copy()
                df_1h = fetcher.get_stored_data(symbol, "1h").resample("D").last()
                df_4h = fetcher.get_stored_data(symbol, "4h").resample("D").last()

                if not df_1h.empty and not df_4h.empty:
                    df["rsi_1h"] = df_1h["RSI"].reindex(df.index).ffill()
                    df["macd_4h"] = df_4h["MACD"].reindex(df.index).ffill()
                else:
                    _log(f"⚠️ Secondary timeframes empty for {symbol} - using main data only", "info")

                symbols_data[symbol] = df.dropna(how="any")
            except Exception as e:
                _log(f"❌ Error processing {symbol}: {e} - skipping", "error")

        if not symbols_data:
            raise ValueError("❌ No valid data found for training!")

        # 1. Train simple models
        try:
            simple_trainer = SimpleMLTrainer()
            simple_models = simple_trainer.train_all_models(symbols_data)
            if simple_models:
                _log("✅ Simple models trained successfully")
                ml_manager.models.update(simple_models)
        except Exception as e:
            _log(f"⚠️ Error in simple model training: {e} - continuing", "warning")

        # 2. Train advanced models
        try:
            advanced_trainer = AdvancedMLTrainer()
            ml_manager = enhanced_ml_training_pipeline(symbols_data, ml_manager)
            _log("✅ Advanced models trained successfully")
        except Exception as e:
            _log(f"⚠️ Error in advanced training: {e} - continuing", "warning")

        # 3. Train RL models
        try:
            rl_integration = RLIntegration()  # Will create traders per symbol internally
            for symbol, df in symbols_data.items():
                try:
                    feature_cols = [col for col in df.columns if col not in ["close", "target", "datetime", "Date"]]
                    state_size = len(feature_cols)
                    _log(f"🎯 Training RL for {symbol} with state_size={state_size}")

                    rl_integration.train_rl_trader(df, symbol, episodes=250)

                    if symbol in rl_integration.rl_traders:
                        model = rl_integration.rl_traders[symbol].model
                        ml_manager.models[symbol + "_rl"] = (model, None, feature_cols, "keras")
                        _log(f"✅ RL model for {symbol} saved with state_size={state_size}")
                except Exception as e:
                    _log(f"❌ RL error for {symbol}: {e}", "error")
        except Exception as e:
            _log(f"⚠️ General RL error: {e} - continuing without RL", "warning")

        # 4. Validate models
        try:
            _log("🔬 Validating models...")
            validation_results = validate_strategy(symbols, ml_manager.models)
            for symbol, res in validation_results.items():
                oos = res.get("out_of_sample", {}).get("strategy_return", 0)
                wf_avg = res.get("walk_forward", [{}])[0].get("total_return", 0) if res.get("walk_forward") else 0
                _log(f"📊 {symbol}: OOS {oos:+.2f}% | WF Avg {wf_avg:+.2f}%")
        except Exception as e:
            _log(f"⚠️ Validation error: {e} - continuing", "warning")

        # Save models
        try:
            ml_manager.save_models()
            _log("💾 All models saved successfully")
        except Exception as e:
            _log(f"❌ Error saving models: {e}", "error")
            return None

        return ml_manager

    except ValueError as ve:
        _log(f"❌ Value error: {ve}", "error")
        return None
    except Exception as e:
        _log(f"❌ Unexpected error: {e}", "critical")
        return None


def enhanced_backtest(
    symbols: List[str],
    ml_models: Optional[Dict] = None,
    initial_capital: float = 1000.0,
    logger_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Advanced backtest with ML signal generation.
    
    Args:
        symbols: Trading symbols
        ml_models: Pre-trained ML models dict
        initial_capital: Starting capital
        logger_callback: Optional progress logger
    
    Returns:
        Dict with enhanced backtest results per symbol
    """
    def _log(msg: str):
        if logger_callback:
            logger_callback(msg)
        else:
            logger.info(msg)

    from strategies.signal_generator import enhanced_signal_generation
    from backtest.backtester import run_backtest
    from backtest.performance import analyze_performance

    _log("🎯 Starting enhanced backtest with ML...")
    results = {}
    capital_per_symbol = initial_capital / len(symbols)

    # Fetch data
    _log("📥 Fetching historical data...")
    fetcher = DataFetcher()
    fetcher.fetch_multiple_symbols_data(symbols)
    
    symbols_data = {}
    for symbol in symbols:
        df = fetcher.get_stored_data(symbol, "1d")
        if not df.empty:
            symbols_data[symbol] = df

    if not symbols_data:
        _log("❌ No data received!")
        return results

    for symbol, df in symbols_data.items():
        _log(f"\n{'='*60}\n🚀 Advanced analysis: {symbol}\n{'='*60}")

        # Generate signals
        _log("🔍 Generating advanced signals...")
        df = enhanced_signal_generation(df, symbol, ml_models, None)

        # Run backtest
        _log("⚡ Running backtest...")
        trades, equity_curve, final_capital, final_position = run_backtest(df, capital_per_symbol)

        # Analyze
        if len(trades) > 0:
            if final_position > 0 and len(df) > 0:
                final_value = final_capital + (final_position * df["close"].iloc[-1])
            else:
                final_value = final_capital

            total_return, max_drawdown = analyze_performance(
                trades, equity_curve, capital_per_symbol, final_value, df
            )

            results[symbol] = {
                "total_return": float(total_return),
                "max_drawdown": float(max_drawdown),
                "trades": trades,
                "final_value": float(final_value),
                "num_trades": len([t for t in trades if t[0] == "BUY"]),
                "equity_curve": equity_curve.tolist() if hasattr(equity_curve, 'tolist') else equity_curve,
                "symbol": symbol,
            }
            _log(f"✅ {symbol}: Return {total_return:+.2f}% | Trades: {len(trades)}")
        else:
            _log(f"⚠️ No trades executed for {symbol}")

    _log("✅ Enhanced backtest completed")
    return results


def validate_models(
    symbols: List[str],
    ml_models: Dict,
    initial_capital: float = 1000.0,
    logger_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Validate models with walk-forward, OOS, Monte Carlo.
    
    Returns:
        Validation results dict
    """
    def _log(msg: str):
        if logger_callback:
            logger_callback(msg)
        else:
            logger.info(msg)

    _log("🔬 Starting model validation...")
    results = validate_strategy(symbols, ml_models, initial_capital)
    _log("✅ Validation completed")
    return results


def format_validation_report(results: Dict[str, Any]) -> str:
    """
    Format validation results as human-readable string.
    Pure function — no side effects.
    """
    if not results:
        return "❌ No validation results available!"

    lines = [f"\n🎯 Model Validation Report:\n{'='*80}"]
    
    for symbol, validations in results.items():
        lines.append(f"\n📊 {symbol}:")
        
        if "out_of_sample" in validations:
            oos = validations["out_of_sample"]
            lines.append(f"  📈 Out-of-Sample:")
            lines.append(f"     Strategy: {oos['strategy_return']:+.2f}% | Buy&Hold: {oos['buy_hold_return']:+.2f}%")
            lines.append(f"     Outperformance: {oos['outperformance']:+.2f}% | Max DD: {oos['max_drawdown']:.2f}%")

        if "walk_forward" in validations:
            wf_results = validations["walk_forward"]
            avg_return = np.mean([r["total_return"] for r in wf_results])
            lines.append(f"  🔄 Walk-Forward: Average Return {avg_return:+.2f}%")

        if "monte_carlo" in validations:
            mc = validations["monte_carlo"]
            lines.append(f"  🎲 Monte Carlo:")
            lines.append(f"     Mean: {mc['mean_return']:+.2f}% | Std: {mc['std_return']:.2f}% | VaR 95%: {mc['var_95']:+.2f}%")
            if "successful_simulations" in mc:
                lines.append(f"     Successful Simulations: {mc['successful_simulations']}")
    
    return "\n".join(lines)