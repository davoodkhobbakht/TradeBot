# -*- coding: utf-8 -*-
# backtest/validation.py

import pandas as pd
import numpy as np
from backtest.backtester import run_backtest
from backtest.performance import analyze_performance
from config import TRADE_SETTINGS
from strategies.signal_generator import enhanced_signal_generation
import logging

# تنظیم لاگینگ برای ارور هندلینگ
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def walk_forward_validation(
    df, ml_models, symbol, window_size=252, step_size=21, initial_capital=1000.0
):
    """Walk-forward validation with rolling windows"""
    print("🔄 شروع Walk-Forward Validation...")

    results = []
    start_idx = 0

    while start_idx + window_size < len(df):
        train_end = start_idx + window_size
        test_end = min(train_end + step_size, len(df))

        # Train on window
        train_df = df.iloc[start_idx:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()
        test_df.name = symbol  # Preserve symbol name

        if len(test_df) < 50:
            break

        # Generate signals on test data (assume models are trained on train)
        test_df = enhanced_signal_generation(
            test_df, symbol, ml_models, None, verbose=False
        )

        # Run backtest on test period
        trades, equity_curve, final_capital, final_position = run_backtest(
            test_df, initial_capital
        )

        if len(trades) > 0:
            if final_position > 0:
                final_value = final_capital + (
                    final_position * test_df["close"].iloc[-1]
                )
            else:
                final_value = final_capital

            total_return, max_drawdown = analyze_performance(
                trades, equity_curve, initial_capital, final_value, test_df
            )

            results.append(
                {
                    "period_start": test_df.index[0],
                    "period_end": test_df.index[-1],
                    "total_return": total_return,
                    "max_drawdown": max_drawdown,
                    "num_trades": len([t for t in trades if t[0] == "BUY"]),
                    "final_value": final_value,
                }
            )

        start_idx += step_size

    return results


def out_of_sample_validation(
    symbol, ml_models, train_end_date="2023-01-01", initial_capital=1000.0
):
    """Out-of-sample validation"""
    from data.data_fetcher import DataFetcher

    print(f"📊 شروع Out-of-Sample Validation برای {symbol}...")

    try:
        # Fetch data up to recent

        fetcher = DataFetcher()
        df = fetcher.get_stored_data(symbol, "1d")

        if df.empty or len(df) < 200:
            print(f"⚠️ داده کافی برای {symbol} وجود ندارد")
            return None

        # Split at train_end_date
        train_df = df[df.index < train_end_date].copy()
        test_df = df[df.index >= train_end_date].copy()
        test_df.name = symbol  # Preserve symbol name

        if len(test_df) < 50:
            print(f"⚠️ داده تست کافی برای {symbol} وجود ندارد ({len(test_df)} کندل)")
            return None

        # Generate signals on test data
        test_df = enhanced_signal_generation(
            test_df, symbol, ml_models, None, verbose=True
        )

        # Run backtest
        trades, equity_curve, final_capital, final_position = run_backtest(
            test_df, initial_capital
        )

        if len(trades) == 0:
            print(f"⚠️ هیچ معامله‌ای برای {symbol} انجام نشد")
            return None

        if final_position > 0:
            final_value = final_capital + (final_position * test_df["close"].iloc[-1])
        else:
            final_value = final_capital

        total_return, max_drawdown = analyze_performance(
            trades, equity_curve, initial_capital, final_value, test_df
        )

        # Compare to buy & hold
        test_start_price = test_df["close"].iloc[0]
        test_end_price = test_df["close"].iloc[-1]
        buy_hold_return = ((test_end_price - test_start_price) / test_start_price) * 100

        return {
            "symbol": symbol,
            "strategy_return": total_return,
            "buy_hold_return": buy_hold_return,
            "outperformance": total_return - buy_hold_return,
            "max_drawdown": max_drawdown,
            "num_trades": len([t for t in trades if t[0] == "BUY"]),
            "test_period_days": (test_df.index[-1] - test_df.index[0]).days,
            "final_value": final_value,
        }
    except Exception as e:
        print(f"❌ خطا در اعتبارسنجی Out-of-Sample برای {symbol}: {e}")
        return None


def monte_carlo_simulation(
    df, ml_models, symbol, num_simulations=50, initial_capital=1000.0
):
    """Monte Carlo simulation for robustness testing"""
    print(f"🎲 شروع Monte Carlo Simulation ({num_simulations} شبیه‌سازی)...")

    returns = []
    successful_sims = 0

    for sim in range(num_simulations):
        try:
            # Bootstrap sample with replacement
            sampled_indices = np.random.choice(len(df), size=len(df), replace=True)
            sampled_df = df.iloc[sampled_indices].sort_index()
            sampled_df.name = symbol  # Preserve symbol name

            # Generate signals
            sampled_df = enhanced_signal_generation(
                sampled_df, symbol, ml_models, None, verbose=False
            )

            # Run backtest
            trades, equity_curve, final_capital, final_position = run_backtest(
                sampled_df, initial_capital
            )

            if len(trades) > 0:
                if final_position > 0:
                    final_value = final_capital + (
                        final_position * sampled_df["close"].iloc[-1]
                    )
                else:
                    final_value = final_capital

                total_return = (final_value - initial_capital) / initial_capital * 100
                returns.append(total_return)
                successful_sims += 1
        except Exception as e:
            print(f"⚠️ خطا در شبیه‌سازی Monte Carlo #{sim+1}: {e}")
            continue

    print(f"✅ شبیه‌سازی‌های موفق: {successful_sims}/{num_simulations}")

    if returns:
        return {
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "min_return": np.min(returns),
            "max_return": np.max(returns),
            "var_95": np.percentile(returns, 5),  # 95% VaR
            "sharpe_ratio": (
                np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            ),
            "successful_simulations": successful_sims,
        }
    return None


def regime_change_detection(df):
    """Detect market regime changes using volatility clustering"""
    # Simple regime detection using rolling volatility
    volatility = df["close"].pct_change().rolling(20).std()
    high_vol_threshold = volatility.quantile(0.8)

    regimes = []
    current_regime = "normal"

    for i, vol in enumerate(volatility):
        if pd.isna(vol):
            continue

        if vol > high_vol_threshold:
            regime = "high_volatility"
        else:
            regime = "normal"

        if regime != current_regime:
            regimes.append({"date": df.index[i], "regime": regime, "volatility": vol})
            current_regime = regime

    return regimes


def validate_strategy(symbols, ml_models, initial_capital=1000.0):
    """Comprehensive validation framework"""
    print("🔬 شروع Validation Framework...")

    validation_results = {}

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"📈 Validating {symbol}")
        print(f"{'='*60}")

        try:
            # Out-of-sample validation
            oos_result = out_of_sample_validation(
                symbol, ml_models, initial_capital=initial_capital
            )
            if oos_result:
                validation_results[symbol] = {"out_of_sample": oos_result}

                print("📊 Out-of-Sample Results:")
                print(f"   Strategy Return: {oos_result['strategy_return']:+.2f}%")
                print(f"   Buy & Hold Return: {oos_result['buy_hold_return']:+.2f}%")
                print(f"   Outperformance: {oos_result['outperformance']:+.2f}%")
                print(f"   Max Drawdown: {oos_result['max_drawdown']:.2f}%")
                print(f"   Number of Trades: {oos_result['num_trades']}")

            # Walk-forward validation (sample)
            from data.data_fetcher import DataFetcher

            # In functions:
            fetcher = DataFetcher()
            df = fetcher.get_stored_data(symbol, "1d")
            if df.empty:
                logger.warning(f"⚠️ No data for {symbol}")
                return None
            if not df.empty and len(df) > 500:
                wf_results = walk_forward_validation(
                    df, ml_models, symbol, initial_capital=initial_capital
                )
                if wf_results:
                    avg_return = np.mean([r["total_return"] for r in wf_results])
                    print(f"🔄 Walk-Forward Average Return: {avg_return:+.2f}%")
                    validation_results[symbol]["walk_forward"] = wf_results

            # Monte Carlo
            if not df.empty and len(df) > 100:
                mc_results = monte_carlo_simulation(
                    df,
                    ml_models,
                    symbol,
                    num_simulations=50,
                    initial_capital=initial_capital,
                )
                if mc_results:
                    print("🎲 Monte Carlo Results:")
                    print(f"   Mean Return: {mc_results['mean_return']:+.2f}%")
                    print(f"   Std Dev: {mc_results['std_return']:.2f}%")
                    print(f"   95% VaR: {mc_results['var_95']:+.2f}%")
                    print(
                        f"   Successful Simulations: {mc_results['successful_simulations']}"
                    )
                    validation_results[symbol]["monte_carlo"] = mc_results

        except Exception as e:
            print(f"❌ خطا در اعتبارسنجی {symbol}: {e}")
            continue

    return validation_results
