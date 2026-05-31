# main.py — CLI entry point only
# -*- coding: utf-8 -*-
"""
CLI wrapper for TradeBot.
All business logic delegated to trading_engine.py.
"""

import argparse
import sys
import os
import logging

# Ensure project root in path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import pure logic
from trading_engine import (
    simple_backtest,
    train_ml_models,
    enhanced_backtest,
    validate_models,
    format_validation_report,
)
from config import TRADE_SETTINGS
from ml.base_ml import MLModelManager

# Configure CLI logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def print_simple_results(results: dict):
    """CLI-only: Print simple backtest results"""
    if results:
        print(f"\n🎉 Simple Backtest Results:\n{'='*60}")
        for symbol, res in results.items():
            icon = "🟢" if res["buy_hold_return"] > 0 else "🔴"
            print(f"{icon} {symbol}:")
            print(f"   📅 Data Points: {res['data_points']}")
            print(f"   💰 Entry: {res['initial_price']:.2f} → Exit: {res['final_price']:.2f}")
            print(f"   📈 Buy & Hold: {res['buy_hold_return']:+.2f}%\n")


def print_enhanced_results(results: dict, initial_capital: float):
    """CLI-only: Print enhanced backtest results"""
    if results:
        total_final = sum(res["final_value"] for res in results.values())
        portfolio_return = (total_final - initial_capital) / initial_capital * 100
        total_trades = sum(res["num_trades"] for res in results.values())

        print(f"\n🎉 Enhanced Backtest Results:\n{'='*70}")
        print(f"💰 Initial Capital: {initial_capital:,.0f} USDT")
        print(f"💰 Final Value: {total_final:,.0f} USDT")
        print(f"📈 Portfolio Return: {portfolio_return:+.2f}%")
        print(f"🔢 Total Trades: {total_trades}\n")

        sorted_results = sorted(results.items(), key=lambda x: x[1]["total_return"], reverse=True)
        print("🏆 Symbol Ranking:")
        for i, (symbol, res) in enumerate(sorted_results, 1):
            icon = "🟢" if res["total_return"] > 0 else "🔴"
            print(f"{i}. {icon} {symbol}: {res['total_return']:+.1f}% "
                  f"(Trades: {res['num_trades']}, Max DD: {res['max_drawdown']:.1f}%)")
    else:
        print("❌ No results to display!")


def cli_logger(msg: str):
    """Adapter: trading_engine log → CLI print"""
    print(msg)


def main():
    """CLI entry point — delegates to trading_engine"""
    parser = argparse.ArgumentParser(description="TradeBot CLI")
    parser.add_argument("--simple", action="store_true", help="Run simple backtest")
    parser.add_argument("--train", action="store_true", help="Train ML models")
    parser.add_argument("--enhanced", action="store_true", help="Run enhanced backtest with ML")
    parser.add_argument("--live", action="store_true", help="Run live testnet trading")
    parser.add_argument("--validate", action="store_true", help="Run model validation")
    parser.add_argument("--api_key", type=str, help="Binance API Key (live mode)")
    parser.add_argument("--api_secret", type=str, help="Binance API Secret (live mode)")
    args = parser.parse_args()

    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    initial_capital = TRADE_SETTINGS["initial_capital"]

    print("🚀 Starting TradeBot CLI...")

    if args.simple:
        results = simple_backtest(symbols, initial_capital, logger_callback=cli_logger)
        print_simple_results(results)
        return

    if args.train:
        ml_manager = train_ml_models(symbols, logger_callback=cli_logger)
        if ml_manager:
            print("✅ Training completed successfully")
        return

    if args.enhanced:
        ml_manager = MLModelManager()
        if not ml_manager.load_models():
            print("❌ No ML models found. Train first: python main.py --train")
            return
        results = enhanced_backtest(symbols, ml_manager.models, initial_capital, logger_callback=cli_logger)
        print_enhanced_results(results, initial_capital)
        return

    if args.validate:
        ml_manager = MLModelManager()
        if not ml_manager.load_models():
            print("❌ No ML models found. Train first: python main.py --train")
            return
        results = validate_models(symbols, ml_manager.models, initial_capital, logger_callback=cli_logger)
        print(format_validation_report(results))
        return

    # Default: show help
    print("\n🎯 Available Commands:")
    print("   python main.py --simple     # Quick system test")
    print("   python main.py --train      # Train ML models")
    print("   python main.py --enhanced   # Enhanced backtest with ML")
    print("   python main.py --validate   # Model validation")
    print("   python main.py --live       # Live testnet trading (requires API keys)")


if __name__ == "__main__":
    main()