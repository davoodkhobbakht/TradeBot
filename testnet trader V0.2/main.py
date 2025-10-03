# -*- coding: utf-8 -*-
# main.py

import argparse
import time
import sys
import os
import numpy as np

# اضافه کردن مسیر پوشه‌ها به sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "ml"))
sys.path.append(os.path.join(os.path.dirname(__file__), "strategies"))
sys.path.append(os.path.join(os.path.dirname(__file__), "backtest"))

from data.data_fetcher import fetch_multiple_symbols_data
from ml.base_ml import MLModelManager
from ml.advanced_ml import enhanced_ml_training_pipeline
from config import TRADE_SETTINGS


def simple_backtest(symbols, initial_capital=1000.0):
    """یک بک‌تست ساده برای تست سیستم"""
    print("🎯 شروع بک‌تست ساده برای تست...")

    results = {}
    capital_per_symbol = initial_capital / len(symbols)

    # دریافت داده‌ها
    print("📥 دریافت داده‌های تاریخی...")
    symbols_data = fetch_multiple_symbols_data(symbols)

    if not symbols_data:
        print("❌ هیچ داده‌ای دریافت نشد!")
        return results

    for symbol, df in symbols_data.items():
        print(f"\n{'='*50}")
        print(f"📊 تحلیل نماد: {symbol}")
        print(f"{'='*50}")

        # نمایش اطلاعات پایه
        print(
            f"📅 دوره داده: {df.index[0].strftime('%Y-%m-%d')} تا {df.index[-1].strftime('%Y-%m-%d')}"
        )
        print(f"📈 قیمت فعلی: {df['close'].iloc[-1]:.2f}")
        print(f"📊 RSI: {df['RSI'].iloc[-1]:.2f}")
        print(f"📊 MACD: {df['MACD'].iloc[-1]:.4f}")
        print(f"📊 ADX: {df['ADX'].iloc[-1]:.2f}")

        # محاسبه سود Buy & Hold
        initial_price = df["close"].iloc[0]
        final_price = df["close"].iloc[-1]
        buy_hold_return = ((final_price - initial_price) / initial_price) * 100

        print(f"💰 سود Buy & Hold: {buy_hold_return:+.2f}%")

        results[symbol] = {
            "buy_hold_return": buy_hold_return,
            "initial_price": initial_price,
            "final_price": final_price,
            "data_points": len(df),
        }

    return results


def train_ml_models(symbols):
    """آموزش مدل‌های ML"""
    print("🤖 شروع آموزش مدل‌های یادگیری ماشین...")

    # دریافت داده‌ها
    print("📥 دریافت داده‌های تاریخی برای آموزش...")
    symbols_data = fetch_multiple_symbols_data(
        symbols, start_date="2020-01-01T00:00:00Z"
    )

    if not symbols_data:
        print("❌ هیچ داده‌ای برای آموزش دریافت نشد!")
        return None

    # ابتدا مدل ساده رو امتحان کن
    from ml.simple_ml import SimpleMLTrainer

    simple_trainer = SimpleMLTrainer()
    simple_models = simple_trainer.train_all_models(symbols_data)

    if simple_models:
        print("✅ مدل‌های ساده با موفقیت آموزش داده شدند")
        # ایجاد مدیر مدل و اضافه کردن مدل‌های ساده
        ml_manager = MLModelManager()
        ml_manager.models = simple_models
        ml_manager.save_models()
        return ml_manager
    else:
        print("❌ مدل‌های ساده هم آموزش داده نشدند!")
        return None


def enhanced_backtest(symbols, ml_models=None, initial_capital=1000.0):
    """بک‌تست پیشرفته با ML"""
    print("🎯 شروع بک‌تست پیشرفته با ML...")

    from strategies.signal_generator import enhanced_signal_generation
    from backtest.backtester import run_backtest
    from backtest.performance import analyze_performance

    results = {}
    capital_per_symbol = initial_capital / len(symbols)

    # دریافت داده‌ها
    print("📥 دریافت داده‌های تاریخی...")
    symbols_data = fetch_multiple_symbols_data(symbols)

    if not symbols_data:
        print("❌ هیچ داده‌ای دریافت نشد!")
        return results

    for symbol, df in symbols_data.items():
        print(f"\n{'='*60}")
        print(f"🚀 تحلیل پیشرفته نماد: {symbol}")
        print(f"{'='*60}")

        # تولید سیگنال پیشرفته
        print("🔍 تولید سیگنال‌های پیشرفته...")
        df = enhanced_signal_generation(df, symbol, ml_models, None)

        # اجرای بک‌تست
        print("⚡ اجرای بک‌تست...")
        trades, equity_curve, final_capital, final_position = run_backtest(
            df, capital_per_symbol
        )

        # تحلیل نتایج
        if len(trades) > 0:
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
                "final_value": final_value,
                "num_trades": len([t for t in trades if t[0] == "BUY"]),
            }
        else:
            print(f"⚠️ هیچ معامله‌ای برای {symbol} انجام نشد")

    return results


def validate_models(symbols, ml_models, initial_capital=1000.0):
    """اعتبارسنجی مدل‌ها با روش‌های پیشرفته"""
    print("🔬 شروع اعتبارسنجی مدل‌ها...")

    from backtest.validation import validate_strategy

    results = validate_strategy(symbols, ml_models, initial_capital)
    print_validation_results(results)
    return results


def print_validation_results(results):
    """چاپ نتایج اعتبارسنجی"""
    if not results:
        print("❌ هیچ نتیجه اعتبارسنجی وجود ندارد!")
        return

    print(f"\n🎯 نتایج اعتبارسنجی مدل‌ها:")
    print(f"{'='*80}")

    for symbol, validations in results.items():
        print(f"\n📊 {symbol}:")

        if "out_of_sample" in validations:
            oos = validations["out_of_sample"]
            print(f"  📈 Out-of-Sample:")
            print(
                f"     Strategy: {oos['strategy_return']:+.2f}% | Buy&Hold: {oos['buy_hold_return']:+.2f}%"
            )
            print(
                f"     Outperformance: {oos['outperformance']:+.2f}% | Max DD: {oos['max_drawdown']:.2f}%"
            )

        if "walk_forward" in validations:
            wf_results = validations["walk_forward"]
            avg_return = np.mean([r["total_return"] for r in wf_results])
            print(f"  🔄 Walk-Forward: Average Return {avg_return:+.2f}%")

        if "monte_carlo" in validations:
            mc = validations["monte_carlo"]
            print(f"  🎲 Monte Carlo:")
            print(
                f"     Mean: {mc['mean_return']:+.2f}% | Std: {mc['std_return']:.2f}% | VaR 95%: {mc['var_95']:+.2f}%"
            )
            if "successful_simulations" in mc:
                print(f"     Successful Simulations: {mc['successful_simulations']}")


def main():
    """تابع اصلی اجرای برنامه"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple", action="store_true", help="Run simple test first")
    parser.add_argument("--train", action="store_true", help="Train ML models")
    parser.add_argument(
        "--enhanced", action="store_true", help="Run enhanced backtest with ML"
    )
    parser.add_argument("--live", action="store_true", help="Run in live testnet mode")
    parser.add_argument(
        "--validate", action="store_true", help="Run comprehensive validation"
    )
    parser.add_argument("--api_key", type=str, help="Binance API Key for live")
    parser.add_argument("--api_secret", type=str, help="Binance API Secret for live")
    args = parser.parse_args()

    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    initial_capital = TRADE_SETTINGS["initial_capital"]

    print("🚀 در حال راه‌اندازی سیستم معاملاتی...")

    if args.simple:
        # اجرای تست ساده
        results = simple_backtest(symbols, initial_capital)
        print_results(results)
        return

    if args.train:
        # آموزش مدل‌های ML
        ml_manager = train_ml_models(symbols)
        return

    if args.enhanced:
        # بارگذاری مدل‌های ML
        ml_manager = MLModelManager()
        models_loaded = ml_manager.load_models()

        if not models_loaded:
            print("❌ مدل‌های ML یافت نشد. ابتدا مدل‌ها را آموزش دهید:")
            print("   python main.py --train")
            return

        # اجرای بک‌تست پیشرفته
        results = enhanced_backtest(symbols, ml_manager.models, initial_capital)
        print_enhanced_results(results, initial_capital)
        return

    if args.validate:
        # بارگذاری مدل‌های ML
        ml_manager = MLModelManager()
        models_loaded = ml_manager.load_models()

        if not models_loaded:
            print("❌ مدل‌های ML یافت نشد. ابتدا مدل‌ها را آموزش دهید:")
            print("   python main.py --train")
            return

        # اجرای اعتبارسنجی
        validation_results = validate_models(
            symbols, ml_manager.models, initial_capital
        )
        return

    # مدیریت مدل‌های ML
    ml_manager = MLModelManager()

    # سعی در بارگذاری مدل‌های موجود
    try:
        models_loaded = ml_manager.load_models()
        if models_loaded:
            print(f"✅ مدل‌های ML بارگذاری شدند ({len(ml_manager.models)} مدل)")
        else:
            print("📚 مدل‌های ذخیره شده یافت نشد")
    except Exception as e:
        print(f"⚠️ خطا در بارگذاری مدل‌ها: {e}")

    if args.live:
        if not args.api_key or not args.api_secret:
            print("❌ API Key و Secret لازم است برای حالت لایو.")
        else:
            print("🟢 حالت لایو (Testnet) فعال شد.")
    else:
        print("\n🎯 دستورات موجود:")
        print("   python main.py --simple        # تست ساده سیستم")
        print("   python main.py --train         # آموزش مدل‌های ML")
        print("   python main.py --enhanced      # بک‌تست پیشرفته با ML")
        print("   python main.py --validate      # اعتبارسنجی مدل‌ها")
        print("   python main.py --live          # حالت لایو (نیاز به API)")


def print_results(results):
    """چاپ نتایج تست ساده"""
    if results:
        print(f"\n🎉 نتایج تست ساده:")
        print(f"{'='*60}")
        for symbol, res in results.items():
            icon = "🟢" if res["buy_hold_return"] > 0 else "🔴"
            print(f"{icon} {symbol}:")
            print(f"   📅 داده‌ها: {res['data_points']} کندل")
            print(f"   💰 قیمت اولیه: {res['initial_price']:.2f}")
            print(f"   💰 قیمت نهایی: {res['final_price']:.2f}")
            print(f"   📈 سود Buy & Hold: {res['buy_hold_return']:+.2f}%")
            print()


def print_enhanced_results(results, initial_capital):
    """چاپ نتایج بک‌تست پیشرفته"""
    if results:
        total_final_value = sum(res["final_value"] for res in results.values())
        portfolio_return = (total_final_value - initial_capital) / initial_capital * 100
        total_trades = sum(res["num_trades"] for res in results.values())

        print(f"\n🎉 نتایج بک‌تست پیشرفته:")
        print(f"{'='*70}")
        print(f"💰 سرمایه اولیه: {initial_capital:,.0f} USDT")
        print(f"💰 سرمایه نهایی: {total_final_value:,.0f} USDT")
        print(f"📈 بازدهی پرتفوی: {portfolio_return:+.2f}%")
        print(f"🔢 کل معاملات: {total_trades}")

        # رتبه‌بندی نمادها
        sorted_results = sorted(
            results.items(), key=lambda x: x[1]["total_return"], reverse=True
        )

        print(f"\n🏆 رتبه‌بندی نمادها:")
        for i, (symbol, res) in enumerate(sorted_results, 1):
            icon = "🟢" if res["total_return"] > 0 else "🔴"
            print(
                f"{i}. {icon} {symbol}: {res['total_return']:+.1f}% "
                f"(معاملات: {res['num_trades']}, Max DD: {res['max_drawdown']:.1f}%)"
            )
    else:
        print("❌ هیچ نتیجه‌ای برای نمایش وجود ندارد!")


if __name__ == "__main__":
    main()
