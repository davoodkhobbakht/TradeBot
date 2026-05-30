# -*- coding: utf-8 -*-
# main.py

import argparse
import time
import sys
import os
import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Dict
from data.data_fetcher import DataFetcher
from ml.base_ml import MLModelManager
from ml.simple_ml import SimpleMLTrainer
from ml.advanced_ml import enhanced_ml_training_pipeline, AdvancedMLTrainer
from ml.rl_trader import RLIntegration
from backtest.validation import validate_strategy

# اضافه کردن مسیر پوشه‌ها به sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "ml"))
sys.path.append(os.path.join(os.path.dirname(__file__), "strategies"))
sys.path.append(os.path.join(os.path.dirname(__file__), "backtest"))

from data.data_fetcher import DataFetcher
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

    fetcher = DataFetcher()
    fetcher.fetch_multiple_symbols_data(symbols)  # Fetches & stores
    symbols_data = {}
    for symbol in symbols:
        df = fetcher.get_stored_data(symbol, "1d")  # Or your default timeframe
        if not df.empty:
            symbols_data[symbol] = df

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


# تنظیم لاگینگ برای ارور هندلینگ
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_ml_models(symbols: List[str]) -> Optional[MLModelManager]:
    """آموزش کامل مدل‌های ML با هندلینگ ارور پیشرفته و تمام قابلیت‌ها"""
    logger.info("🤖 شروع آموزش پیشرفته مدل‌های یادگیری ماشین...")
    ml_manager = MLModelManager()

    try:
        # دریافت داده‌ها با retry و چند timeframe
        logger.info("📥 دریافت داده‌های تاریخی چندتایم‌فریم برای آموزش...")
        fetcher = DataFetcher()
        timeframes = ["5m", "15m", "30m", "1h", "4h"]
        for attempt in range(3):  # Retry تا 3 بار
            try:
                fetcher.fetch_multiple_symbols_data(
                    symbols, timeframes=timeframes, start_date="2020-01-01T00:00:00Z"
                )
                break
            except Exception as e:
                logger.warning(f"تلاش {attempt+1} شکست: {e}. Retry...")
                if attempt == 2:
                    raise ValueError("❌ شکست در دریافت داده‌ها پس از 3 تلاش!")

        symbols_data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                df_main = fetcher.get_stored_data(symbol, "1d")
                if len(df_main) < 200:  # حداقل داده برای آموزش
                    logger.warning(f"⚠️ داده ناکافی برای {symbol} (<200 ردیف) - رد شد")
                    continue

                # ترکیب ویژگی‌های چندتایم‌فریم با چک NaN
                df = df_main.copy()
                df_1h = fetcher.get_stored_data(symbol, "1h").resample("D").last()
                df_4h = fetcher.get_stored_data(symbol, "4h").resample("D").last()

                if not df_1h.empty and not df_4h.empty:
                    df["rsi_1h"] = df_1h["RSI"].reindex(df.index).fillna(method="ffill")
                    df["macd_4h"] = (
                        df_4h["MACD"].reindex(df.index).fillna(method="ffill")
                    )
                else:
                    logger.info(
                        f"⚠️ تایم‌فریم‌های فرعی خالی برای {symbol} - ادامه با داده اصلی"
                    )

                symbols_data[symbol] = df.dropna(how="any")  # حذف ردیف‌های NaN

            except Exception as e:
                logger.error(f"❌ ارور در پردازش داده {symbol}: {e} - رد شد")

        if not symbols_data:
            raise ValueError("❌ هیچ داده معتبری برای آموزش یافت نشد!")

        # 1. آموزش مدل‌های ساده با هندلینگ
        try:
            simple_trainer = SimpleMLTrainer()
            simple_models = simple_trainer.train_all_models(symbols_data)
            if simple_models:
                logger.info("✅ مدل‌های ساده آموزش داده شدند")
                ml_manager.models.update(simple_models)
        except Exception as e:
            logger.warning(f"⚠️ ارور در آموزش ساده: {e} - ادامه بدون مدل ساده")

        # 2. آموزش مدل‌های پیشرفته با پایپلاین و آستانه پایین‌تر
        try:
            advanced_trainer = AdvancedMLTrainer()
            # پایین آوردن آستانه کلاس‌ها به 20
            # (در advanced_ml.py, if target_counts.get(1, 0) > 20 and ... )
            # اینجا فرض می‌کنیم پچ شده - اگر نه، دستی اضافه کن
            ml_manager = enhanced_ml_training_pipeline(symbols_data, ml_manager)
            logger.info("✅ مدل‌های پیشرفته آموزش داده شدند")
        except Exception as e:
            logger.warning(f"⚠️ ارور در آموزش پیشرفته: {e} - ادامه بدون مدل پیشرفته")

        # 3. آموزش RL برای هر نماد با state_size پویا
        try:
            for symbol, df in symbols_data.items():
                try:
                    # تعداد دقیق ویژگی‌ها بعد از ترکیب چندتایم‌فریم
                    feature_cols = [
                        col
                        for col in df.columns
                        if col not in ["close", "target", "datetime", "Date"]
                    ]
                    state_size = len(feature_cols)  # دقیقاً همون چیزی که به مدل می‌ره

                    logger.info(
                        f"🎯 آموزش RL برای {symbol} با state_size={state_size} ({feature_cols})"
                    )

                    rl_integration = RLIntegration(
                        state_size=state_size
                    )  # حالا 24 یا 25 هر چی باشه درست میشه
                    rl_integration.train_rl_trader(
                        df, symbol, episodes=250
                    )  # کمی بیشتر برای یادگیری بهتر

                    # ذخیره درست RL
                    if symbol in rl_integration.rl_traders:
                        model = rl_integration.rl_traders[symbol].model
                        ml_manager.models[symbol + "_rl"] = (
                            model,
                            None,
                            feature_cols,
                            "keras",
                        )
                        logger.info(
                            f"✅ RL مدل {symbol} با state_size={state_size} ذخیره شد"
                        )

                except Exception as e:
                    logger.error(f"❌ ارور RL برای {symbol}: {e}")

            # ذخیره RL مدل‌ها
            for symbol in symbols:
                if (
                    symbol in rl_integration.rl_traders
                ):  # rl_integration محلی نیست - پچ به کلاس
                    model = rl_integration.rl_traders[symbol].model
                    ml_manager.models[symbol + "_rl"] = (model, None, [], "keras")
                    logger.info(f"✅ RL مدل {symbol} ذخیره شد")
        except Exception as e:
            logger.warning(f"⚠️ ارور کلی در RL: {e} - ادامه بدون RL")

        # 4. اعتبارسنجی کامل مدل‌ها با هندلینگ
        try:
            logger.info("🔬 اعتبارسنجی مدل‌ها...")
            validation_results = validate_strategy(symbols, ml_manager.models)
            for symbol, res in validation_results.items():
                oos = res.get("out_of_sample", {}).get("strategy_return", 0)
                wf_avg = (
                    res.get("walk_forward", [{}])[0].get("total_return", 0)
                    if res.get("walk_forward")
                    else 0
                )
                logger.info(
                    f"📊 {symbol}: OOS Return {oos:+.2f}% | WF Avg {wf_avg:+.2f}%"
                )
        except Exception as e:
            logger.warning(f"⚠️ ارور در اعتبارسنجی: {e} - ادامه بدون validation")

        # ذخیره همه مدل‌ها با فرمت جدید keras
        try:
            ml_manager.save_models()  # در base_ml.py, برای keras: model.save(f"{path}{symbol}_model.keras")
            logger.info("💾 تمام مدل‌ها ذخیره شدند")
        except Exception as e:
            logger.error(f"❌ ارور در ذخیره مدل‌ها: {e}")
            return None  # خروج اگر ذخیره شکست

        return ml_manager

    except ValueError as ve:
        logger.error(f"❌ ارور ارزشی: {ve}")
        return None
    except Exception as e:
        logger.critical(f"❌ ارور غیرمنتظره کلی: {e}")
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
    fetcher = DataFetcher()
    fetcher.fetch_multiple_symbols_data(symbols)  # Fetches & stores
    symbols_data = {}
    for symbol in symbols:
        df = fetcher.get_stored_data(symbol, "1d")  # Or your default timeframe
        if not df.empty:
            symbols_data[symbol] = df

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
