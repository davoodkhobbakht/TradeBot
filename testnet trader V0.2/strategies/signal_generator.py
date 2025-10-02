# -*- coding: utf-8 -*-
# strategies/signal_generator.py

import pandas as pd
import numpy as np
from strategies.multi_strategy import MultiStrategyEngine
from data.data_processor import extract_advanced_features
from ml.base_ml import predict_with_ml
from strategies.base_strategy import generate_signals

# اضافه کردن import برای تحلیل چند تایم‌فریمی
try:
    from utils.multi_timeframe import confirm_signal_with_timeframes

    MULTI_TIMEFRAME_AVAILABLE = False
except ImportError:
    print("⚠️ تحلیل چند تایم‌فریمی در دسترس نیست")
    MULTI_TIMEFRAME_AVAILABLE = False


def enhanced_signal_generation(df, ml_models=None, rl_integration=None):
    """تولید سیگنال پیشرفته با ترکیب تمام استراتژی‌ها"""
    df = df.copy()
    symbol = getattr(df, "name", "BTC/USDT")

    print(f"\n🔍 تولید سیگنال برای {symbol}...")

    # اول سیگنال‌های پایه رو تولید کن تا ستون position ایجاد بشه
    df = generate_signals(df)

    # موتور استراتژی ترکیبی
    strategy_engine = MultiStrategyEngine()

    # تشخیص شرایط بازار
    market_regime = strategy_engine.detect_market_regime(df)
    print(f"📊 شرایط بازار {symbol}: {market_regime}")

    # سیگنال استراتژی ترکیبی
    strategy_signal, strategy_scores = strategy_engine.calculate_combined_signal(
        df, market_regime
    )

    print(f"🎯 امتیاز استراتژی‌های {symbol}:")
    for strategy, score in strategy_scores.items():
        print(f"   {strategy}: {score:.3f}")

    # سیگنال ML (اگر موجود باشد)
    ml_signal = 0
    ml_confidence = 0.5

    if ml_models and symbol in ml_models:
        model, scaler, features, model_type = ml_models[symbol]
        try:
            # استخراج ویژگی‌ها برای ML
            df_enhanced = extract_advanced_features(df, None, symbol)

            # اطمینان از وجود ویژگی‌ها
            available_features = [f for f in features if f in df_enhanced.columns]
            if len(available_features) > len(features) * 0.7:
                ml_confidence = predict_with_ml(
                    model, scaler, features, df_enhanced, model_type
                )

                # برای مدل‌های باینری ساده
                if model_type == "sklearn":
                    ml_signal = 1 if ml_confidence > 0.6 else 0
                else:
                    ml_signal = (
                        1 if ml_confidence > 0.6 else (-1 if ml_confidence < 0.4 else 0)
                    )

                print(
                    f"🤖 سیگنال ML {symbol}: {ml_signal} (اعتماد: {ml_confidence:.3f})"
                )
            else:
                print(f"⚠️ ویژگی‌های کافی برای ML {symbol} موجود نیست")

        except Exception as e:
            print(f"⚠️ خطا در پیش‌بینی ML برای {symbol}: {e}")

    # ترکیب نهایی سیگنال‌ها
    final_signal = 0
    signal_source = "هیچکدام"

    # منطق ترکیبی بهبود یافته - فقط سیگنال‌های 0 یا 1
    if ml_signal != 0 and ml_confidence > 0.65:
        final_signal = ml_signal
        signal_source = "ML"
    elif abs(strategy_signal) > 0.3:  # آستانه متعادل
        final_signal = 1 if strategy_signal > 0.3 else -1
        signal_source = "استراتژی"
    else:
        final_signal = 0
        signal_source = "هیچکدام"

    print(f"🎯 سیگنال اولیه {symbol}: {final_signal} (منبع: {signal_source})")

    # ==================== تحلیل چند تایم‌فریمی ====================
    """if MULTI_TIMEFRAME_AVAILABLE and final_signal != 0:
        try:
            confirmed_signal, confirmation_message = confirm_signal_with_timeframes(
                final_signal, symbol, df
            )
            print(f"✅ تایید چند تایم‌فریمی: {confirmation_message}")

            # فقط سیگنال‌های کامل 0 یا 1 رو قبول کن
            if confirmed_signal == 0:
                final_signal = 0
                signal_source = "تایید نشد"
                print(f"❌ سیگنال {symbol} توسط تحلیل چند تایم‌فریمی تایید نشد")
            elif abs(confirmed_signal) < 1:  # اگر سیگنال ضعیف‌تر شده (مثلاً 0.5)
                final_signal = 0  # کاملاً حذف کن
                signal_source = "تایید نشد"
                print(f"❌ سیگنال {symbol} ضعیف تشخیص داده شد")
        except Exception as e:
            print(f"⚠️ خطا در تحلیل چند تایم‌فریمی: {e}")
    """
    # ==================== پایان بخش جدید ====================

    print(f"🎯 سیگنال نهایی {symbol}: {final_signal} (منبع: {signal_source})")

    # ایجاد position جدید با فیلترهای بهتر
    df["final_signal"] = final_signal
    df["new_position"] = 0

    # فیلترهای پیشرفته برای کاهش over-trading
    min_distance = 5  # حداقل 5 روز بین معاملات
    last_position_change = -min_distance

    for i in range(1, len(df)):
        current_signal = df["final_signal"].iloc[i]
        prev_position = df["new_position"].iloc[i - 1]

        # فقط اگر فاصله کافی از آخرین تغییر گذشته باشد
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

    # جایگزینی position قدیمی
    df["position"] = df["new_position"]
    df.drop("new_position", axis=1, inplace=True)

    # ذخیره سیگنال
    df["strategy_signal"] = strategy_signal
    df["ml_signal"] = ml_signal
    df["ml_confidence"] = ml_confidence

    # نمایش تعداد سیگنال‌های واقعی
    position_changes = (df["position"] != df["position"].shift(1)).sum()
    buy_positions = len(df[df["position"] == 1])
    sell_positions = len(df[df["position"] == -1])

    print(f"📊 موقعیت‌های واقعی {symbol}: {buy_positions} خرید, {sell_positions} فروش")
    print(f"🔄 تعداد تغییرات موقعیت: {position_changes}")

    return df


# تابع قدیمی برای سازگاری
def generate_signals_with_ml(df, ml_models=None, cross_data=None):
    """تولید سیگنال‌ها با ترکیب تحلیل تکنیکال و ML"""
    return enhanced_signal_generation(df, ml_models, None)
