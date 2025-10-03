# -*- coding: utf-8 -*-
# strategies/signal_generator.py

import pandas as pd
import numpy as np
from strategies.multi_strategy import MultiStrategyEngine
from data.data_processor import extract_advanced_features
from ml.base_ml import predict_with_ml
from strategies.base_strategy import generate_signals
from config import TRADE_SETTINGS

# اضافه کردن import برای تحلیل چند تایم‌فریمی
try:
    from utils.multi_timeframe import confirm_signal_with_timeframes

    MULTI_TIMEFRAME_AVAILABLE = True
except ImportError:
    print("⚠️ تحلیل چند تایم‌فریمی در دسترس نیست")
    MULTI_TIMEFRAME_AVAILABLE = False


def enhanced_signal_generation(
    df, symbol="BTC/USDT", ml_models=None, rl_integration=None, verbose=True
):
    """تولید سیگنال پیشرفته با ترکیب تمام استراتژی‌ها"""
    df = df.copy()

    # Clean up symbol name for display
    display_symbol = str(symbol).split("/")[0] if "/" in str(symbol) else str(symbol)

    if verbose:
        print(f"\n🔍 تولید سیگنال برای {display_symbol}...")

    # اول سیگنال‌های پایه رو تولید کن تا ستون position ایجاد بشه
    df = generate_signals(df)

    # موتور استراتژی ترکیبی
    strategy_engine = MultiStrategyEngine()

    # تشخیص شرایط بازار
    market_regime = strategy_engine.detect_market_regime(df)
    print(f"📊 شرایط بازار {display_symbol}: {market_regime}")

    # سیگنال استراتژی ترکیبی
    strategy_signal, strategy_scores = strategy_engine.calculate_combined_signal(
        df, market_regime
    )

    print(f"🎯 امتیاز استراتژی‌های {display_symbol}:")
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
                    f"🤖 سیگنال ML {display_symbol}: {ml_signal} (اعتماد: {ml_confidence:.3f})"
                )
            else:
                print(f"⚠️ ویژگی‌های کافی برای ML {display_symbol} موجود نیست")

        except Exception as e:
            print(f"⚠️ خطا در پیش‌بینی ML برای {display_symbol}: {e}")

    # ترکیب نهایی سیگنال‌ها با حل تعارض - بهبود یافته برای بازار گاوی
    final_signal = 0
    signal_source = "هیچکدام"

    # تشخیص شرایط بازار برای تنظیم آستانه
    market_regime = strategy_engine.detect_market_regime(df)
    bullish_bias = market_regime in ["trending_bull", "high_volatility"]

    # امتیاز نهایی برای تصمیم‌گیری
    ml_strength = ml_confidence if ml_signal != 0 else 0
    strategy_strength = abs(strategy_signal)

    # آستانه‌های پویا بر اساس شرایط بازار
    ml_threshold = 0.55 if bullish_bias else 0.6
    strategy_threshold = 0.2 if bullish_bias else 0.3

    if (
        ml_signal != 0
        and strategy_signal != 0
        and ml_signal != np.sign(strategy_signal)
    ):
        # تعارض: مقایسه قدرت سیگنال‌ها
        if ml_strength > 0.7 and strategy_strength < 0.5:
            final_signal = ml_signal
            signal_source = "ML (برنده تعارض)"
        elif strategy_strength > 0.5 and ml_strength < 0.6:
            final_signal = 1 if strategy_signal > 0 else -1
            signal_source = "استراتژی (برنده تعارض)"
        else:
            # تعارض شدید: هیچ سیگنال
            final_signal = 0
            signal_source = "تعارض - رد سیگنال"
    elif ml_signal != 0 and ml_strength > ml_threshold:
        final_signal = ml_signal
        signal_source = "ML"
    elif strategy_signal != 0 and strategy_strength > strategy_threshold:
        # بایاس گاوی: ترجیح سیگنال خرید
        if strategy_signal > 0 or bullish_bias:
            final_signal = 1 if strategy_signal > 0 else -1
            signal_source = "استراتژی"
        else:
            final_signal = 0
    else:
        # در بازار گاوی، سیگنال خنثی را به خرید ملایم تبدیل کن
        if bullish_bias and strategy_strength > 0.1:
            final_signal = 1
            signal_source = "استراتژی (بایاس گاوی)"
        else:
            final_signal = 0
            signal_source = "هیچکدام"

    print(f"🎯 سیگنال اولیه {display_symbol}: {final_signal} (منبع: {signal_source})")

    # ==================== تحلیل چند تایم‌فریمی ====================
    if MULTI_TIMEFRAME_AVAILABLE and final_signal != 0 and symbol != "UNKNOWN":
        try:
            confirmed_signal, confirmation_message = confirm_signal_with_timeframes(
                final_signal, symbol, df
            )
            print(f"✅ تایید چند تایم‌فریمی: {confirmation_message}")

            # فقط سیگنال‌های کامل 0 یا 1 رو قبول کن
            if confirmed_signal == 0:
                final_signal = 0
                signal_source = "تایید نشد"
                print(f"❌ سیگنال {display_symbol} توسط تحلیل چند تایم‌فریمی تایید نشد")
            elif abs(confirmed_signal) < 0.8:  # آستانه سخت‌گیرانه‌تر
                final_signal = 0  # کاملاً حذف کن
                signal_source = "تایید نشد"
                print(f"❌ سیگنال {display_symbol} ضعیف تشخیص داده شد")
            else:
                # اگر قوی‌تر شد، به‌روزرسانی کن
                final_signal = 1 if confirmed_signal > 0 else -1
                signal_source += " + چندتایم‌فریمی"
        except Exception as e:
            print(f"⚠️ خطا در تحلیل چند تایم‌فریمی: {e}")
    elif symbol == "UNKNOWN":
        print("⚠️ نماد نامشخص - تحلیل چند تایم‌فریمی رد شد")
    # ==================== پایان بخش جدید ====================

    print(f"🎯 سیگنال نهایی {display_symbol}: {final_signal} (منبع: {signal_source})")

    # ایجاد position جدید با فیلترهای بهتر
    df["final_signal"] = final_signal
    df["new_position"] = 0

    # فیلترهای پیشرفته برای کاهش over-trading
    min_distance = TRADE_SETTINGS.get(
        "min_distance_between_trades", 5
    )  # حداقل فاصله بین معاملات
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

    print(
        f"📊 موقعیت‌های واقعی {display_symbol}: {buy_positions} خرید, {sell_positions} فروش"
    )
    print(f"🔄 تعداد تغییرات موقعیت: {position_changes}")

    df.name = symbol
    return df


# تابع قدیمی برای سازگاری
def generate_signals_with_ml(df, ml_models=None, cross_data=None):
    """تولید سیگنال‌ها با ترکیب تحلیل تکنیکال و ML"""
    return enhanced_signal_generation(df, ml_models, None)
