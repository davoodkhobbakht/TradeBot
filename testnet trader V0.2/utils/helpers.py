# -*- coding: utf-8 -*-
# utils/helpers.py

import time
import pandas as pd
import numpy as np
from datetime import datetime


def timer_decorator(func):
    """دکوراتور برای زمان‌سنجی اجرای توابع"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱ زمان اجرای {func.__name__}: {end_time - start_time:.2f} ثانیه")
        return result

    return wrapper


def debug_signals(df):
    """بررسی و نمایش اطلاعات سیگنال‌ها"""
    print("\n" + "=" * 60)
    print("🔍 دیباگ سیگنال‌ها")
    print("=" * 60)

    print(f"تعداد کل داده‌ها: {len(df)}")
    print(f"سیگنال‌های خرید (signal=1): {len(df[df['signal'] == 1])}")
    print(f"سیگنال‌های فروش (signal=-1): {len(df[df['signal'] == -1])}")

    if "position" in df.columns:
        print(f"موقعیت‌های خرید (position=1): {len(df[df['position'] == 1])}")
        print(f"موقعیت‌های فروش (position=-1): {len(df[df['position'] == -1])}")

    if len(df[df.get("position", 0) != 0]) > 0:
        print("\n📅 نقاط معاملاتی شناسایی شده:")
        trade_points = df[df["position"] != 0]
        for i, (idx, row) in enumerate(trade_points.iterrows()):
            action = "خرید" if row["position"] == 1 else "فروش"
            print(
                f"  {i+1}. {idx.strftime('%Y-%m-%d')}: {action} - قیمت: {row['close']:.0f}"
            )
    else:
        print("\n⚠️ هیچ نقطه معاملاتی شناسایی نشد!")
