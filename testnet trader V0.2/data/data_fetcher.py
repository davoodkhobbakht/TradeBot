# -*- coding: utf-8 -*-
# data/data_fetcher.py

import ccxt
import pandas as pd
import time
import sys
import os

# اضافه کردن مسیر پوشه utils به sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.indicators import calculate_indicators


def fetch_ohlcv(symbol, timeframe, since, limit=1000):
    """دریافت داده‌های تاریخی از صرافی"""
    exchange = ccxt.binance()
    data = []
    retries = 0
    max_retries = 5

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if len(ohlcv) == 0:
                break
            since = ohlcv[-1][0] + 1
            data.extend(ohlcv)
            print(f"تعداد داده‌های دریافت شده برای {symbol}: {len(data)}")
            if len(ohlcv) < limit:
                break
            time.sleep(0.5)
        except Exception as e:
            print(f"خطا در دریافت داده {symbol}: {e}")
            retries += 1
            if retries >= max_retries:
                break
            time.sleep(3 * retries)
    return data


def fetch_historical_data(symbol, timeframe="1d", start_date="2023-01-01T00:00:00Z"):
    """دریافت داده‌های تاریخی کامل برای یک نماد"""
    exchange = ccxt.binance()
    since = exchange.parse8601(start_date)

    print(f"📥 دریافت داده‌های تاریخی برای {symbol}...")
    ohlcv = fetch_ohlcv(symbol, timeframe, since)

    if not ohlcv or len(ohlcv) < 300:
        print(f"❌ داده‌های ناکافی برای {symbol}")
        return None

    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df.name = symbol
    df = calculate_indicators(df)

    print(f"✅ داده‌های {symbol} آماده شد ({len(ohlcv)} کندل)")
    return df


def fetch_multiple_symbols_data(
    symbols, timeframe="1d", start_date="2023-01-01T00:00:00Z"
):
    """دریافت داده‌های چندین نماد"""
    symbols_data = {}

    for symbol in symbols:
        df = fetch_historical_data(symbol, timeframe, start_date)
        if df is not None:
            symbols_data[symbol] = df

    return symbols_data
