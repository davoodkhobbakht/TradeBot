# -*- coding: utf-8 -*-
# backtest/backtester.py

import pandas as pd
import numpy as np
from config import SYMBOL_OPTIMIZED_PARAMS, TRADE_SETTINGS
from utils.indicators import dynamic_trailing_stop


def run_backtest(df, initial_capital=1000.0):
    """اجرای بک‌تست با مدیریت سرمایه"""
    capital = initial_capital
    position = 0
    trades = []
    entry_price = 0
    entry_date = None
    trailing_high = 0
    in_position = False

    equity_curve = [initial_capital] * len(df)

    # استفاده از پارامترهای بهینه‌شده برای هر نماد
    current_symbol = getattr(df, "name", "BTC/USDT")
    if current_symbol in SYMBOL_OPTIMIZED_PARAMS:
        params = SYMBOL_OPTIMIZED_PARAMS[current_symbol]
        stop_loss_pct = params["stop_loss"]
        take_profit_pct = params["take_profit"]
        position_size_percent = params["position_size"]
    else:
        stop_loss_pct = 0.08
        take_profit_pct = 0.20
        position_size_percent = 0.10

    trade_fee = TRADE_SETTINGS["trade_fee"]
    slippage = TRADE_SETTINGS["slippage"]
    max_trades = TRADE_SETTINGS["max_trades_per_symbol"]

    print(
        f"⚙️ پارامترهای معاملاتی {current_symbol}: "
        f"استاپ لاس {stop_loss_pct*100}%، "
        f"تیک پروفیت {take_profit_pct*100}%، "
        f"سایز پوزیشن {position_size_percent*100}%"
    )

    start_idx = 200
    trade_count = 0

    for i in range(start_idx, len(df)):
        current_price = df["close"].iloc[i] * (
            1 + slippage if in_position else 1 - slippage
        )
        current_date = df.index[i]
        volatility = df["BB_Bandwidth"].iloc[i] if i < len(df) else 0.02

        if in_position:
            pnl_pct = (current_price - entry_price) / entry_price
            trailing_high = max(trailing_high, current_price)

            # ترییلینگ استاپ پویا
            trailing_stop_price = dynamic_trailing_stop(
                current_price, entry_price, trailing_high, volatility
            )
            stop_loss_price = entry_price * (1 - stop_loss_pct)

            # شرایط خروج
            exit_condition = False
            exit_type = ""

            if pnl_pct >= take_profit_pct:
                exit_condition = True
                exit_type = "TAKE_PROFIT"
            elif current_price <= min(trailing_stop_price, stop_loss_price):
                exit_condition = True
                exit_type = "STOP_LOSS"
            elif df["position"].iloc[i] == -1:
                exit_condition = True
                exit_type = "STRATEGY"

            if exit_condition:
                exit_value = position * current_price * (1 - trade_fee)
                profit = exit_value - (position * entry_price)
                profit_pct = (
                    (profit / (position * entry_price)) * 100
                    if position * entry_price > 0
                    else 0
                )

                capital += exit_value
                trades.append(
                    (
                        "SELL",
                        exit_type,
                        current_date,
                        current_price,
                        exit_value,
                        profit,
                        profit_pct,
                        (current_date - entry_date).days,
                    )
                )
                position = 0
                in_position = False

                profit_icon = "🟢" if profit > 0 else "🔴"
                print(
                    f"{profit_icon} خروج {exit_type}: {current_date.strftime('%Y-%m-%d')} - "
                    f"قیمت: {current_price:.0f} - سود: {profit:+.0f} USDT ({profit_pct:+.1f}%)"
                )

        # شرایط ورود
        if (
            df["position"].iloc[i] == 1
            and not in_position
            and capital > 10
            and trade_count < max_trades
        ):

            investment_amount = capital * position_size_percent
            min_investment = initial_capital * 0.01
            max_investment = initial_capital * 0.20

            investment_amount = max(
                min_investment, min(investment_amount, max_investment)
            )

            if investment_amount > capital:
                continue

            position = investment_amount / current_price
            entry_price = current_price
            entry_date = current_date
            entry_capital = investment_amount
            capital -= investment_amount
            trailing_high = entry_price
            in_position = True
            trade_count += 1

            trades.append(
                ("BUY", "ENTRY", current_date, entry_price, entry_capital, 0, 0, 0)
            )
            print(
                f"🟢 خرید #{trade_count}: {current_date.strftime('%Y-%m-%d')} - "
                f"قیمت: {entry_price:.0f} - سرمایه: {entry_capital:.0f} USDT"
            )

        # محاسبه ارزش فعلی سبد
        if in_position:
            current_equity = capital + (position * current_price)
        else:
            current_equity = capital

        equity_curve[i] = current_equity

    # بستن پوزیشن باز در پایان دوره
    if in_position and len(df) > 0:
        current_price = df["close"].iloc[-1]
        current_date = df.index[-1]
        exit_value = position * current_price * (1 - trade_fee)
        profit = exit_value - (position * entry_price)
        profit_pct = (
            (profit / (position * entry_price)) * 100
            if position * entry_price > 0
            else 0
        )

        capital += exit_value
        trades.append(
            (
                "SELL",
                "END_OF_PERIOD",
                current_date,
                current_price,
                exit_value,
                profit,
                profit_pct,
                (current_date - entry_date).days,
            )
        )
        profit_icon = "🟢" if profit > 0 else "🔴"
        print(
            f"{profit_icon} بستن پوزیشن در پایان دوره: سود {profit:+.0f} USDT ({profit_pct:+.1f}%)"
        )

    final_value = capital
    if in_position and len(df) > 0:
        final_value = capital + (position * df["close"].iloc[-1])

    if len(trades) == 0:
        print("⚠️ هشدار: هیچ معامله‌ای انجام نشد!")
    else:
        total_profit = final_value - initial_capital
        print(
            f"📊 نتیجه نهایی: سرمایه اولیه {initial_capital:.0f} → سرمایه نهایی {final_value:.0f}"
        )
        print(
            f"📈 سود/ضرر کل: {total_profit:+.0f} USDT ({(total_profit/initial_capital*100):+.1f}%)"
        )

    return trades, equity_curve, capital, position
