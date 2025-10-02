# -*- coding: utf-8 -*-
# backtest/performance.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze_performance(trades, equity_curve, initial_capital, final_value, df):
    """تحلیل جامع عملکرد استراتژی"""
    print("\n" + "=" * 80)
    print("📊 تحلیل پیشرفته عملکرد")
    print("=" * 80)

    total_return = ((final_value - initial_capital) / initial_capital) * 100
    buy_hold_return = (
        (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]
    ) * 100

    print(f"💰 سرمایه اولیه: {initial_capital:,.0f} USDT")
    print(f"💰 ارزش نهایی: {final_value:,.0f} USDT")
    print(f"📈 سود/ضرر استراتژی: {total_return:+.2f}%")
    print(f"📊 سود/ضرر Buy & Hold: {buy_hold_return:+.2f}%")

    # تحلیل معاملات
    buy_trades = [t for t in trades if t[0] == "BUY"]
    sell_trades = [t for t in trades if t[0] == "SELL"]
    total_trades = len(buy_trades)

    print(f"\n🔢 آمار معاملات:")
    print(f"   • تعداد کل معاملات: {total_trades}")

    if total_trades == 0:
        print("⚠️ هیچ معامله‌ای برای تحلیل وجود ندارد!")
        return total_return, 0

    # تحلیل سود و ضرر
    profits = []
    profits_pct = []
    durations = []

    for sell_trade in sell_trades:
        if len(sell_trade) > 5:
            profit = sell_trade[5]
            profits.append(profit)

            if len(sell_trade) > 6:
                profit_pct = sell_trade[6]
                profits_pct.append(profit_pct)

            if len(sell_trade) > 7:
                duration = sell_trade[7]
                durations.append(duration)

    winning_trades = [p for p in profits if p > 0]
    losing_trades = [p for p in profits if p < 0]

    # Win Rate
    win_rate = (len(winning_trades) / len(profits)) * 100 if profits else 0

    print(f"\n🎯 کارایی معاملاتی:")
    print(f"   • Win Rate: {win_rate:.1f}% ({len(winning_trades)}/{len(profits)})")
    print(f"   • معاملات برنده: {len(winning_trades)}")
    print(f"   • معاملات بازنده: {len(losing_trades)}")

    if winning_trades and losing_trades:
        avg_win = np.mean(winning_trades)
        avg_loss = np.mean(losing_trades)
        profit_factor = (
            abs(sum(winning_trades)) / abs(sum(losing_trades))
            if losing_trades
            else float("inf")
        )

        print(f"   • میانگین سود معاملات برنده: {avg_win:+.0f} USDT")
        print(f"   • میانگین ضرر معاملات بازنده: {avg_loss:.0f} USDT")
        print(f"   • نسبت سود به ضرر (Profit Factor): {profit_factor:.2f}")

    # تحلیل drawdown
    if equity_curve:
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdowns = (rolling_max - equity_series) / rolling_max
        max_drawdown = drawdowns.max() * 100

        print(f"\n📉 تحلیل کاهش سرمایه:")
        print(f"   • حداکثر کاهش سرمایه (Max Drawdown): {max_drawdown:.2f}%")

    return total_return, max_drawdown if "max_drawdown" in locals() else 0


def plot_results(df, equity_curve, trades, initial_capital):
    """رسم نمودارهای بهبود یافته"""
    if len(equity_curve) != len(df):
        extended_equity = [initial_capital] * len(df)
        for i in range(len(equity_curve)):
            if i < len(df):
                extended_equity[i] = equity_curve[i]
        equity_curve = extended_equity

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

    # نمودار قیمت و معاملات
    ax1.plot(df.index, df["close"], label="قیمت", linewidth=1, color="black")
    ax1.plot(df.index, df["SMA_50"], label="SMA 50", alpha=0.7, color="blue")
    ax1.plot(df.index, df["SMA_200"], label="SMA 200", alpha=0.7, color="red")

    # سیگنال‌های خرید و فروش
    buy_signals = df[df["position"] == 1]
    sell_signals = df[df["position"] == -1]

    if not buy_signals.empty:
        ax1.scatter(
            buy_signals.index,
            buy_signals["close"],
            color="green",
            marker="^",
            s=100,
            label="سیگنال خرید",
            zorder=5,
        )

    if not sell_signals.empty:
        ax1.scatter(
            sell_signals.index,
            sell_signals["close"],
            color="red",
            marker="v",
            s=100,
            label="سیگنال فروش",
            zorder=5,
        )

    ax1.set_title("نمودار قیمت و سیگنال‌های معاملاتی")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # نمودار RSI
    ax2.plot(df.index, df["RSI"], label="RSI", color="purple", linewidth=1)
    ax2.axhline(70, color="red", linestyle="--", alpha=0.5, label="اشباع خرید")
    ax2.axhline(30, color="green", linestyle="--", alpha=0.5, label="اشباع فروش")
    ax2.axhline(50, color="gray", linestyle="--", alpha=0.3)
    ax2.set_ylim(0, 100)
    ax2.set_title("اندیکاتور RSI")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # نمودار MACD
    ax3.plot(df.index, df["MACD"], label="MACD", color="blue", linewidth=1)
    ax3.plot(
        df.index, df["Signal_Line"], label="Signal Line", color="orange", linewidth=1
    )
    ax3.bar(
        df.index,
        df["MACD_Histogram"],
        label="Histogram",
        color="gray",
        alpha=0.3,
        width=1,
    )
    ax3.axhline(0, color="black", linestyle="-", alpha=0.5)
    ax3.set_title("اندیکاتور MACD")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # نمودار سرمایه
    ax4.plot(df.index, equity_curve, label="سرمایه", linewidth=2, color="green")
    ax4.axhline(initial_capital, color="red", linestyle="--", label="سرمایه اولیه")

    for trade in trades:
        if trade[0] == "BUY":
            ax4.scatter(trade[2], trade[4], color="blue", marker="^", s=80, zorder=5)
        elif trade[0] == "SELL":
            color = "green" if trade[5] > 0 else "red"
            ax4.scatter(trade[2], trade[4], color=color, marker="v", s=80, zorder=5)

    ax4.set_title("منحنی رشد سرمایه")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
