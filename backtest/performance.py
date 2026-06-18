# -*- coding: utf-8 -*-
"""
Comprehensive Performance Analysis with Advanced Metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def analyze_performance(
    trades: List,
    equity_curve: np.ndarray,
    initial_capital: float,
    final_value: float,
    df: pd.DataFrame,
    metrics: Dict = None
) -> Tuple[float, float, Dict]:
    """
    Comprehensive performance analysis with advanced metrics

    Returns:
        total_return: Strategy total return percentage
        max_drawdown: Maximum drawdown percentage
        metrics: Complete metrics dictionary
    """
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Basic returns
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    buy_hold_return = ((df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]) * 100

    print(f"\n💰 Capital Summary:")
    print(f"   Initial Capital: {initial_capital:,.0f} USDT")
    print(f"   Final Value: {final_value:,.0f} USDT")
    print(f"   Strategy Return: {total_return:+.2f}%")
    print(f"   Buy & Hold Return: {buy_hold_return:+.2f}%")
    print(f"   Outperformance: {total_return - buy_hold_return:+.2f}%")

    # Trade statistics
    buy_trades = [t for t in trades if t[0] == "BUY"]
    sell_trades = [t for t in trades if t[0] == "SELL"]
    total_trades = len(buy_trades)

    print(f"\n🔢 Trade Statistics:")
    print(f"   Total Trades: {total_trades}")

    if total_trades == 0:
        print("⚠️ No trades to analyze!")
        return total_return, 0, {}

    # Profit/Loss analysis
    profits = [t[5] for t in sell_trades if len(t) > 5]
    profits_pct = [t[6] for t in sell_trades if len(t) > 6]
    durations = [t[7] for t in sell_trades if len(t) > 7]

    winning_trades = [p for p in profits if p > 0]
    losing_trades = [p for p in profits if p < 0]

    win_rate = (len(winning_trades) / len(profits)) * 100 if profits else 0

    print(f"\n🎯 Trading Efficiency:")
    print(f"   Win Rate: {win_rate:.1f}% ({len(winning_trades)}/{len(profits)})")
    print(f"   Winning Trades: {len(winning_trades)}")
    print(f"   Losing Trades: {len(losing_trades)}")

    if winning_trades and losing_trades:
        avg_win = np.mean(winning_trades)
        avg_loss = np.mean(losing_trades)
        profit_factor = abs(sum(winning_trades)) / abs(sum(losing_trades))

        print(f"   Avg Win: {avg_win:+.0f} USDT")
        print(f"   Avg Loss: {avg_loss:.0f} USDT")
        print(f"   Profit Factor: {profit_factor:.2f}")

        # Expectancy
        expectancy = (win_rate/100 * avg_win) + ((1-win_rate/100) * avg_loss)
        print(f"   Expectancy per Trade: {expectancy:+.0f} USDT")

    # Drawdown analysis
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.expanding().max()
    drawdowns = (rolling_max - equity_series) / rolling_max
    max_drawdown = drawdowns.max() * 100

    # Max drawdown duration
    in_drawdown = drawdowns > 0
    max_dd_duration = 0
    current_duration = 0

    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_dd_duration = max(max_dd_duration, current_duration)
        else:
            current_duration = 0

    print(f"\n📉 Drawdown Analysis:")
    print(f"   Max Drawdown: {max_drawdown:.2f}%")
    print(f"   Max DD Duration: {max_dd_duration} periods")

    # Risk-adjusted metrics
    daily_returns = equity_series.pct_change().dropna().values

    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        # Sharpe Ratio
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

        # Sortino Ratio
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino = np.mean(daily_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        else:
            sortino = float('inf')

        # Calmar Ratio
        years = len(df) / 252
        if years > 0 and max_drawdown > 0:
            annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100
            calmar = annualized_return / max_drawdown
        else:
            calmar = 0

        print(f"\n📈 Risk-Adjusted Metrics:")
        print(f"   Sharpe Ratio: {sharpe:.2f}")
        print(f"   Sortino Ratio: {sortino:.2f}")
        print(f"   Calmar Ratio: {calmar:.2f}")
    else:
        sharpe = sortino = calmar = 0

    # Trade duration analysis
    if durations:
        avg_duration = np.mean(durations)
        max_duration = max(durations)
        min_duration = min(durations)

        print(f"\n⏱️ Trade Duration:")
        print(f"   Average: {avg_duration:.1f} days")
        print(f"   Max: {max_duration} days")
        print(f"   Min: {min_duration} days")

    # Consecutive wins/losses
    if profits:
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for profit in profits:
            if profit > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)

        print(f"\n🔥 Streaks:")
        print(f"   Max Consecutive Wins: {max_consecutive_wins}")
        print(f"   Max Consecutive Losses: {max_consecutive_losses}")

    # Compile metrics
    metrics = {
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "outperformance": total_return - buy_hold_return,
        "num_trades": total_trades,
        "win_rate": win_rate,
        "avg_win": np.mean(winning_trades) if winning_trades else 0,
        "avg_loss": np.mean(losing_trades) if losing_trades else 0,
        "profit_factor": profit_factor if winning_trades and losing_trades else 0,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration": max_dd_duration,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "avg_trade_duration": np.mean(durations) if durations else 0,
        "final_value": final_value,
    }

    print("\n" + "=" * 80)

    return total_return, max_drawdown, metrics


def plot_results(df, equity_curve, trades, initial_capital):
    """Enhanced visualization with multiple subplots"""
    try:
        import matplotlib.pyplot as plt

        if len(equity_curve) != len(df):
            extended_equity = np.full(len(df), initial_capital)
            for i in range(min(len(equity_curve), len(df))):
                extended_equity[i] = equity_curve[i]
            equity_curve = extended_equity

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

        # Price chart with signals
        ax1.plot(df.index, df["close"], label="Price", linewidth=1, color="black")

        if "SMA_50" in df.columns:
            ax1.plot(df.index, df["SMA_50"], label="SMA 50", alpha=0.7, color="blue")
        if "SMA_200" in df.columns:
            ax1.plot(df.index, df["SMA_200"], label="SMA 200", alpha=0.7, color="red")

        if "position" in df.columns:
            buy_signals = df[df["position"] == 1]
            sell_signals = df[df["position"] == -1]

            if not buy_signals.empty:
                ax1.scatter(buy_signals.index, buy_signals["close"],
                          color="green", marker="^", s=100, label="Buy Signal", zorder=5)
            if not sell_signals.empty:
                ax1.scatter(sell_signals.index, sell_signals["close"],
                          color="red", marker="v", s=100, label="Sell Signal", zorder=5)

        ax1.set_title("Price Chart with Trading Signals")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # RSI
        if "RSI" in df.columns:
            ax2.plot(df.index, df["RSI"], label="RSI", color="purple", linewidth=1)
            ax2.axhline(70, color="red", linestyle="--", alpha=0.5, label="Overbought")
            ax2.axhline(30, color="green", linestyle="--", alpha=0.5, label="Oversold")
            ax2.axhline(50, color="gray", linestyle="--", alpha=0.3)
            ax2.set_ylim(0, 100)
            ax2.set_title("RSI Indicator")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Equity curve
        ax3.plot(df.index, equity_curve, label="Equity", linewidth=2, color="green")
        ax3.axhline(initial_capital, color="red", linestyle="--", label="Initial Capital")

        for trade in trades:
            if trade[0] == "BUY":
                ax3.scatter(trade[2], trade[4], color="blue", marker="^", s=80, zorder=5)
            elif trade[0] == "SELL":
                color = "green" if trade[5] > 0 else "red"
                ax3.scatter(trade[2], trade[4], color=color, marker="v", s=80, zorder=5)

        ax3.set_title("Equity Curve")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdowns = (rolling_max - equity_series) / rolling_max * 100

        ax4.fill_between(df.index, drawdowns, 0, color="red", alpha=0.3)
        ax4.plot(df.index, drawdowns, color="red", linewidth=1)
        ax4.set_title("Drawdown (%)")
        ax4.set_ylabel("Drawdown %")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("⚠️ Matplotlib not available - skipping plots")
