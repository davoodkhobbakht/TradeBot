# -*- coding: utf-8 -*-
"""
Advanced Backtest Engine with Portfolio Risk Management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from config import SYMBOL_OPTIMIZED_PARAMS, TRADE_SETTINGS

try:
    from utils.indicators import dynamic_trailing_stop
except ImportError:
    def dynamic_trailing_stop(current_price, entry_price, trailing_high, volatility):
        """Fallback trailing stop calculation"""
        return trailing_high * (1 - volatility * 2)


class EnhancedBacktester:
    """Advanced backtester with adaptive position sizing and comprehensive metrics"""

    def __init__(self, initial_capital=1000.0, enable_short_selling=False):
        self.initial_capital = initial_capital
        self.enable_short_selling = enable_short_selling
        self.trade_fee = TRADE_SETTINGS["trade_fee"]
        self.slippage = TRADE_SETTINGS["slippage"]
        self.bid_ask_spread = TRADE_SETTINGS.get("bid_ask_spread", 0.0)
        self.max_trades = TRADE_SETTINGS["max_trades_per_symbol"]

    def calculate_position_size_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.25
    ) -> float:
        """
        Calculate position size using Kelly Criterion
        Kelly % = (Win% × Avg Win - Loss% × Avg Loss) / Avg Win
        Uses fractional Kelly (0.25) for safety
        """
        if avg_loss == 0 or win_rate == 0:
            return 0.01

        loss_rate = 1 - win_rate
        kelly_pct = (win_rate * avg_win - loss_rate * abs(avg_loss)) / avg_win
        kelly_pct = max(0, min(kelly_pct * kelly_fraction, 0.20))

        return max(0.01, kelly_pct)

    def calculate_position_size_volatility(
        self,
        capital: float,
        atr: float,
        price: float,
        risk_per_trade: float = 0.02
    ) -> float:
        """
        Calculate position size based on volatility (ATR-based risk sizing)
        """
        if atr <= 0 or price <= 0:
            return 0.01

        risk_amount = capital * risk_per_trade
        stop_distance = atr * 2
        position_size = risk_amount / stop_distance
        position_pct = (position_size * price) / capital
        position_pct = max(0.01, min(position_pct, 0.20))

        return position_pct

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)

        return float(sharpe)

    def calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Sortino ratio (downside deviation only)"""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return float('inf')

        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0

        sortino = np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)
        return float(sortino)

    def calculate_calmar_ratio(
        self,
        total_return: float,
        max_drawdown: float,
        years: float
    ) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)"""
        if max_drawdown == 0 or years == 0:
            return 0.0

        annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
        calmar = annualized_return / max_drawdown

        return float(calmar)

    def calculate_max_drawdown_duration(self, equity_curve: np.ndarray) -> int:
        """Calculate maximum drawdown duration in periods"""
        if len(equity_curve) < 2:
            return 0

        rolling_max = np.maximum.accumulate(equity_curve)
        drawdowns = (rolling_max - equity_curve) / rolling_max

        in_drawdown = drawdowns > 0
        max_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def run_backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float,
        use_kelly: bool = True,
        use_volatility_sizing: bool = True,
        signal_confidence: Optional[np.ndarray] = None
    ) -> Tuple[List, np.ndarray, float, float, Dict]:
        """
        Enhanced backtest with advanced position sizing and risk management

        Returns:
            trades: List of trade tuples
            equity_curve: Array of equity values
            final_capital: Final cash amount
            final_position: Final position size
            metrics: Dictionary of performance metrics
        """
        capital = initial_capital
        position = 0
        trades = []
        entry_price = 0
        entry_date = None
        trailing_high = 0
        in_position = False

        equity_curve = np.full(len(df), initial_capital, dtype=float)

        # Get symbol-specific parameters
        current_symbol = getattr(df, "name", "BTC/USDT")
        if current_symbol in SYMBOL_OPTIMIZED_PARAMS:
            params = SYMBOL_OPTIMIZED_PARAMS[current_symbol]
            stop_loss_pct = params["stop_loss"]
            take_profit_pct = params["take_profit"]
            base_position_size = params["position_size"]
        else:
            stop_loss_pct = 0.08
            take_profit_pct = 0.20
            base_position_size = 0.10

        start_idx = 200
        trade_count = 0
        trade_returns = []

        print(f"\n⚙️ Backtest Parameters for {current_symbol}:")
        print(f"   Stop Loss: {stop_loss_pct*100:.1f}% | Take Profit: {take_profit_pct*100:.1f}%")
        print(f"   Base Position Size: {base_position_size*100:.1f}%")
        print(f"   Kelly Sizing: {'Enabled' if use_kelly else 'Disabled'}")
        print(f"   Volatility Sizing: {'Enabled' if use_volatility_sizing else 'Disabled'}")

        for i in range(start_idx, len(df)):
            base_price = df["close"].iloc[i]

            # Apply slippage realistically
            if in_position:
                current_price = base_price * (1 - self.slippage - self.bid_ask_spread / 2)
            else:
                current_price = base_price * (1 + self.slippage + self.bid_ask_spread / 2)

            current_date = df.index[i]
            volatility = df["BB_Bandwidth"].iloc[i] if "BB_Bandwidth" in df.columns else 0.02
            atr = df["atr_14"].iloc[i] if "atr_14" in df.columns else base_price * 0.02

            if in_position:
                pnl_pct = (current_price - entry_price) / entry_price
                trailing_high = max(trailing_high, current_price)

                # Dynamic trailing stop
                trailing_stop_price = dynamic_trailing_stop(
                    current_price, entry_price, trailing_high, volatility
                )
                stop_loss_price = entry_price * (1 - stop_loss_pct)

                # Exit conditions
                exit_condition = False
                exit_type = ""

                if pnl_pct >= take_profit_pct:
                    exit_condition = True
                    exit_type = "TAKE_PROFIT"
                elif current_price <= min(trailing_stop_price, stop_loss_price):
                    exit_condition = True
                    exit_type = "STOP_LOSS"
                elif "position" in df.columns and df["position"].iloc[i] == -1:
                    exit_condition = True
                    exit_type = "STRATEGY"

                if exit_condition:
                    exit_value = position * current_price * (1 - self.trade_fee)
                    profit = exit_value - (position * entry_price)
                    profit_pct = (profit / (position * entry_price)) * 100 if position * entry_price > 0 else 0

                    capital += exit_value
                    trade_returns.append(profit_pct)

                    trades.append((
                        "SELL", exit_type, current_date, current_price,
                        exit_value, profit, profit_pct,
                        (current_date - entry_date).days
                    ))

                    position = 0
                    in_position = False

                    icon = "🟢" if profit > 0 else "🔴"
                    print(f"{icon} Exit {exit_type}: {current_date.strftime('%Y-%m-%d')} - "
                          f"Price: {current_price:.0f} - P/L: {profit:+.0f} USDT ({profit_pct:+.1f}%)")

            # Entry conditions
            if (
                "position" in df.columns
                and df["position"].iloc[i] == 1
                and not in_position
                and capital > 10
                and trade_count < self.max_trades
            ):
                # Calculate dynamic position size
                position_size_pct = base_position_size

                # Kelly criterion sizing
                if use_kelly and len(trade_returns) >= 5:
                    wins = [r for r in trade_returns if r > 0]
                    losses = [r for r in trade_returns if r < 0]

                    if wins and losses:
                        win_rate = len(wins) / len(trade_returns)
                        avg_win = np.mean(wins)
                        avg_loss = np.mean(losses)
                        kelly_size = self.calculate_position_size_kelly(
                            win_rate, avg_win, avg_loss
                        )
                        position_size_pct = kelly_size

                # Volatility-based sizing
                if use_volatility_sizing and atr > 0:
                    vol_size = self.calculate_position_size_volatility(
                        capital, atr, current_price, risk_per_trade=0.02
                    )
                    position_size_pct = min(position_size_pct, vol_size)

                # Signal confidence adjustment
                if signal_confidence is not None and i < len(signal_confidence):
                    confidence = signal_confidence[i]
                    position_size_pct *= confidence

                # Apply bounds
                min_investment = initial_capital * 0.01
                max_investment = initial_capital * 0.20
                investment_amount = capital * position_size_pct
                investment_amount = max(min_investment, min(investment_amount, max_investment))

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

                trades.append((
                    "BUY", "ENTRY", current_date, entry_price,
                    entry_capital, 0, 0, 0
                ))

                print(f"🟢 Entry #{trade_count}: {current_date.strftime('%Y-%m-%d')} - "
                      f"Price: {entry_price:.0f} - Investment: {entry_capital:.0f} USDT "
                      f"(Size: {position_size_pct*100:.1f}%)")

            # Update equity curve
            if in_position:
                current_equity = capital + (position * current_price)
            else:
                current_equity = capital

            equity_curve[i] = current_equity

        # Close open position at end
        if in_position and len(df) > 0:
            current_price = df["close"].iloc[-1]
            current_date = df.index[-1]
            exit_value = position * current_price * (1 - self.trade_fee)
            profit = exit_value - (position * entry_price)
            profit_pct = (profit / (position * entry_price)) * 100 if position * entry_price > 0 else 0

            capital += exit_value
            trade_returns.append(profit_pct)

            trades.append((
                "SELL", "END_OF_PERIOD", current_date, current_price,
                exit_value, profit, profit_pct,
                (current_date - entry_date).days
            ))

            icon = "🟢" if profit > 0 else "🔴"
            print(f"{icon} Close position at period end: P/L {profit:+.0f} USDT ({profit_pct:+.1f}%)")

        final_value = capital if not in_position else capital + (position * df["close"].iloc[-1])

        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(
            trades, equity_curve, initial_capital, final_value, df, trade_returns
        )

        return trades, equity_curve, capital, position, metrics

    def _calculate_metrics(
        self,
        trades: List,
        equity_curve: np.ndarray,
        initial_capital: float,
        final_value: float,
        df: pd.DataFrame,
        trade_returns: List[float]
    ) -> Dict:
        """Calculate comprehensive performance metrics"""

        total_return = ((final_value - initial_capital) / initial_capital) * 100
        buy_hold_return = ((df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]) * 100

        # Trade statistics
        buy_trades = [t for t in trades if t[0] == "BUY"]
        sell_trades = [t for t in trades if t[0] == "SELL"]

        profits = [t[5] for t in sell_trades if len(t) > 5]
        profits_pct = [t[6] for t in sell_trades if len(t) > 6]
        durations = [t[7] for t in sell_trades if len(t) > 7]

        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]

        win_rate = (len(winning_trades) / len(profits)) * 100 if profits else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = abs(sum(winning_trades)) / abs(sum(losing_trades)) if losing_trades else float('inf')

        # Drawdown analysis
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdowns = (rolling_max - equity_series) / rolling_max
        max_drawdown = drawdowns.max() * 100
        max_dd_duration = self.calculate_max_drawdown_duration(equity_curve)

        # Risk-adjusted metrics
        daily_returns = equity_series.pct_change().dropna().values
        years = len(df) / 252

        sharpe = self.calculate_sharpe_ratio(daily_returns)
        sortino = self.calculate_sortino_ratio(daily_returns)
        calmar = self.calculate_calmar_ratio(total_return, max_drawdown, years) if years > 0 else 0

        metrics = {
            "total_return": total_return,
            "buy_hold_return": buy_hold_return,
            "outperformance": total_return - buy_hold_return,
            "num_trades": len(buy_trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": max_dd_duration,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "avg_trade_duration": np.mean(durations) if durations else 0,
            "final_value": final_value,
        }

        return metrics


def run_backtest(df, initial_capital=1000.0):
    """Backward-compatible wrapper for old API"""
    backtester = EnhancedBacktester(initial_capital)
    trades, equity_curve, final_capital, final_position, metrics = backtester.run_backtest(
        df, initial_capital
    )

    # Print summary
    print(f"\n📊 Final Results:")
    print(f"   Initial Capital: {initial_capital:,.0f} USDT")
    print(f"   Final Value: {metrics['final_value']:,.0f} USDT")
    print(f"   Total Return: {metrics['total_return']:+.2f}%")
    print(f"   Buy & Hold: {metrics['buy_hold_return']:+.2f}%")
    print(f"   Outperformance: {metrics['outperformance']:+.2f}%")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"   Max Drawdown: {metrics['max_drawdown']:.2f}%")

    return trades, equity_curve, final_capital, final_position
