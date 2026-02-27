"""
Performance Tracker for paper trading system.

Issue #332: Comprehensive Decision Logging & Full Trading Pipeline Integration
Stream C: Performance Tracker

This module provides real-time performance metrics tracking including:
- Win rate, profit factor, Sharpe ratio
- Max drawdown tracking
- Consecutive win/loss statistics
- Daily summaries
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""

    timestamp: datetime
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade_duration: timedelta = field(default_factory=timedelta)
    best_trade: float = 0.0
    worst_trade: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0


@dataclass
class TradeResult:
    """Result of a completed trade."""

    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_percent: float
    duration: timedelta
    exit_reason: str  # "take_profit", "stop_loss", "manual"
    timestamp: datetime


class PerformanceTracker:
    """Tracks and calculates real-time performance metrics."""

    def __init__(self, initial_balance: float = 10000.0):
        """
        Initialize the performance tracker.

        Args:
            initial_balance: Starting balance for the trading account.
        """
        self._initial_balance = initial_balance
        self._current_balance = initial_balance
        self._peak_balance = initial_balance
        self._trades: list[TradeResult] = []
        self._daily_returns: list[float] = []
        # Track balance history for drawdown calculation
        self._balance_history: list[float] = [initial_balance]

    def record_trade(self, trade: TradeResult) -> None:
        """
        Record a completed trade and update metrics.

        Args:
            trade: The completed trade result.
        """
        self._trades.append(trade)

        # Update current balance
        self._current_balance += trade.pnl

        # Update peak balance if new high
        if self._current_balance > self._peak_balance:
            self._peak_balance = self._current_balance

        # Track balance history
        self._balance_history.append(self._current_balance)

        # Track daily returns
        self._daily_returns.append(trade.pnl_percent)

    def get_metrics(self) -> PerformanceMetrics:
        """
        Calculate and return current performance metrics.

        Returns:
            PerformanceMetrics object with all calculated metrics.
        """
        now = datetime.now(timezone.utc)

        if not self._trades:
            return PerformanceMetrics(timestamp=now)

        # Calculate basic stats
        total_trades = len(self._trades)
        winning_trades = sum(1 for t in self._trades if t.pnl > 0)
        losing_trades = sum(1 for t in self._trades if t.pnl < 0)

        # Calculate PnL aggregates
        total_pnl = sum(t.pnl for t in self._trades)
        gross_profit = sum(t.pnl for t in self._trades if t.pnl > 0)
        gross_loss = sum(t.pnl for t in self._trades if t.pnl < 0)

        # Calculate averages
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0.0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0.0

        # Best and worst trades
        pnls = [t.pnl for t in self._trades]
        best_trade = max(pnls)
        worst_trade = min(pnls)

        # Average trade duration
        total_duration = sum((t.duration for t in self._trades), start=timedelta())
        avg_duration = total_duration / total_trades

        # Calculate derived metrics
        win_rate = self._calculate_win_rate()
        profit_factor = self._calculate_profit_factor()
        sharpe_ratio = self._calculate_sharpe_ratio()
        max_dd_abs, max_dd_pct = self._calculate_max_drawdown()
        (
            current_wins,
            current_losses,
            max_wins,
            max_losses,
        ) = self._calculate_consecutive_stats()

        return PerformanceMetrics(
            timestamp=now,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_dd_abs,
            max_drawdown_percent=max_dd_pct,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade_duration=avg_duration,
            best_trade=best_trade,
            worst_trade=worst_trade,
            consecutive_wins=current_wins,
            consecutive_losses=current_losses,
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
        )

    def _calculate_win_rate(self) -> float:
        """
        Calculate win rate percentage.

        Returns:
            Win rate as percentage (0-100).
        """
        if not self._trades:
            return 0.0

        winning_trades = sum(1 for t in self._trades if t.pnl > 0)
        return (winning_trades / len(self._trades)) * 100

    def _calculate_profit_factor(self) -> float:
        """
        Calculate profit factor (gross profit / abs(gross loss)).

        Returns:
            Profit factor ratio. Returns inf if no losses, 0 if no profits.
        """
        if not self._trades:
            return 0.0

        gross_profit = sum(t.pnl for t in self._trades if t.pnl > 0)
        gross_loss = sum(t.pnl for t in self._trades if t.pnl < 0)

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / abs(gross_loss)

    def _calculate_sharpe_ratio(self) -> float:
        """
        Calculate Sharpe ratio.

        Sharpe = (mean_return - risk_free_rate) / std_return * sqrt(252)
        Uses pnl_percent as returns, assumes risk_free_rate = 0.

        Returns:
            Annualized Sharpe ratio.
        """
        if len(self._trades) < 2:
            return 0.0

        returns = [t.pnl_percent for t in self._trades]
        mean_return = sum(returns) / len(returns)

        # Calculate population standard deviation
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_return = math.sqrt(variance)

        if std_return == 0:
            return 0.0

        # Annualize: multiply by sqrt(252) for daily trading days
        return (mean_return / std_return) * math.sqrt(252)

    def _calculate_max_drawdown(self) -> tuple[float, float]:
        """
        Calculate max drawdown in $ and %.

        Returns:
            Tuple of (max_drawdown_absolute, max_drawdown_percent).
        """
        if len(self._balance_history) < 2:
            return 0.0, 0.0

        max_dd_abs = 0.0
        max_dd_pct = 0.0
        peak = self._balance_history[0]

        for balance in self._balance_history[1:]:
            if balance > peak:
                peak = balance
            else:
                dd_abs = peak - balance
                dd_pct = (dd_abs / peak) * 100 if peak > 0 else 0.0

                if dd_abs > max_dd_abs:
                    max_dd_abs = dd_abs
                    max_dd_pct = dd_pct

        return max_dd_abs, max_dd_pct

    def _calculate_consecutive_stats(self) -> tuple[int, int, int, int]:
        """
        Calculate consecutive wins/losses statistics.

        Returns:
            Tuple of (current_wins, current_losses, max_wins, max_losses).
        """
        if not self._trades:
            return 0, 0, 0, 0

        current_wins = 0
        current_losses = 0
        max_wins = 0
        max_losses = 0

        win_streak = 0
        loss_streak = 0

        for trade in self._trades:
            if trade.pnl > 0:
                win_streak += 1
                loss_streak = 0
                max_wins = max(max_wins, win_streak)
            elif trade.pnl < 0:
                loss_streak += 1
                win_streak = 0
                max_losses = max(max_losses, loss_streak)
            else:
                # Breakeven trade resets both streaks
                win_streak = 0
                loss_streak = 0

        # Current streak
        if win_streak > 0:
            current_wins = win_streak
        elif loss_streak > 0:
            current_losses = loss_streak

        return current_wins, current_losses, max_wins, max_losses

    def get_daily_summary(self) -> dict:
        """
        Get summary of today's trading performance.

        Returns:
            Dictionary with today's trading summary.
        """
        today = datetime.now(timezone.utc).date()

        # Filter trades from today
        today_trades = [t for t in self._trades if t.timestamp.date() == today]

        total_trades = len(today_trades)
        winning_trades = sum(1 for t in today_trades if t.pnl > 0)
        losing_trades = sum(1 for t in today_trades if t.pnl < 0)
        pnl = sum(t.pnl for t in today_trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        return {
            "date": today,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "pnl": pnl,
            "win_rate": win_rate,
        }

    def reset(self) -> None:
        """Reset tracker to initial state."""
        self._current_balance = self._initial_balance
        self._peak_balance = self._initial_balance
        self._trades = []
        self._daily_returns = []
        self._balance_history = [self._initial_balance]
