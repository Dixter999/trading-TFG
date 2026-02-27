"""
Backtesting metrics calculation functions.

This module provides comprehensive metrics calculation for backtesting results,
including Sharpe ratio, Sortino ratio, maximum drawdown, profit factor, and
a complete calculate_backtest_metrics function.

All metrics use numpy for efficient calculations.
"""

import math
from typing import List

import numpy as np

from backtesting.models import BacktestMetrics, Trade


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio for a series of returns.

    The Sharpe ratio measures risk-adjusted return by comparing excess return
    (return above risk-free rate) to the volatility (standard deviation) of returns.

    Formula: Sharpe = (mean(returns) - risk_free_rate) / std(returns)

    Args:
        returns: List of return values (e.g., P&L per trade or period).
        risk_free_rate: Risk-free rate to subtract from returns (default: 0.0).

    Returns:
        Sharpe ratio. Returns 0.0 for empty returns, single return, or zero volatility.

    Examples:
        >>> returns = [0.02, 0.015, 0.01, 0.025, 0.03]
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> sharpe > 0  # Positive returns should yield positive Sharpe
        True
    """
    if not returns or len(returns) < 2:
        return 0.0

    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_returns = np.std(excess_returns, ddof=1)  # Sample standard deviation

    # Handle zero volatility case
    if std_returns == 0:
        return 0.0

    return float(mean_excess_return / std_returns)


def calculate_sortino_ratio(
    returns: List[float], risk_free_rate: float = 0.0, target_return: float = 0.0
) -> float:
    """
    Calculate Sortino ratio for a series of returns.

    The Sortino ratio is similar to Sharpe but only penalizes downside volatility,
    making it better for strategies with asymmetric returns.

    Formula: Sortino = (mean(returns) - target_return) / downside_deviation

    Args:
        returns: List of return values.
        risk_free_rate: Risk-free rate (default: 0.0).
        target_return: Minimum acceptable return (default: 0.0).

    Returns:
        Sortino ratio. Returns 0.0 for empty returns or when downside deviation is zero.

    Examples:
        >>> returns = [0.05, 0.03, -0.01, 0.04]
        >>> sortino = calculate_sortino_ratio(returns)
        >>> sortino > 0  # Net positive returns should yield positive Sortino
        True
    """
    if not returns:
        return 0.0

    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    mean_excess_return = np.mean(excess_returns)

    # Calculate downside deviation (only negative returns relative to target)
    downside_returns = np.minimum(returns_array - target_return, 0)
    downside_deviation = np.sqrt(np.mean(downside_returns**2))

    # Handle zero downside deviation case
    if downside_deviation == 0:
        # If all returns are positive, Sortino should be very high
        # Return mean / small epsilon to avoid division by zero
        if mean_excess_return > 0:
            return float(mean_excess_return / 1e-10)
        return 0.0

    return float(mean_excess_return / downside_deviation)


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    Calculate maximum drawdown from an equity curve.

    Maximum drawdown is the largest peak-to-trough decline in equity,
    expressed as a negative percentage.

    Formula: max_dd = min((equity[i] - peak[i]) / peak[i])

    Args:
        equity_curve: List of equity values over time (e.g., running account balance).

    Returns:
        Maximum drawdown as a negative decimal (e.g., -0.20 for 20% drawdown).
        Returns 0.0 if no drawdown exists (always increasing equity) or empty curve.

    Examples:
        >>> equity = [1000, 1100, 900, 1050, 850, 950]
        >>> max_dd = calculate_max_drawdown(equity)
        >>> max_dd < 0  # Drawdown should be negative
        True
        >>> abs(max_dd) > 0.2  # Should be around 22.7% (1100 -> 850)
        True
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0

    equity_array = np.array(equity_curve)

    # Calculate running maximum (peak)
    running_max = np.maximum.accumulate(equity_array)

    # Calculate drawdown at each point
    drawdown = (equity_array - running_max) / running_max

    # Maximum drawdown is the minimum (most negative) drawdown
    max_dd = float(np.min(drawdown))

    return max_dd


def calculate_profit_factor(trades: List[Trade]) -> float:
    """
    Calculate profit factor for a list of trades.

    Profit factor is the ratio of gross profits to gross losses.
    A value > 1.0 indicates a profitable system.

    Formula: PF = gross_profit / abs(gross_loss)

    Args:
        trades: List of Trade objects with P&L values.

    Returns:
        Profit factor. Returns:
        - 0.0 for empty trades or no winning trades
        - infinity for all winning trades (no losses)
        - ratio of gross profit to gross loss otherwise

    Examples:
        >>> # Assuming create_test_trade exists
        >>> trades = [Trade(pnl=100), Trade(pnl=-50), Trade(pnl=150)]
        >>> pf = calculate_profit_factor(trades)
        >>> pf > 1.0  # More wins than losses
        True
    """
    if not trades:
        return 0.0

    gross_profit = sum(trade.pnl for trade in trades if trade.pnl > 0)
    gross_loss = abs(sum(trade.pnl for trade in trades if trade.pnl < 0))

    # Handle edge cases
    if gross_profit == 0:
        return 0.0

    if gross_loss == 0:
        return float("inf")

    return float(gross_profit / gross_loss)


def calculate_backtest_metrics(
    trades: List[Trade],
    initial_capital: float = 10000.0,
    total_signals: int | None = None,
    trading_days: int | None = None,
) -> BacktestMetrics:
    """
    Calculate comprehensive backtest metrics from a list of trades.

    This function computes all performance metrics required for backtest analysis,
    including profitability metrics, risk-adjusted returns, drawdown analysis,
    and trade statistics.

    Args:
        trades: List of executed Trade objects with P&L and metadata.
        initial_capital: Starting capital for equity curve calculation (default: 10000.0).
        total_signals: Total signals generated including filtered ones (optional).
        trading_days: Number of trading days in backtest period (optional).

    Returns:
        BacktestMetrics object containing comprehensive performance statistics.

    Examples:
        >>> trades = [Trade(...), Trade(...)]  # List of trades
        >>> metrics = calculate_backtest_metrics(trades)
        >>> assert 0 <= metrics.win_rate <= 1.0
        >>> assert metrics.total_trades == len(trades)
    """
    # Handle empty trades
    if not trades:
        return BacktestMetrics(
            total_pnl=0.0,
            total_pnl_pct=0.0,
            win_count=0,
            loss_count=0,
            win_rate=0.0,
            profit_factor=0.0,
            payoff_ratio=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            avg_drawdown=0.0,
            drawdown_duration=0,
            total_trades=0,
            avg_win=0.0,
            avg_loss=0.0,
            consecutive_wins=0,
            consecutive_losses=0,
            avg_trade_duration=0.0,
            total_signals=total_signals or 0,
            signals_per_day=0.0,
            avg_confidence=0.0,
            successful_signals=0.0,
        )

    # Basic profitability metrics
    total_pnl = sum(trade.pnl for trade in trades)
    total_pnl_pct = (total_pnl / initial_capital) * 100.0

    win_trades = [trade for trade in trades if trade.pnl > 0]
    loss_trades = [trade for trade in trades if trade.pnl < 0]
    breakeven_trades = [trade for trade in trades if trade.pnl == 0]

    win_count = len(win_trades)
    loss_count = len(loss_trades)
    total_trades = len(trades)
    win_rate = win_count / total_trades if total_trades > 0 else 0.0

    # Profit factor
    profit_factor = calculate_profit_factor(trades)

    # Average win/loss
    avg_win = (
        sum(trade.pnl for trade in win_trades) / win_count if win_count > 0 else 0.0
    )
    avg_loss = (
        sum(trade.pnl for trade in loss_trades) / loss_count if loss_count > 0 else 0.0
    )

    # Payoff ratio (avg win / abs(avg loss))
    payoff_ratio = (
        avg_win / abs(avg_loss)
        if avg_loss != 0
        else (float("inf") if avg_win > 0 else 0.0)
    )

    # Risk-adjusted metrics
    returns = [trade.pnl for trade in trades]
    sharpe_ratio = calculate_sharpe_ratio(returns)
    sortino_ratio = calculate_sortino_ratio(returns)

    # Equity curve and drawdown analysis
    equity_curve = [initial_capital]
    for trade in trades:
        equity_curve.append(equity_curve[-1] + trade.pnl)

    max_drawdown = calculate_max_drawdown(equity_curve)

    # Calculate average drawdown
    equity_array = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_array)
    drawdowns = (equity_array - running_max) / running_max
    drawdowns = drawdowns[drawdowns < 0]  # Only negative drawdowns
    avg_drawdown = float(np.mean(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Drawdown duration (simplified: consecutive periods in drawdown)
    in_drawdown = equity_array < running_max
    max_duration = 0
    current_duration = 0
    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    drawdown_duration = max_duration

    # Calmar ratio (return / abs(max_drawdown))
    calmar_ratio = (
        (total_pnl_pct / 100.0) / abs(max_drawdown)
        if max_drawdown != 0
        else (float("inf") if total_pnl > 0 else 0.0)
    )

    # Consecutive wins/losses
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0

    for trade in trades:
        if trade.pnl > 0:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        elif trade.pnl < 0:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
        else:
            # Breakeven - reset both
            current_wins = 0
            current_losses = 0

    # Average trade duration (in hours)
    durations = [
        (trade.exit_time - trade.entry_time).total_seconds() / 3600.0
        for trade in trades
    ]
    avg_trade_duration = float(np.mean(durations)) if durations else 0.0

    # Signal metrics
    if total_signals is None:
        total_signals = total_trades

    signals_per_day = (
        total_signals / trading_days if trading_days and trading_days > 0 else 0.0
    )

    avg_confidence = (
        float(np.mean([trade.signal.confidence for trade in trades])) if trades else 0.0
    )

    successful_signals = win_rate  # Same as win rate for traded signals

    return BacktestMetrics(
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
        win_count=win_count,
        loss_count=loss_count,
        win_rate=win_rate,
        profit_factor=profit_factor,
        payoff_ratio=payoff_ratio,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown=max_drawdown,
        avg_drawdown=avg_drawdown,
        drawdown_duration=drawdown_duration,
        total_trades=total_trades,
        avg_win=avg_win,
        avg_loss=avg_loss,
        consecutive_wins=max_consecutive_wins,
        consecutive_losses=max_consecutive_losses,
        avg_trade_duration=avg_trade_duration,
        total_signals=total_signals,
        signals_per_day=signals_per_day,
        avg_confidence=avg_confidence,
        successful_signals=successful_signals,
    )
