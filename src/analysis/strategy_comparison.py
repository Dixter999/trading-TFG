"""
Strategy comparison utilities for analyzing multiple trading strategies.

Author: python-backend-engineer (Stream A)
Issue: #255
Created: 2025-11-12 (GREEN phase - minimal implementation)
Refactored: 2025-11-12 (REFACTOR phase - extract helper functions)

This module provides utilities to:
- Calculate comprehensive strategy metrics from backtest results
- Compare multiple strategies side-by-side
- Generate comparison tables for reports

TDD Cycle:
1. RED: Tests written first (test_strategy_comparison.py)
2. GREEN: Minimal implementation to pass tests
3. REFACTOR: Extracted helper functions for better readability
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd


@dataclass
class StrategyMetrics:
    """
    Performance metrics for a trading strategy.

    Attributes:
        name: Strategy name
        total_return: Total return percentage
        sharpe_ratio: Sharpe ratio (risk-adjusted return)
        sortino_ratio: Sortino ratio (downside risk-adjusted return)
        max_drawdown: Maximum drawdown percentage (negative value)
        calmar_ratio: Calmar ratio (return / max drawdown)
        win_rate: Winning trade percentage (0-100)
        total_trades: Total number of trades executed
        avg_win: Average winning trade P&L
        avg_loss: Average losing trade P&L (negative value)
        profit_factor: Ratio of total wins to total losses
    """

    name: str
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    total_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float

    def to_dict(self) -> dict:
        """
        Convert StrategyMetrics to dictionary.

        Returns:
            Dictionary with all metrics
        """
        return {
            "name": self.name,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
        }


def _calculate_returns_series(equity_curve: np.ndarray) -> pd.Series:
    """
    Calculate returns series from equity curve.

    Args:
        equity_curve: Array of equity values

    Returns:
        Series of returns
    """
    if len(equity_curve) > 1:
        returns = pd.Series(np.diff(equity_curve) / equity_curve[:-1])
    else:
        returns = pd.Series([])
    return returns


def _calculate_sharpe_ratio(returns: pd.Series) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Series of returns

    Returns:
        Sharpe ratio (annualized assuming 252 trading days)
    """
    if len(returns) > 0 and returns.std() > 0:
        return (returns.mean() / returns.std()) * np.sqrt(252)
    return 0.0


def _calculate_sortino_ratio(returns: pd.Series, sharpe_ratio: float) -> float:
    """
    Calculate annualized Sortino ratio (only penalizes downside volatility).

    Args:
        returns: Series of returns
        sharpe_ratio: Sharpe ratio (used as fallback if no downside risk)

    Returns:
        Sortino ratio (annualized assuming 252 trading days)
    """
    if len(returns) > 0:
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std()
        if downside_std > 0 and not np.isnan(downside_std):
            return (returns.mean() / downside_std) * np.sqrt(252)
        # No downside risk - use Sharpe as fallback
        return sharpe_ratio if sharpe_ratio > 0 else 0.0
    return 0.0


def _calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown percentage.

    Args:
        equity_curve: Array of equity values

    Returns:
        Maximum drawdown as negative percentage
    """
    peak = equity_curve[0]
    max_dd = 0.0

    for equity in equity_curve:
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = ((peak - equity) / peak) * 100.0
            if dd > max_dd:
                max_dd = dd

    return -max_dd  # Return as negative value


def _calculate_trade_statistics(
    trades: pd.DataFrame,
) -> Tuple[float, float, float, float]:
    """
    Calculate trade statistics from trades DataFrame.

    Args:
        trades: DataFrame with 'pnl' and 'trade_outcome' columns

    Returns:
        Tuple of (win_rate, avg_win, avg_loss, profit_factor)
    """
    if len(trades) == 0:
        return 0.0, 0.0, 0.0, 0.0

    wins = trades[trades["trade_outcome"] == "WIN"]
    losses = trades[trades["trade_outcome"] == "LOSS"]

    # Win rate
    win_rate = (len(wins) / len(trades)) * 100.0

    # Average win
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0.0

    # Average loss
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0.0

    # Profit factor
    total_wins = wins["pnl"].sum() if len(wins) > 0 else 0.0
    total_losses = abs(losses["pnl"].sum()) if len(losses) > 0 else 0.0

    if total_losses > 0:
        profit_factor = total_wins / total_losses
    else:
        profit_factor = 0.0

    return win_rate, avg_win, avg_loss, profit_factor


def calculate_strategy_metrics(
    backtest_results: pd.DataFrame, strategy_name: str
) -> StrategyMetrics:
    """
    Calculate comprehensive metrics from backtest results.

    Args:
        backtest_results: DataFrame with columns:
            - date: Trading date
            - pnl: Profit/loss for the period
            - equity: Current equity value
            - trade_outcome: Trade result ('WIN', 'LOSS', None for no trade)
        strategy_name: Name of the strategy

    Returns:
        StrategyMetrics object with all performance metrics

    Raises:
        ValueError: If backtest_results is empty
    """
    if len(backtest_results) == 0:
        raise ValueError("backtest_results DataFrame cannot be empty")

    # Extract equity curve and calculate total return
    equity_curve = backtest_results["equity"].values
    initial_equity = equity_curve[0]
    final_equity = equity_curve[-1]

    if initial_equity == 0:
        total_return = 0.0
    else:
        total_return = ((final_equity - initial_equity) / initial_equity) * 100.0

    # Calculate risk-adjusted metrics
    returns = _calculate_returns_series(equity_curve)
    sharpe_ratio = _calculate_sharpe_ratio(returns)
    sortino_ratio = _calculate_sortino_ratio(returns, sharpe_ratio)

    # Calculate drawdown metrics
    max_drawdown = _calculate_max_drawdown(equity_curve)
    max_dd_abs = abs(max_drawdown)
    calmar_ratio = total_return / max_dd_abs if max_dd_abs > 0 else 0.0

    # Calculate trade statistics
    trades = backtest_results[backtest_results["trade_outcome"].notna()]
    win_rate, avg_win, avg_loss, profit_factor = _calculate_trade_statistics(trades)

    return StrategyMetrics(
        name=strategy_name,
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        win_rate=win_rate,
        total_trades=len(trades),
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
    )


def create_comparison_table(strategies: List[StrategyMetrics]) -> pd.DataFrame:
    """
    Create comparison table from multiple strategies.

    Args:
        strategies: List of StrategyMetrics objects

    Returns:
        DataFrame with formatted comparison table

    Example:
        >>> strategies = [metrics1, metrics2, metrics3]
        >>> table = create_comparison_table(strategies)
        >>> print(table)
        |  Strategy  | Return (%) | Sharpe | ... |
        |------------|------------|--------|-----|
        | RL Strategy|    7.67    |  0.89  | ... |
    """
    if len(strategies) == 0:
        return pd.DataFrame()

    data = []
    for s in strategies:
        data.append(
            {
                "Strategy": s.name,
                "Return (%)": f"{s.total_return:.2f}",
                "Sharpe": f"{s.sharpe_ratio:.2f}",
                "Sortino": f"{s.sortino_ratio:.2f}",
                "Max DD (%)": f"{s.max_drawdown:.2f}",
                "Calmar": f"{s.calmar_ratio:.2f}",
                "Win Rate (%)": f"{s.win_rate:.1f}",
                "Trades": s.total_trades,
                "Profit Factor": f"{s.profit_factor:.2f}",
            }
        )

    return pd.DataFrame(data)
