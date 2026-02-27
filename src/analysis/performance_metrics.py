"""
Performance metrics calculation module.

Author: python-backend-engineer (Stream B)
Issue: #249
Created: 2025-11-11 (GREEN phase - stub to see failures)

Calculates:
- Total return and annualized return
- Sharpe ratio (risk-adjusted return)
- Sortino ratio (downside risk-adjusted return)
- Calmar ratio (return to max drawdown)
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pandas as pd


def calculate_total_return(initial_capital: float, final_capital: float) -> float:
    """
    Calculate total return percentage.

    Args:
        initial_capital: Starting capital
        final_capital: Ending capital

    Returns:
        Total return as percentage

    Raises:
        ValueError: If initial_capital is zero or negative
    """
    if initial_capital <= 0:
        raise ValueError("Initial capital must be greater than zero")

    return ((final_capital - initial_capital) / initial_capital) * 100.0


def calculate_annualized_return(total_return: float, days: int) -> float:
    """
    Calculate annualized return from total return.

    Args:
        total_return: Total return percentage
        days: Number of days in period

    Returns:
        Annualized return percentage

    Raises:
        ValueError: If days is zero or negative
    """
    if days <= 0:
        raise ValueError("Days must be greater than zero")

    # Convert total return to decimal
    total_return_decimal = total_return / 100.0

    # Annualize using compound growth formula: (1 + r)^(365/days) - 1
    years = days / 365.0
    annualized_decimal = (1.0 + total_return_decimal) ** (1.0 / years) - 1.0

    return annualized_decimal * 100.0


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default 0.0)

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate

    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    # Handle zero volatility
    if std_excess == 0:
        return 0.0

    # Annualize (assuming 252 trading days)
    sharpe = (mean_excess / std_excess) * np.sqrt(252)

    return sharpe


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio (only penalizes downside volatility).

    Uses target downside deviation (TDD) which considers all returns,
    but only penalizes those below the target (risk-free rate).

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default 0.0)

    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    # Calculate excess returns
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf

    mean_excess = excess_returns.mean()

    # Calculate target downside deviation (TDD)
    # Square all negative excess returns, average them, then sqrt
    downside_squared = np.where(excess_returns < 0, excess_returns**2, 0)
    tdd = np.sqrt(np.mean(downside_squared))

    if tdd == 0 or np.isnan(tdd):
        # No downside risk - return very high value (positive returns dominate)
        # But to keep it finite and consistent, return a high fixed value
        return mean_excess * np.sqrt(252) * 100 if mean_excess > 0 else 0.0

    # Annualize (assuming 252 trading days)
    sortino = (mean_excess / tdd) * np.sqrt(252)

    return sortino


def calculate_calmar_ratio(equity_curve: np.ndarray, days: int) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        equity_curve: Array of equity values over time
        days: Number of days in period

    Returns:
        Calmar ratio
    """
    if len(equity_curve) == 0:
        return 0.0

    # Calculate total return
    initial_capital = equity_curve[0]
    final_capital = equity_curve[-1]

    if initial_capital <= 0:
        return 0.0

    total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100.0

    # Annualize return
    if days <= 0:
        return 0.0

    years = days / 365.0
    annualized_return = (
        (1.0 + total_return_pct / 100.0) ** (1.0 / years) - 1.0
    ) * 100.0

    # Calculate max drawdown
    peak = equity_curve[0]
    max_dd = 0.0

    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = ((peak - equity) / peak) * 100.0
        if dd > max_dd:
            max_dd = dd

    # Handle zero drawdown
    if max_dd == 0:
        return 0.0

    # Calmar ratio
    calmar = annualized_return / max_dd

    return calmar


@dataclass
class PerformanceMetrics:
    """
    Container for performance metrics.

    Attributes:
        total_return: Total return percentage
        annualized_return: Annualized return percentage
        sharpe_ratio: Sharpe ratio
        sortino_ratio: Sortino ratio
        calmar_ratio: Calmar ratio
        max_drawdown: Maximum drawdown percentage
    """

    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float

    @classmethod
    def from_backtest_results(
        cls,
        initial_capital: float,
        final_capital: float,
        equity_curve: np.ndarray,
        trades: List[Dict],
        days: int,
        risk_free_rate: float = 0.0,
    ) -> "PerformanceMetrics":
        """
        Create PerformanceMetrics from backtest results.

        Args:
            initial_capital: Starting capital
            final_capital: Ending capital
            equity_curve: Array of equity values
            trades: List of trade dictionaries
            days: Number of days in backtest
            risk_free_rate: Annual risk-free rate

        Returns:
            PerformanceMetrics instance
        """
        # Calculate total return
        total_return = calculate_total_return(initial_capital, final_capital)

        # Calculate annualized return
        annualized_return = calculate_annualized_return(total_return, days)

        # Calculate returns series from equity curve
        if len(equity_curve) > 1:
            returns = pd.Series(np.diff(equity_curve) / equity_curve[:-1])
        else:
            returns = pd.Series([])

        # Calculate risk-adjusted metrics
        sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate)
        sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate)
        calmar_ratio = calculate_calmar_ratio(equity_curve, days)

        # Calculate max drawdown
        peak = equity_curve[0] if len(equity_curve) > 0 else initial_capital
        max_dd = 0.0

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = ((peak - equity) / peak) * 100.0
            if dd > max_dd:
                max_dd = dd

        return cls(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_dd,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
        }
