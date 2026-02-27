"""
Risk metrics calculation module.

Author: python-backend-engineer (Stream B)
Issue: #249
Created: 2025-11-11 (RED phase - stub)

Calculates:
- Maximum drawdown and drawdown duration
- Drawdown periods analysis
- Current drawdown status
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown percentage.

    Args:
        equity_curve: Array of equity values over time

    Returns:
        Maximum drawdown as percentage
    """
    if len(equity_curve) == 0:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0

    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = ((peak - equity) / peak) * 100.0
        if dd > max_dd:
            max_dd = dd

    return max_dd


def calculate_drawdown_duration(equity_curve: np.ndarray) -> int:
    """
    Calculate duration of maximum drawdown.

    Args:
        equity_curve: Array of equity values over time

    Returns:
        Duration in number of periods
    """
    if len(equity_curve) <= 1:
        return 0

    # Track peaks and drawdown periods
    peak = equity_curve[0]
    peak_idx = 0
    in_drawdown = False
    drawdown_start = 0
    max_duration = 0
    current_duration = 0

    for i, equity in enumerate(equity_curve):
        if equity > peak:
            # New peak - end any drawdown
            if in_drawdown:
                duration = i - drawdown_start
                if duration > max_duration:
                    max_duration = duration
                in_drawdown = False
            peak = equity
            peak_idx = i
        elif equity < peak:
            # In drawdown
            if not in_drawdown:
                in_drawdown = True
                drawdown_start = peak_idx
            current_duration = i - drawdown_start

    # Check if still in drawdown at end
    if in_drawdown:
        duration = len(equity_curve) - 1 - drawdown_start
        if duration > max_duration:
            max_duration = duration

    return max_duration


def find_drawdown_periods(equity_curve: np.ndarray) -> List[Dict]:
    """
    Find all drawdown periods in equity curve.

    Args:
        equity_curve: Array of equity values over time

    Returns:
        List of dictionaries with drawdown period information
    """
    if len(equity_curve) <= 1:
        return []

    periods = []
    peak = equity_curve[0]
    peak_idx = 0
    in_drawdown = False
    drawdown_start = 0
    max_dd = 0.0

    for i, equity in enumerate(equity_curve):
        if equity > peak:
            # New peak - end any drawdown
            if in_drawdown:
                end_idx = i - 1
                duration = end_idx - drawdown_start + 1
                periods.append(
                    {
                        "start_idx": drawdown_start,
                        "end_idx": end_idx,
                        "drawdown_pct": max_dd,
                        "duration": duration,
                    }
                )
                in_drawdown = False
                max_dd = 0.0
            peak = equity
            peak_idx = i
        elif equity < peak:
            # In drawdown
            if not in_drawdown:
                in_drawdown = True
                drawdown_start = peak_idx

            # Calculate current drawdown
            dd = ((peak - equity) / peak) * 100.0
            if dd > max_dd:
                max_dd = dd

    # Check if still in drawdown at end
    if in_drawdown:
        periods.append(
            {
                "start_idx": drawdown_start,
                "end_idx": len(equity_curve) - 1,
                "drawdown_pct": max_dd,
                "duration": len(equity_curve) - drawdown_start,
            }
        )

    return periods


@dataclass
class RiskMetrics:
    """
    Container for risk metrics.

    Attributes:
        max_drawdown: Maximum drawdown percentage
        max_drawdown_duration: Duration of max drawdown in periods
        drawdown_periods: List of all drawdown periods
        current_drawdown: Current drawdown percentage (if in drawdown)
    """

    max_drawdown: float
    max_drawdown_duration: int
    drawdown_periods: List[Dict]
    current_drawdown: float

    @classmethod
    def from_equity_curve(cls, equity_curve: np.ndarray) -> "RiskMetrics":
        """
        Create RiskMetrics from equity curve.

        Args:
            equity_curve: Array of equity values

        Returns:
            RiskMetrics instance
        """
        max_dd = calculate_max_drawdown(equity_curve)
        duration = calculate_drawdown_duration(equity_curve)
        periods = find_drawdown_periods(equity_curve)

        # Calculate current drawdown (if in drawdown at end)
        if len(equity_curve) > 0:
            current_peak = max(equity_curve)
            current_equity = equity_curve[-1]
            current_dd = (
                ((current_peak - current_equity) / current_peak) * 100.0
                if current_peak > 0
                else 0.0
            )
        else:
            current_dd = 0.0

        return cls(
            max_drawdown=max_dd,
            max_drawdown_duration=duration,
            drawdown_periods=periods,
            current_drawdown=current_dd,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "drawdown_periods": self.drawdown_periods,
            "current_drawdown": self.current_drawdown,
        }
