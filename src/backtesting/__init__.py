"""
Backtesting framework for trading strategies.

This package provides comprehensive backtesting capabilities for evaluating
confluence-based trading strategies on historical data.
"""

from backtesting.models import BacktestMetrics, BacktestResults, Trade

__all__ = [
    "Trade",
    "BacktestMetrics",
    "BacktestResults",
]
