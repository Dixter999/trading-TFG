"""
Backtesting reporting and comparison module.

This module provides functions for comparing backtest configurations,
generating reports, and creating visualization data.
"""

from typing import Any, TypedDict

from backtesting.models import BacktestResults

# Report formatting constants
HEADER_WIDTH = 60
SECTION_SEPARATOR = "=" * HEADER_WIDTH
SUBSECTION_SEPARATOR = "-" * HEADER_WIDTH


class ComparisonReport(TypedDict):
    """Type definition for comparison report."""

    rankings: list[dict[str, Any]]
    best_config: dict[str, Any] | None
    comparisons: dict[str, float]


class EquityCurveData(TypedDict):
    """Type definition for equity curve data."""

    timestamps: list[str]
    values: list[float]
    config_name: str


def rank_configurations(
    results: dict[str, BacktestResults]
) -> list[tuple[str, BacktestResults]]:
    """
    Rank configurations by Sharpe ratio in descending order.

    Args:
        results: Dictionary mapping config names to BacktestResults.

    Returns:
        List of (config_name, BacktestResults) tuples sorted by Sharpe ratio descending.
    """
    if not results:
        return []

    # Sort by Sharpe ratio descending
    ranked = sorted(
        results.items(), key=lambda x: x[1].metrics.sharpe_ratio, reverse=True
    )

    return ranked


def compare_configurations(results: dict[str, BacktestResults]) -> ComparisonReport:
    """
    Compare multiple backtest configurations.

    Args:
        results: Dictionary mapping config names to BacktestResults.

    Returns:
        Comparison report dictionary containing:
        - rankings: List of configs sorted by performance
        - best_config: Best performing configuration
        - comparisons: Metric differences between configs
    """
    if not results:
        return ComparisonReport(
            rankings=[], best_config=None, comparisons={}
        )

    # Rank configurations
    ranked = rank_configurations(results)

    # Build rankings list with key metrics
    rankings = [
        {
            "name": name,
            "sharpe_ratio": result.metrics.sharpe_ratio,
            "win_rate": result.metrics.win_rate,
            "profit_factor": result.metrics.profit_factor,
            "total_pnl": result.metrics.total_pnl,
        }
        for name, result in ranked
    ]

    # Best config is first in rankings
    best_config = rankings[0] if rankings else None

    # Calculate comparisons (differences from best to worst)
    comparisons: dict[str, float] = {}
    if len(ranked) >= 2:
        best_result = ranked[0][1]
        worst_result = ranked[-1][1]

        comparisons = {
            "sharpe_ratio_diff": best_result.metrics.sharpe_ratio
            - worst_result.metrics.sharpe_ratio,
            "win_rate_diff": best_result.metrics.win_rate
            - worst_result.metrics.win_rate,
            "profit_factor_diff": best_result.metrics.profit_factor
            - worst_result.metrics.profit_factor,
            "total_pnl_diff": best_result.metrics.total_pnl
            - worst_result.metrics.total_pnl,
        }

    return ComparisonReport(
        rankings=rankings, best_config=best_config, comparisons=comparisons
    )


def generate_equity_curve_data(results: BacktestResults) -> EquityCurveData:
    """
    Generate equity curve visualization data.

    Args:
        results: Backtest results to extract equity curve from.

    Returns:
        Dictionary containing:
        - timestamps: List of ISO format timestamp strings
        - values: List of equity values
        - config_name: Configuration name
    """
    # Convert timestamps to ISO format strings
    timestamps = [ts.isoformat() for ts in results.timestamps]

    return EquityCurveData(
        timestamps=timestamps,
        values=results.equity_curve,
        config_name=results.config_name,
    )


def generate_backtest_report(results: BacktestResults) -> str:
    """
    Generate formatted text report for backtest results.

    Args:
        results: Backtest results to generate report from.

    Returns:
        Formatted text report string with all sections and metrics.
    """
    metrics = results.metrics

    # Format dates
    start_date = results.start_date.strftime("%Y-%m-%d")
    end_date = results.end_date.strftime("%Y-%m-%d")

    # Build report sections
    report_lines = [
        "CONFLUENCE BACKTEST RESULTS",
        SECTION_SEPARATOR,
        "",
        f"Configuration: {results.config_name}",
        f"Period: {start_date} to {end_date}",
        f"Symbol: {results.symbol} {results.timeframe}",
        "",
        "PROFITABILITY",
        SUBSECTION_SEPARATOR,
        f"Total P&L:        {metrics.total_pnl:,.2f} pips",
        f"Win Rate:         {metrics.win_rate * 100:.2f}%",
        f"Profit Factor:    {metrics.profit_factor:.2f}",
        f"Total Trades:     {metrics.total_trades}",
        "",
        "RISK-ADJUSTED RETURNS",
        SUBSECTION_SEPARATOR,
        f"Sharpe Ratio:     {metrics.sharpe_ratio:.2f}",
        f"Sortino Ratio:    {metrics.sortino_ratio:.2f}",
        f"Max Drawdown:     {metrics.max_drawdown:.2f}%",
        f"Drawdown Duration: {metrics.drawdown_duration} candles",
        "",
        "TRADE STATISTICS",
        SUBSECTION_SEPARATOR,
        f"Average Win:      {metrics.avg_win:.2f} pips",
        f"Average Loss:     {metrics.avg_loss:.2f} pips",
        f"Consecutive Wins: {metrics.consecutive_wins}",
        f"Consecutive Losses: {metrics.consecutive_losses}",
        f"Avg Trade Duration: {metrics.avg_trade_duration:.2f} hours",
        "",
        "SIGNAL ANALYSIS",
        SUBSECTION_SEPARATOR,
        f"Total Signals:    {metrics.total_signals}",
        f"Signals/Day:      {metrics.signals_per_day:.2f}",
        f"Winning Signals:  {int(metrics.successful_signals * metrics.total_signals)} ({metrics.successful_signals * 100:.2f}%)",
        f"Avg Confidence:   {metrics.avg_confidence:.2f}",
        "",
    ]

    return "\n".join(report_lines)
