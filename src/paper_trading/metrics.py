"""
Performance metrics calculation for paper trading (Issue #517).

Provides comprehensive trading metrics including:
- Win/Loss statistics (win rate, avg win, avg loss)
- Expectancy calculation
- Profit Factor
- Sharpe Ratio (risk-adjusted return)
- Maximum Drawdown (pips and %)
- Trading costs (spread deduction)

Author: python-backend-engineer
Created: 2025-01-04
Phase: REFACTOR - Improved code structure and documentation
"""

import statistics
from decimal import Decimal
from typing import Any


class PerformanceMetrics:
    """Calculate comprehensive performance metrics from trade history.

    This class provides methods to analyze trading performance across multiple
    dimensions including profitability, risk-adjusted returns, and drawdown analysis.

    Args:
        trades: List of trade dictionaries with at least 'pnl' and 'outcome' fields
        initial_balance_pips: Starting balance in pips (for drawdown % calculation)

    Example:
        >>> trades = [
        ...     {"pnl": Decimal("30"), "outcome": "win"},
        ...     {"pnl": Decimal("25"), "outcome": "win"},
        ...     {"pnl": Decimal("-30"), "outcome": "loss"},
        ... ]
        >>> metrics = PerformanceMetrics(trades)
        >>> metrics.calculate_expectancy()
        Decimal('13.33')
    """

    def __init__(
        self,
        trades: list[dict[str, Any]],
        initial_balance_pips: Decimal = Decimal("10000")
    ):
        self.trades = trades
        self.initial_balance_pips = initial_balance_pips
        self._wins = [t["pnl"] for t in trades if t["outcome"] == "win"]
        self._losses = [t["pnl"] for t in trades if t["outcome"] == "loss"]

    def calculate_win_rate(self) -> float:
        """Calculate win rate percentage.

        Returns:
            Win rate as percentage (0-100)
        """
        if not self.trades:
            return 0.0

        wins = sum(1 for t in self.trades if t["outcome"] == "win")
        return (wins / len(self.trades)) * 100

    def calculate_average_win(self) -> Decimal:
        """Calculate average winning trade in pips.

        Returns:
            Average win in pips (0 if no wins)
        """
        if not self._wins:
            return Decimal("0")

        return sum(self._wins) / len(self._wins)

    def calculate_average_loss(self) -> Decimal:
        """Calculate average losing trade in pips.

        Returns:
            Average loss in pips (negative value, 0 if no losses)
        """
        if not self._losses:
            return Decimal("0")

        return sum(self._losses) / len(self._losses)

    def calculate_expectancy(self) -> Decimal:
        """Calculate expectancy (expected pips per trade).

        Formula: (Avg Win × Win%) - (Avg Loss × Loss%)

        Returns:
            Expectancy in pips per trade
        """
        if not self.trades:
            return Decimal("0")

        win_rate = Decimal(str(self.calculate_win_rate())) / Decimal("100")
        loss_rate = Decimal("1") - win_rate

        avg_win = self.calculate_average_win()
        avg_loss = abs(self.calculate_average_loss())

        expectancy = (avg_win * win_rate) - (avg_loss * loss_rate)
        return expectancy

    def calculate_profit_factor(self) -> Decimal | float:
        """Calculate profit factor (total wins / total losses).

        The profit factor measures the ratio of gross profits to gross losses.
        Values >1.0 indicate profitability, >1.2 is considered good.

        Returns:
            Profit factor (Decimal or float('inf') if no losses)
        """
        if not self.trades:
            return Decimal("0")

        total_wins = sum(self._wins) if self._wins else Decimal("0")
        total_losses = abs(sum(self._losses)) if self._losses else Decimal("0")

        if total_losses == 0:
            return float("inf")  # No losses - infinite profit factor

        if total_wins == 0:
            return Decimal("0")  # No wins - zero profit factor

        return total_wins / total_losses

    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (risk-adjusted return).

        The Sharpe ratio measures risk-adjusted performance by dividing the mean
        return by the standard deviation of returns. Higher values indicate better
        risk-adjusted performance.

        Formula: Mean(returns) / StdDev(returns)

        Returns:
            Sharpe ratio (0 if insufficient data or zero volatility)
        """
        if not self.trades or len(self.trades) < 2:
            return 0.0

        returns = [float(t["pnl"]) for t in self.trades]
        mean_return = statistics.mean(returns)

        try:
            std_dev = statistics.stdev(returns)
            if std_dev == 0:
                return 0.0  # Zero volatility - undefined Sharpe
            return mean_return / std_dev
        except statistics.StatisticsError:
            return 0.0

    def calculate_max_drawdown_pips(self) -> Decimal:
        """Calculate maximum drawdown in pips.

        Maximum drawdown measures the largest peak-to-trough decline in cumulative
        PnL during the trading period. This is a key risk metric.

        Returns:
            Maximum peak-to-trough decline in pips
        """
        if not self.trades:
            return Decimal("0")

        # Build cumulative PnL curve
        cumulative_pnl = self._calculate_cumulative_pnl()

        # Track peak and max drawdown
        max_dd = Decimal("0")
        peak = cumulative_pnl[0]

        for pnl in cumulative_pnl:
            peak = max(peak, pnl)
            drawdown = peak - pnl
            max_dd = max(max_dd, drawdown)

        return max_dd

    def _calculate_cumulative_pnl(self) -> list[Decimal]:
        """Calculate cumulative PnL curve.

        Returns:
            List of cumulative PnL values after each trade
        """
        cumulative = []
        current = Decimal("0")
        for trade in self.trades:
            current += trade["pnl"]
            cumulative.append(current)
        return cumulative

    def calculate_max_drawdown_percentage(self) -> float:
        """Calculate maximum drawdown as percentage of peak equity.

        Expresses the maximum drawdown as a percentage of the peak equity value.
        This normalizes the drawdown to account for account size.

        Returns:
            Maximum drawdown percentage (0-100)
        """
        if not self.trades:
            return 0.0

        # Build equity curve including initial balance
        equity = [self.initial_balance_pips]
        for trade in self.trades:
            equity.append(equity[-1] + trade["pnl"])

        # Track peak and max drawdown percentage
        max_dd_pct = 0.0
        peak = equity[0]

        for eq in equity:
            peak = max(peak, eq)

            if peak > 0:
                drawdown_pct = float((peak - eq) / peak * 100)
                max_dd_pct = max(max_dd_pct, drawdown_pct)

        return max_dd_pct

    def calculate_total_pnl(self) -> Decimal:
        """Calculate total profit/loss in pips.

        Returns:
            Total PnL in pips
        """
        if not self.trades:
            return Decimal("0")

        return sum(t["pnl"] for t in self.trades)

    def apply_spread_cost(
        self,
        total_pnl: Decimal,
        spread_pips: Decimal = Decimal("2.0")
    ) -> Decimal:
        """Apply spread cost to total PnL.

        Args:
            total_pnl: Total PnL before costs
            spread_pips: Spread cost per trade in pips

        Returns:
            Total PnL after spread costs
        """
        num_trades = len(self.trades)
        total_spread_cost = Decimal(str(num_trades)) * spread_pips
        return total_pnl - total_spread_cost

    def calculate_risk_reward_ratio(self) -> Decimal | float:
        """Calculate risk/reward ratio (avg win / avg loss).

        The R:R ratio measures the average reward per unit of risk. A ratio >1.0
        means average wins are larger than average losses.

        Returns:
            R:R ratio (Decimal or float('inf') if no losses)
        """
        avg_win = self.calculate_average_win()
        avg_loss = abs(self.calculate_average_loss())

        if avg_loss == 0:
            return float("inf")  # No losses - infinite R:R

        return avg_win / avg_loss

    def get_all_metrics(
        self,
        spread_pips: Decimal = Decimal("2.0")
    ) -> dict[str, Any]:
        """Get comprehensive metrics dictionary.

        Args:
            spread_pips: Spread cost per trade in pips

        Returns:
            Dictionary with all calculated metrics
        """
        total_pnl = self.calculate_total_pnl()

        return {
            "total_trades": len(self.trades),
            "win_rate": self.calculate_win_rate(),
            "avg_win": self.calculate_average_win(),
            "avg_loss": self.calculate_average_loss(),
            "expectancy": self.calculate_expectancy(),
            "profit_factor": self.calculate_profit_factor(),
            "sharpe_ratio": self.calculate_sharpe_ratio(),
            "max_drawdown_pips": self.calculate_max_drawdown_pips(),
            "max_drawdown_pct": self.calculate_max_drawdown_percentage(),
            "total_pnl": total_pnl,
            "total_pnl_after_spread": self.apply_spread_cost(total_pnl, spread_pips),
            "risk_reward_ratio": self.calculate_risk_reward_ratio(),
        }
