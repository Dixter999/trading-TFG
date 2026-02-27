"""
Database-integrated Performance Tracker for Paper Trading Engine.

Issue #433: Performance Metrics Calculation
Track 6: Paper Trading

This module provides:
- PerformanceMetrics dataclass for storing calculated metrics
- DBPerformanceTracker for database-integrated performance tracking
- Validation thresholds for paper-trading-ready criteria

Tables used:
- paper_trades: Source of closed trade data
- paper_performance: Storage for aggregated metrics
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class DatabaseManager(Protocol):
    """Protocol for database manager dependency."""

    def execute_query(
        self, db_name: str, query: str, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Execute a query and return results."""
        ...

    def insert_data(self, table: str, data: dict[str, Any]) -> None:
        """Insert data into a table."""
        ...


# Validation thresholds for paper-trading-ready criteria
VALIDATION_THRESHOLDS: dict[str, float | int] = {
    "profit_factor": 1.0,
    "win_rate": 0.40,
    "max_drawdown_pips": 50,
    "min_weekly_trades": 5,
}

# Decimal constants
DECIMAL_ZERO = Decimal("0")
DECIMAL_ONE = Decimal("1")
DECIMAL_HUNDRED = Decimal("100")
DECIMAL_INFINITY = Decimal("999.999")  # Representation for infinity in DB


@dataclass
class PerformanceMetrics:
    """Performance metrics for a trading period.

    All monetary values are in pips for consistency with paper trading.

    Attributes:
        symbol: Trading symbol (e.g., "EURUSD") or "ALL" for overall
        period_start: Start of the measurement period
        period_end: End of the measurement period
        total_trades: Total number of closed trades
        winning_trades: Number of trades with positive P&L
        losing_trades: Number of trades with negative P&L
        total_pnl_pips: Sum of all trade P&L in pips
        gross_profit_pips: Sum of winning trades P&L
        gross_loss_pips: Sum of losing trades P&L (as positive value)
        profit_factor: Gross profit / Gross loss
        win_rate: Winning trades / Total trades as percentage
        avg_win_pips: Average winning trade in pips
        avg_loss_pips: Average losing trade in pips (as positive value)
        max_drawdown_pips: Maximum peak-to-trough decline in pips
        max_consecutive_wins: Longest winning streak
        max_consecutive_losses: Longest losing streak
        expectancy: Expected value per trade in pips
    """

    symbol: str
    period_start: datetime
    period_end: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl_pips: Decimal
    gross_profit_pips: Decimal
    gross_loss_pips: Decimal
    profit_factor: Decimal
    win_rate: Decimal
    avg_win_pips: Decimal
    avg_loss_pips: Decimal
    max_drawdown_pips: Decimal
    max_consecutive_wins: int
    max_consecutive_losses: int
    expectancy: Decimal

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion.

        Returns:
            Dictionary with all fields suitable for database insertion.
        """
        return {
            "symbol": self.symbol,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl_pips": self.total_pnl_pips,
            "gross_profit_pips": self.gross_profit_pips,
            "gross_loss_pips": self.gross_loss_pips,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "avg_win_pips": self.avg_win_pips,
            "avg_loss_pips": self.avg_loss_pips,
            "max_drawdown_pips": self.max_drawdown_pips,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "expectancy": self.expectancy,
        }


class DBPerformanceTracker:
    """Database-integrated performance tracker.

    Reads closed trades from paper_trades table and calculates comprehensive
    performance metrics. Results can be stored in paper_performance table.

    Attributes:
        _db_manager: Database manager for executing queries
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize the performance tracker.

        Args:
            db_manager: Database manager with execute_query and insert_data methods
        """
        self._db_manager = db_manager
        logger.debug("DBPerformanceTracker initialized")

    def calculate_metrics(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> PerformanceMetrics:
        """Calculate performance metrics for a symbol within a time period.

        Queries closed trades from paper_trades table and calculates all
        performance metrics.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            start_date: Start of period (inclusive)
            end_date: End of period (inclusive)

        Returns:
            PerformanceMetrics with all calculated values
        """
        logger.debug(
            "Calculating metrics for %s from %s to %s", symbol, start_date, end_date
        )

        # Query trades from database
        trades = self._get_trades(symbol, start_date, end_date)

        # Calculate metrics from trades
        metrics = self._calculate_metrics_from_trades(trades, symbol, start_date, end_date)

        logger.info(
            "Calculated metrics for %s: %d trades, PF=%.3f, WR=%.1f%%",
            symbol,
            metrics.total_trades,
            float(metrics.profit_factor),
            float(metrics.win_rate),
        )

        return metrics

    def _get_trades(
        self,
        symbol: str | None,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict[str, Any]]:
        """Query trades from paper_trades table.

        Args:
            symbol: Trading symbol or None for all symbols
            start_date: Start of period
            end_date: End of period

        Returns:
            List of trade dictionaries with pnl_pips values
        """
        if symbol is not None:
            query = """
                SELECT id, symbol, pnl_pips
                FROM paper_trades
                WHERE symbol = :symbol
                  AND exit_time >= :start_date
                  AND exit_time <= :end_date
                ORDER BY exit_time ASC
            """
            params = {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
            }
        else:
            query = """
                SELECT id, symbol, pnl_pips
                FROM paper_trades
                WHERE exit_time >= :start_date
                  AND exit_time <= :end_date
                ORDER BY exit_time ASC
            """
            params = {
                "start_date": start_date,
                "end_date": end_date,
            }

        result = self._db_manager.execute_query("ai_model", query, params)
        logger.debug("Retrieved %d trades from database", len(result))
        return result

    @staticmethod
    def _extract_pnl_values(trades: list[dict[str, Any]]) -> list[Decimal]:
        """Extract and convert P&L values from trade dictionaries.

        Args:
            trades: List of trade dictionaries with pnl_pips field

        Returns:
            List of Decimal P&L values
        """
        return [
            Decimal(str(t["pnl_pips"])) if t["pnl_pips"] is not None else DECIMAL_ZERO
            for t in trades
        ]

    def _calculate_metrics_from_trades(
        self,
        trades: list[dict[str, Any]],
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> PerformanceMetrics:
        """Calculate metrics from a list of trades.

        Args:
            trades: List of trade dictionaries with pnl_pips
            symbol: Symbol for the metrics
            start_date: Period start
            end_date: Period end

        Returns:
            Calculated PerformanceMetrics
        """
        if not trades:
            return PerformanceMetrics(
                symbol=symbol,
                period_start=start_date,
                period_end=end_date,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_pnl_pips=DECIMAL_ZERO,
                gross_profit_pips=DECIMAL_ZERO,
                gross_loss_pips=DECIMAL_ZERO,
                profit_factor=DECIMAL_ZERO,
                win_rate=DECIMAL_ZERO,
                avg_win_pips=DECIMAL_ZERO,
                avg_loss_pips=DECIMAL_ZERO,
                max_drawdown_pips=DECIMAL_ZERO,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                expectancy=DECIMAL_ZERO,
            )

        # Extract P&L values using helper method
        pnl_values = self._extract_pnl_values(trades)

        # Basic counts
        total_trades = len(trades)
        winning_trades = sum(1 for pnl in pnl_values if pnl > DECIMAL_ZERO)
        losing_trades = sum(1 for pnl in pnl_values if pnl < DECIMAL_ZERO)

        # P&L aggregates
        total_pnl_pips = sum(pnl_values, DECIMAL_ZERO)
        gross_profit_pips = sum(pnl for pnl in pnl_values if pnl > DECIMAL_ZERO)
        gross_loss_pips = abs(sum(pnl for pnl in pnl_values if pnl < DECIMAL_ZERO))

        # Profit factor
        if gross_loss_pips == DECIMAL_ZERO:
            profit_factor = DECIMAL_INFINITY if gross_profit_pips > DECIMAL_ZERO else DECIMAL_ZERO
        else:
            profit_factor = gross_profit_pips / gross_loss_pips

        # Win rate
        win_rate = (
            Decimal(str(winning_trades)) / Decimal(str(total_trades)) * DECIMAL_HUNDRED
            if total_trades > 0
            else DECIMAL_ZERO
        )

        # Average win/loss
        avg_win_pips = (
            gross_profit_pips / Decimal(str(winning_trades))
            if winning_trades > 0
            else DECIMAL_ZERO
        )
        avg_loss_pips = (
            gross_loss_pips / Decimal(str(losing_trades))
            if losing_trades > 0
            else DECIMAL_ZERO
        )

        # Max drawdown
        max_drawdown_pips = self._calculate_max_drawdown(pnl_values)

        # Consecutive stats
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_stats(
            pnl_values
        )

        # Expectancy: (WR * avg_win) - ((1-WR) * avg_loss)
        win_rate_decimal = win_rate / DECIMAL_HUNDRED
        expectancy = (win_rate_decimal * avg_win_pips) - (
            (DECIMAL_ONE - win_rate_decimal) * avg_loss_pips
        )

        return PerformanceMetrics(
            symbol=symbol,
            period_start=start_date,
            period_end=end_date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl_pips=total_pnl_pips,
            gross_profit_pips=gross_profit_pips,
            gross_loss_pips=gross_loss_pips,
            profit_factor=profit_factor.quantize(Decimal("0.001")),
            win_rate=win_rate.quantize(Decimal("0.1")),
            avg_win_pips=avg_win_pips.quantize(Decimal("0.01")),
            avg_loss_pips=avg_loss_pips.quantize(Decimal("0.01")),
            max_drawdown_pips=max_drawdown_pips,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            expectancy=expectancy.quantize(Decimal("0.01")),
        )

    def _calculate_max_drawdown(self, pnl_values: list[Decimal]) -> Decimal:
        """Calculate maximum drawdown from P&L sequence.

        Drawdown is the peak-to-trough decline during a specific period.

        Args:
            pnl_values: List of trade P&L values in order

        Returns:
            Maximum drawdown in pips
        """
        if not pnl_values:
            return DECIMAL_ZERO

        cumulative = DECIMAL_ZERO
        peak = DECIMAL_ZERO
        max_drawdown = DECIMAL_ZERO

        for pnl in pnl_values:
            cumulative += pnl

            if cumulative > peak:
                peak = cumulative

            drawdown = peak - cumulative
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def _calculate_consecutive_stats(
        self, pnl_values: list[Decimal]
    ) -> tuple[int, int]:
        """Calculate maximum consecutive wins and losses.

        Args:
            pnl_values: List of trade P&L values in order

        Returns:
            Tuple of (max_consecutive_wins, max_consecutive_losses)
        """
        if not pnl_values:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for pnl in pnl_values:
            if pnl > DECIMAL_ZERO:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif pnl < DECIMAL_ZERO:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                # Breakeven resets streaks
                current_wins = 0
                current_losses = 0

        return max_wins, max_losses

    def update_daily_metrics(self, symbol: str) -> None:
        """Update daily performance metrics in paper_performance table.

        Calculates metrics for today and stores them in the database.

        Args:
            symbol: Trading symbol to update
        """
        now = datetime.now(timezone.utc)
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=999999)

        metrics = self.calculate_metrics(symbol, start_of_day, end_of_day)

        # Prepare data for insertion
        data = {
            "symbol": metrics.symbol,
            "period_start": metrics.period_start.date(),
            "period_end": metrics.period_end.date(),
            "total_trades": metrics.total_trades,
            "winning_trades": metrics.winning_trades,
            "total_pnl_pips": metrics.total_pnl_pips,
            "max_drawdown": metrics.max_drawdown_pips,
            "profit_factor": metrics.profit_factor,
            "win_rate": metrics.win_rate,
        }

        self._db_manager.insert_data("paper_performance", data)

    def get_current_drawdown(self, symbol: str) -> Decimal:
        """Get current drawdown from peak for a symbol.

        Calculates from the beginning of available history to now.

        Args:
            symbol: Trading symbol

        Returns:
            Current drawdown in pips
        """
        # Query all historical trades for this symbol
        query = """
            SELECT pnl_pips
            FROM paper_trades
            WHERE symbol = :symbol
              AND exit_time IS NOT NULL
            ORDER BY exit_time ASC
        """
        params = {"symbol": symbol}

        trades = self._db_manager.execute_query("ai_model", query, params)

        if not trades:
            logger.debug("No trades found for %s, returning zero drawdown", symbol)
            return DECIMAL_ZERO

        pnl_values = self._extract_pnl_values(trades)

        # Calculate current drawdown from peak
        cumulative = DECIMAL_ZERO
        peak = DECIMAL_ZERO

        for pnl in pnl_values:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative

        # Current drawdown is peak - current
        current_drawdown = peak - cumulative
        logger.debug("Current drawdown for %s: %.2f pips", symbol, float(current_drawdown))
        return current_drawdown

    def check_validation_thresholds(
        self, symbol: str
    ) -> tuple[bool, dict[str, dict[str, Any]]]:
        """Check if symbol meets paper-trading-ready criteria.

        Validates against VALIDATION_THRESHOLDS for the past week.

        Args:
            symbol: Trading symbol to validate

        Returns:
            Tuple of (all_passed, details) where details contains
            per-threshold results with 'passed', 'value', and 'threshold'.
        """
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)

        metrics = self.calculate_metrics(symbol, week_ago, now)

        details: dict[str, dict[str, Any]] = {}

        # Profit factor check
        pf_threshold = VALIDATION_THRESHOLDS["profit_factor"]
        pf_passed = float(metrics.profit_factor) >= pf_threshold
        details["profit_factor"] = {
            "passed": pf_passed,
            "value": float(metrics.profit_factor),
            "threshold": pf_threshold,
        }

        # Win rate check (threshold is decimal, metrics is percentage)
        wr_threshold = VALIDATION_THRESHOLDS["win_rate"]
        wr_value = float(metrics.win_rate) / 100  # Convert to decimal
        wr_passed = wr_value >= wr_threshold
        details["win_rate"] = {
            "passed": wr_passed,
            "value": wr_value,
            "threshold": wr_threshold,
        }

        # Max drawdown check
        dd_threshold = VALIDATION_THRESHOLDS["max_drawdown_pips"]
        dd_passed = float(metrics.max_drawdown_pips) <= dd_threshold
        details["max_drawdown_pips"] = {
            "passed": dd_passed,
            "value": float(metrics.max_drawdown_pips),
            "threshold": dd_threshold,
        }

        # Minimum weekly trades check
        trades_threshold = VALIDATION_THRESHOLDS["min_weekly_trades"]
        trades_passed = metrics.total_trades >= trades_threshold
        details["min_weekly_trades"] = {
            "passed": trades_passed,
            "value": metrics.total_trades,
            "threshold": trades_threshold,
        }

        all_passed = all(d["passed"] for d in details.values())

        if all_passed:
            logger.info("Symbol %s PASSED all validation thresholds", symbol)
        else:
            failed = [k for k, v in details.items() if not v["passed"]]
            logger.warning("Symbol %s FAILED validation: %s", symbol, ", ".join(failed))

        return all_passed, details

    def get_overall_metrics(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> PerformanceMetrics:
        """Get aggregated metrics across all symbols.

        Args:
            start_date: Optional start of period (default: 30 days ago)
            end_date: Optional end of period (default: now)

        Returns:
            PerformanceMetrics aggregated across all symbols
        """
        now = datetime.now(timezone.utc)

        if end_date is None:
            end_date = now
        if start_date is None:
            start_date = now - timedelta(days=30)

        # Query trades without symbol filter
        trades = self._get_trades(None, start_date, end_date)

        return self._calculate_metrics_from_trades(trades, "ALL", start_date, end_date)
