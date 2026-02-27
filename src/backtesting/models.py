"""
Backtesting data models.

This module defines dataclasses for backtesting, including Trade, BacktestMetrics,
and BacktestResults. These models support the complete backtesting framework.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pattern_system.confluence.models import ConfluenceSignal


@dataclass
class Trade:
    """
    Single trade execution record.

    Attributes:
        entry_time: Timestamp when trade was entered.
        entry_price: Price at trade entry.
        exit_time: Timestamp when trade was exited.
        exit_price: Price at trade exit.
        direction: Trade direction - "LONG" or "SHORT".
        quantity: Position size (lots or units).
        pnl: Profit/loss in pips or currency units.
        pnl_pct: Profit/loss as percentage of entry price.
        signal: ConfluenceSignal that triggered the trade.
        exit_reason: Why trade was closed - "TAKE_PROFIT", "STOP_LOSS", or "TIME_EXIT".
    """

    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    direction: str
    quantity: float
    pnl: float
    pnl_pct: float
    signal: ConfluenceSignal
    exit_reason: str

    def __post_init__(self):
        """Validate Trade fields after initialization."""
        # Validate direction
        valid_directions = {"LONG", "SHORT"}
        if self.direction not in valid_directions:
            raise ValueError(
                f"direction must be one of {valid_directions}, got '{self.direction}'"
            )

        # Validate exit_reason
        valid_exit_reasons = {"TAKE_PROFIT", "STOP_LOSS", "TIME_EXIT"}
        if self.exit_reason not in valid_exit_reasons:
            raise ValueError(
                f"exit_reason must be one of {valid_exit_reasons}, "
                f"got '{self.exit_reason}'"
            )

        # Validate exit_time is after entry_time
        if self.exit_time <= self.entry_time:
            raise ValueError(
                f"exit_time must be after entry_time. "
                f"Got entry_time={self.entry_time}, exit_time={self.exit_time}"
            )

        # Validate quantity is positive
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert Trade to dictionary.

        Returns:
            Dictionary representation of the trade.
        """
        return {
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_time": self.exit_time.isoformat(),
            "exit_price": self.exit_price,
            "direction": self.direction,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "signal": self.signal.to_dict(),
            "exit_reason": self.exit_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trade":
        """
        Create Trade from dictionary.

        Args:
            data: Dictionary containing trade fields.

        Returns:
            Trade instance.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If field values are invalid.
        """
        return cls(
            entry_time=datetime.fromisoformat(data["entry_time"]),
            entry_price=data["entry_price"],
            exit_time=datetime.fromisoformat(data["exit_time"]),
            exit_price=data["exit_price"],
            direction=data["direction"],
            quantity=data["quantity"],
            pnl=data["pnl"],
            pnl_pct=data["pnl_pct"],
            signal=ConfluenceSignal.from_dict(data["signal"]),
            exit_reason=data["exit_reason"],
        )


@dataclass
class BacktestMetrics:
    """
    Comprehensive backtest performance metrics.

    Attributes:
        total_pnl: Total profit/loss in currency or pips.
        total_pnl_pct: Total P&L as percentage.
        win_count: Number of winning trades.
        loss_count: Number of losing trades.
        win_rate: Percentage of winning trades (0.0-1.0).
        profit_factor: Gross profit / gross loss ratio.
        payoff_ratio: Average win / average loss ratio.
        sharpe_ratio: Risk-adjusted return (excess return / volatility).
        sortino_ratio: Downside risk-adjusted return.
        calmar_ratio: Return / max drawdown ratio.
        max_drawdown: Maximum peak-to-trough decline (negative value).
        avg_drawdown: Average drawdown (negative value).
        drawdown_duration: Longest drawdown period in candles.
        total_trades: Total number of trades executed.
        avg_win: Average profit per winning trade.
        avg_loss: Average loss per losing trade (negative value).
        consecutive_wins: Maximum consecutive winning trades.
        consecutive_losses: Maximum consecutive losing trades.
        avg_trade_duration: Average time in trade (hours).
        total_signals: Total signals generated (traded + filtered).
        signals_per_day: Average signals per trading day.
        avg_confidence: Average signal confidence score (0.0-1.0).
        successful_signals: Percentage of signals that were profitable (0.0-1.0).
    """

    # Profitability
    total_pnl: float
    total_pnl_pct: float
    win_count: int
    loss_count: int
    win_rate: float
    profit_factor: float
    payoff_ratio: float

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: int

    # Trade statistics
    total_trades: int
    avg_win: float
    avg_loss: float
    consecutive_wins: int
    consecutive_losses: int
    avg_trade_duration: float

    # Signal metrics
    total_signals: int
    signals_per_day: float
    avg_confidence: float
    successful_signals: float

    def __post_init__(self):
        """Validate BacktestMetrics fields after initialization."""
        # Validate win_rate range
        if not 0.0 <= self.win_rate <= 1.0:
            raise ValueError(
                f"win_rate must be between 0.0 and 1.0, got {self.win_rate}"
            )

        # Validate total_trades is non-negative
        if self.total_trades < 0:
            raise ValueError(
                f"total_trades must be non-negative, got {self.total_trades}"
            )

        # Validate avg_confidence range
        if not 0.0 <= self.avg_confidence <= 1.0:
            raise ValueError(
                f"avg_confidence must be between 0.0 and 1.0, got {self.avg_confidence}"
            )

        # Validate successful_signals range
        if not 0.0 <= self.successful_signals <= 1.0:
            raise ValueError(
                f"successful_signals must be between 0.0 and 1.0, "
                f"got {self.successful_signals}"
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert BacktestMetrics to dictionary.

        Returns:
            Dictionary representation of metrics.
        """
        return {
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl_pct,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "payoff_ratio": self.payoff_ratio,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "avg_drawdown": self.avg_drawdown,
            "drawdown_duration": self.drawdown_duration,
            "total_trades": self.total_trades,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "avg_trade_duration": self.avg_trade_duration,
            "total_signals": self.total_signals,
            "signals_per_day": self.signals_per_day,
            "avg_confidence": self.avg_confidence,
            "successful_signals": self.successful_signals,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BacktestMetrics":
        """
        Create BacktestMetrics from dictionary.

        Args:
            data: Dictionary containing metrics fields.

        Returns:
            BacktestMetrics instance.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If field values are invalid.
        """
        return cls(
            total_pnl=data["total_pnl"],
            total_pnl_pct=data["total_pnl_pct"],
            win_count=data["win_count"],
            loss_count=data["loss_count"],
            win_rate=data["win_rate"],
            profit_factor=data["profit_factor"],
            payoff_ratio=data["payoff_ratio"],
            sharpe_ratio=data["sharpe_ratio"],
            sortino_ratio=data["sortino_ratio"],
            calmar_ratio=data["calmar_ratio"],
            max_drawdown=data["max_drawdown"],
            avg_drawdown=data["avg_drawdown"],
            drawdown_duration=data["drawdown_duration"],
            total_trades=data["total_trades"],
            avg_win=data["avg_win"],
            avg_loss=data["avg_loss"],
            consecutive_wins=data["consecutive_wins"],
            consecutive_losses=data["consecutive_losses"],
            avg_trade_duration=data["avg_trade_duration"],
            total_signals=data["total_signals"],
            signals_per_day=data["signals_per_day"],
            avg_confidence=data["avg_confidence"],
            successful_signals=data["successful_signals"],
        )


@dataclass
class BacktestResults:
    """
    Complete backtest results for a configuration.

    Attributes:
        config_name: Name of the configuration tested (e.g., "3-Pattern Confluence").
        symbol: Trading symbol (e.g., "EURUSD").
        timeframe: Candle timeframe (e.g., "H1", "D1").
        start_date: Backtest start date.
        end_date: Backtest end date.
        pattern_types: List of pattern types used (e.g., ["pin_bar", "supply_demand"]).
        regime_aware: Whether regime filtering was applied.
        metrics: Aggregate performance metrics.
        trades: List of executed trades.
        signals: List of generated signals (all, not just traded).
        equity_curve: Running equity values over time.
        timestamps: Timestamps corresponding to equity_curve values.
    """

    config_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    pattern_types: list[str]
    regime_aware: bool

    metrics: BacktestMetrics
    trades: list[Trade]
    signals: list[ConfluenceSignal]
    equity_curve: list[float]
    timestamps: list[datetime]

    def __post_init__(self):
        """Validate BacktestResults fields after initialization."""
        # Validate config_name is not empty
        if not self.config_name or not self.config_name.strip():
            raise ValueError("config_name cannot be empty")

        # Validate end_date is after start_date
        if self.end_date <= self.start_date:
            raise ValueError(
                f"end_date must be after start_date. "
                f"Got start_date={self.start_date}, end_date={self.end_date}"
            )

        # Validate pattern_types is not empty
        if not self.pattern_types:
            raise ValueError("pattern_types cannot be empty")

        # Validate equity_curve and timestamps have same length
        if len(self.equity_curve) != len(self.timestamps):
            raise ValueError(
                f"equity_curve and timestamps must have same length. "
                f"Got equity_curve={len(self.equity_curve)}, "
                f"timestamps={len(self.timestamps)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert BacktestResults to dictionary.

        Returns:
            Dictionary representation of results.
        """
        return {
            "config_name": self.config_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "pattern_types": self.pattern_types,
            "regime_aware": self.regime_aware,
            "metrics": self.metrics.to_dict(),
            "trades": [t.to_dict() for t in self.trades],
            "signals": [s.to_dict() for s in self.signals],
            "equity_curve": self.equity_curve,
            "timestamps": [t.isoformat() for t in self.timestamps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BacktestResults":
        """
        Create BacktestResults from dictionary.

        Args:
            data: Dictionary containing results fields.

        Returns:
            BacktestResults instance.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If field values are invalid.
        """
        return cls(
            config_name=data["config_name"],
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            start_date=datetime.fromisoformat(data["start_date"]),
            end_date=datetime.fromisoformat(data["end_date"]),
            pattern_types=data["pattern_types"],
            regime_aware=data["regime_aware"],
            metrics=BacktestMetrics.from_dict(data["metrics"]),
            trades=[Trade.from_dict(t) for t in data["trades"]],
            signals=[ConfluenceSignal.from_dict(s) for s in data["signals"]],
            equity_curve=data["equity_curve"],
            timestamps=[datetime.fromisoformat(t) for t in data["timestamps"]],
        )
