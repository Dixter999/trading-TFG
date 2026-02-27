"""
Data models for Paper Trading Engine (Issue #328, Issue #333).

This module defines:
- PositionState: Enum for position states (FLAT, LONG, SHORT)
- PositionDirection: Enum for position direction (LONG, SHORT)
- PositionStatus: Enum for position status (OPEN, CLOSED, PARTIAL_CLOSED)
- ExitReason: Enum for trade exit reasons
- PartialClose: Dataclass for partial close records
- Position: Open position data model with multi-TP support
- Trade: Closed trade record model
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

# EURUSD pip value: 1 pip = 0.0001
EURUSD_PIP_VALUE = Decimal("0.0001")

# Pip values for different symbol types (Issue #431, #571)
# XXX/USD pairs: 1 pip = 0.0001
# XXX/JPY pairs: 1 pip = 0.01
# Metals (XAU/XAG): 1 pip = 0.01 (Issue #506)
# CAD pairs: 1 pip = 0.0001 (Issue #571)
PIP_VALUES: dict[str, Decimal] = {
    # Standard USD pairs
    "EURUSD": Decimal("0.0001"),
    "GBPUSD": Decimal("0.0001"),
    "USDCAD": Decimal("0.0001"),
    "USDCHF": Decimal("0.0001"),
    # Cross pairs
    "EURCAD": Decimal("0.0001"),
    "EURGBP": Decimal("0.0001"),
    # JPY pairs (0.01 pip value)
    "USDJPY": Decimal("0.01"),
    "EURJPY": Decimal("0.01"),
    # Metals (0.01 pip value, Issue #506)
    "XAUUSD": Decimal("0.01"),  # Gold
    "XAGUSD": Decimal("0.01"),  # Silver
    "GOLD": Decimal("0.01"),    # Gold alias
    "SILVER": Decimal("0.01"),  # Silver alias
    # Crypto
    "BTCUSD": Decimal("1.0"),   # BTC uses $1 as pip
    # DEFAULT fallback for any unlisted symbol
    "DEFAULT": Decimal("0.0001"),
}

# 30/30 Scaffold Constants (FROZEN - DO NOT MODIFY)
# These define the fixed SL and TP distances in pips for the baseline strategy
SL_PIPS: int = 30
TP_PIPS: int = 30


class PositionState(Enum):
    """Position state machine states."""

    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


class PositionDirection(Enum):
    """Position direction (long or short)."""

    LONG = "long"
    SHORT = "short"


class ExitReason(Enum):
    """Reasons for exiting a position."""

    TP_HIT = "tp_hit"
    SL_HIT = "sl_hit"
    MODEL_EXIT = "model_exit"
    TIMEOUT = "timeout"
    MANUAL = "manual"
    BROKER_CLOSED = "broker_closed"


class PositionStatus(Enum):
    """Status of a trading position."""

    OPEN = "open"
    CLOSED = "closed"
    PARTIAL_CLOSED = "partial_closed"


@dataclass
class PartialClose:
    """Record of a partial position close at a take profit level.

    Attributes:
        timestamp: Time when partial close occurred
        close_price: Price at which the partial close was executed
        size_closed: Size of the portion that was closed
        remaining_size: Size remaining after the partial close
        tp_level: Take profit level (1, 2, or 3)
        pnl_pips: Realized P&L in pips for this partial close
    """

    timestamp: datetime
    close_price: Decimal
    size_closed: Decimal
    remaining_size: Decimal
    tp_level: int
    pnl_pips: Decimal


@dataclass
class Position:
    """Represents an open trading position with multi-TP support.

    Attributes:
        symbol: Trading symbol (e.g., "EURUSD")
        direction: Position direction (LONG or SHORT)
        entry_price: Price at which the position was entered
        entry_time: Timestamp when position was opened
        size: Current position size in lots (may decrease with partial closes)
        tp_price: Take profit price level (optional, legacy single TP)
        sl_price: Stop loss price level (optional)
        id: Unique position identifier (auto-generated)

        Multi-TP fields (Issue #333):
        tp1_price: First take profit price level
        tp2_price: Second take profit price level
        tp3_price: Third take profit price level
        tp1_hit: Flag indicating TP1 was hit
        tp2_hit: Flag indicating TP2 was hit
        tp3_hit: Flag indicating TP3 was hit
        original_size: Original position size before any partial closes
        status: Position status (OPEN, PARTIAL_CLOSED, CLOSED)
        partial_close_history: List of partial close records

        Entry tracking (Issue #495):
        entry_model: Model version used for entry decision
    """

    symbol: str
    direction: PositionDirection
    entry_price: Decimal
    entry_time: datetime
    size: Decimal
    tp_price: Decimal | None = None
    sl_price: Decimal | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Multi-TP fields (Issue #333)
    tp1_price: Decimal | None = None
    tp2_price: Decimal | None = None
    tp3_price: Decimal | None = None
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    original_size: Decimal | None = None
    status: PositionStatus = PositionStatus.OPEN
    partial_close_history: list[PartialClose] = field(default_factory=list)

    # Entry model tracking (Issue #495)
    entry_model: str | None = None

    # Signal timeframe tracking (e.g., "H2", "H4") - stored at open time
    signal_timeframe: str | None = None

    # Live trading fields (Issue #596)
    ticket: int | None = None  # MT5 broker ticket number (None for paper trading)
    is_live: bool = False      # True if opened via MT5 Gateway, False for paper trading

    def __post_init__(self) -> None:
        """Set original_size if not provided."""
        if self.original_size is None:
            self.original_size = self.size

    def calculate_unrealized_pnl_pips(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L in pips.

        For EURUSD: 1 pip = 0.0001

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L in pips (positive = profit, negative = loss)
        """
        price_diff = current_price - self.entry_price

        if self.direction == PositionDirection.LONG:
            # Long position: profit when price goes up
            pnl = price_diff
        else:
            # Short position: profit when price goes down
            pnl = -price_diff

        # Convert to pips using symbol-specific pip value
        pip_value = PIP_VALUES.get(self.symbol, EURUSD_PIP_VALUE)
        pnl_pips = pnl / pip_value
        return pnl_pips

    def is_tp_hit(self, current_price: Decimal) -> bool:
        """Check if take profit level is hit.

        Args:
            current_price: Current market price

        Returns:
            True if TP is hit, False otherwise
        """
        if self.tp_price is None:
            return False

        if self.direction == PositionDirection.LONG:
            # Long position: TP is above entry
            return current_price >= self.tp_price
        else:
            # Short position: TP is below entry
            return current_price <= self.tp_price

    def is_sl_hit(self, current_price: Decimal) -> bool:
        """Check if stop loss level is hit.

        Args:
            current_price: Current market price

        Returns:
            True if SL is hit, False otherwise
        """
        if self.sl_price is None:
            return False

        if self.direction == PositionDirection.LONG:
            # Long position: SL is below entry
            return current_price <= self.sl_price
        else:
            # Short position: SL is above entry
            return current_price >= self.sl_price

    def is_sl_hit_realtime(self, bid: Decimal, ask: Decimal) -> bool:
        """Check SL using proper bid/ask for direction (real-time).

        This method uses the correct bid/ask price based on position direction:
        - LONG position: SL hit when BID <= sl_price (we sell at bid)
        - SHORT position: SL hit when ASK >= sl_price (we buy at ask)

        This is more accurate than using a single "current price" because
        it accounts for the spread and uses the actual execution price.

        Args:
            bid: Current bid price (price at which we can sell)
            ask: Current ask price (price at which we can buy)

        Returns:
            True if SL is hit, False otherwise
        """
        if self.sl_price is None:
            return False

        if self.direction == PositionDirection.LONG:
            # LONG position: SL hit when BID drops to/below SL price
            # We would exit by selling at the bid price
            return bid <= self.sl_price
        else:
            # SHORT position: SL hit when ASK rises to/above SL price
            # We would exit by buying at the ask price
            return ask >= self.sl_price

    def is_tp_hit_realtime(self, bid: Decimal, ask: Decimal) -> bool:
        """Check TP using proper bid/ask for direction (real-time).

        This method uses the correct bid/ask price based on position direction:
        - LONG position: TP hit when BID >= tp_price (we sell at bid)
        - SHORT position: TP hit when ASK <= tp_price (we buy at ask)

        This is more accurate than using a single "current price" because
        it accounts for the spread and uses the actual execution price.

        Args:
            bid: Current bid price (price at which we can sell)
            ask: Current ask price (price at which we can buy)

        Returns:
            True if TP is hit, False otherwise
        """
        if self.tp_price is None:
            return False

        if self.direction == PositionDirection.LONG:
            # LONG position: TP hit when BID rises to/above TP price
            # We would exit by selling at the bid price
            return bid >= self.tp_price
        else:
            # SHORT position: TP hit when ASK drops to/below TP price
            # We would exit by buying at the ask price
            return ask <= self.tp_price


@dataclass
class Trade:
    """Represents a closed trade record.

    Attributes:
        symbol: Trading symbol (e.g., "EURUSD")
        direction: Position direction (LONG or SHORT)
        entry_time: Timestamp when position was opened
        exit_time: Timestamp when position was closed
        entry_price: Price at which the position was entered
        exit_price: Price at which the position was closed
        size: Position size in lots
        tp_price: Take profit price level (optional)
        sl_price: Stop loss price level (optional)
        pnl_pips: Realized P&L in pips
        exit_reason: Reason for closing the position
        model_version: Version of the model that made exit decision (optional)
        entry_model: Model version used for entry decision (copied from position)
        id: Unique trade identifier (auto-generated)
    """

    symbol: str
    direction: PositionDirection
    entry_time: datetime
    exit_time: datetime
    entry_price: Decimal
    exit_price: Decimal
    size: Decimal
    pnl_pips: Decimal
    exit_reason: ExitReason
    tp_price: Decimal | None = None
    sl_price: Decimal | None = None
    model_version: str | None = None
    entry_model: str | None = None  # Preserve entry strategy info from position
    signal_timeframe: str | None = None  # Timeframe of the signal that opened this trade
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def from_position(
        cls,
        position: Position,
        exit_price: Decimal,
        exit_time: datetime,
        exit_reason: ExitReason,
        model_version: str | None = None,
    ) -> "Trade":
        """Create a Trade record from a closed Position.

        Args:
            position: The position being closed
            exit_price: Price at which position is closed
            exit_time: Time of exit
            exit_reason: Reason for exit
            model_version: Optional model version identifier

        Returns:
            Trade record representing the closed position
        """
        pnl_pips = position.calculate_unrealized_pnl_pips(exit_price)

        return cls(
            symbol=position.symbol,
            direction=position.direction,
            entry_time=position.entry_time,
            exit_time=exit_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            tp_price=position.tp_price,
            sl_price=position.sl_price,
            pnl_pips=pnl_pips,
            exit_reason=exit_reason,
            model_version=model_version,
            entry_model=position.entry_model,  # Preserve entry strategy info
            signal_timeframe=position.signal_timeframe,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert trade to dictionary for database insertion.

        Returns:
            Dictionary with all trade fields, enums converted to strings
        """
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "tp_price": self.tp_price,
            "sl_price": self.sl_price,
            "pnl_pips": self.pnl_pips,
            "exit_reason": self.exit_reason.value,
            "model_version": self.model_version,
            "entry_model": self.entry_model,  # Preserve entry strategy info
            "signal_timeframe": self.signal_timeframe,
        }
