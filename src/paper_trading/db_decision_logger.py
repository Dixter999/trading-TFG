"""
Trade Decision DB Logger for Paper Trading Engine.

Issue #427: Trade Logging and Decision Recording
Stream B: Core Logger Implementation

This module provides PostgreSQL logging for all trading decisions,
including entry signals (accepted and rejected), position opens/closes,
and risk violations. All decisions are stored with full context for
audit and analysis.

Table Schema (paper_decision_log):
    id SERIAL PRIMARY KEY
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
    log_type VARCHAR(30) NOT NULL
    symbol VARCHAR(20)
    direction VARCHAR(10)
    entry_price DECIMAL(18, 8)
    signal_confidence DECIMAL(6, 4)
    rejection_reason VARCHAR(100)
    context_data JSONB
    created_at TIMESTAMP DEFAULT NOW()
"""

import json
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from psycopg2.extras import Json


class LogEntryType(Enum):
    """Types of log entries for trade decisions.

    Attributes:
        SIGNAL_GENERATED: An entry signal was generated and accepted
        SIGNAL_REJECTED: An entry signal was generated but rejected
        POSITION_OPENED: A position was opened
        POSITION_CLOSED: A position was closed
        RISK_VIOLATION: A risk rule was violated
        SYSTEM_ERROR: A system error occurred
        SYMBOL_DISABLED: A symbol was disabled due to missing validated model
    """

    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_REJECTED = "signal_rejected"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    RISK_VIOLATION = "risk_violation"
    SYSTEM_ERROR = "system_error"
    SYMBOL_DISABLED = "symbol_disabled"
    RL_EXIT_DECISION = "rl_exit_decision"


class TradeDecisionDBLogger:
    """Logs all trading decisions to PostgreSQL for audit and analysis.

    This class provides:
    - Signal logging (accepted and rejected)
    - Position open/close logging
    - Risk violation logging
    - Decision history querying with filters

    Attributes:
        db: Reference to database session for database operations

    Example:
        >>> db = DatabaseManager()
        >>> logger = TradeDecisionDBLogger(db_session=db)
        >>> logger.log_signal(signal, accepted=True)
    """

    TABLE_NAME = "paper_decision_log"

    def __init__(self, db_session: Any) -> None:
        """Initialize TradeDecisionDBLogger with database session.

        Args:
            db_session: Database session/manager for database operations
        """
        self.db = db_session

    def log_signal(
        self,
        signal: Any,
        accepted: bool,
        reason: str | None = None,
    ) -> None:
        """Log entry signal generation.

        Logs both accepted and rejected signals with full context.

        Args:
            signal: Signal object containing symbol, direction, entry_price, etc.
            accepted: Whether the signal was accepted (True) or rejected (False)
            reason: Reason for rejection (only used when accepted=False)
        """
        log_type = (
            LogEntryType.SIGNAL_GENERATED if accepted else LogEntryType.SIGNAL_REJECTED
        )

        # Get timestamp from signal or use current UTC time
        timestamp = getattr(signal, "timestamp", None)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Get direction - could be string or enum
        direction = getattr(signal, "direction", None)
        if direction is not None and hasattr(direction, "value"):
            direction = direction.value
        elif direction is not None:
            direction = str(direction)

        # Build context data
        context_data = self._build_signal_context(signal)

        data = {
            "timestamp": timestamp,
            "log_type": log_type.value,
            "symbol": getattr(signal, "symbol", None),
            "direction": direction,
            "entry_price": getattr(signal, "entry_price", None),
            "signal_confidence": getattr(signal, "confidence", None),
            "rejection_reason": reason if not accepted else None,
            "context_data": Json(context_data),
        }

        self.db.insert_data(self.TABLE_NAME, data)

    def log_position_open(self, position: Any, signal: Any) -> None:
        """Log position open with full context.

        Args:
            position: Position object being opened
            signal: Signal that triggered the position open
        """
        # Get timestamp from position entry_time or use current UTC time
        timestamp = getattr(position, "entry_time", None)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Get direction - could be enum with .value
        direction = getattr(position, "direction", None)
        if direction is not None and hasattr(direction, "value"):
            direction = direction.value
        elif direction is not None:
            direction = str(direction)

        # Build context data including both position and signal info
        context_data = self._build_position_open_context(position, signal)

        data = {
            "timestamp": timestamp,
            "log_type": LogEntryType.POSITION_OPENED.value,
            "symbol": getattr(position, "symbol", None),
            "direction": direction,
            "entry_price": getattr(position, "entry_price", None),
            "signal_confidence": getattr(signal, "confidence", None),
            "rejection_reason": None,
            "context_data": Json(context_data),
        }

        self.db.insert_data(self.TABLE_NAME, data)

    def log_position_close(self, trade: Any) -> None:
        """Log position close with result.

        Args:
            trade: Trade object representing the closed position
        """
        # Get timestamp from trade exit_time or use current UTC time
        timestamp = getattr(trade, "exit_time", None)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Get direction - could be enum with .value
        direction = getattr(trade, "direction", None)
        if direction is not None and hasattr(direction, "value"):
            direction = direction.value
        elif direction is not None:
            direction = str(direction)

        # Build context data with trade result
        context_data = self._build_position_close_context(trade)

        data = {
            "timestamp": timestamp,
            "log_type": LogEntryType.POSITION_CLOSED.value,
            "symbol": getattr(trade, "symbol", None),
            "direction": direction,
            "entry_price": getattr(trade, "entry_price", None),
            "signal_confidence": None,
            "rejection_reason": None,
            "context_data": Json(context_data),
        }

        self.db.insert_data(self.TABLE_NAME, data)

    def log_risk_violation(self, signal: Any, rule: str) -> None:
        """Log risk rule violation.

        Args:
            signal: Signal that would have been traded
            rule: Name of the risk rule that was violated
        """
        # Get timestamp from signal or use current UTC time
        timestamp = getattr(signal, "timestamp", None)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Get direction - could be string or enum
        direction = getattr(signal, "direction", None)
        if direction is not None and hasattr(direction, "value"):
            direction = direction.value
        elif direction is not None:
            direction = str(direction)

        # Build context data
        context_data = self._build_signal_context(signal)
        context_data["violated_rule"] = rule

        data = {
            "timestamp": timestamp,
            "log_type": LogEntryType.RISK_VIOLATION.value,
            "symbol": getattr(signal, "symbol", None),
            "direction": direction,
            "entry_price": getattr(signal, "entry_price", None),
            "signal_confidence": getattr(signal, "confidence", None),
            "rejection_reason": rule,
            "context_data": Json(context_data),
        }

        self.db.insert_data(self.TABLE_NAME, data)

    def log_symbol_disabled(self, symbol: str, reason: str) -> None:
        """Log when a symbol is disabled due to missing validated model.

        This is a SAFETY event - records that a symbol was blocked from trading
        because no validated model exists for it.

        Args:
            symbol: Trading symbol that was disabled (e.g., "XAUUSD", "BTCUSD")
            reason: Reason the symbol was disabled
        """
        timestamp = datetime.now(timezone.utc)

        context_data = {
            "disabled_reason": reason,
            "safety_action": "symbol_disabled",
            "severity": "critical",
        }

        data = {
            "timestamp": timestamp,
            "log_type": LogEntryType.SYMBOL_DISABLED.value,
            "symbol": symbol,
            "direction": None,
            "entry_price": None,
            "signal_confidence": None,
            "rejection_reason": reason,
            "context_data": Json(context_data),
        }

        self.db.insert_data(self.TABLE_NAME, data)

    def log_exit_decision(self, exit_signal: Any, position: Any) -> None:
        """Log RL exit evaluation decision (HOLD or CLOSE) with ensemble vote breakdown.

        Args:
            exit_signal: HybridExitSignal with action, confidence, ensemble_meta
            position: Position object being evaluated
        """
        timestamp = getattr(exit_signal, "timestamp", None)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        direction = getattr(exit_signal, "direction", None)
        if direction is None:
            direction = getattr(position, "direction", None)
            if direction is not None and hasattr(direction, "value"):
                direction = direction.value

        action_str = "CLOSE" if getattr(exit_signal, "action", 0) == 1 else "HOLD"
        context_data = {
            "action": action_str,
            "confidence": self._serialize_value(getattr(exit_signal, "confidence", None)),
            "reason": getattr(exit_signal, "reason", None),
            "position_bars": getattr(exit_signal, "position_bars", None),
            "unrealized_pnl_pips": self._serialize_value(
                getattr(exit_signal, "unrealized_pnl_pips", None)
            ),
            "model_version": getattr(exit_signal, "model_version", None),
            "position_id": getattr(position, "id", None),
            "entry_price": self._serialize_value(getattr(position, "entry_price", None)),
        }

        ensemble_meta = getattr(exit_signal, "ensemble_meta", None)
        if ensemble_meta:
            context_data["ensemble_votes"] = ensemble_meta

        data = {
            "timestamp": timestamp,
            "log_type": LogEntryType.RL_EXIT_DECISION.value,
            "symbol": getattr(exit_signal, "symbol", None),
            "direction": direction,
            "entry_price": getattr(position, "entry_price", None),
            "signal_confidence": getattr(exit_signal, "confidence", None),
            "rejection_reason": None,
            "context_data": Json(context_data),
        }

        self.db.insert_data(self.TABLE_NAME, data)

    def get_decision_history(
        self,
        symbol: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        log_type: LogEntryType | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query decision history with optional filters.

        Args:
            symbol: Filter by trading symbol (e.g., "EURUSD")
            start_date: Filter by timestamp >= start_date
            end_date: Filter by timestamp <= end_date
            log_type: Filter by log entry type
            limit: Maximum number of records to return

        Returns:
            List of decision history records as dictionaries
        """
        conditions = []
        params: dict[str, Any] = {}

        if symbol:
            conditions.append("symbol = :symbol")
            params["symbol"] = symbol

        if start_date:
            conditions.append("timestamp >= :start_date")
            params["start_date"] = start_date

        if end_date:
            conditions.append("timestamp <= :end_date")
            params["end_date"] = end_date

        if log_type:
            conditions.append("log_type = :log_type")
            params["log_type"] = log_type.value

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT * FROM {self.TABLE_NAME}
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT :limit
        """
        params["limit"] = limit

        return self.db.execute_query("ai_model", query, params)

    def _build_signal_context(self, signal: Any) -> dict:
        """Build context data dictionary from signal object.

        Args:
            signal: Signal object

        Returns:
            Dictionary with signal context data
        """
        context: dict[str, Any] = {}

        # Extract common signal attributes
        for attr in [
            "indicators",
            "pattern_type",
            "timeframe",
            "model_confidence",
            "ensemble_agreement",
        ]:
            value = getattr(signal, attr, None)
            if value is not None:
                context[attr] = self._serialize_value(value)

        return context

    def _build_position_open_context(self, position: Any, signal: Any) -> dict:
        """Build context data for position open.

        Args:
            position: Position object
            signal: Signal that triggered the open

        Returns:
            Dictionary with position and signal context
        """
        context: dict[str, Any] = {}

        # Position data
        for attr in [
            "id",
            "size",
            "tp_price",
            "sl_price",
            "tp1_price",
            "tp2_price",
            "tp3_price",
            "original_size",
        ]:
            value = getattr(position, attr, None)
            if value is not None:
                context[f"position_{attr}"] = self._serialize_value(value)

        # Signal data
        signal_context = self._build_signal_context(signal)
        for key, value in signal_context.items():
            context[f"signal_{key}"] = value

        return context

    def _build_position_close_context(self, trade: Any) -> dict:
        """Build context data for position close.

        Args:
            trade: Trade object representing closed position

        Returns:
            Dictionary with trade result context
        """
        context: dict[str, Any] = {}

        # Trade result data
        for attr in [
            "id",
            "entry_time",
            "exit_time",
            "exit_price",
            "size",
            "pnl_pips",
            "exit_reason",
            "model_version",
            "tp_price",
            "sl_price",
        ]:
            value = getattr(trade, attr, None)
            if value is not None:
                context[attr] = self._serialize_value(value)

        return context

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for JSON storage.

        Handles Decimals, datetimes, Enums, and other special types.

        Args:
            value: Value to serialize

        Returns:
            JSON-serializable value
        """
        if isinstance(value, Decimal):
            return str(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._serialize_value(v) for v in value]
        else:
            return value
