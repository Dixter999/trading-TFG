"""
Position Manager for Paper Trading Engine (Issue #328).

This module provides the PositionManager class for tracking open positions
with a state machine approach: FLAT -> LONG/SHORT -> FLAT

State Machine:
    FLAT -> [Entry Signal] -> LONG/SHORT
    LONG/SHORT -> [Exit Signal/TP/SL Hit] -> FLAT
"""

from __future__ import annotations

import logging
import threading
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from src.paper_trading.models import (
    EURUSD_PIP_VALUE,
    PIP_VALUES,
    SL_PIPS,
    TP_PIPS,
    ExitReason,
    PartialClose,
    Position,
    PositionDirection,
    PositionState,
    PositionStatus,
    Trade,
)

if TYPE_CHECKING:
    from src.database.connection_manager import DatabaseManager

logger = logging.getLogger(__name__)


class PositionManager:
    """Manages trading positions with bidirectional support (Issue #510).

    This class provides:
    - Position state machine (FLAT -> LONG/SHORT -> FLAT)
    - Bidirectional trading: supports both LONG and SHORT positions per symbol
    - P&L calculation
    - Position lifecycle: open -> update -> close
    - Database persistence (paper_positions and paper_trades tables)

    Attributes:
        _positions: Dictionary mapping "symbol_direction" to open Position
        db: Optional database session for persistence
    """

    POSITIONS_TABLE = "paper_positions"
    TRADES_TABLE = "paper_trades"

    def __init__(self, db_session: DatabaseManager | None = None) -> None:
        """Initialize PositionManager with positions loaded from database.

        Args:
            db_session: Optional database session for persistence.
                        If provided, positions and trades will be persisted to DB.
        """
        # Changed from Dict[str, Position] to support bidirectional trading
        # Key format: "SYMBOL_DIRECTION" (e.g., "EURUSD_LONG", "EURUSD_SHORT")
        self._positions: dict[str, Position] = {}
        self.db = db_session

        # Lifecycle client for performance reporting (set externally)
        self.lifecycle_client: Any = None
        # Live performance tracker for adaptive sizing (set externally)
        self.live_tracker: Any = None
        # Per-signal trade counters for rolling WR: {signal_id: {"wins": N, "total": N}}
        self._signal_trade_counts: dict[str, dict[str, int]] = {}

        # Issue #628: Lock to prevent race conditions when closing positions
        # This prevents duplicate trades when RealtimeSLChecker and main loop
        # try to close the same position simultaneously
        self._close_lock = threading.Lock()

        # Load existing positions from database on startup
        if self.db is not None:
            self._load_positions_from_db()

    def _load_positions_from_db(self) -> None:
        """Load existing open positions from database on startup.

        This ensures that positions are tracked after service restart.
        Positions are stored by their UUID key (Issue #631+).
        """
        if self.db is None:
            return

        try:
            query = f"""
                SELECT id, symbol, direction, entry_time, entry_price,
                       sl_price, tp_price, size, entry_model, ticket, is_live,
                       signal_timeframe
                FROM {self.POSITIONS_TABLE}
            """
            result = self.db.execute_query("ai_model", query, {})

            if not result:
                logger.info("No existing positions found in database")
                return

            loaded_count = 0
            for row in result:
                try:
                    # Parse direction
                    direction = PositionDirection(row["direction"])

                    # Create Position object with database ID if available
                    position = Position(
                        symbol=row["symbol"],
                        direction=direction,
                        entry_price=Decimal(str(row["entry_price"])),
                        entry_time=row["entry_time"],
                        size=Decimal(str(row["size"])),
                        sl_price=Decimal(str(row["sl_price"])) if row["sl_price"] else None,
                        tp_price=Decimal(str(row["tp_price"])) if row["tp_price"] else None,
                        entry_model=row.get("entry_model"),
                        signal_timeframe=row.get("signal_timeframe"),
                        ticket=row.get("ticket"),
                        is_live=row.get("is_live", False),
                    )

                    # If row has an id from database, use it; otherwise use the auto-generated one
                    if row.get("id"):
                        # Replace the auto-generated ID with the one from DB
                        object.__setattr__(position, 'id', str(row["id"]))

                    # Store by position UUID (Issue #631+)
                    self._positions[position.id] = position
                    loaded_count += 1

                    logger.info(
                        f"Loaded position from DB: {position.symbol} {position.direction.value} "
                        f"entry={position.entry_price} SL={position.sl_price} TP={position.tp_price} "
                        f"id={position.id}"
                    )
                except Exception as e:
                    logger.error(f"Failed to load position row: {row}, error: {e}")

            logger.info(f"Loaded {loaded_count} positions from database")

        except Exception as e:
            logger.error(f"Failed to load positions from database: {e}")

    @staticmethod
    def _make_position_key(symbol: str, direction: PositionDirection | str) -> str:
        """Create compound key for position storage.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            direction: Position direction (LONG or SHORT)

        Returns:
            Compound key in format "SYMBOL_DIRECTION"
        """
        dir_str = direction.value if isinstance(direction, PositionDirection) else direction
        return f"{symbol}_{dir_str.upper()}"

    def _persist_position(self, position: Position) -> None:
        """Persist position to database.

        Args:
            position: Position to persist
        """
        if self.db is None:
            return

        try:
            # Don't include 'id' - let database auto-generate it (id is INTEGER)
            data = {
                "symbol": position.symbol,
                "direction": position.direction.value,
                "entry_time": position.entry_time,
                "entry_price": position.entry_price,
                "sl_price": position.sl_price,
                "tp_price": position.tp_price,
                "size": position.size,
                "unrealized_pnl": Decimal("0"),
                "updated_at": datetime.now(timezone.utc),
                "entry_model": position.entry_model,
                "signal_timeframe": position.signal_timeframe,
                "ticket": position.ticket,
                "is_live": position.is_live,
            }
            self.db.insert_data(self.POSITIONS_TABLE, data)

            # Fetch the auto-generated id and update the position
            if position.ticket:
                query = f"""
                    SELECT id FROM {self.POSITIONS_TABLE}
                    WHERE ticket = :ticket
                    ORDER BY id DESC LIMIT 1
                """
                params = {"ticket": position.ticket}
            else:
                query = f"""
                    SELECT id FROM {self.POSITIONS_TABLE}
                    WHERE symbol = :symbol AND direction = :direction
                    ORDER BY entry_time DESC LIMIT 1
                """
                params = {
                    "symbol": position.symbol,
                    "direction": position.direction.value
                }
            result = self.db.execute_query("ai_model", query, params)
            if result and len(result) > 0:
                db_id = str(result[0]["id"])
                old_id = position.id
                object.__setattr__(position, 'id', db_id)
                # Replace old UUID key with DB integer key
                if old_id != db_id and old_id in self._positions:
                    del self._positions[old_id]
                self._positions[db_id] = position

            logger.info(f"Persisted position to DB: {position.symbol} {position.direction.value} model={position.entry_model} id={position.id}")
        except Exception as e:
            logger.error(f"Failed to persist position to DB: {e}")

    def _delete_position_from_db(self, position_id: str) -> None:
        """Delete position from database by position ID.

        Args:
            position_id: ID of the position to delete (integer string from DB)
        """
        if self.db is None:
            return

        try:
            # Check if position_id is a valid integer (from DB) or UUID (not in DB)
            try:
                int(position_id)  # Will succeed for DB integer ids like "316"
                is_db_id = True
            except ValueError:
                is_db_id = False  # UUID format, not in DB

            if is_db_id:
                query = f"""
                    DELETE FROM {self.POSITIONS_TABLE}
                    WHERE id = :position_id
                """
                self.db.execute_query(
                    "ai_model",
                    query,
                    {"position_id": int(position_id)}  # Cast to int for DB
                )
                logger.info(f"Deleted position from DB: id={position_id}")
            else:
                logger.info(f"Skipping DB delete for UUID position: id={position_id[:8]}...")
        except Exception as e:
            logger.error(f"Failed to delete position from DB: {e}")

    def _persist_trade(self, trade: Trade) -> None:
        """Persist completed trade to database.

        Args:
            trade: Trade to persist

        Note:
            Issue #628: Added deduplication check to prevent duplicate trades
            when multiple processes (RealtimeSLChecker + main loop) try to
            close the same position simultaneously.
        """
        if self.db is None:
            return

        try:
            # Issue #628+: Check for duplicate trade before inserting.
            # Primary check: same symbol+direction+entry_time = same position,
            # regardless of exit_time (catches closures via different code paths).
            dedup_query = f"""
                SELECT id, exit_reason FROM {self.TRADES_TABLE}
                WHERE symbol = :symbol
                  AND direction = :direction
                  AND entry_time = :entry_time
                LIMIT 1
            """

            existing = self.db.execute_query(
                "ai_model",
                dedup_query,
                {
                    "symbol": trade.symbol,
                    "direction": trade.direction.value,
                    "entry_time": trade.entry_time,
                },
            )

            if existing and len(existing) > 0:
                logger.warning(
                    f"Duplicate trade detected, skipping insert: "
                    f"{trade.symbol} {trade.direction.value} "
                    f"entry={trade.entry_time} exit={trade.exit_time} "
                    f"(existing id={existing[0].get('id')}, "
                    f"exit_reason={existing[0].get('exit_reason')})"
                )
                return

            data = {
                "symbol": trade.symbol,
                "direction": trade.direction.value,
                "entry_time": trade.entry_time,
                "entry_price": trade.entry_price,
                "exit_time": trade.exit_time,
                "exit_price": trade.exit_price,
                "sl_price": trade.sl_price,
                "tp_price": trade.tp_price,
                "size": trade.size,
                "pnl_pips": trade.pnl_pips,
                "exit_reason": trade.exit_reason.value if trade.exit_reason else None,
                "entry_model": trade.entry_model,  # Preserve entry strategy info
                "signal_timeframe": trade.signal_timeframe,
                "created_at": datetime.now(timezone.utc),
            }
            self.db.insert_data(self.TRADES_TABLE, data)
            logger.info(
                f"Persisted trade to DB: {trade.symbol} {trade.direction.value} "
                f"PnL={trade.pnl_pips} pips model={trade.entry_model}"
            )

            # Update live performance tracker for adaptive sizing
            if self.live_tracker is not None:
                try:
                    from src.paper_trading.live_performance_tracker import LivePerformanceTracker
                    sig_key = LivePerformanceTracker.normalize_signal_key(
                        symbol=trade.symbol,
                        direction=trade.direction.value,
                        entry_model=trade.entry_model,
                    )
                    if sig_key is not None:
                        pnl = float(trade.pnl_pips) if trade.pnl_pips is not None else 0.0
                        self.live_tracker.record_trade(sig_key, pnl)
                        logger.info(
                            f"LiveTracker updated: {sig_key} pnl={pnl:.1f} "
                            f"weight={self.live_tracker.get_weight(sig_key):.3f}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to update live tracker: {e}")

            # Report performance to lifecycle if client available
            self._report_lifecycle_performance(trade)
        except Exception as e:
            logger.error(f"Failed to persist trade to DB: {e}")

    def _report_lifecycle_performance(self, trade: Trade) -> None:
        """Report signal performance to lifecycle after trade close.

        Tracks per-signal win/loss counts and reports rolling WR
        every 10th trade for a signal.

        Args:
            trade: Completed trade with entry_model containing signal info
        """
        if self.lifecycle_client is None:
            return

        # Derive signal_id from entry_model (format: {symbol}_{direction}_{signal}_{tf})
        signal_id = trade.entry_model
        if not signal_id:
            return

        try:
            # Update per-signal counters
            if signal_id not in self._signal_trade_counts:
                self._signal_trade_counts[signal_id] = {"wins": 0, "total": 0}

            counts = self._signal_trade_counts[signal_id]
            counts["total"] += 1
            if trade.pnl_pips is not None and trade.pnl_pips > 0:
                counts["wins"] += 1

            # Report every 10th trade
            if self.lifecycle_client.should_report_performance(counts["total"]):
                wr = self.lifecycle_client.calculate_rolling_wr(
                    wins=counts["wins"], total=counts["total"]
                )
                self.lifecycle_client.report_signal_performance(
                    signal_id=signal_id,
                    current_wr=wr,
                    total_trades=counts["total"],
                )
                logger.info(
                    f"Lifecycle performance reported: {signal_id} "
                    f"WR={wr:.1%} trades={counts['total']}"
                )
        except Exception as e:
            logger.warning(f"Failed to report lifecycle performance: {e}")

    def get_state(self, symbol: str) -> PositionState:
        """Get the current state for a symbol.

        With multiple positions per symbol feature, returns the state based on
        the first position found for the symbol. If both LONG and SHORT positions
        exist, returns LONG (LONG takes precedence).

        Args:
            symbol: Trading symbol (e.g., "EURUSD")

        Returns:
            PositionState.FLAT if no position, otherwise LONG or SHORT
        """
        # Search for positions with this symbol
        positions = self.get_positions(symbol)
        if not positions:
            return PositionState.FLAT

        # Check if any LONG positions exist (takes precedence)
        for pos in positions:
            if pos.direction == PositionDirection.LONG:
                return PositionState.LONG

        # Otherwise return SHORT (if we have any SHORT positions)
        return PositionState.SHORT

    def get_positions(self, symbol: str, direction: PositionDirection | str | None = None) -> list[Position]:
        """Get all positions for a symbol, optionally filtered by direction.

        This method supports the multiple positions per symbol feature (Issue #631+).

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            direction: Optional position direction filter (LONG or SHORT).
                       If None, returns ALL positions for the symbol.

        Returns:
            List of Position objects matching the criteria
        """
        result = []
        for position in self._positions.values():
            if position.symbol == symbol:
                if direction is None:
                    result.append(position)
                else:
                    dir_value = direction.value if isinstance(direction, PositionDirection) else direction
                    if position.direction.value.upper() == dir_value.upper():
                        result.append(position)
        return result

    def get_position(self, symbol: str, direction: PositionDirection | str | None = None) -> Position | None:
        """Get the first open position for a symbol and direction.

        For backward compatibility, returns the first matching position.
        Use get_positions() to get all matching positions when multiple exist.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            direction: Optional position direction (LONG or SHORT).
                       If None, returns first position found for symbol (backward compatibility).

        Returns:
            Position if exists, None otherwise
        """
        # Use get_positions and return first match
        positions = self.get_positions(symbol, direction)
        return positions[0] if positions else None

    def has_position(self, symbol: str, direction: PositionDirection | str | None = None) -> bool:
        """Check if a position exists for a symbol and optionally a direction.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            direction: Optional position direction (LONG or SHORT).
                       If None, checks if ANY position exists for symbol.

        Returns:
            True if at least one matching position exists, False otherwise
        """
        # Use get_positions to check for existence
        positions = self.get_positions(symbol, direction)
        return len(positions) > 0

    def force_remove_positions(
        self, symbol: str, direction: PositionDirection | str | None = None
    ) -> int:
        """Force-remove all in-memory positions for a symbol/direction.

        Used for ghost position cleanup when broker confirms position
        no longer exists but local state is stuck.

        Args:
            symbol: Trading symbol
            direction: Optional direction filter

        Returns:
            Number of positions removed
        """
        keys_to_remove = []
        for k, v in self._positions.items():
            if v.symbol == symbol:
                if direction is None:
                    keys_to_remove.append(k)
                else:
                    dir_value = direction.value if isinstance(direction, PositionDirection) else direction
                    if v.direction.value.upper() == dir_value.upper():
                        keys_to_remove.append(k)
        for k in keys_to_remove:
            del self._positions[k]
        if keys_to_remove:
            logger.info(
                f"Force-removed {len(keys_to_remove)} ghost position(s) for "
                f"{symbol} {direction or 'ALL'}: keys={keys_to_remove}"
            )
        return len(keys_to_remove)

    def open_position(
        self,
        symbol: str,
        direction: PositionDirection,
        entry_price: Decimal,
        entry_time: datetime,
        size: Decimal,
        tp_price: Decimal | None = None,
        sl_price: Decimal | None = None,
        tp1_price: Decimal | None = None,
        tp2_price: Decimal | None = None,
        tp3_price: Decimal | None = None,
        entry_model: str | None = None,
        signal_timeframe: str | None = None,
        ticket: int | None = None,      # NEW: MT5 broker ticket
        is_live: bool = False,          # NEW: Live trading flag
    ) -> Position:
        """Open a new position for a symbol with optional multi-TP levels.

        Multiple positions per symbol are now supported (Issue #631+).
        Each position is stored by its unique UUID, allowing multiple
        positions with the same symbol+direction from different timeframes/signals.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            direction: Position direction (LONG or SHORT)
            entry_price: Price at which position is entered
            entry_time: Time of entry
            size: Position size in lots
            tp_price: Optional legacy single take profit price
            sl_price: Optional stop loss price
            tp1_price: Optional first take profit price (50% close)
            tp2_price: Optional second take profit price (30% close)
            tp3_price: Optional third take profit price (20% close)
            entry_model: Optional model version used for entry (Issue #495)
            signal_timeframe: Optional signal timeframe (e.g., "H2", "H4")
            ticket: Optional MT5 broker ticket
            is_live: Optional live trading flag

        Returns:
            The newly created Position
        """
        # Multiple positions per symbol now allowed - no duplicate check
        # Each position stored by its unique UUID

        position = Position(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=entry_time,
            size=size,
            tp_price=tp_price,
            sl_price=sl_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            tp3_price=tp3_price,
            entry_model=entry_model,
            signal_timeframe=signal_timeframe,
            ticket=ticket,
            is_live=is_live,
        )

        # Store by position UUID, not compound key (Issue #631+)
        self._positions[position.id] = position

        # Persist to database
        self._persist_position(position)

        return position

    def close_position(
        self,
        symbol: str,
        exit_price: Decimal,
        exit_time: datetime,
        exit_reason: ExitReason,
        model_version: str | None = None,
        direction: PositionDirection | str | None = None,
        position_id: str | None = None,
    ) -> Trade:
        """Close an existing position and return a Trade record.

        Supports closing by position_id for multiple positions per symbol (Issue #631+).

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            exit_price: Price at which position is closed
            exit_time: Time of exit
            exit_reason: Reason for closing the position
            model_version: Optional model version identifier
            direction: Optional direction to close (for bidirectional trading).
                       If None and no position_id, closes first position found for symbol.
            position_id: Optional position UUID to close a specific position.
                        Takes precedence over symbol+direction lookup.

        Returns:
            Trade record for the closed position

        Raises:
            ValueError: If no position exists for the criteria

        Note:
            Issue #628: Uses lock to prevent race conditions when multiple
            processes (RealtimeSLChecker + main loop) try to close the same
            position simultaneously.
        """
        # Issue #628: Acquire lock to prevent race condition between
        # RealtimeSLChecker and main loop's check_exits
        with self._close_lock:
            # Get position - by ID if provided, otherwise by symbol/direction
            if position_id:
                position = self._positions.get(position_id)
                if position is None:
                    raise ValueError(f"No position exists with id {position_id}")
            else:
                # Backward compatible: get by symbol+direction
                position = self.get_position(symbol, direction)
                if position is None:
                    dir_str = f" {direction}" if direction else ""
                    raise ValueError(f"No position exists for {symbol}{dir_str}")

            trade = Trade.from_position(
                position=position,
                exit_price=exit_price,
                exit_time=exit_time,
                exit_reason=exit_reason,
                model_version=model_version,
            )

            # Delete from in-memory dict using position UUID (Issue #631+)
            # Handle stale keys: position may be stored under a different key
            # than position.id (e.g., old UUID key after DB id replacement)
            if position.id in self._positions:
                del self._positions[position.id]
            else:
                # Brute-force: find and remove by object identity
                keys_to_remove = [
                    k for k, v in self._positions.items() if v is position
                ]
                for k in keys_to_remove:
                    del self._positions[k]
                if not keys_to_remove:
                    logger.warning(
                        f"Position {position.symbol} id={position.id} not found "
                        f"in _positions dict during close (already removed?)"
                    )

            # Persist to database - delete position and save trade
            self._delete_position_from_db(position.id)
            self._persist_trade(trade)

            return trade

    def update_position(
        self,
        symbol: str,
        tp_price: Decimal | None = None,
        sl_price: Decimal | None = None,
        direction: PositionDirection | str | None = None,
        position_id: str | None = None,
    ) -> Position:
        """Update TP/SL levels for an existing position.

        Supports updating by position_id for multiple positions per symbol (Issue #631+).

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            tp_price: New take profit price (None to keep current)
            sl_price: New stop loss price (None to keep current)
            direction: Optional direction (for bidirectional trading).
                       If None and no position_id, updates first position found for symbol.
            position_id: Optional position UUID to update a specific position.

        Returns:
            Updated Position

        Raises:
            ValueError: If no position exists for the criteria
        """
        # Get position by ID if provided, otherwise by symbol/direction
        if position_id:
            position = self._positions.get(position_id)
            if position is None:
                raise ValueError(f"No position exists with id {position_id}")
        else:
            position = self.get_position(symbol, direction)
            if position is None:
                dir_str = f" {direction}" if direction else ""
                raise ValueError(f"No position exists for {symbol}{dir_str}")

        # Use dataclass replace to create updated position
        updates = {}
        if tp_price is not None:
            updates["tp_price"] = tp_price
        if sl_price is not None:
            updates["sl_price"] = sl_price

        if updates:
            updated_position = replace(position, **updates)
            # Use position UUID as key (Issue #631+)
            self._positions[position.id] = updated_position
            return updated_position

        return position

    def get_unrealized_pnl_pips(
        self, symbol: str, current_price: Decimal, direction: PositionDirection | str | None = None
    ) -> Decimal | None:
        """Get unrealized P&L in pips for a position.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            current_price: Current market price
            direction: Optional direction (for bidirectional trading).
                       If None, uses first position found for symbol.

        Returns:
            Unrealized P&L in pips, or None if no position exists
        """
        position = self.get_position(symbol, direction)
        if position is None:
            return None

        return position.calculate_unrealized_pnl_pips(current_price)

    def update_unrealized_pnl(self, current_prices: dict[str, Decimal]) -> None:
        """Update unrealized P&L in database for all open positions.

        This method should be called periodically to keep the frontend
        dashboard updated with live P&L values.

        Args:
            current_prices: Dictionary mapping symbol to current price
                           (e.g., {"EURCAD": Decimal("1.62180")})
        """
        if self.db is None:
            return

        # Issue #631+: _positions uses UUID keys
        for position_id, position in self._positions.items():
            if position.symbol not in current_prices:
                continue

            current_price = current_prices[position.symbol]
            pnl_pips = position.calculate_unrealized_pnl_pips(current_price)

            try:
                # Check if position.id is a valid integer (from DB) or UUID (in-memory only)
                try:
                    int(position.id)  # Will succeed for DB integer ids like "316"
                    is_db_id = True
                except ValueError:
                    is_db_id = False  # UUID format, not in DB yet

                if is_db_id:
                    # Update by position ID (for positions loaded from DB)
                    query = f"""
                        UPDATE {self.POSITIONS_TABLE}
                        SET unrealized_pnl = :pnl, updated_at = :updated_at
                        WHERE id = :position_id
                    """
                    params = {
                        "pnl": pnl_pips,
                        "position_id": int(position.id),  # Cast to int for DB
                        "updated_at": datetime.now(timezone.utc),
                    }
                    self.db.execute_query("ai_model", query, params)
                    logger.debug(f"Updated unrealized P&L for {position.symbol} {position.direction.value} (id={position.id}): {pnl_pips:.1f} pips")
                else:
                    # Position has UUID id (not saved to DB yet), skip DB update
                    logger.debug(f"Skipping DB update for {position.symbol} (UUID id={position.id[:8]}...)")
            except Exception as e:
                logger.error(f"Failed to update unrealized P&L for {position.symbol}: {e}")

    def get_all_positions(self) -> dict[str, Position]:
        """Get all open positions.

        Returns:
            Dictionary mapping symbol to Position
        """
        return dict(self._positions)

    def partial_close_at_tp(
        self,
        symbol: str,
        tp_level: int,
        close_price: Decimal,
        close_percentage: Decimal,
    ) -> PartialClose:
        """Execute partial close at take profit level.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            tp_level: Which TP level hit (1, 2, or 3)
            close_price: Current market price
            close_percentage: Percentage of original size to close (0.5, 0.3, or 0.2)

        Returns:
            PartialClose object with details

        Raises:
            ValueError: If no position exists, invalid TP level, or TP already hit
        """
        position = self._positions.get(symbol)
        if position is None:
            raise ValueError(f"No position exists for {symbol}")

        # Validate TP level
        if tp_level not in (1, 2, 3):
            raise ValueError(f"Invalid TP level: {tp_level}. Must be 1, 2, or 3")

        # Check if TP level already hit
        if tp_level == 1 and position.tp1_hit:
            raise ValueError("TP level 1 already hit")
        if tp_level == 2 and position.tp2_hit:
            raise ValueError("TP level 2 already hit")
        if tp_level == 3 and position.tp3_hit:
            raise ValueError("TP level 3 already hit")

        # Calculate size to close based on original size
        size_to_close = position.original_size * close_percentage
        remaining_size = position.size - size_to_close

        # Calculate P&L for the closed portion
        pnl_pips = self._calculate_pnl_for_size(position, close_price, size_to_close)

        # Create partial close record
        partial_close = PartialClose(
            timestamp=datetime.now(timezone.utc),
            close_price=close_price,
            size_closed=size_to_close,
            remaining_size=remaining_size,
            tp_level=tp_level,
            pnl_pips=pnl_pips,
        )

        # Update position
        tp1_hit = position.tp1_hit or (tp_level == 1)
        tp2_hit = position.tp2_hit or (tp_level == 2)
        tp3_hit = position.tp3_hit or (tp_level == 3)

        # Determine new status
        if remaining_size <= Decimal("0"):
            # Position fully closed - remove it
            del self._positions[symbol]
        else:
            # Update position with partial close
            new_history = list(position.partial_close_history)
            new_history.append(partial_close)

            updated_position = replace(
                position,
                size=remaining_size,
                tp1_hit=tp1_hit,
                tp2_hit=tp2_hit,
                tp3_hit=tp3_hit,
                status=PositionStatus.PARTIAL_CLOSED,
                partial_close_history=new_history,
            )
            self._positions[symbol] = updated_position

        return partial_close

    def check_multi_tp(self, position: Position, current_price: Decimal) -> int | None:
        """Check if any TP level should trigger.

        Returns the lowest unhit TP level that current price has reached.
        TPs must trigger in order (TP1 before TP2, TP2 before TP3).

        Args:
            position: The position to check
            current_price: Current market price

        Returns:
            TP level (1, 2, 3) if hit, None otherwise
        """
        if position.direction == PositionDirection.LONG:
            # LONG: TP hit when price >= TP level
            if not position.tp1_hit and position.tp1_price is not None:
                if current_price >= position.tp1_price:
                    return 1
            if (
                position.tp1_hit
                and not position.tp2_hit
                and position.tp2_price is not None
            ):
                if current_price >= position.tp2_price:
                    return 2
            if (
                position.tp2_hit
                and not position.tp3_hit
                and position.tp3_price is not None
            ):
                if current_price >= position.tp3_price:
                    return 3
        else:
            # SHORT: TP hit when price <= TP level
            if not position.tp1_hit and position.tp1_price is not None:
                if current_price <= position.tp1_price:
                    return 1
            if (
                position.tp1_hit
                and not position.tp2_hit
                and position.tp2_price is not None
            ):
                if current_price <= position.tp2_price:
                    return 2
            if (
                position.tp2_hit
                and not position.tp3_hit
                and position.tp3_price is not None
            ):
                if current_price <= position.tp3_price:
                    return 3

        return None

    def _calculate_pnl_for_size(
        self,
        position: Position,
        close_price: Decimal,
        size: Decimal,
    ) -> Decimal:
        """Calculate P&L in pips for a specific size.

        Args:
            position: The position
            close_price: Price at which to calculate P&L
            size: Size for which to calculate P&L

        Returns:
            P&L in pips for the given size
        """
        price_diff = close_price - position.entry_price

        if position.direction == PositionDirection.LONG:
            pnl = price_diff
        else:
            pnl = -price_diff

        # Convert to pips using symbol-specific pip value
        pip_value = PIP_VALUES.get(position.symbol, EURUSD_PIP_VALUE)
        pnl_pips = pnl / pip_value
        return pnl_pips

    def calculate_exit_prices(
        self, entry_price: Decimal, direction: PositionDirection, symbol: str
    ) -> tuple[Decimal, Decimal]:
        """Calculate SL and TP prices using 30/30 scaffold (Issue #431).

        Uses the frozen 30/30 pip scaffold where:
        - Stop Loss: 30 pips from entry (against position direction)
        - Take Profit: 30 pips from entry (with position direction)

        Args:
            entry_price: The entry price of the position
            direction: Position direction (LONG or SHORT)
            symbol: Trading symbol (e.g., "EURUSD", "USDJPY")

        Returns:
            Tuple of (sl_price, tp_price)

        Raises:
            ValueError: If symbol is not recognized
        """
        # Get pip value from PIP_VALUES with DEFAULT fallback (Issue #571)
        # PIP_VALUES now has all symbols + DEFAULT key
        pip_value = PIP_VALUES.get(symbol, PIP_VALUES.get("DEFAULT", EURUSD_PIP_VALUE))
        pip_offset = pip_value * SL_PIPS  # 30 pips in price terms

        if direction == PositionDirection.LONG:
            # LONG: SL below entry, TP above entry
            sl_price = entry_price - pip_offset
            tp_price = entry_price + pip_offset
        else:
            # SHORT: SL above entry, TP below entry
            sl_price = entry_price + pip_offset
            tp_price = entry_price - pip_offset

        return (sl_price, tp_price)

    def check_exits(self, current_prices: dict[str, Decimal]) -> list[Trade]:
        """Check all positions against current prices for SL/TP hits (Issue #431).

        Iterates through all open positions and checks if the current price
        has hit the stop loss or take profit level. Positions without SL/TP
        set are ignored.

        Args:
            current_prices: Dictionary mapping symbol to current price
                           (e.g., {"EURCAD": Decimal("1.62180")})

        Returns:
            List of closed Trade records for positions that hit SL or TP
        """
        closed_trades: list[Trade] = []
        exit_time = datetime.now(timezone.utc)

        # Get list of position keys to check (avoid modifying dict during iteration)
        # Issue #571: _positions uses compound keys like "EURCAD_LONG"
        position_keys_to_check = list(self._positions.keys())

        for position_key in position_keys_to_check:
            position = self._positions.get(position_key)
            if position is None:
                continue

            # Issue #571: Use position.symbol (e.g., "EURCAD") not position_key (e.g., "EURCAD_LONG")
            # current_prices uses simple symbol keys
            current_price = current_prices.get(position.symbol)
            if current_price is None:
                continue

            # Check SL first (higher priority than TP)
            if position.is_sl_hit(current_price):
                trade = self.close_position(
                    symbol=position.symbol,
                    direction=position.direction,
                    exit_price=current_price,
                    exit_time=exit_time,
                    exit_reason=ExitReason.SL_HIT,
                    position_id=position.id,  # Use position ID for multiple positions per symbol
                )
                closed_trades.append(trade)
                continue

            # Check TP
            if position.is_tp_hit(current_price):
                trade = self.close_position(
                    symbol=position.symbol,
                    direction=position.direction,
                    exit_price=current_price,
                    exit_time=exit_time,
                    exit_reason=ExitReason.TP_HIT,
                    position_id=position.id,  # Use position ID for multiple positions per symbol
                )
                closed_trades.append(trade)

        return closed_trades

    def check_sl_realtime(
        self,
        prices: dict[str, tuple[Decimal, Decimal]],
    ) -> list[Trade]:
        """Check SL only using real-time bid/ask prices.

        This method is designed for high-frequency SL checking using WebSocket
        tick data. It only checks SL conditions (not TP) because:
        - TP/profit exits are handled by RL model in main 60s loop
        - SL is a hard protection that should trigger immediately

        Uses proper bid/ask for direction:
        - LONG: SL hit when BID <= sl_price (we sell at bid)
        - SHORT: SL hit when ASK >= sl_price (we buy at ask)

        Args:
            prices: Dictionary mapping symbol to (bid, ask) tuple

        Returns:
            List of closed Trade records for positions that hit SL
        """
        closed_trades: list[Trade] = []
        exit_time = datetime.now(timezone.utc)

        # Get list of position keys to check (avoid modifying dict during iteration)
        keys_to_check = list(self._positions.keys())

        for key in keys_to_check:
            position = self._positions.get(key)
            if position is None:
                continue

            # Get bid/ask for this symbol
            bid_ask = prices.get(position.symbol)
            if bid_ask is None:
                continue

            bid, ask = bid_ask

            # Check SL using proper bid/ask for direction
            if position.is_sl_hit_realtime(bid, ask):
                # Use actual exit price based on direction
                # LONG: exit at bid (we sell)
                # SHORT: exit at ask (we buy to cover)
                if position.direction == PositionDirection.LONG:
                    exit_price = bid
                else:
                    exit_price = ask

                trade = self.close_position(
                    symbol=position.symbol,
                    exit_price=exit_price,
                    exit_time=exit_time,
                    exit_reason=ExitReason.SL_HIT,
                    direction=position.direction,
                    position_id=position.id,  # Use position ID for multiple positions per symbol
                )
                closed_trades.append(trade)
                logger.info(
                    f"Real-time SL hit: {position.symbol} {position.direction.value} "
                    f"exit_price={exit_price} (bid={bid}, ask={ask})"
                )

        return closed_trades

    def check_tp_realtime(
        self,
        prices: dict[str, tuple[Decimal, Decimal]],
    ) -> list[Trade]:
        """Check TP only using real-time bid/ask prices.

        This method is designed for high-frequency TP checking using WebSocket
        tick data. It catches intra-candle TP hits that would be missed by
        the 60-second polling loop.

        Example scenario (Issue #535):
        - GBPUSD SHORT opened at 1.33986, TP at 1.33686 (+30 pips)
        - M30 candle: Low=1.33662 (hit TP!), but Close=1.33780
        - Without real-time check: TP missed, profit reduced
        - With real-time check: TP hit at 1.33662, +30 pips secured

        Uses proper bid/ask for direction:
        - LONG: TP hit when BID >= tp_price (we sell at bid)
        - SHORT: TP hit when ASK <= tp_price (we buy at ask)

        Args:
            prices: Dictionary mapping symbol to (bid, ask) tuple

        Returns:
            List of closed Trade records for positions that hit TP
        """
        closed_trades: list[Trade] = []
        exit_time = datetime.now(timezone.utc)

        # Get list of position keys to check (avoid modifying dict during iteration)
        keys_to_check = list(self._positions.keys())

        for key in keys_to_check:
            position = self._positions.get(key)
            if position is None:
                continue

            # Get bid/ask for this symbol
            bid_ask = prices.get(position.symbol)
            if bid_ask is None:
                continue

            bid, ask = bid_ask

            # Check TP using proper bid/ask for direction
            if position.is_tp_hit_realtime(bid, ask):
                # Use actual exit price based on direction
                # LONG: exit at bid (we sell)
                # SHORT: exit at ask (we buy to cover)
                if position.direction == PositionDirection.LONG:
                    exit_price = bid
                else:
                    exit_price = ask

                trade = self.close_position(
                    symbol=position.symbol,
                    exit_price=exit_price,
                    exit_time=exit_time,
                    exit_reason=ExitReason.TP_HIT,
                    direction=position.direction,
                )
                closed_trades.append(trade)
                logger.info(
                    f"Real-time TP hit: {position.symbol} {position.direction.value} "
                    f"exit_price={exit_price} (bid={bid}, ask={ask})"
                )

        return closed_trades
