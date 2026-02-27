"""
Paper Trading Engine (Issue #328).

This module provides the main trading loop that processes candle data
and makes entry/exit decisions based on model signals.

Main Loop Flow:
    1. Receive new candle data
    2. Check if position exists for symbol
    3. If position exists:
       a. Check TP/SL levels first (immediate exit)
       b. Call inference service for model decision
       c. If exit signal: close position and log trade
       d. If hold: optionally update SL/TP levels
    4. If no position: (future) check for entry signals
"""

from decimal import Decimal
from typing import Any

from src.paper_trading.models import ExitReason, Position
from src.paper_trading.position_manager import PositionManager
from src.paper_trading.trade_logger import TradeLogger


class PaperTradingEngine:
    """Main trading loop for paper trading.

    This class orchestrates the paper trading workflow:
    - Processes candle data
    - Checks exit conditions (TP/SL and model)
    - Manages position lifecycle
    - Logs completed trades

    Attributes:
        _inference_client: Client for model inference service
        _position_manager: Manager for open positions
        _trade_logger: Logger for closed trades
        _model_version: Version identifier for the model
    """

    def __init__(
        self,
        inference_client: Any,
        position_manager: PositionManager,
        trade_logger: TradeLogger,
        model_version: str | None = None,
    ) -> None:
        """Initialize PaperTradingEngine with dependencies.

        Args:
            inference_client: Client for calling inference service
            position_manager: Manager for tracking positions
            trade_logger: Logger for recording trades
            model_version: Optional model version identifier
        """
        self._inference_client = inference_client
        self._position_manager = position_manager
        self._trade_logger = trade_logger
        self._model_version = model_version

    async def check_exit_conditions(
        self,
        position: Position,
        current_price: Decimal,
    ) -> ExitReason | None:
        """Check if position should be exited.

        Checks in order:
        1. TP hit (immediate exit)
        2. SL hit (immediate exit)
        3. Model decision (exit or hold)

        Args:
            position: The open position to check
            current_price: Current market price

        Returns:
            ExitReason if exit should occur, None if holding
        """
        # Check TP first
        if position.is_tp_hit(current_price):
            return ExitReason.TP_HIT

        # Check SL second
        if position.is_sl_hit(current_price):
            return ExitReason.SL_HIT

        # Call model for decision
        prediction = await self._inference_client.predict(
            symbol=position.symbol,
            position_type=position.direction.value,
            entry_price=float(position.entry_price),
            current_price=float(current_price),
        )

        if prediction.get("action") == "exit":
            return ExitReason.MODEL_EXIT

        return None

    async def on_new_candle(self, candle_data: dict[str, Any]) -> None:
        """Process new candle data.

        Main entry point for the trading loop. Called when a new
        candle is received for a symbol.

        Args:
            candle_data: Dictionary containing:
                - symbol: Trading symbol
                - close: Closing price
                - high: High price
                - low: Low price
                - timestamp: Candle timestamp
        """
        symbol = candle_data["symbol"]
        current_price = candle_data["close"]
        timestamp = candle_data["timestamp"]

        # Check if we have a position for this symbol
        if not self._position_manager.has_position(symbol):
            # No position - future: check for entry signals
            return

        # Get the position
        position = self._position_manager.get_position(symbol)
        if position is None:
            return

        # Check exit conditions
        exit_reason = await self.check_exit_conditions(position, current_price)

        if exit_reason is not None:
            # Close the position
            trade = self._position_manager.close_position(
                symbol=symbol,
                exit_price=current_price,
                exit_time=timestamp,
                exit_reason=exit_reason,
                model_version=self._model_version,
            )

            # Log the trade
            self._trade_logger.log_trade(trade)
        else:
            # Position is held - check if we should update SL/TP from model
            await self._update_sltp_from_model(position, current_price)

    async def _update_sltp_from_model(
        self,
        position: Position,
        current_price: Decimal,
    ) -> None:
        """Update SL/TP levels based on model predictions.

        Called when holding a position to potentially adjust
        stop loss and take profit levels.

        Args:
            position: The open position
            current_price: Current market price
        """
        # Get updated SL/TP from model
        prediction = await self._inference_client.predict(
            symbol=position.symbol,
            position_type=position.direction.value,
            entry_price=float(position.entry_price),
            current_price=float(current_price),
        )

        sl_distance = prediction.get("sl_distance")
        tp_distance = prediction.get("tp_distance")

        if sl_distance is not None or tp_distance is not None:
            # Calculate new SL/TP prices based on current price and distances
            new_sl = None
            new_tp = None

            if sl_distance is not None:
                sl_decimal = Decimal(str(sl_distance))
                if position.direction.value == "long":
                    new_sl = current_price - sl_decimal
                else:
                    new_sl = current_price + sl_decimal

            if tp_distance is not None:
                tp_decimal = Decimal(str(tp_distance))
                if position.direction.value == "long":
                    new_tp = current_price + tp_decimal
                else:
                    new_tp = current_price - tp_decimal

            # Update position with new levels
            self._position_manager.update_position(
                symbol=position.symbol,
                tp_price=new_tp,
                sl_price=new_sl,
            )
