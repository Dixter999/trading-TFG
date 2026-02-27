"""RiskCalculator - ATR-based stop-loss and take-profit calculator.

This module provides risk management calculations for the MTF trading environment:
- ATR-normalized stop-loss calculation for OB pullback and breakout patterns
- Take-profit levels based on risk multiples (1R, 2R)
- Position sizing for partial exits (TP1, TP2, runner)

All calculations use ATR% for temporal portability across different timeframes.
"""



class RiskCalculator:
    """Calculate ATR-based stop-loss and take-profit levels."""

    # Pattern type constants
    PATTERN_OB_PULLBACK = "OB_PULLBACK"
    PATTERN_BREAKOUT = "BREAKOUT"

    # Direction constants
    DIRECTION_LONG = "LONG"
    DIRECTION_SHORT = "SHORT"

    # ATR multipliers for SL calculation
    ATR_MULTIPLIER_OB_PULLBACK = 0.3
    ATR_MULTIPLIER_BREAKOUT = 0.5

    def __init__(self):
        """Initialize risk calculator."""
        pass

    def calculate_stop_loss(
        self,
        pattern_type: str,
        direction: str,
        entry_price: float,
        atr: float,
        critical_level: float,
    ) -> float:
        """
        Calculate stop-loss level based on pattern type and direction.

        For OB_PULLBACK:
            LONG: SL = critical_level - 0.3×ATR
            SHORT: SL = critical_level + 0.3×ATR

        For BREAKOUT:
            LONG: SL = critical_level - 0.5×ATR
            SHORT: SL = critical_level + 0.5×ATR

        Args:
            pattern_type: 'OB_PULLBACK' or 'BREAKOUT'
            direction: 'LONG' or 'SHORT'
            entry_price: Entry price for the trade
            atr: Current ATR value
            critical_level: OB boundary (pullback) or swing level (breakout)

        Returns:
            Stop-loss price level

        Raises:
            ValueError: If pattern_type or direction is invalid
        """
        # Determine ATR multiplier based on pattern type
        if pattern_type == self.PATTERN_OB_PULLBACK:
            atr_multiplier = self.ATR_MULTIPLIER_OB_PULLBACK
        elif pattern_type == self.PATTERN_BREAKOUT:
            atr_multiplier = self.ATR_MULTIPLIER_BREAKOUT
        else:
            raise ValueError(
                f"Invalid pattern_type: {pattern_type}. Must be 'OB_PULLBACK' or 'BREAKOUT'"
            )

        # Calculate SL distance from critical level
        sl_distance = atr_multiplier * atr

        # Calculate SL based on direction
        if direction == self.DIRECTION_LONG:
            # LONG: SL below critical level
            stop_loss = critical_level - sl_distance
        elif direction == self.DIRECTION_SHORT:
            # SHORT: SL above critical level
            stop_loss = critical_level + sl_distance
        else:
            raise ValueError(
                f"Invalid direction: {direction}. Must be 'LONG' or 'SHORT'"
            )

        return stop_loss

    def calculate_take_profit_levels(
        self, direction: str, entry_price: float, stop_loss: float, atr: float
    ) -> dict[str, float]:
        """
        Calculate take-profit levels based on R multiples.

        TP1: 1.0R (risk:reward 1:1) - 40-50% position exit
        TP2: 2.0R (risk:reward 1:2) - 30-40% position exit
        Runner: 1.0×ATR trailing stop - 10-20% position

        Args:
            direction: 'LONG' or 'SHORT'
            entry_price: Entry price for the trade
            stop_loss: Stop-loss level
            atr: Current ATR value for trailing stop calculation

        Returns:
            Dictionary with keys:
                - 'tp1': TP1 price level (1.0R)
                - 'tp2': TP2 price level (2.0R)
                - 'runner_trail_distance': Trailing stop distance (1.0×ATR)

        Raises:
            ValueError: If direction is invalid
        """
        # Calculate risk (R) based on direction
        if direction == self.DIRECTION_LONG:
            # LONG: Risk = entry - SL
            risk = entry_price - stop_loss
            # TP levels above entry
            tp1 = entry_price + (1.0 * risk)
            tp2 = entry_price + (2.0 * risk)
        elif direction == self.DIRECTION_SHORT:
            # SHORT: Risk = SL - entry
            risk = stop_loss - entry_price
            # TP levels below entry
            tp1 = entry_price - (1.0 * risk)
            tp2 = entry_price - (2.0 * risk)
        else:
            raise ValueError(
                f"Invalid direction: {direction}. Must be 'LONG' or 'SHORT'"
            )

        # Runner trailing stop distance = 1.0×ATR
        runner_trail_distance = 1.0 * atr

        return {"tp1": tp1, "tp2": tp2, "runner_trail_distance": runner_trail_distance}

    def calculate_position_sizes(self) -> dict[str, float]:
        """
        Get position sizing percentages for partial exits.

        Position allocation:
            - TP1: 45% (middle of 40-50% range) - First target at 1.0R
            - TP2: 35% (middle of 30-40% range) - Second target at 2.0R
            - Runner: 20% (middle of 10-20% range) - Trailing stop

        Returns:
            Dictionary with keys:
                - 'tp1_pct': Percentage for TP1 exit (0.45 = 45%)
                - 'tp2_pct': Percentage for TP2 exit (0.35 = 35%)
                - 'runner_pct': Percentage for runner (0.20 = 20%)
        """
        return {"tp1_pct": 0.45, "tp2_pct": 0.35, "runner_pct": 0.20}

    def should_move_sl_to_breakeven(
        self,
        current_price: float,
        entry_price: float,
        tp1_level: float,
        direction: str,
    ) -> bool:
        """
        Check if TP1 hit and SL should move to breakeven.

        Stop-loss moves to breakeven (entry price) when price reaches or exceeds TP1.
        This locks in a risk-free trade after the first target is hit.

        For LONG trades:
            - TP1 hit when current_price >= tp1_level
        For SHORT trades:
            - TP1 hit when current_price <= tp1_level

        Args:
            current_price: Current market price
            entry_price: Original entry price
            tp1_level: TP1 price level
            direction: 'LONG' or 'SHORT'

        Returns:
            True if TP1 reached and SL should move to breakeven

        Raises:
            ValueError: If direction is invalid
        """
        if direction == self.DIRECTION_LONG:
            # LONG: TP1 hit when price >= tp1_level
            return current_price >= tp1_level
        elif direction == self.DIRECTION_SHORT:
            # SHORT: TP1 hit when price <= tp1_level
            return current_price <= tp1_level
        else:
            raise ValueError(
                f"Invalid direction: {direction}. Must be 'LONG' or 'SHORT'"
            )

    def update_trailing_stop(
        self,
        current_price: float,
        current_trailing_stop: float,
        atr: float,
        direction: str,
    ) -> float:
        """
        Update trailing stop for runner position.

        Trailing stop follows price movement in favorable direction only.
        Uses 1.0×ATR distance from current price.

        For LONG trades:
            - Trailing stop = current_price - 1.0×ATR
            - Only moves UP (never down)
        For SHORT trades:
            - Trailing stop = current_price + 1.0×ATR
            - Only moves DOWN (never up)

        Args:
            current_price: Current market price
            current_trailing_stop: Current trailing stop level
            atr: Current ATR value
            direction: 'LONG' or 'SHORT'

        Returns:
            Updated trailing stop level (moves only in favorable direction)

        Raises:
            ValueError: If direction is invalid
        """
        if direction == self.DIRECTION_LONG:
            # LONG: Trailing stop = price - ATR, only moves UP
            new_trailing_stop = current_price - atr
            # Only update if new stop is higher (favorable)
            return max(new_trailing_stop, current_trailing_stop)
        elif direction == self.DIRECTION_SHORT:
            # SHORT: Trailing stop = price + ATR, only moves DOWN
            new_trailing_stop = current_price + atr
            # Only update if new stop is lower (favorable)
            return min(new_trailing_stop, current_trailing_stop)
        else:
            raise ValueError(
                f"Invalid direction: {direction}. Must be 'LONG' or 'SHORT'"
            )
