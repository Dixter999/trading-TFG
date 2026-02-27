"""
Hybrid action space implementation for RL-refined Order Block trading.

The RL agent receives Order Block signals as input and chooses one of 5 actions
to refine the trading execution. This allows the agent to learn which Order Blocks
to trade and how to optimize entry/exit parameters.
"""

from enum import IntEnum
from typing import Dict, Optional, Union
from copy import deepcopy


class HybridAction(IntEnum):
    """
    Discrete action space for hybrid RL strategy.

    Actions:
        SKIP (0): Ignore this Order Block signal (no entry)
        ENTER (1): Accept baseline entry parameters as-is
        ENTER_2X (2): Double the position size
        ADJUST_SL (3): Tighten stop-loss by 30% (multiply distance by 0.7)
        ADJUST_TP (4): Extend take-profit by 50% (multiply distance by 1.5)
    """

    SKIP = 0
    ENTER = 1
    ENTER_2X = 2
    ADJUST_SL = 3
    ADJUST_TP = 4


def apply_hybrid_action(
    action: Union[HybridAction, int],
    baseline_entry: Dict[str, float],
) -> Optional[Dict[str, float]]:
    """
    Apply a hybrid action to baseline entry parameters.

    Args:
        action: HybridAction enum or integer (0-4)
        baseline_entry: Dictionary with keys:
            - direction: "BUY" or "SELL"
            - entry_price: Entry price
            - stop_loss: Stop-loss price
            - take_profit: Take-profit price
            - position_size: Position size multiplier

    Returns:
        Modified entry parameters dict, or None if action is SKIP

    Examples:
        >>> baseline = {
        ...     "direction": "BUY",
        ...     "entry_price": 1.1000,
        ...     "stop_loss": 1.0950,
        ...     "take_profit": 1.1100,
        ...     "position_size": 1.0,
        ... }
        >>> apply_hybrid_action(HybridAction.SKIP, baseline)
        None
        >>> result = apply_hybrid_action(HybridAction.ENTER, baseline)
        >>> result["position_size"]
        1.0
        >>> result = apply_hybrid_action(HybridAction.ENTER_2X, baseline)
        >>> result["position_size"]
        2.0
    """
    # Convert integer to enum if needed
    if isinstance(action, int):
        action = HybridAction(action)

    # SKIP: Return None (no entry)
    if action == HybridAction.SKIP:
        return None

    # Create a deep copy to avoid modifying the original baseline
    entry = deepcopy(baseline_entry)

    # ENTER: Return baseline as-is
    if action == HybridAction.ENTER:
        return entry

    # ENTER_2X: Double position size
    if action == HybridAction.ENTER_2X:
        entry["position_size"] *= 2.0
        return entry

    # ADJUST_SL: Tighten stop-loss by 30%
    if action == HybridAction.ADJUST_SL:
        entry_price = entry["entry_price"]
        stop_loss = entry["stop_loss"]
        direction = entry["direction"]

        # Calculate original distance from entry to stop-loss
        if direction == "BUY":
            # For BUY: SL is below entry, distance is positive
            distance = entry_price - stop_loss
            # Tighten by 30%: multiply distance by 0.7
            new_distance = distance * 0.7
            # New SL is closer to entry
            entry["stop_loss"] = entry_price - new_distance
        else:  # SELL
            # For SELL: SL is above entry, distance is positive
            distance = stop_loss - entry_price
            # Tighten by 30%: multiply distance by 0.7
            new_distance = distance * 0.7
            # New SL is closer to entry
            entry["stop_loss"] = entry_price + new_distance

        return entry

    # ADJUST_TP: Extend take-profit by 50%
    if action == HybridAction.ADJUST_TP:
        entry_price = entry["entry_price"]
        take_profit = entry["take_profit"]
        direction = entry["direction"]

        # Calculate original distance from entry to take-profit
        if direction == "BUY":
            # For BUY: TP is above entry, distance is positive
            distance = take_profit - entry_price
            # Extend by 50%: multiply distance by 1.5
            new_distance = distance * 1.5
            # New TP is farther from entry
            entry["take_profit"] = entry_price + new_distance
        else:  # SELL
            # For SELL: TP is below entry, distance is positive
            distance = entry_price - take_profit
            # Extend by 50%: multiply distance by 1.5
            new_distance = distance * 1.5
            # New TP is farther from entry
            entry["take_profit"] = entry_price - new_distance

        return entry

    # Should never reach here if all enum values are handled
    raise ValueError(f"Unhandled action: {action}")
