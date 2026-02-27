"""
Trade execution and signal processing for backtesting.

This module implements the execute_confluence_signal function that converts
confluence signals into Trade objects, along with signal filtering logic.
"""

from datetime import datetime, timedelta
from typing import List

from backtesting.models import Trade
from pattern_system.confluence.models import ConfluenceSignal

# Forex pip value for standard pairs (EURUSD, GBPUSD, etc.)
PIP_VALUE = 0.0001


def execute_confluence_signal(
    signal: ConfluenceSignal,
    entry_price: float,
    stop_loss_pips: int = 20,
    take_profit_ratio: float = 1.5,
    slippage_pips: int = 0,
    exit_reason: str = "STOP_LOSS",
    quantity: float = 1.0,
) -> Trade:
    """
    Execute confluence signal as a trade.

    Args:
        signal: ConfluenceSignal to execute.
        entry_price: Entry price for the trade.
        stop_loss_pips: Stop loss distance in pips (default: 20).
        take_profit_ratio: Take profit as multiple of stop loss (default: 1.5).
        slippage_pips: Slippage to apply to entry price (default: 0).
        exit_reason: Reason for trade exit - "STOP_LOSS" or "TAKE_PROFIT".
        quantity: Position size in lots (default: 1.0).

    Returns:
        Trade object with entry and exit details.

    Raises:
        ValueError: If signal direction is NEUTRAL or parameters are invalid.
    """
    # Validate parameters
    if signal.direction == "NEUTRAL":
        raise ValueError("Cannot execute trade with NEUTRAL direction")

    if entry_price <= 0:
        raise ValueError("entry_price must be positive")

    if stop_loss_pips <= 0:
        raise ValueError("stop_loss_pips must be positive")

    if take_profit_ratio <= 0:
        raise ValueError("take_profit_ratio must be positive")

    if slippage_pips < 0:
        raise ValueError("slippage_pips cannot be negative")

    # Apply slippage to entry price
    if signal.direction == "LONG":
        # For LONG, positive slippage means worse fill (higher price)
        actual_entry_price = entry_price + (slippage_pips * PIP_VALUE)
    else:  # SHORT
        # For SHORT, positive slippage means worse fill (lower price)
        actual_entry_price = entry_price - (slippage_pips * PIP_VALUE)

    # Calculate exit price based on exit reason
    if exit_reason == "STOP_LOSS":
        if signal.direction == "LONG":
            # LONG stop loss is below entry
            exit_price = actual_entry_price - (stop_loss_pips * PIP_VALUE)
        else:  # SHORT
            # SHORT stop loss is above entry
            exit_price = actual_entry_price + (stop_loss_pips * PIP_VALUE)
    elif exit_reason == "TAKE_PROFIT":
        if signal.direction == "LONG":
            # LONG take profit is above entry
            exit_price = actual_entry_price + (
                stop_loss_pips * take_profit_ratio * PIP_VALUE
            )
        else:  # SHORT
            # SHORT take profit is below entry
            exit_price = actual_entry_price - (
                stop_loss_pips * take_profit_ratio * PIP_VALUE
            )
    else:
        raise ValueError(f"Invalid exit_reason: {exit_reason}")

    # Calculate P&L in pips
    if signal.direction == "LONG":
        pnl_pips = (exit_price - actual_entry_price) / PIP_VALUE
    else:  # SHORT
        pnl_pips = (actual_entry_price - exit_price) / PIP_VALUE

    pnl = pnl_pips

    # Calculate P&L percentage
    pnl_pct = (abs(exit_price - actual_entry_price) / actual_entry_price) * 100
    if pnl < 0:
        pnl_pct = -pnl_pct

    # Set exit time (for backtesting, assume immediate fill for now)
    # In real backtesting, this would be determined by actual candle data
    exit_time = signal.timestamp + timedelta(hours=1)

    # Create and return Trade
    return Trade(
        entry_time=signal.timestamp,
        entry_price=actual_entry_price,
        exit_time=exit_time,
        exit_price=exit_price,
        direction=signal.direction,
        quantity=quantity,
        pnl=pnl,
        pnl_pct=pnl_pct,
        signal=signal,
        exit_reason=exit_reason,
    )


def filter_signals_by_confidence(
    signals: List[ConfluenceSignal],
    min_confidence: float,
) -> List[ConfluenceSignal]:
    """
    Filter signals by minimum confidence threshold.

    Args:
        signals: List of signals to filter.
        min_confidence: Minimum confidence threshold (0.0-1.0).

    Returns:
        Filtered list of signals with confidence >= min_confidence.
    """
    return [s for s in signals if s.confidence >= min_confidence]


def filter_signals_by_pattern_count(
    signals: List[ConfluenceSignal],
    min_pattern_count: int,
) -> List[ConfluenceSignal]:
    """
    Filter signals by minimum number of patterns.

    Args:
        signals: List of signals to filter.
        min_pattern_count: Minimum number of patterns required.

    Returns:
        Filtered list of signals with pattern_count >= min_pattern_count.
    """
    return [s for s in signals if s.pattern_count >= min_pattern_count]


def filter_signals_by_rate_limit(
    signals: List[ConfluenceSignal],
    max_signals_per_hour: int,
) -> List[ConfluenceSignal]:
    """
    Filter signals to limit rate to max signals per hour.

    Implements a true sliding window rate limiter that keeps only signals
    if there are fewer than max_signals_per_hour in the preceding 1-hour window.

    Args:
        signals: List of signals to filter (must be sorted by timestamp).
        max_signals_per_hour: Maximum signals allowed per hour.

    Returns:
        Filtered list of signals respecting rate limit.
    """
    if not signals:
        return []

    filtered = []

    for signal in signals:
        # Count how many accepted signals are within the last hour from this signal
        one_hour_ago = signal.timestamp - timedelta(hours=1)

        # Count signals in filtered list that are within the window
        count_in_window = sum(1 for s in filtered if s.timestamp > one_hour_ago)

        # Only accept if we haven't exceeded the limit
        if count_in_window < max_signals_per_hour:
            filtered.append(signal)

    return filtered
