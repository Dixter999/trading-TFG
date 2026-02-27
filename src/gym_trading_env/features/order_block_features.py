"""
Order Block feature extraction for RL trading environment.

Author: python-backend-engineer (Stream A)
Issue: #254
Created: 2025-11-12

This module extracts Order Block features from price data for use as
observations in the gym_trading_env RL environment. It integrates with
the existing OrderBlockDetector to provide actionable signals.

TDD Cycle Status: ✅ Complete (RED → GREEN → REFACTOR)
"""

from dataclasses import dataclass
import math
import pandas as pd
from functools import lru_cache

from trading_patterns.detectors.order_block_detector import OrderBlockDetector

# Global cached detector instance (singleton pattern)
_CACHED_DETECTOR = None


def _get_detector() -> OrderBlockDetector:
    """
    Get cached OrderBlockDetector instance (singleton).

    This prevents creating a new detector at every step, which is expensive.

    Returns:
        Cached OrderBlockDetector instance
    """
    global _CACHED_DETECTOR
    if _CACHED_DETECTOR is None:
        _CACHED_DETECTOR = OrderBlockDetector(
            min_body_ratio=0.5,
            periods=5,
            threshold=0.0,
            usewicks=False
        )
    return _CACHED_DETECTOR


@dataclass
class OrderBlockSignal:
    """
    Order Block signal data structure for RL environment observations.

    Represents a detected Order Block with all relevant features needed
    for the RL agent to make trading decisions.

    Attributes:
        direction (str): Signal direction - "LONG", "SHORT", or "NONE"
        strength (float): Signal strength from 0.0 to 1.0
        age_bars (int): Number of bars since OB formation
        distance_pips (float): Distance from current price in pips
        ob_high (float): High price of the Order Block zone
        ob_low (float): Low price of the Order Block zone
        is_valid (bool): Whether the signal is valid for trading

    Example:
        >>> signal = OrderBlockSignal(
        ...     direction="LONG",
        ...     strength=0.8,
        ...     age_bars=5,
        ...     distance_pips=10.5,
        ...     ob_high=1.1000,
        ...     ob_low=1.0950,
        ...     is_valid=True
        ... )
    """
    direction: str  # "LONG", "SHORT", or "NONE"
    strength: float  # 0.0 to 1.0
    age_bars: int  # Bars since OB formation
    distance_pips: float  # Distance from current price
    ob_high: float  # OB zone high
    ob_low: float  # OB zone low
    is_valid: bool  # Whether signal is usable


def _create_none_signal() -> OrderBlockSignal:
    """
    Create an invalid NONE signal (no Order Block detected).

    Returns:
        OrderBlockSignal with is_valid=False and direction="NONE"
    """
    return OrderBlockSignal(
        direction="NONE",
        strength=0.0,
        age_bars=0,
        distance_pips=0.0,
        ob_high=0.0,
        ob_low=0.0,
        is_valid=False
    )


def _calculate_distance_pips(
    current_price: float,
    ob_low: float,
    ob_high: float,
    direction: str
) -> float:
    """
    Calculate distance from current price to Order Block zone in pips.

    Args:
        current_price: Current market price
        ob_low: Order Block zone low boundary
        ob_high: Order Block zone high boundary
        direction: "LONG" or "SHORT"

    Returns:
        Distance in pips (0.0 if inside zone)
    """
    # Distance to nearest OB boundary
    if current_price < ob_low:
        distance = ob_low - current_price
    elif current_price > ob_high:
        distance = current_price - ob_high
    else:
        distance = 0.0  # Inside zone

    return distance * 10000  # Convert to pips (4 decimal places)


def _calculate_strength(age_bars: int, decay_factor: float = 50.0) -> float:
    """
    Calculate signal strength based on age.

    Fresher Order Blocks are stronger. Strength decays exponentially with age.

    Args:
        age_bars: Number of bars since OB formation
        decay_factor: Controls decay rate (default: 50.0)

    Returns:
        Strength value between 0.0 and 1.0
    """
    strength = math.exp(-age_bars / decay_factor)
    return max(0.0, min(1.0, strength))  # Clamp to [0, 1]


def _search_order_blocks(
    detector: OrderBlockDetector,
    df: pd.DataFrame,
    current_idx: int,
    start_idx: int
) -> tuple[dict | None, dict | None]:
    """
    Search for bullish and bearish Order Blocks within the lookback window.

    Args:
        detector: OrderBlockDetector instance
        df: OHLCV DataFrame
        current_idx: Current bar index
        start_idx: Start of lookback window

    Returns:
        Tuple of (bullish_ob, bearish_ob) dictionaries, or None if not found
    """
    bullish_ob = None
    bearish_ob = None

    # Search from current_idx backwards to start_idx
    for idx in range(current_idx, start_idx, -1):
        # Try bullish if not yet found
        if bullish_ob is None:
            ob = detector.detect_bullish_ob(df, idx)
            if ob is not None and ob['ob_index'] >= start_idx:
                bullish_ob = ob

        # Try bearish if not yet found
        if bearish_ob is None:
            ob = detector.detect_bearish_ob(df, idx)
            if ob is not None and ob['ob_index'] >= start_idx:
                bearish_ob = ob

        # Early exit if both found
        if bullish_ob is not None and bearish_ob is not None:
            break

    return bullish_ob, bearish_ob


def _select_most_recent_ob(
    bullish_ob: dict | None,
    bearish_ob: dict | None
) -> tuple[dict | None, str]:
    """
    Select the most recent Order Block from bullish and bearish candidates.

    Args:
        bullish_ob: Bullish OB detection result or None
        bearish_ob: Bearish OB detection result or None

    Returns:
        Tuple of (selected_ob, direction)
    """
    if bullish_ob is None and bearish_ob is None:
        return None, "NONE"

    if bullish_ob is not None and bearish_ob is None:
        return bullish_ob, "LONG"

    if bearish_ob is not None and bullish_ob is None:
        return bearish_ob, "SHORT"

    # Both exist, pick most recent
    if bullish_ob['ob_index'] > bearish_ob['ob_index']:
        return bullish_ob, "LONG"
    else:
        return bearish_ob, "SHORT"


def extract_order_block_features(
    df: pd.DataFrame,
    current_idx: int,
    lookback_bars: int = 50
) -> OrderBlockSignal:
    """
    Extract Order Block features for RL environment observation.

    Scans for Order Blocks within the lookback window and returns
    the most relevant signal with all necessary features.

    Args:
        df: OHLCV DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close']
        current_idx: Current bar index to extract features for
        lookback_bars: Number of bars to look back for OB detection (default: 50)

    Returns:
        OrderBlockSignal with detected features or NONE signal if no valid OB found

    Example:
        >>> signal = extract_order_block_features(df, current_idx=100, lookback_bars=50)
        >>> if signal.is_valid and signal.direction == "LONG":
        ...     print(f"Bullish OB at distance {signal.distance_pips} pips")
    """
    # Validate sufficient data
    if current_idx < 6 or len(df) < 6:
        return _create_none_signal()

    # Get cached detector (singleton - prevents expensive re-initialization)
    detector = _get_detector()

    # Define search window
    start_idx = max(0, current_idx - lookback_bars)

    # Search for Order Blocks within lookback window
    bullish_ob, bearish_ob = _search_order_blocks(
        detector, df, current_idx, start_idx
    )

    # Select most recent Order Block
    selected_ob, direction = _select_most_recent_ob(bullish_ob, bearish_ob)

    # Return NONE signal if no OB found
    if selected_ob is None:
        return _create_none_signal()

    # Extract OB properties
    ob_idx = selected_ob['ob_index']
    ob_low, ob_high = selected_ob['price_zone']

    # Calculate features
    age_bars = current_idx - ob_idx
    current_price = df.iloc[current_idx]['close']
    distance_pips = _calculate_distance_pips(current_price, ob_low, ob_high, direction)
    strength = _calculate_strength(age_bars)

    return OrderBlockSignal(
        direction=direction,
        strength=strength,
        age_bars=age_bars,
        distance_pips=distance_pips,
        ob_high=ob_high,
        ob_low=ob_low,
        is_valid=True
    )
