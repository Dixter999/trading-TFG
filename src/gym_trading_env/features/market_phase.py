"""
Market Phase Detection for RL Trading Environment.

Author: python-backend-engineer (Stream A)
Issue: #258
Created: 2025-11-13

This module detects market phases (ACCUMULATION, DISTRIBUTION, MARKUP, MARKDOWN)
based on price structure, volatility (ATR), volume profile, and price change percentage.

Market Phases:
1. ACCUMULATION: Consolidation with low volatility (sideways, waiting for breakout)
2. DISTRIBUTION: Consolidation after move (potential reversal, caution)
3. MARKUP: Bullish trend with +5% move (higher highs/lows, long bias)
4. MARKDOWN: Bearish trend with -5% move (lower highs/lows, short bias)

TDD Cycle Status: REFACTOR phase (optimized and formatted)
"""

from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np


class MarketPhase(Enum):
    """
    Market phase classification for trading strategy selection.

    Phases determine agent behavior:
    - ACCUMULATION: Wait for breakout (patient state)
    - DISTRIBUTION: Wait for reversal confirmation (cautious state)
    - MARKUP: Long bias (bullish trend state)
    - MARKDOWN: Short bias (bearish trend state)
    """

    ACCUMULATION = 0
    DISTRIBUTION = 1
    MARKUP = 2
    MARKDOWN = 3


def detect_market_phase(
    data: pd.DataFrame,
    current_idx: int,
    lookback_bars: int = 30,
    price_threshold: float = 0.05,  # 5% threshold for markup/markdown
    volatility_threshold: float = 0.002,  # 0.2% ATR threshold for consolidation
) -> MarketPhase:
    """
    Detect current market phase using price structure, volatility, and trend.

    Algorithm:
    1. Validate data and handle edge cases
    2. Calculate ATR (Average True Range) for volatility
    3. Calculate price change percentage over lookback period
    4. Detect price structure (higher highs/lows or lower highs/lows)
    5. Classify phase based on trend + volatility combination

    Decision Logic:
    - If price down > -5%: MARKDOWN (bearish trend)
    - If price up > +5%: MARKUP (bullish trend)
    - If low volatility after uptrend: DISTRIBUTION (potential top)
    - If low volatility: ACCUMULATION (consolidation)

    Parameters:
        data: DataFrame with OHLCV columns (open, high, low, close, volume)
        current_idx: Current bar index in the DataFrame
        lookback_bars: Number of bars to analyze (default: 30)
        price_threshold: Price change % to trigger trend phases (default: 0.05 = 5%)
        volatility_threshold: ATR threshold for consolidation detection (default: 0.002)

    Returns:
        MarketPhase: Detected market phase

    Edge Cases:
    - Empty data → ACCUMULATION (safe default)
    - Insufficient data → ACCUMULATION
    - NaN values → Handled via dropna()
    - Index out of bounds → ACCUMULATION

    Example:
        >>> data = pd.DataFrame({...})  # OHLCV data
        >>> phase = detect_market_phase(data, current_idx=50, lookback_bars=20)
        >>> print(phase)  # MarketPhase.MARKUP
    """
    # Edge case: Empty DataFrame
    if data.empty or len(data) == 0:
        return MarketPhase.ACCUMULATION

    # Edge case: Index out of bounds
    if current_idx < 0 or current_idx >= len(data):
        return MarketPhase.ACCUMULATION

    # Edge case: Insufficient data
    if current_idx < 1:  # Need at least 2 bars for calculations
        return MarketPhase.ACCUMULATION

    # Determine actual lookback (can't exceed available data)
    start_idx = max(0, current_idx - lookback_bars)
    lookback_data = data.iloc[start_idx : current_idx + 1].copy()

    # Edge case: Still insufficient after slicing
    if len(lookback_data) < 2:
        return MarketPhase.ACCUMULATION

    # Edge case: All NaN values
    if lookback_data[["open", "high", "low", "close"]].isna().all().all():
        return MarketPhase.ACCUMULATION

    # Calculate ATR (Average True Range) for volatility measurement
    atr = _calculate_atr(lookback_data)

    # Edge case: ATR calculation failed (NaN)
    if pd.isna(atr) or atr == 0:
        atr = 0.0

    # Calculate price change percentage
    price_change_pct = _calculate_price_change(lookback_data)

    # Edge case: Price change calculation failed
    if pd.isna(price_change_pct):
        return MarketPhase.ACCUMULATION

    # Detect price structure (trending or consolidating)
    is_uptrend = _detect_uptrend(lookback_data)
    is_downtrend = _detect_downtrend(lookback_data)

    # Get current price for volatility normalization
    current_price = lookback_data["close"].iloc[-1]
    if pd.isna(current_price) or current_price == 0:
        return MarketPhase.ACCUMULATION

    # Convert Decimal to float (PostgreSQL returns Decimal for numeric columns)
    current_price = float(current_price)

    # Normalize ATR to price (ATR as % of price)
    normalized_atr = atr / current_price if current_price > 0 else 0

    # Check if we had a recent significant move (for distribution detection)
    had_recent_uptrend = price_change_pct > price_threshold
    had_recent_downtrend = price_change_pct < -price_threshold

    # Check recent behavior (for consolidation detection)
    # Look at last 20% of bars to see if currently consolidating
    recent_bars = max(5, int(len(lookback_data) * 0.2))
    recent_data = lookback_data.iloc[-recent_bars:]
    recent_price_change = _calculate_price_change(recent_data)
    is_currently_trending_up = _detect_uptrend(recent_data)
    is_currently_trending_down = _detect_downtrend(recent_data)

    # Phase Classification Logic
    # Priority 1: Strong trends with threshold crossed (MARKUP/MARKDOWN)
    if price_change_pct <= -price_threshold:
        return MarketPhase.MARKDOWN

    if price_change_pct >= price_threshold:
        return MarketPhase.MARKUP

    # Priority 2: Distribution (consolidation after uptrend)
    # Requires: overall uptrend but currently consolidating
    if is_uptrend and not is_currently_trending_up:
        # Currently sideways after an uptrend = distribution
        if abs(recent_price_change) < price_threshold * 0.3:
            return MarketPhase.DISTRIBUTION

    # Alternative: Had strong uptrend earlier, now consolidating with low volatility
    if had_recent_uptrend and normalized_atr < volatility_threshold * 3:
        if abs(recent_price_change) < price_threshold * 0.2:
            return MarketPhase.DISTRIBUTION

    # Priority 3: Clear trends without threshold (approaching markup/markdown)
    # If strong trend signal but not 5% yet
    if is_uptrend and price_change_pct > price_threshold * 0.3:
        return MarketPhase.MARKUP

    if is_downtrend and price_change_pct < -price_threshold * 0.3:
        return MarketPhase.MARKDOWN

    # Priority 4: Accumulation (low volatility consolidation without prior trend)
    if normalized_atr < volatility_threshold * 2:
        return MarketPhase.ACCUMULATION

    # Default: If trending but weak signal
    if is_uptrend:
        return MarketPhase.MARKUP
    elif is_downtrend:
        return MarketPhase.MARKDOWN

    # Final default: ACCUMULATION
    return MarketPhase.ACCUMULATION


def _calculate_atr(data: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate Average True Range (ATR) for volatility measurement.

    ATR measures volatility by calculating average of true ranges:
    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))

    Parameters:
        data: DataFrame with high, low, close columns
        period: ATR period (default: 14 bars)

    Returns:
        float: Average True Range value (0.0 if calculation fails)
    """
    try:
        # Need at least 2 bars for ATR
        if len(data) < 2:
            return 0.0

        # Clean data (drop rows with NaN in critical columns)
        clean_data = data[["high", "low", "close"]].dropna()

        if len(clean_data) < 2:
            return 0.0

        # Calculate True Range components
        high_low = clean_data["high"] - clean_data["low"]
        high_close = (clean_data["high"] - clean_data["close"].shift(1)).abs()
        low_close = (clean_data["low"] - clean_data["close"].shift(1)).abs()

        # True Range = max of three components
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Average True Range (simple moving average)
        atr_period = min(period, len(true_range))
        atr = true_range.rolling(window=atr_period, min_periods=1).mean().iloc[-1]

        return float(atr) if not pd.isna(atr) else 0.0

    except Exception:
        return 0.0


def _calculate_price_change(data: pd.DataFrame) -> float:
    """
    Calculate percentage price change over the period.

    Price change = (current_close - start_close) / start_close

    Parameters:
        data: DataFrame with close column

    Returns:
        float: Price change percentage (0.0 if calculation fails)
    """
    try:
        clean_closes = data["close"].dropna()

        if len(clean_closes) < 2:
            return 0.0

        start_price = clean_closes.iloc[0]
        end_price = clean_closes.iloc[-1]

        if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
            return 0.0

        # Convert Decimal to float (PostgreSQL returns Decimal for numeric columns)
        start_price = float(start_price)
        end_price = float(end_price)

        price_change = (end_price - start_price) / start_price

        return float(price_change) if not pd.isna(price_change) else 0.0

    except Exception:
        return 0.0


def _detect_uptrend(data: pd.DataFrame, swing_period: int = 5) -> bool:
    """
    Detect uptrend using price structure analysis.

    Uptrend indicators:
    1. Positive price slope
    2. Higher highs and higher lows
    3. Close trending above open

    Parameters:
        data: DataFrame with high, low, close columns
        swing_period: Period for detecting swings (default: 5)

    Returns:
        bool: True if uptrend detected, False otherwise
    """
    try:
        if len(data) < 2:
            return False

        clean_data = data[["high", "low", "close"]].dropna()

        if len(clean_data) < 2:
            return False

        # Method 1: Simple slope check (most reliable)
        closes = clean_data["close"].values
        # Convert Decimal to float (PostgreSQL returns Decimal for numeric columns)
        closes = closes.astype(float)
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]

        # Method 2: Compare first half vs second half
        mid = len(clean_data) // 2
        first_half_avg = clean_data["close"].iloc[:mid].mean()
        second_half_avg = clean_data["close"].iloc[mid:].mean()

        # Method 3: Higher highs and higher lows (simplified)
        recent_high = clean_data["high"].iloc[-swing_period:].max()
        old_high = clean_data["high"].iloc[:swing_period].max()
        recent_low = clean_data["low"].iloc[-swing_period:].min()
        old_low = clean_data["low"].iloc[:swing_period].min()

        higher_highs = recent_high > old_high
        higher_lows = recent_low > old_low

        # Require at least 2 of 3 methods to agree
        signals = [
            slope > 0,
            second_half_avg > first_half_avg,
            higher_highs and higher_lows,
        ]

        return sum(signals) >= 2

    except Exception:
        return False


def _detect_downtrend(data: pd.DataFrame, swing_period: int = 5) -> bool:
    """
    Detect downtrend using price structure analysis.

    Downtrend indicators:
    1. Negative price slope
    2. Lower highs and lower lows
    3. Close trending below open

    Parameters:
        data: DataFrame with high, low, close columns
        swing_period: Period for detecting swings (default: 5)

    Returns:
        bool: True if downtrend detected, False otherwise
    """
    try:
        if len(data) < 2:
            return False

        clean_data = data[["high", "low", "close"]].dropna()

        if len(clean_data) < 2:
            return False

        # Method 1: Simple slope check (most reliable)
        closes = clean_data["close"].values
        # Convert Decimal to float (PostgreSQL returns Decimal for numeric columns)
        closes = closes.astype(float)
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]

        # Method 2: Compare first half vs second half
        mid = len(clean_data) // 2
        first_half_avg = clean_data["close"].iloc[:mid].mean()
        second_half_avg = clean_data["close"].iloc[mid:].mean()

        # Method 3: Lower highs and lower lows (simplified)
        recent_high = clean_data["high"].iloc[-swing_period:].max()
        old_high = clean_data["high"].iloc[:swing_period].max()
        recent_low = clean_data["low"].iloc[-swing_period:].min()
        old_low = clean_data["low"].iloc[:swing_period].min()

        lower_highs = recent_high < old_high
        lower_lows = recent_low < old_low

        # Require at least 2 of 3 methods to agree
        signals = [
            slope < 0,
            second_half_avg < first_half_avg,
            lower_highs and lower_lows,
        ]

        return sum(signals) >= 2

    except Exception:
        return False


def _simple_trend_check(data: pd.DataFrame, direction: str = "up") -> bool:
    """
    Fallback simple trend detection using linear regression slope.

    Parameters:
        data: DataFrame with close column
        direction: 'up' or 'down'

    Returns:
        bool: True if trend matches direction
    """
    try:
        closes = data["close"].dropna().values

        if len(closes) < 2:
            return False

        # Calculate simple slope
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]

        if direction == "up":
            return slope > 0
        else:
            return slope < 0

    except Exception:
        return False
