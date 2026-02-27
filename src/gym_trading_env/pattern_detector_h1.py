"""
H1 Pattern Detector for trading pattern recognition.

This module implements the PatternDetectorH1 class that analyzes H1 timeframe
data from technical_indicators table to detect trading patterns with strict
anti-leakage guarantees.

TDD Phase: REFACTOR - Improving code structure and maintainability
"""

from datetime import datetime

import pandas as pd

from trading_patterns.detectors import CHOCHDetector, OrderBlockDetector, LiquiditySweepDetector
from trading_patterns.indicators import TrendBiasIndicator

# Trend bias constants
TREND_LONG = "LONG"
TREND_SHORT = "SHORT"
TREND_NEUTRAL = "NEUTRAL"

# Required columns for H1 data validation
REQUIRED_COLUMNS = frozenset(
    [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "sma_50",
        "sma_200",
        "atr_14",
        "macd_line",
        "macd_signal",
        "rsi_14",
        "stoch_k",
        "stoch_d",
        "bb_upper_20",
        "bb_lower_20",
        "ob_bullish_high",
        "ob_bullish_low",
        "ob_bearish_high",
        "ob_bearish_low",
    ]
)


class PatternDetectorH1:
    """
    H1 pattern detector using technical_indicators data.
    Guarantees zero data leakage through timestamp validation.

    This class analyzes H1 timeframe data to detect:
    - Trend Bias (SMA50/200 crossover)
    - Swing Points (k=5 confirmation)
    - Break of Structure (BOS)
    - Change of Character (CHOCH)
    - Order Block touches
    - Breakouts (Bollinger Bands + swing levels)
    - Momentum confirmation (MACD/RSI/Stochastic)

    All pattern detection methods ensure zero data leakage by validating
    that only data with timestamp <= current_time is accessed.
    """

    def __init__(self, h1_data: pd.DataFrame):
        """
        Initialize PatternDetectorH1 with H1 DataFrame.

        Args:
            h1_data: DataFrame from technical_indicators filtered by timeframe='H1'
                    Must be sorted by timestamp ascending and contain all required columns

        Raises:
            TypeError: If h1_data is not a pandas DataFrame
            ValueError: If h1_data validation fails (unsorted, missing columns, etc.)
        """
        self._validate_input_data(h1_data)
        self.h1_data = h1_data
        self._initialize_state()

        # Initialize shared library detectors
        self.choch_detector = CHOCHDetector(swing_confirmation=5)
        self.ob_detector = OrderBlockDetector(min_body_ratio=0.5, periods=5)
        self.trend_bias_indicator = TrendBiasIndicator(threshold=0.0001)
        self.liquidity_sweep_detector = LiquiditySweepDetector(lookback_periods=20, wick_threshold=0.5)

    def _validate_input_data(self, h1_data: pd.DataFrame) -> None:
        """
        Validate input DataFrame structure and content.

        Args:
            h1_data: DataFrame to validate

        Raises:
            TypeError: If h1_data is not a pandas DataFrame
            ValueError: If validation fails
        """
        # Validate type
        if not isinstance(h1_data, pd.DataFrame):
            raise TypeError("h1_data must be a pandas DataFrame")

        # Validate timestamp column exists
        if "timestamp" not in h1_data.columns:
            raise ValueError("h1_data must contain 'timestamp' column")

        # Validate all required columns exist
        missing_columns = REQUIRED_COLUMNS - set(h1_data.columns)
        if missing_columns:
            raise ValueError(
                f"h1_data is missing required columns: {sorted(missing_columns)}"
            )

        # Validate data is sorted by timestamp
        self._validate_sorted_timestamps(h1_data)

    def _validate_sorted_timestamps(self, h1_data: pd.DataFrame) -> None:
        """
        Validate that DataFrame is sorted by timestamp in ascending order.

        Args:
            h1_data: DataFrame to validate

        Raises:
            ValueError: If timestamps are not sorted
        """
        if len(h1_data) > 1:
            timestamps = h1_data["timestamp"].tolist()
            if timestamps != sorted(timestamps):
                raise ValueError("h1_data must be sorted by timestamp ascending")

    def _initialize_state(self) -> None:
        """Initialize internal state tracking variables."""
        self.swings = {"highs": [], "lows": []}
        self.last_bos_direction = None

    def _get_row_at_timestamp(self, timestamp: datetime) -> pd.Series | None:
        """
        Get the most recent row at or before the given timestamp.

        This method implements anti-leakage by only returning data where
        the row's timestamp <= the given timestamp.

        Args:
            timestamp: Current time to query data for

        Returns:
            Series with the row data, or None if no data available
        """
        # Filter data to only include rows at or before timestamp
        valid_data = self.h1_data[self.h1_data["timestamp"] <= timestamp]

        if len(valid_data) == 0:
            return None

        # Return the most recent row (last row in filtered data)
        return valid_data.iloc[-1]

    def _get_index_for_timestamp(self, timestamp: datetime) -> int | None:
        """
        Get the index of the most recent row at or before the given timestamp.

        This method implements anti-leakage by only returning index where
        the row's timestamp <= the given timestamp.

        Args:
            timestamp: Current time to query data for

        Returns:
            Index of the row, or None if no data available
        """
        # Filter data to only include rows at or before timestamp
        valid_data = self.h1_data[self.h1_data["timestamp"] <= timestamp]

        if len(valid_data) == 0:
            return None

        # Return the index of the most recent row
        return len(valid_data) - 1

    def _detect_trend_bias(self, timestamp: datetime) -> str:
        """
        Detect trend bias using SMA50/200 analysis.

        LONG bias: close > sma_50 > sma_200
        SHORT bias: close < sma_50 < sma_200
        NEUTRAL: Mixed or unclear conditions

        Args:
            timestamp: Current H1 candle close time

        Returns:
            'LONG', 'SHORT', or 'NEUTRAL'
        """
        # Get index for current timestamp
        idx = self._get_index_for_timestamp(timestamp)

        if idx is None:
            return TREND_NEUTRAL

        # Delegate to shared library
        return self.trend_bias_indicator.calculate(self.h1_data, idx)

    # Swing Detection (Stream B)
    def _detect_swings(self, timestamp: datetime, k: int = 5) -> dict:
        """
        Detect swing highs and lows with k-period confirmation.

        A swing high at index i is higher than k bars before and k bars after.
        A swing low at index i is lower than k bars before and k bars after.

        Args:
            timestamp: Current H1 candle close time
            k: Confirmation period (default: 5)

        Returns:
            Dictionary with detected swings:
            {
                'highs': [{'timestamp': datetime, 'price': float}, ...],
                'lows': [{'timestamp': datetime, 'price': float}, ...]
            }
        """
        # Get data up to timestamp (anti-leakage)
        valid_data = self.h1_data[self.h1_data["timestamp"] <= timestamp]

        if len(valid_data) < (2 * k + 1):
            # Not enough data for k-period confirmation
            return {"highs": [], "lows": []}

        detected_highs = []
        detected_lows = []

        # Check each point that can be a swing (needs k before and k after)
        for i in range(k, len(valid_data) - k):
            current_row = valid_data.iloc[i]

            if self._is_swing_high(valid_data, i, k):
                detected_highs.append({
                    "timestamp": current_row["timestamp"],
                    "price": current_row["high"]
                })

            if self._is_swing_low(valid_data, i, k):
                detected_lows.append({
                    "timestamp": current_row["timestamp"],
                    "price": current_row["low"]
                })

        return {"highs": detected_highs, "lows": detected_lows}

    def _is_swing_high(self, data: pd.DataFrame, index: int, k: int) -> bool:
        """
        Check if the point at given index is a swing high.

        Args:
            data: DataFrame with OHLC data
            index: Index to check
            k: Number of bars to look before and after

        Returns:
            True if swing high detected, False otherwise
        """
        current_high = data.iloc[index]["high"]

        # Check k bars before and after
        for j in range(1, k + 1):
            if data.iloc[index - j]["high"] >= current_high:
                return False
            if data.iloc[index + j]["high"] >= current_high:
                return False

        return True

    def _is_swing_low(self, data: pd.DataFrame, index: int, k: int) -> bool:
        """
        Check if the point at given index is a swing low.

        Args:
            data: DataFrame with OHLC data
            index: Index to check
            k: Number of bars to look before and after

        Returns:
            True if swing low detected, False otherwise
        """
        current_low = data.iloc[index]["low"]

        # Check k bars before and after
        for j in range(1, k + 1):
            if data.iloc[index - j]["low"] <= current_low:
                return False
            if data.iloc[index + j]["low"] <= current_low:
                return False

        return True

    # BOS Detection (Stream B)
    def _detect_bos(self, timestamp: datetime) -> str | None:
        """
        Detect Break of Structure (BOS).

        BOS occurs when price breaks above the most recent swing high (bullish BOS)
        or below the most recent swing low (bearish BOS), using a dynamic buffer.

        Buffer = max(0.0005, 0.25 * atr_pct) where atr_pct = atr_14 / close

        Args:
            timestamp: Current H1 candle close time

        Returns:
            'UP' for bullish BOS, 'DOWN' for bearish BOS, None if no BOS detected
        """
        # Get current row (anti-leakage)
        row = self._get_row_at_timestamp(timestamp)

        if row is None:
            return None

        # Calculate dynamic buffer
        buffer_pct = self._calculate_buffer(row)

        # Get current close price
        current_close = row["close"]

        # Detect swings up to current timestamp
        swings = self._detect_swings(timestamp, k=5)

        # Check for bullish BOS (break above swing high)
        if len(swings["highs"]) > 0:
            # Get most recent swing high
            most_recent_swing_high = swings["highs"][-1]["price"]
            threshold_high = most_recent_swing_high * (1 + buffer_pct)

            if current_close > threshold_high:
                self.last_bos_direction = "UP"
                return "UP"

        # Check for bearish BOS (break below swing low)
        if len(swings["lows"]) > 0:
            # Get most recent swing low
            most_recent_swing_low = swings["lows"][-1]["price"]
            threshold_low = most_recent_swing_low * (1 - buffer_pct)

            if current_close < threshold_low:
                self.last_bos_direction = "DOWN"
                return "DOWN"

        return None

    def _calculate_buffer(self, row: pd.Series) -> float:
        """
        Calculate dynamic buffer for BOS/CHOCH detection.

        Buffer = max(0.0005, 0.25 * atr_pct) where atr_pct = atr_14 / close

        Args:
            row: DataFrame row with OHLC and ATR data

        Returns:
            Buffer percentage (e.g., 0.001 for 0.1%)
        """
        atr_pct = row["atr_14"] / row["close"]
        return max(0.0005, 0.25 * atr_pct)

    # CHOCH Detection (Stream B)
    def _detect_choch(self, timestamp: datetime) -> bool:
        """
        Detect Change of Character (CHOCH).

        CHOCH occurs when price breaks a swing level in the OPPOSITE direction
        to the last BOS. For example:
        - If last BOS was UP, CHOCH is a break below a swing low
        - If last BOS was DOWN, CHOCH is a break above a swing high

        Args:
            timestamp: Current H1 candle close time

        Returns:
            True if CHOCH detected, False otherwise
        """
        # Sync BOS state to CHOCHDetector
        self.choch_detector.last_bos_direction = self.last_bos_direction

        # Get index for current timestamp
        idx = self._get_index_for_timestamp(timestamp)

        if idx is None:
            return False

        # Delegate to shared library
        result = self.choch_detector.detect(self.h1_data, idx)

        # Return True if CHOCH detected, False otherwise
        return result is not None

    def _detect_ob_touch(self, timestamp: datetime) -> dict:
        """
        Detect Order Block touch validation.

        Checks if current candle's OHLC intersects with Order Block zones.

        Args:
            timestamp: Current H1 candle close time

        Returns:
            Dictionary with:
            - bullish: bool (True if bullish OB touched)
            - bearish: bool (True if bearish OB touched)
            - touched: bool (True if any OB touched)
        """
        # Get most recent row at or before timestamp (anti-leakage)
        row = self._get_row_at_timestamp(timestamp)

        if row is None:
            return {"bullish": False, "bearish": False, "touched": False}

        # Get candle OHLC
        candle_low = row["low"]
        candle_high = row["high"]

        # Check bullish OB touch using shared library
        bullish_touch = False
        if pd.notna(row["ob_bullish_low"]) and pd.notna(row["ob_bullish_high"]):
            bullish_touch = self.ob_detector.check_zone_touch(
                candle_low, candle_high, row["ob_bullish_low"], row["ob_bullish_high"]
            )

        # Check bearish OB touch using shared library
        bearish_touch = False
        if pd.notna(row["ob_bearish_low"]) and pd.notna(row["ob_bearish_high"]):
            bearish_touch = self.ob_detector.check_zone_touch(
                candle_low, candle_high, row["ob_bearish_low"], row["ob_bearish_high"]
            )

        return {
            "bullish": bullish_touch,
            "bearish": bearish_touch,
            "touched": bullish_touch or bearish_touch,
        }

    def _detect_breakout(self, timestamp: datetime) -> dict:
        """
        Detect breakout (Bollinger Bands or swing level penetration).

        Checks if close breaks above BB upper, below BB lower, or breaks swing levels.

        Args:
            timestamp: Current H1 candle close time

        Returns:
            Dictionary with:
            - type: str ('BB_UPPER', 'BB_LOWER', 'SWING_HIGH', 'SWING_LOW', or None)
            - detected: bool (True if breakout detected)
        """
        # Get most recent row at or before timestamp (anti-leakage)
        row = self._get_row_at_timestamp(timestamp)

        if row is None:
            return {"type": None, "detected": False}

        # Check BB breakouts first (faster)
        bb_breakout = self._check_bb_breakout(row)
        if bb_breakout["detected"]:
            return bb_breakout

        # Check swing level breakouts
        swing_breakout = self._check_swing_breakout(timestamp, row["close"])
        if swing_breakout["detected"]:
            return swing_breakout

        return {"type": None, "detected": False}

    def _check_bb_breakout(self, row: pd.Series) -> dict:
        """
        Check for Bollinger Band breakout.

        Args:
            row: DataFrame row with OHLC and Bollinger Band data

        Returns:
            Dictionary with breakout type and detection status
        """
        close = row["close"]

        if pd.notna(row["bb_upper_20"]) and close > row["bb_upper_20"]:
            return {"type": "BB_UPPER", "detected": True}

        if pd.notna(row["bb_lower_20"]) and close < row["bb_lower_20"]:
            return {"type": "BB_LOWER", "detected": True}

        return {"type": None, "detected": False}

    def _check_swing_breakout(self, timestamp: datetime, close: float) -> dict:
        """
        Check for swing level breakout.

        Args:
            timestamp: Current H1 candle close time
            close: Current close price

        Returns:
            Dictionary with breakout type and detection status
        """
        swings = self._detect_swings(timestamp, k=5)

        # Check swing high breakout
        if len(swings["highs"]) > 0:
            recent_swing_high = swings["highs"][-1]["price"]
            if close > recent_swing_high:
                return {"type": "SWING_HIGH", "detected": True}

        # Check swing low breakout
        if len(swings["lows"]) > 0:
            recent_swing_low = swings["lows"][-1]["price"]
            if close < recent_swing_low:
                return {"type": "SWING_LOW", "detected": True}

        return {"type": None, "detected": False}

    def _detect_momentum_confirmation(self, timestamp: datetime) -> dict:
        """
        Detect momentum confirmation using MACD/RSI/Stochastic.

        Bullish momentum: At least 2/3 indicators show bullish signals
        - MACD histogram > 0 (macd_line > macd_signal)
        - RSI > 50
        - Stochastic K > 50

        Bearish momentum: At least 2/3 indicators show bearish signals
        - MACD histogram < 0 (macd_line < macd_signal)
        - RSI < 50
        - Stochastic K < 50

        Neutral: Less than 2/3 agreement or missing data

        Args:
            timestamp: Current H1 candle close time

        Returns:
            Dictionary with:
            - bullish: bool (True if bullish momentum detected)
            - bearish: bool (True if bearish momentum detected)
            - neutral: bool (True if no clear direction)
        """
        # Get most recent row at or before timestamp (anti-leakage)
        row = self._get_row_at_timestamp(timestamp)

        if row is None:
            return self._neutral_momentum()

        # Check if we have all required momentum data
        if not self._has_momentum_data(row):
            return self._neutral_momentum()

        # Count bullish/bearish signals from all momentum indicators
        bullish_count, bearish_count = self._count_momentum_signals(row)

        # Return momentum direction based on signal consensus (2/3 agreement)
        return self._determine_momentum_direction(bullish_count, bearish_count)

    def _has_momentum_data(self, row: pd.Series) -> bool:
        """
        Check if row has all required momentum indicator data.

        Args:
            row: DataFrame row with indicator data

        Returns:
            True if all momentum indicators present, False otherwise
        """
        required = ["macd_line", "macd_signal", "rsi_14", "stoch_k"]
        return not any(pd.isna(row[col]) for col in required)

    def _count_momentum_signals(self, row: pd.Series) -> tuple[int, int]:
        """
        Count bullish and bearish signals from momentum indicators.

        Args:
            row: DataFrame row with indicator data

        Returns:
            Tuple of (bullish_count, bearish_count)
        """
        bullish_signals = 0
        bearish_signals = 0

        # MACD histogram (macd_line - macd_signal)
        macd_hist = row["macd_line"] - row["macd_signal"]
        if macd_hist > 0:
            bullish_signals += 1
        elif macd_hist < 0:
            bearish_signals += 1

        # RSI (neutral zone at 50)
        if row["rsi_14"] > 50:
            bullish_signals += 1
        elif row["rsi_14"] < 50:
            bearish_signals += 1

        # Stochastic K (neutral zone at 50)
        if row["stoch_k"] > 50:
            bullish_signals += 1
        elif row["stoch_k"] < 50:
            bearish_signals += 1

        return bullish_signals, bearish_signals

    def _determine_momentum_direction(
        self, bullish_count: int, bearish_count: int
    ) -> dict:
        """
        Determine momentum direction based on signal counts.

        Requires at least 2/3 signals in agreement for directional momentum.

        Args:
            bullish_count: Number of bullish momentum signals
            bearish_count: Number of bearish momentum signals

        Returns:
            Dictionary with bullish/bearish/neutral flags
        """
        if bullish_count >= 2:
            return {"bullish": True, "bearish": False, "neutral": False}
        elif bearish_count >= 2:
            return {"bullish": False, "bearish": True, "neutral": False}
        else:
            return self._neutral_momentum()

    def _neutral_momentum(self) -> dict:
        """
        Return neutral momentum result.

        Returns:
            Dictionary indicating neutral momentum state
        """
        return {"bullish": False, "bearish": False, "neutral": True}

    def _detect_liquidity_sweep(self, timestamp: datetime) -> dict:
        """
        Detect liquidity sweep pattern at the given timestamp.

        A liquidity sweep occurs when price sweeps above/below recent highs/lows
        with a wick and then reverses. This pattern indicates institutional
        players hunting stop losses before reversing.

        Args:
            timestamp: Current H1 candle close time

        Returns:
            Dictionary with:
            - detected: bool (True if liquidity sweep detected)
            - type: str ('bullish_sweep' or 'bearish_sweep' or None)
            - swept_level: float or None (the price level that was swept)
            - wick_ratio: float or None (ratio of wick to total range)
        """
        # Get current index (anti-leakage)
        current_idx = self._get_index_for_timestamp(timestamp)

        if current_idx is None:
            return {"detected": False, "type": None, "swept_level": None, "wick_ratio": None}

        # Use shared library detector
        result = self.liquidity_sweep_detector.detect(self.h1_data, current_idx)

        if result is None:
            return {"detected": False, "type": None, "swept_level": None, "wick_ratio": None}

        return {
            "detected": True,
            "type": result["type"],
            "swept_level": result["swept_level"],
            "wick_ratio": result["wick_ratio"],
        }

    def detect(self, timestamp: datetime) -> dict:
        """
        Main detection method - analyzes all patterns at given timestamp.

        Calculates confluence score (0-5) from 5 pattern types:
        1. Trend bias (LONG/SHORT/NEUTRAL)
        2. Break of Structure (BOS)
        3. Order Block touch
        4. Breakout (Bollinger Bands or swing levels)
        5. Momentum confirmation (MACD/RSI/Stoch)

        Opens trading window when score >= 3.

        Args:
            timestamp: Current H1 candle close time

        Returns:
            Dictionary with detection results including:
            - timestamp: datetime
            - window_long: bool
            - window_short: bool
            - window_type: str ('OB_PULLBACK', 'BREAKOUT', or 'NONE')
            - score: int (0-5 confluence score)
            - patterns: dict with all pattern detection results
            - levels: dict with support/resistance levels
            - atr_pct: float
            - buffer: float

        Example:
            >>> detector = PatternDetectorH1(h1_data)
            >>> signal = detector.detect(datetime(2024, 1, 15, 10, 0))
            >>> if signal['window_long'] and signal['score'] >= 3:
            ...     print("Long window opened with confluence score:", signal['score'])
        """
        # Get current row (anti-leakage)
        row = self._get_row_at_timestamp(timestamp)

        if row is None:
            return self._build_empty_signal(timestamp)

        # Detect all patterns
        patterns = {
            "trend_bias": self._detect_trend_bias(timestamp),
            "bos": self._detect_bos(timestamp),
            "ob_touch": self._detect_ob_touch(timestamp),
            "breakout": self._detect_breakout(timestamp),
            "momentum": self._detect_momentum_confirmation(timestamp),
            "liquidity_sweep": self._detect_liquidity_sweep(timestamp),
        }

        # Calculate confluence score (0-5)
        score = self._calculate_confluence_score(patterns)

        # Determine window type and direction based on patterns
        window_long, window_short, window_type = self._determine_window_state(
            score, patterns
        )

        # Extract swing levels
        swings = self._detect_swings(timestamp, k=5)

        # Build complete signal structure
        return {
            "timestamp": timestamp,
            "window_long": window_long,
            "window_short": window_short,
            "window_type": window_type,
            "score": score,
            "patterns": patterns,
            "levels": {
                "swing_high": swings["highs"][-1]["price"] if swings["highs"] else None,
                "swing_low": swings["lows"][-1]["price"] if swings["lows"] else None,
                "ob_bullish_high": row.get("ob_bullish_high"),
                "ob_bullish_low": row.get("ob_bullish_low"),
                "ob_bearish_high": row.get("ob_bearish_high"),
                "ob_bearish_low": row.get("ob_bearish_low"),
            },
            "atr_pct": row["atr_14"] / row["close"] if row["close"] != 0 else 0,
            "buffer": self._calculate_buffer(row),
        }

    def _build_empty_signal(self, timestamp: datetime) -> dict:
        """
        Build empty signal structure when no data available.

        Args:
            timestamp: Current timestamp

        Returns:
            Empty signal dictionary
        """
        return {
            "timestamp": timestamp,
            "window_long": False,
            "window_short": False,
            "window_type": "NONE",
            "score": 0,
            "patterns": {},
            "levels": {},
            "atr_pct": 0.0,
            "buffer": 0.0,
        }

    def _calculate_confluence_score(self, patterns: dict) -> int:
        """
        Calculate confluence score from all detected patterns.

        Score calculation:
        - +1 if trend bias is LONG or SHORT (not NEUTRAL)
        - +1 if BOS detected (UP or DOWN)
        - +1 if OB touched
        - +1 if breakout detected
        - +1 if momentum confirmed (bullish or bearish)

        Args:
            patterns: Dictionary of all pattern detection results

        Returns:
            Confluence score (0-5)
        """
        score = 0

        # Trend bias (+1 if directional)
        if patterns["trend_bias"] in [TREND_LONG, TREND_SHORT]:
            score += 1

        # BOS (+1 if detected)
        if patterns["bos"] in ["UP", "DOWN"]:
            score += 1

        # OB touch (+1 if any OB touched)
        if patterns["ob_touch"].get("touched", False):
            score += 1

        # Breakout (+1 if detected)
        if patterns["breakout"].get("detected", False):
            score += 1

        # Momentum (+1 if bullish or bearish)
        if patterns["momentum"].get("bullish") or patterns["momentum"].get("bearish"):
            score += 1

        return score

    def _determine_window_state(
        self, score: int, patterns: dict
    ) -> tuple[bool, bool, str]:
        """
        Determine if windows should open based on confluence score and pattern alignment.

        Opens long window if:
        - score >= 3 AND
        - majority of patterns are bullish

        Opens short window if:
        - score >= 3 AND
        - majority of patterns are bearish

        Args:
            score: Confluence score (0-5)
            patterns: Dictionary of all pattern detection results

        Returns:
            Tuple of (window_long, window_short, window_type)
        """
        if score < 3:
            return False, False, "NONE"

        # Count bullish vs bearish signals
        bullish_count = 0
        bearish_count = 0

        # Trend bias
        if patterns["trend_bias"] == TREND_LONG:
            bullish_count += 1
        elif patterns["trend_bias"] == TREND_SHORT:
            bearish_count += 1

        # BOS
        if patterns["bos"] == "UP":
            bullish_count += 1
        elif patterns["bos"] == "DOWN":
            bearish_count += 1

        # OB touch
        if patterns["ob_touch"].get("bullish"):
            bullish_count += 1
        if patterns["ob_touch"].get("bearish"):
            bearish_count += 1

        # Breakout
        breakout_type = patterns["breakout"].get("type")
        if breakout_type in ["BB_UPPER", "SWING_HIGH"]:
            bullish_count += 1
        elif breakout_type in ["BB_LOWER", "SWING_LOW"]:
            bearish_count += 1

        # Momentum
        if patterns["momentum"].get("bullish"):
            bullish_count += 1
        elif patterns["momentum"].get("bearish"):
            bearish_count += 1

        # Determine window direction
        if bullish_count > bearish_count:
            # Determine window type based on OB touch or breakout
            window_type = "OB_PULLBACK" if patterns["ob_touch"].get("bullish") else "BREAKOUT"
            return True, False, window_type
        elif bearish_count > bullish_count:
            window_type = "OB_PULLBACK" if patterns["ob_touch"].get("bearish") else "BREAKOUT"
            return False, True, window_type
        else:
            # Tie - no clear direction
            return False, False, "NONE"
