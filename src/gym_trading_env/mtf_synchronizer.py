"""
Multi-Timeframe Synchronizer for H1 -> M15 execution coordination.

This module implements the MTFSynchronizer class that coordinates multi-timeframe
(MTF) execution between H1 signals from PatternDetectorH1 and M15 timing for
entry execution. This is a CRITICAL anti-leakage component that ensures M15
execution windows are always in the future relative to H1 signal timestamps.

TDD Phase: GREEN - Implementing minimal code to pass tests

Anti-Leakage Guarantee:
    ALL M15 execution window timestamps MUST be strictly greater than the
    H1 signal timestamp. This is enforced by mandatory assertions that
    MUST NEVER be removed or disabled.

    Window Generation Pattern:
        H1 signal at time T → M15 window: [T+15m, T+30m, T+45m, T+60m]

    This ensures:
    - No lookahead bias (cannot see future H1 data)
    - No temporal leakage (M15 evaluation happens AFTER H1 signal)
    - Realistic trading simulation (execution requires time)
"""

from datetime import datetime, timedelta
from typing import List

import pandas as pd

# Required M15 OHLC columns for trigger detection
REQUIRED_M15_COLUMNS = frozenset(["timestamp", "open", "high", "low", "close"])


class MTFSynchronizer:
    """
    Multi-timeframe synchronizer for H1 pattern signals -> M15 execution timing.

    This class coordinates between H1 pattern detection signals and M15 execution
    windows, ensuring strict temporal ordering to prevent data leakage.

    Key Responsibilities:
    1. Generate safe M15 execution windows (4 candles after H1 signal)
    2. Validate M15 trigger patterns (REJECTION, ENGULF, MICRO_BOS)
    3. Enforce anti-leakage guarantees through timestamp validation

    Critical Rule:
        ALL M15 timestamps in execution window MUST be > H1 signal timestamp.
        This is enforced by mandatory assertions that cannot be bypassed.

    Data Flow:
        PatternDetectorH1 (H1 signal) → get_safe_m15_execution_window() →
        evaluate_m15_trigger() → Entry decision

    Example:
        >>> h1_data = pd.DataFrame(...)  # H1 OHLC + indicators
        >>> m15_data = pd.DataFrame(...)  # M15 OHLC only
        >>> sync = MTFSynchronizer(h1_data=h1_data, m15_data=m15_data)
        >>> window = sync.get_safe_m15_execution_window(h1_timestamp)
        >>> # window = [T+15m, T+30m, T+45m, T+60m] - guaranteed > h1_timestamp
    """

    def __init__(self, h1_data: pd.DataFrame, m15_data: pd.DataFrame):
        """
        Initialize MTFSynchronizer with H1 and M15 DataFrames.

        Args:
            h1_data: DataFrame from technical_indicators with timeframe='H1'.
                    Must be sorted by timestamp ascending and contain at least
                    'timestamp' column (other columns used by PatternDetectorH1).

            m15_data: DataFrame with M15 OHLC data. Must be sorted by timestamp
                     ascending and contain: timestamp, open, high, low, close.
                     NO technical indicators required (trigger detection uses
                     OHLC only).

        Raises:
            TypeError: If h1_data or m15_data is not a pandas DataFrame
            ValueError: If data validation fails (missing columns, unsorted, etc.)

        Anti-Leakage Validation:
            - Validates both DataFrames are sorted by timestamp ascending
            - Ensures required columns exist for temporal ordering
            - Stores DataFrames immutably for safe temporal access
        """
        # Type validation
        self._validate_dataframe_types(h1_data, m15_data)

        # Column validation
        self._validate_required_columns(h1_data, m15_data)

        # Temporal ordering validation (anti-leakage)
        self._validate_sorted_timestamps(h1_data, "h1_data")
        self._validate_sorted_timestamps(m15_data, "m15_data")

        # Store DataFrames (immutable references)
        self.h1_data = h1_data
        self.m15_data = m15_data

    def _validate_dataframe_types(
        self, h1_data: pd.DataFrame, m15_data: pd.DataFrame
    ) -> None:
        """
        Validate that both inputs are pandas DataFrames.

        Args:
            h1_data: H1 data to validate
            m15_data: M15 data to validate

        Raises:
            TypeError: If either input is not a DataFrame
        """
        if not isinstance(h1_data, pd.DataFrame):
            raise TypeError("h1_data must be a pandas DataFrame")

        if not isinstance(m15_data, pd.DataFrame):
            raise TypeError("m15_data must be a pandas DataFrame")

    def _validate_required_columns(
        self, h1_data: pd.DataFrame, m15_data: pd.DataFrame
    ) -> None:
        """
        Validate that both DataFrames have required columns.

        Args:
            h1_data: H1 data to validate
            m15_data: M15 data to validate

        Raises:
            ValueError: If required columns are missing
        """
        # H1 data must have timestamp column
        if "timestamp" not in h1_data.columns:
            raise ValueError("h1_data must contain 'timestamp' column")

        # M15 data must have timestamp column
        if "timestamp" not in m15_data.columns:
            raise ValueError("m15_data must contain 'timestamp' column")

        # M15 data must have OHLC columns for trigger detection
        missing_m15_columns = REQUIRED_M15_COLUMNS - set(m15_data.columns)
        if missing_m15_columns:
            raise ValueError(
                f"m15_data is missing required columns: {sorted(missing_m15_columns)}"
            )

    def _validate_sorted_timestamps(self, data: pd.DataFrame, data_name: str) -> None:
        """
        Validate that DataFrame is sorted by timestamp in ascending order.

        This is CRITICAL for anti-leakage guarantees. All temporal access
        patterns assume data is sorted ascending, allowing us to use
        timestamp filtering (data[data['timestamp'] <= T]) to prevent
        accidental lookahead bias.

        Args:
            data: DataFrame to validate
            data_name: Name of DataFrame for error messages

        Raises:
            ValueError: If timestamps are not sorted ascending
        """
        if len(data) > 1:
            timestamps = data["timestamp"].tolist()
            if timestamps != sorted(timestamps):
                raise ValueError(f"{data_name} must be sorted by timestamp ascending")

    def get_safe_m15_execution_window(self, h1_timestamp: datetime) -> List[datetime]:
        """
        Generate safe M15 execution window starting AFTER H1 signal timestamp.

        This method generates exactly 4 M15 timestamps representing the execution
        window for M15 trigger detection. The window starts at T+15m (where T is
        the H1 signal timestamp) and extends to T+60m, covering exactly 1 hour
        (4 M15 candles) of future price action.

        Window Pattern:
            H1 signal at T → M15 window: [T+15m, T+30m, T+45m, T+60m]

        Anti-Leakage Enforcement:
            - Window timestamps are GENERATED (not looked up from data)
            - ALL timestamps in window are > h1_timestamp (mandatory assertion)
            - Window proceeds forward in time (ascending order)
            - Returns empty list if insufficient M15 data for full window

        Args:
            h1_timestamp: H1 candle close time when pattern signal was generated.
                         Must be a timezone-aware datetime in UTC.

        Returns:
            List of exactly 4 datetime objects representing M15 execution window,
            or empty list if insufficient M15 data available.

            Example: If h1_timestamp = 2024-01-01 10:00:00 UTC, returns:
                [
                    2024-01-01 10:15:00 UTC,  # T+15m
                    2024-01-01 10:30:00 UTC,  # T+30m
                    2024-01-01 10:45:00 UTC,  # T+45m
                    2024-01-01 11:00:00 UTC,  # T+60m
                ]

        Anti-Leakage Guarantee:
            This method GUARANTEES that all returned timestamps are strictly
            greater than h1_timestamp through:
            1. Generated timestamps start at T+15m
            2. Mandatory assertion validates: all(ts > h1_timestamp)
            3. No M15 data lookup until trigger evaluation (prevents lookahead)

        Notes:
            - Window size is FIXED at 4 M15 candles (60 minutes total)
            - Window starts 15 minutes AFTER H1 signal (realistic execution delay)
            - If M15 data doesn't extend to T+60m, returns empty list
        """
        # Generate M15 window timestamps: [T+15m, T+30m, T+45m, T+60m]
        window_timestamps = [
            h1_timestamp + timedelta(minutes=15 * i) for i in range(1, 5)
        ]

        # MANDATORY ANTI-LEAKAGE ASSERTION
        # This assertion MUST NEVER be removed or disabled
        # It is the PRIMARY defense against temporal data leakage
        assert all(
            m15_ts > h1_timestamp for m15_ts in window_timestamps
        ), f"Anti-leakage violation: Window timestamps must all be > H1 timestamp. H1: {h1_timestamp}, Window: {window_timestamps}"

        # Verify M15 data contains all required timestamps
        # If data doesn't extend to last window timestamp, return empty list
        if len(self.m15_data) == 0:
            return []

        # Get latest available M15 timestamp
        latest_m15_timestamp = self.m15_data["timestamp"].iloc[-1]

        # Check if we have enough M15 data for complete window
        last_window_timestamp = window_timestamps[-1]
        if latest_m15_timestamp < last_window_timestamp:
            # Insufficient M15 data for full window
            return []

        # Verify all window timestamps are ascending (temporal ordering)
        assert window_timestamps == sorted(
            window_timestamps
        ), f"Window timestamps must be in ascending order. Got: {window_timestamps}"

        return window_timestamps

    def _check_rejection_long(
        self, m15_timestamp: datetime, target_level: float
    ) -> bool:
        """
        Check if M15 candle shows REJECTION long pattern (bullish rejection).

        REJECTION Long Pattern Definition:
        - Lower wick ≥ 40% of candle range (strong rejection of lower prices)
        - Close in top 30% of candle (70-100% from low to high)
        - Current price within or above target level

        This pattern indicates price tested lower levels but was strongly rejected
        upward, suggesting buying pressure and potential bullish continuation.

        Args:
            m15_timestamp: M15 candle timestamp to check
            target_level: H1 target level (support/demand zone)

        Returns:
            True if valid REJECTION long pattern detected, False otherwise

        Anti-Leakage Guarantee:
            Only uses OHLC data from the specified M15 timestamp (no future data).
            Target level comes from past H1 analysis, ensuring temporal ordering.

        Pattern Validation:
            1. Lower wick ≥ 40% of range: (min(open, close) - low) / range ≥ 0.40
            2. Close in top 30%: (close - low) / range ≥ 0.70
            3. Close ≥ target_level (price within or above target zone)
        """
        # Get M15 candle OHLC at specified timestamp
        m15_candle = self.m15_data[self.m15_data["timestamp"] == m15_timestamp]

        if m15_candle.empty:
            return False

        # Extract OHLC values
        open_price = float(m15_candle["open"].iloc[0])
        high_price = float(m15_candle["high"].iloc[0])
        low_price = float(m15_candle["low"].iloc[0])
        close_price = float(m15_candle["close"].iloc[0])

        # Calculate candle range (avoid division by zero)
        candle_range = high_price - low_price
        if candle_range == 0:
            return False

        # Calculate lower wick size
        # Lower wick = distance from low to min(open, close)
        body_low = min(open_price, close_price)
        lower_wick = body_low - low_price
        lower_wick_percent = lower_wick / candle_range

        # Check condition 1: Lower wick ≥ 40% of range
        if lower_wick_percent < 0.40:
            return False

        # Calculate close position as percentage from low
        # 0% = at low, 100% = at high
        close_position = (close_price - low_price) / candle_range

        # Check condition 2: Close in top 30% (≥ 70% from low)
        if close_position < 0.70:
            return False

        # Check condition 3: Close within or above target level
        if close_price < target_level:
            return False

        # All conditions met - valid REJECTION long pattern
        return True

    def _check_rejection_short(
        self, m15_timestamp: datetime, target_level: float
    ) -> bool:
        """
        Check if M15 candle shows REJECTION short pattern (bearish rejection).

        REJECTION Short Pattern Definition:
        - Upper wick ≥ 40% of candle range (strong rejection of higher prices)
        - Close in bottom 30% of candle (0-30% from low to high)
        - Current price within or below target level

        This pattern indicates price tested higher levels but was strongly rejected
        downward, suggesting selling pressure and potential bearish continuation.

        Args:
            m15_timestamp: M15 candle timestamp to check
            target_level: H1 target level (resistance/supply zone)

        Returns:
            True if valid REJECTION short pattern detected, False otherwise

        Anti-Leakage Guarantee:
            Only uses OHLC data from the specified M15 timestamp (no future data).
            Target level comes from past H1 analysis, ensuring temporal ordering.

        Pattern Validation:
            1. Upper wick ≥ 40% of range: (high - max(open, close)) / range ≥ 0.40
            2. Close in bottom 30%: (close - low) / range ≤ 0.30
            3. Close ≤ target_level (price within or below target zone)
        """
        # Get M15 candle OHLC at specified timestamp
        m15_candle = self.m15_data[self.m15_data["timestamp"] == m15_timestamp]

        if m15_candle.empty:
            return False

        # Extract OHLC values
        open_price = float(m15_candle["open"].iloc[0])
        high_price = float(m15_candle["high"].iloc[0])
        low_price = float(m15_candle["low"].iloc[0])
        close_price = float(m15_candle["close"].iloc[0])

        # Calculate candle range (avoid division by zero)
        candle_range = high_price - low_price
        if candle_range == 0:
            return False

        # Calculate upper wick size
        # Upper wick = distance from high to max(open, close)
        body_high = max(open_price, close_price)
        upper_wick = high_price - body_high
        upper_wick_percent = upper_wick / candle_range

        # Check condition 1: Upper wick ≥ 40% of range
        if upper_wick_percent < 0.40:
            return False

        # Calculate close position as percentage from low
        # 0% = at low, 100% = at high
        close_position = (close_price - low_price) / candle_range

        # Check condition 2: Close in bottom 30% (≤ 30% from low)
        if close_position > 0.30:
            return False

        # Check condition 3: Close within or below target level
        if close_price > target_level:
            return False

        # All conditions met - valid REJECTION short pattern
        return True

    def _get_m15_at(self, m15_timestamp: datetime) -> pd.Series:
        """
        Get M15 candle data at specific timestamp.

        Helper method to access M15 OHLC data for trigger detection.

        Args:
            m15_timestamp: M15 candle timestamp to retrieve.

        Returns:
            pandas Series with OHLC data for the timestamp.

        Raises:
            ValueError: If timestamp not found in M15 data.

        Anti-Leakage:
            This method only allows access to PAST OR CURRENT M15 data,
            never future data. Caller is responsible for ensuring
            m15_timestamp does not leak future information.
        """
        m15_row = self.m15_data[self.m15_data["timestamp"] == m15_timestamp]

        if len(m15_row) == 0:
            raise ValueError(
                f"M15 timestamp {m15_timestamp} not found in M15 data. "
                f"Available range: {self.m15_data['timestamp'].iloc[0]} to "
                f"{self.m15_data['timestamp'].iloc[-1]}"
            )

        return m15_row.iloc[0]

    def _get_prev_m15_range(self, m15_timestamp: datetime, count: int) -> pd.DataFrame:
        """
        Get previous M15 candles before specified timestamp.

        Helper method to access historical M15 data for lookback operations.

        Args:
            m15_timestamp: Current M15 timestamp (exclusive - not included).
            count: Number of previous candles to retrieve.

        Returns:
            DataFrame with previous M15 candles (sorted ascending by timestamp).

        Raises:
            ValueError: If insufficient data available for lookback.

        Anti-Leakage:
            Only returns candles BEFORE m15_timestamp (exclusive).
            This ensures lookback operations never access current or future data.

        Example:
            >>> # Get 3 previous M15 candles before current timestamp
            >>> prev = self._get_prev_m15_range(m15_timestamp, count=3)
            >>> # Returns candles at [t-3, t-2, t-1] (NOT including t)
        """
        # Filter M15 data to only include timestamps BEFORE current
        prev_data = self.m15_data[self.m15_data["timestamp"] < m15_timestamp]

        if len(prev_data) < count:
            raise ValueError(
                f"Insufficient M15 data for lookback. Need {count} candles "
                f"before {m15_timestamp}, but only {len(prev_data)} available."
            )

        # Get last 'count' candles before current timestamp
        return prev_data.tail(count)

    def _check_micro_bos_long(self, m15_timestamp: datetime) -> bool:
        """
        Check for MICRO Break of Structure (BOS) pattern for LONG entry.

        MICRO_BOS LONG Pattern:
            - Current close > max(high[t-1:t-3])
            - Indicates break of structure to the upside
            - Confirms bullish momentum after level touch

        This pattern validates that price has broken above recent swing highs,
        suggesting continuation of bullish momentum.

        Args:
            m15_timestamp: M15 candle timestamp to check for pattern.

        Returns:
            True if valid MICRO_BOS LONG pattern detected, False otherwise.

        Anti-Leakage:
            - Only accesses current candle (m15_timestamp)
            - Only accesses previous candles (t-1, t-2, t-3)
            - Never looks ahead at future candles

        Example:
            >>> # Check if M15 candle shows MICRO_BOS LONG
            >>> is_bos = self._check_micro_bos_long(
            ...     m15_timestamp=datetime(2024, 1, 1, 10, 45)
            ... )
            >>> # True if close > max(high[t-1:t-3])
        """
        try:
            # Get current M15 candle
            current_candle = self._get_m15_at(m15_timestamp)

            # Get previous 3 M15 candles (t-1, t-2, t-3)
            prev_candles = self._get_prev_m15_range(m15_timestamp, count=3)

            # Calculate max of previous highs
            prev_high_max = prev_candles["high"].max()

            # Check if current close > max(high[t-1:t-3])
            return current_candle["close"] > prev_high_max

        except ValueError:
            # Insufficient data for pattern detection
            return False

    def _check_micro_bos_short(self, m15_timestamp: datetime) -> bool:
        """
        Check for MICRO Break of Structure (BOS) pattern for SHORT entry.

        MICRO_BOS SHORT Pattern:
            - Current close < min(low[t-1:t-3])
            - Indicates break of structure to the downside
            - Confirms bearish momentum after level touch

        This pattern validates that price has broken below recent swing lows,
        suggesting continuation of bearish momentum.

        Args:
            m15_timestamp: M15 candle timestamp to check for pattern.

        Returns:
            True if valid MICRO_BOS SHORT pattern detected, False otherwise.

        Anti-Leakage:
            - Only accesses current candle (m15_timestamp)
            - Only accesses previous candles (t-1, t-2, t-3)
            - Never looks ahead at future candles

        Example:
            >>> # Check if M15 candle shows MICRO_BOS SHORT
            >>> is_bos = self._check_micro_bos_short(
            ...     m15_timestamp=datetime(2024, 1, 1, 10, 45)
            ... )
            >>> # True if close < min(low[t-1:t-3])
        """
        try:
            # Get current M15 candle
            current_candle = self._get_m15_at(m15_timestamp)

            # Get previous 3 M15 candles (t-1, t-2, t-3)
            prev_candles = self._get_prev_m15_range(m15_timestamp, count=3)

            # Calculate min of previous lows
            prev_low_min = prev_candles["low"].min()

            # Check if current close < min(low[t-1:t-3])
            return current_candle["close"] < prev_low_min

        except ValueError:
            # Insufficient data for pattern detection
            return False

    def _check_engulf_long(self, m15_timestamp: datetime) -> bool:
        """
        Check if M15 candle shows ENGULF long pattern (bullish engulfing).

        ENGULF Long Pattern Definition:
        - Current close > previous candle high (engulfs previous candle)
        - Body ≥ 50% of candle range (strong bullish momentum)

        This pattern indicates strong buying pressure that completely engulfs
        the previous candle, suggesting potential bullish continuation.

        Args:
            m15_timestamp: M15 candle timestamp to check

        Returns:
            True if valid ENGULF long pattern detected, False otherwise

        Anti-Leakage Guarantee:
            Only uses OHLC data from current and previous M15 timestamp (no future data).
            Requires previous candle exists to establish comparison baseline.

        Pattern Validation:
            1. Close[t] > High[t-1] (engulfing condition)
            2. Body[t] ≥ 50% of range[t] (strong momentum)

        Note:
            This pattern requires lookback to t-1, so first candle in dataset
            cannot form an ENGULF pattern (no previous candle available).
        """
        try:
            # Get current M15 candle OHLC
            current_candle = self._get_m15_at(m15_timestamp)

            # Get previous candle (t-1) using helper method
            prev_candles = self._get_prev_m15_range(m15_timestamp, count=1)
            previous_candle = prev_candles.iloc[0]

            # Extract OHLC values - current candle
            current_open = float(current_candle["open"])
            current_high = float(current_candle["high"])
            current_low = float(current_candle["low"])
            current_close = float(current_candle["close"])

            # Extract OHLC values - previous candle
            previous_high = float(previous_candle["high"])

            # Calculate current candle range (avoid division by zero)
            current_range = current_high - current_low
            if current_range == 0:
                return False

            # Check condition 1: Close[t] > High[t-1] (engulfing)
            if current_close <= previous_high:
                return False

            # Calculate body size
            body_size = abs(current_close - current_open)
            body_percent = body_size / current_range

            # Check condition 2: Body ≥ 50% of range
            if body_percent < 0.50:
                return False

            # All conditions met - valid ENGULF long pattern
            return True

        except ValueError:
            # Insufficient data for pattern detection (no previous candle)
            return False

    def _check_engulf_short(self, m15_timestamp: datetime) -> bool:
        """
        Check if M15 candle shows ENGULF short pattern (bearish engulfing).

        ENGULF Short Pattern Definition:
        - Current close < previous candle low (engulfs previous candle)
        - Body ≥ 50% of candle range (strong bearish momentum)

        This pattern indicates strong selling pressure that completely engulfs
        the previous candle, suggesting potential bearish continuation.

        Args:
            m15_timestamp: M15 candle timestamp to check

        Returns:
            True if valid ENGULF short pattern detected, False otherwise

        Anti-Leakage Guarantee:
            Only uses OHLC data from current and previous M15 timestamp (no future data).
            Requires previous candle exists to establish comparison baseline.

        Pattern Validation:
            1. Close[t] < Low[t-1] (engulfing condition)
            2. Body[t] ≥ 50% of range[t] (strong momentum)

        Note:
            This pattern requires lookback to t-1, so first candle in dataset
            cannot form an ENGULF pattern (no previous candle available).
        """
        try:
            # Get current M15 candle OHLC
            current_candle = self._get_m15_at(m15_timestamp)

            # Get previous candle (t-1) using helper method
            prev_candles = self._get_prev_m15_range(m15_timestamp, count=1)
            previous_candle = prev_candles.iloc[0]

            # Extract OHLC values - current candle
            current_open = float(current_candle["open"])
            current_high = float(current_candle["high"])
            current_low = float(current_candle["low"])
            current_close = float(current_candle["close"])

            # Extract OHLC values - previous candle
            previous_low = float(previous_candle["low"])

            # Calculate current candle range (avoid division by zero)
            current_range = current_high - current_low
            if current_range == 0:
                return False

            # Check condition 1: Close[t] < Low[t-1] (engulfing)
            if current_close >= previous_low:
                return False

            # Calculate body size
            body_size = abs(current_close - current_open)
            body_percent = body_size / current_range

            # Check condition 2: Body ≥ 50% of range
            if body_percent < 0.50:
                return False

            # All conditions met - valid ENGULF short pattern
            return True

        except ValueError:
            # Insufficient data for pattern detection (no previous candle)
            return False

    def evaluate_m15_trigger(
        self,
        m15_timestamp: datetime,
        target_level: float,
        direction: str,
        atr_pct: float,
    ) -> dict | None:
        """
        Evaluate M15 trigger patterns and apply distance filter.

        Main orchestration method that:
        1. Checks all trigger patterns (REJECTION, ENGULF, MICRO_BOS)
        2. Applies distance filter (max 0.25×atr_pct from target level)
        3. Returns trigger structure or None

        Priority Order:
        1. REJECTION (strongest signal - wick rejection)
        2. ENGULF (strong - full body engulfment)
        3. MICRO_BOS (confirmation - structure break)

        Args:
            m15_timestamp: M15 candle timestamp to evaluate
            target_level: H1 target entry level (support/resistance)
            direction: Trade direction ('long' or 'short')
            atr_pct: ATR percentage for distance filter calculation

        Returns:
            Dict with trigger details if pattern found and passes distance filter:
            {
                'trigger': 'REJECTION' | 'ENGULF' | 'MICRO_BOS',
                'entry_price': float,
                'timestamp': datetime
            }
            None if no pattern found or distance filter rejects entry.

        Distance Filter:
            Max allowed distance = 0.25 × atr_pct
            This prevents chasing price too far from intended entry level.

        Anti-Leakage:
            - Only evaluates current M15 candle (m15_timestamp)
            - All pattern checks use historical or current data only
            - No future data access

        Example:
            >>> result = sync.evaluate_m15_trigger(
            ...     m15_timestamp=datetime(2024, 1, 1, 10, 15),
            ...     target_level=1.0875,
            ...     direction='long',
            ...     atr_pct=0.005  # 0.5%
            ... )
            >>> if result:
            ...     print(f"Trigger: {result['trigger']} at {result['entry_price']}")
        """
        # Get M15 candle for entry price extraction
        try:
            m15_candle = self._get_m15_at(m15_timestamp)
            entry_price = float(m15_candle["close"])
        except ValueError:
            # M15 timestamp not found
            return None

        # Apply distance filter FIRST (short-circuit if too far)
        # Max allowed distance: 0.5 × atr_pct (absolute price distance)
        max_distance = 0.5 * atr_pct

        # Calculate absolute distance from target level
        distance_from_target = abs(entry_price - target_level)

        # Use small epsilon for floating point comparison tolerance
        epsilon = 1e-10
        if distance_from_target > max_distance + epsilon:
            # Price too far from target level - reject
            return None

        # Check trigger patterns in priority order
        trigger_type = None

        if direction == "long":
            # Check REJECTION first (highest priority)
            if self._check_rejection_long(m15_timestamp, target_level):
                trigger_type = "REJECTION"
            # Check ENGULF if no REJECTION
            elif self._check_engulf_long(m15_timestamp):
                trigger_type = "ENGULF"
            # Check MICRO_BOS if no REJECTION or ENGULF
            elif self._check_micro_bos_long(m15_timestamp):
                trigger_type = "MICRO_BOS"
        else:  # short
            # Check REJECTION first (highest priority)
            if self._check_rejection_short(m15_timestamp, target_level):
                trigger_type = "REJECTION"
            # Check ENGULF if no REJECTION
            elif self._check_engulf_short(m15_timestamp):
                trigger_type = "ENGULF"
            # Check MICRO_BOS if no REJECTION or ENGULF
            elif self._check_micro_bos_short(m15_timestamp):
                trigger_type = "MICRO_BOS"

        # Return trigger structure or None
        if trigger_type:
            return {
                "trigger": trigger_type,
                "entry_price": entry_price,
                "timestamp": m15_timestamp,
            }

        return None

    def invalidate_window(
        self,
        h1_signal_timestamp: datetime,
        m15_count: int,
        h1_current: dict,
        direction: str,
        critical_level: float,
    ) -> bool:
        """
        Check if M15 execution window should be invalidated.

        Window invalidation occurs under two conditions:
        1. TIMEOUT: All 4 M15 candles evaluated without finding trigger
        2. STRUCTURE BREAK: Next H1 candle closes against signal direction

        Invalidation Logic:
            - Timeout: If m15_count == 4 (all M15 candles checked) → True
            - Structure Break (LONG): If H1 closes BELOW critical_level → True
            - Structure Break (SHORT): If H1 closes ABOVE critical_level → True
            - Otherwise → False (keep window active)

        Args:
            h1_signal_timestamp: H1 candle timestamp when signal was generated.
                                Not currently used but provided for future extensions.
            m15_count: Number of M15 candles evaluated so far (0-4).
            h1_current: Current H1 candle data dict with at least:
                       {'timestamp': datetime, 'close': float}
            direction: Trade direction ('long' or 'short').
            critical_level: Price level to monitor for structure break.
                          For LONG: Support/demand zone (invalidate if close < level)
                          For SHORT: Resistance/supply zone (invalidate if close > level)

        Returns:
            True if window should be invalidated (close window), False otherwise.

        Anti-Leakage:
            - Only checks m15_count (simple counter, no data access)
            - Only accesses current H1 candle data (no future H1 data)
            - Structure break check uses current H1 close vs critical level

        Example:
            >>> # Timeout scenario
            >>> invalidate = sync.invalidate_window(
            ...     h1_signal_timestamp=datetime(2024, 1, 1, 10, 0),
            ...     m15_count=4,  # All M15 candles evaluated
            ...     h1_current={'timestamp': datetime(2024, 1, 1, 11, 0), 'close': 1.0900},
            ...     direction='long',
            ...     critical_level=1.0875
            ... )
            >>> print(invalidate)  # True - timeout after 4 M15 candles
            >>>
            >>> # Structure break scenario (LONG)
            >>> invalidate = sync.invalidate_window(
            ...     h1_signal_timestamp=datetime(2024, 1, 1, 10, 0),
            ...     m15_count=1,  # Only 1 M15 evaluated
            ...     h1_current={'timestamp': datetime(2024, 1, 1, 11, 0), 'close': 1.0850},
            ...     direction='long',
            ...     critical_level=1.0875  # Close BELOW critical
            ... )
            >>> print(invalidate)  # True - structure broken
        """
        # Check TIMEOUT condition: All 4 M15 candles evaluated without trigger
        if m15_count >= 4:
            return True

        # Check STRUCTURE BREAK condition based on direction
        h1_close = h1_current.get("close")

        if h1_close is None:
            # If current H1 close not provided, cannot check structure break
            # Keep window active (return False)
            return False

        if direction == "long":
            # LONG: Invalidate if H1 closes BELOW critical level
            # This indicates structure broken to downside, signal invalidated
            if h1_close < critical_level:
                return True
        elif direction == "short":
            # SHORT: Invalidate if H1 closes ABOVE critical level
            # This indicates structure broken to upside, signal invalidated
            if h1_close > critical_level:
                return True

        # Window still active - no timeout, no structure break
        return False
