"""
Normalizer for trading indicators.

Converts all price-based indicators to percentage changes/distances,
making the RL model price-agnostic and robust across different price regimes.
"""

import logging
from typing import Any


class Normalizer:
    """
    Converts trading indicators to percentage-based representation.

    Makes the RL model price-agnostic by normalizing all price-based
    indicators to percentage changes or percentage distances.

    All 26 indicators are normalized:
    - OHLCV: Percentage returns
    - Moving Averages: % distance from close
    - ATR: % of close
    - Bollinger Bands: % distance from close
    - MACD: % of close
    - RSI/Stochastic: Pass-through (already 0-100)
    - Order Blocks: % distance from close (NULL-safe)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def normalize_observation(
        self,
        obs: dict[str, float | None],
        prev_obs: dict[str, float | None] | None = None,
    ) -> dict[str, float | None]:
        """
        Normalize all 26 indicators in observation.

        Args:
            obs: Current observation with raw values
            prev_obs: Previous observation (needed for returns)

        Returns:
            Normalized observation with percentage values

        Example:
            >>> normalizer = Normalizer()
            >>> prev_obs = {'close': 1.1600, 'volume': 1000}
            >>> obs = {'open': 1.1610, 'high': 1.1615, 'low': 1.1605,
            ...        'close': 1.1608, 'volume': 1050, ...}
            >>> normalized = normalizer.normalize_observation(obs, prev_obs)
        """
        result = {}

        # OHLCV (requires prev_obs)
        ohlcv_result = self.normalize_ohlcv(obs, prev_obs)
        result.update(ohlcv_result)

        # Moving Averages
        ma_result = self.normalize_moving_averages(obs)
        result.update(ma_result)

        # Volatility (ATR)
        vol_result = self.normalize_volatility(obs)
        result.update(vol_result)

        # Bollinger Bands
        bb_result = self.normalize_bollinger_bands(obs)
        result.update(bb_result)

        # MACD
        macd_result = self.normalize_macd(obs)
        result.update(macd_result)

        # RSI/Stochastic
        rsi_stoch_result = self.normalize_rsi_stoch(obs)
        result.update(rsi_stoch_result)

        # Order Blocks (legacy)
        ob_result = self.normalize_order_blocks(obs)
        result.update(ob_result)

        # Order Block Features (new hybrid RL features)
        ob_features_result = self.normalize_order_block_features(obs)
        result.update(ob_features_result)

        return result

    def normalize_ohlcv(
        self,
        obs: dict[str, Any],
        prev_obs: dict[str, Any] | None = None,
    ) -> dict[str, float | None]:
        """
        Convert OHLCV to percentage returns.

        Formulas:
            open_pct = (open - prev_close) / prev_close * 100
            high_pct = (high - open) / open * 100
            low_pct = (low - open) / open * 100
            close_pct = (close - open) / open * 100
            volume_pct = (volume - prev_volume) / prev_volume * 100

        Args:
            obs: Current observation
            prev_obs: Previous observation (required for returns)

        Returns:
            Dictionary with OHLCV percentage returns
        """
        result = {}

        # Extract values
        open_val = obs.get("open")
        high_val = obs.get("high")
        low_val = obs.get("low")
        close_val = obs.get("close")
        volume_val = obs.get("volume")

        prev_close = prev_obs.get("close") if prev_obs else None
        prev_volume = prev_obs.get("volume") if prev_obs else None

        # Calculate percentage returns
        result["open_pct"] = self._normalize_distance(open_val, prev_close)
        result["high_pct"] = self._normalize_distance(high_val, open_val)
        result["low_pct"] = self._normalize_distance(low_val, open_val)
        result["close_pct"] = self._normalize_distance(close_val, open_val)
        result["volume_pct"] = self._normalize_distance(volume_val, prev_volume)

        return result

    def normalize_moving_averages(self, obs: dict[str, Any]) -> dict[str, float | None]:
        """
        Convert moving averages to % distance from close.

        Formula:
            ma_pct = (close - ma) / close * 100

        Args:
            obs: Current observation

        Returns:
            Dictionary with MA percentage distances
        """
        close = obs.get("close")

        result = {}
        mas = ["sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "ema_50"]

        for ma_name in mas:
            ma_value = obs.get(ma_name)
            # Formula: (close - ma) / close * 100
            # Note: close is the reference (denominator), not ma_value
            if close is None or ma_value is None:
                result[f"{ma_name}_pct"] = None
            elif close == 0:
                self.logger.warning(
                    f"Division by zero: ({close} - {ma_value}) / {close}"
                )
                result[f"{ma_name}_pct"] = None
            else:
                # Convert Decimal to float (PostgreSQL returns Decimal for numeric columns)
                close_float = float(close)
                ma_float = float(ma_value)
                pct = (close_float - ma_float) / close_float * 100
                if abs(pct) > 1000:
                    self.logger.warning(
                        f"Extreme value detected: {pct:.2f}% (close={close}, ma={ma_value})"
                    )
                result[f"{ma_name}_pct"] = pct

        return result

    def normalize_volatility(self, obs: dict[str, Any]) -> dict[str, float | None]:
        """
        Convert ATR to % of close.

        Formula:
            atr_14_pct = (atr_14 / close) * 100

        Args:
            obs: Current observation

        Returns:
            Dictionary with ATR percentage
        """
        close = obs.get("close")
        atr = obs.get("atr_14")

        return {"atr_14_pct": self._normalize_ratio(atr, close)}

    def normalize_bollinger_bands(self, obs: dict[str, Any]) -> dict[str, float | None]:
        """
        Convert Bollinger Bands to % distance from close.

        Formula:
            bb_pct = (bb - close) / close * 100

        Args:
            obs: Current observation

        Returns:
            Dictionary with BB percentage distances
        """
        close = obs.get("close")

        result = {}
        bbs = ["bb_upper_20", "bb_middle_20", "bb_lower_20"]

        for bb_name in bbs:
            bb_value = obs.get(bb_name)
            result[f"{bb_name}_pct"] = self._normalize_distance(bb_value, close)

        return result

    def normalize_macd(self, obs: dict[str, Any]) -> dict[str, float | None]:
        """
        Convert MACD indicators to % of close.

        Formula:
            macd_pct = (macd / close) * 100

        Args:
            obs: Current observation

        Returns:
            Dictionary with MACD percentages
        """
        close = obs.get("close")

        result = {}
        macds = ["macd_line", "macd_signal", "macd_histogram"]

        for macd_name in macds:
            macd_value = obs.get(macd_name)
            result[f"{macd_name}_pct"] = self._normalize_ratio(macd_value, close)

        return result

    def normalize_rsi_stoch(self, obs: dict[str, Any]) -> dict[str, float | None]:
        """
        Pass-through RSI and Stochastic (already normalized 0-100).

        Args:
            obs: Current observation

        Returns:
            Dictionary with RSI/Stochastic values (unchanged)
        """
        return {
            "rsi_14": obs.get("rsi_14"),
            "stoch_k": obs.get("stoch_k"),
            "stoch_d": obs.get("stoch_d"),
        }

    def normalize_order_blocks(self, obs: dict[str, Any]) -> dict[str, float | None]:
        """
        Convert order blocks to % distance from close (NULL-safe).

        Formula:
            ob_pct = (ob - close) / close * 100 if ob is not None else None

        Args:
            obs: Current observation

        Returns:
            Dictionary with OB percentage distances (NULL-safe)
        """
        close = obs.get("close")

        result = {}
        obs_blocks = [
            "ob_bullish_high",
            "ob_bullish_low",
            "ob_bearish_high",
            "ob_bearish_low",
        ]

        for ob_name in obs_blocks:
            ob_value = obs.get(ob_name)
            result[f"{ob_name}_pct"] = self._normalize_distance(ob_value, close)

        return result

    def normalize_order_block_features(
        self, obs: dict[str, Any]
    ) -> dict[str, float | None]:
        """
        Pass through Order Block features (already normalized).

        Features:
        - ob_direction: -1.0 (SHORT), 0.0 (NONE), +1.0 (LONG)
        - ob_strength: 0.0 to 1.0 (imbalance ratio)
        - ob_distance_pips: 0.0 to ~200 pips (needs scaling)
        - ob_age_bars: 0 to ~100 bars (needs scaling)
        - ob_is_valid: 0.0 or 1.0 (boolean flag)

        Args:
            obs: Current observation

        Returns:
            Dictionary with Order Block features (pass-through or scaled)
        """
        result = {}

        # Direction: already in [-1, 0, +1] range
        result['ob_direction'] = obs.get('ob_direction', 0.0)

        # Strength: already in [0, 1] range
        result['ob_strength'] = obs.get('ob_strength', 0.0)

        # Distance: scale pips to reasonable range (0-200 pips → 0-100 units)
        distance_pips = obs.get('ob_distance_pips', 0.0)
        if distance_pips is not None:
            # Convert Decimal to float (PostgreSQL returns Decimal for numeric columns)
            result['ob_distance_pips'] = min(float(distance_pips) / 2.0, 100.0)
        else:
            result['ob_distance_pips'] = 0.0

        # Age: scale bars to reasonable range (0-100 bars → 0-50 units)
        age_bars = obs.get('ob_age_bars', 0.0)
        if age_bars is not None:
            # Convert Decimal to float (PostgreSQL returns Decimal for numeric columns)
            result['ob_age_bars'] = min(float(age_bars) / 2.0, 50.0)
        else:
            result['ob_age_bars'] = 0.0

        # Valid flag: already in [0, 1] range
        result['ob_is_valid'] = obs.get('ob_is_valid', 0.0)

        return result

    # Helper methods

    def _normalize_distance(
        self, value: float | None, reference: float | None
    ) -> float | None:
        """
        Calculate percentage distance: (value - reference) / reference * 100.

        Clips extreme values to ±100% to prevent neural network slowdown.

        Args:
            value: Value to normalize
            reference: Reference value

        Returns:
            Percentage distance (clipped to ±100%) or None if invalid
        """
        # NULL handling
        if value is None or reference is None:
            return None

        # Convert Decimal to float (PostgreSQL returns Decimal for numeric columns)
        value = float(value)
        reference = float(reference)

        # Division by zero check
        if reference == 0:
            self.logger.warning(
                f"Division by zero: ({value} - {reference}) / {reference}"
            )
            return None

        # Calculate percentage
        pct = (value - reference) / reference * 100

        # Extreme value detection and clipping
        # Commented out to test performance impact
        # if abs(pct) > 1000:
        #     self.logger.warning(
        #         f"Extreme value detected: {pct:.2f}% (value={value}, reference={reference})"
        #     )

        # Clip to ±100% to prevent neural network slowdown
        return max(-100.0, min(100.0, pct))

    def _normalize_ratio(
        self, value: float | None, reference: float | None
    ) -> float | None:
        """
        Calculate percentage ratio: (value / reference) * 100.

        Clips extreme values to ±100% to prevent neural network slowdown.

        Args:
            value: Value to normalize
            reference: Reference value

        Returns:
            Percentage ratio (clipped to ±100%) or None if invalid
        """
        # NULL handling
        if value is None or reference is None:
            return None

        # Convert Decimal to float (PostgreSQL returns Decimal for numeric columns)
        value = float(value)
        reference = float(reference)

        # Division by zero check
        if reference == 0:
            self.logger.warning(f"Division by zero: {value} / {reference}")
            return None

        # Calculate percentage
        pct = (value / reference) * 100

        # Extreme value detection and clipping
        # Commented out to test performance impact
        # if abs(pct) > 1000:
        #     self.logger.warning(
        #         f"Extreme value detected: {pct:.2f}% (value={value}, reference={reference})"
        #     )

        # Clip to ±100% to prevent neural network slowdown
        return max(-100.0, min(100.0, pct))
