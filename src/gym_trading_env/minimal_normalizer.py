"""
Minimal Normalizer - NO Order Block features

This is a simplified version of Normalizer that:
- Normalizes OHLCV, MAs, ATR, BB, MACD, RSI, Stoch
- EXCLUDES all Order Block normalization
- Optimized for Track 7C baseline training

For Track 7D (with OBs), use the original normalizer.py instead.
"""

import logging
from typing import Any


class MinimalNormalizer:
    """
    Converts trading indicators to percentage-based representation (NO Order Blocks).

    Makes the RL model price-agnostic by normalizing all price-based
    indicators to percentage changes or percentage distances.

    Normalizes 17 essential indicators (NO OBs):
    - OHLCV: Percentage returns
    - Moving Averages: % distance from close
    - ATR: % of close
    - Bollinger Bands: % distance from close
    - MACD: % of close
    - RSI/Stochastic: Pass-through (already 0-100)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def normalize_observation(
        self,
        obs: dict[str, float | None],
        prev_obs: dict[str, float | None] | None = None,
    ) -> dict[str, float | None]:
        """
        Normalize 17 essential indicators (NO Order Blocks).

        Args:
            obs: Current observation with raw values
            prev_obs: Previous observation (needed for returns)

        Returns:
            Normalized observation with percentage values
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

        # NO ORDER BLOCK FEATURES - This is the key difference

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
            prev_obs: Previous observation (for computing changes)

        Returns:
            Dict with OHLCV percentage returns
        """
        result = {}

        # Extract current OHLCV
        open_val = obs.get("open")
        high_val = obs.get("high")
        low_val = obs.get("low")
        close_val = obs.get("close")
        volume_val = obs.get("volume")

        # If no previous observation, return 0% for all
        if prev_obs is None:
            result["open_pct"] = 0.0
            result["high_pct"] = 0.0
            result["low_pct"] = 0.0
            result["close_pct"] = 0.0
            result["volume_pct"] = 0.0
            return result

        # Extract previous OHLCV
        prev_close = prev_obs.get("close")
        prev_volume = prev_obs.get("volume")

        # Open return (from prev close)
        if open_val is not None and prev_close is not None and prev_close != 0:
            result["open_pct"] = (open_val - prev_close) / prev_close * 100
        else:
            result["open_pct"] = 0.0

        # High, Low, Close returns (from current open)
        if open_val is not None and open_val != 0:
            result["high_pct"] = (
                (high_val - open_val) / open_val * 100
                if high_val is not None
                else 0.0
            )
            result["low_pct"] = (
                (low_val - open_val) / open_val * 100
                if low_val is not None
                else 0.0
            )
            result["close_pct"] = (
                (close_val - open_val) / open_val * 100
                if close_val is not None
                else 0.0
            )
        else:
            result["high_pct"] = 0.0
            result["low_pct"] = 0.0
            result["close_pct"] = 0.0

        # Volume change
        if volume_val is not None and prev_volume is not None and prev_volume != 0:
            result["volume_pct"] = (volume_val - prev_volume) / prev_volume * 100
        else:
            result["volume_pct"] = 0.0

        return result

    def normalize_moving_averages(self, obs: dict[str, Any]) -> dict[str, float | None]:
        """
        Convert moving averages to % distance from close.

        Formula:
            ma_pct = (ma - close) / close * 100

        Args:
            obs: Current observation

        Returns:
            Dict with MA percentage distances
        """
        result = {}
        close = obs.get("close")

        ma_names = ["sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "ema_50"]

        for ma_name in ma_names:
            ma_value = obs.get(ma_name)
            result[f"{ma_name}_pct"] = self._normalize_distance(ma_value, close)

        return result

    def normalize_volatility(self, obs: dict[str, Any]) -> dict[str, float | None]:
        """
        Convert ATR to % of close.

        Formula:
            atr_pct = (atr / close) * 100

        Args:
            obs: Current observation

        Returns:
            Dict with ATR percentage
        """
        result = {}
        close = obs.get("close")
        atr = obs.get("atr_14")

        if atr is not None and close is not None and close != 0:
            result["atr_14_pct"] = (float(atr) / float(close)) * 100
        else:
            result["atr_14_pct"] = 0.0

        return result

    def normalize_bollinger_bands(self, obs: dict[str, Any]) -> dict[str, float | None]:
        """
        Convert Bollinger Bands to % distance from close.

        Formula:
            bb_pct = (bb - close) / close * 100

        Args:
            obs: Current observation

        Returns:
            Dict with BB percentage distances
        """
        result = {}
        close = obs.get("close")

        bb_names = ["bb_upper_20", "bb_middle_20", "bb_lower_20"]

        for bb_name in bb_names:
            bb_value = obs.get(bb_name)
            result[f"{bb_name}_pct"] = self._normalize_distance(bb_value, close)

        return result

    def normalize_macd(self, obs: dict[str, Any]) -> dict[str, float | None]:
        """
        Convert MACD to % of close.

        Formulas:
            macd_line_pct = (macd_line / close) * 100
            macd_signal_pct = (macd_signal / close) * 100
            macd_histogram_pct = (macd_histogram / close) * 100

        Args:
            obs: Current observation

        Returns:
            Dict with MACD percentage values
        """
        result = {}
        close = obs.get("close")

        macd_names = ["macd_line", "macd_signal", "macd_histogram"]

        for macd_name in macd_names:
            macd_value = obs.get(macd_name)
            if macd_value is not None and close is not None and close != 0:
                result[f"{macd_name}_pct"] = (float(macd_value) / float(close)) * 100
            else:
                result[f"{macd_name}_pct"] = 0.0

        return result

    def normalize_rsi_stoch(self, obs: dict[str, Any]) -> dict[str, float | None]:
        """
        Pass through RSI and Stochastic (already in 0-100 range).

        Args:
            obs: Current observation

        Returns:
            Dict with RSI/Stochastic values
        """
        result = {}

        # RSI (already 0-100) - convert Decimal to float
        rsi_val = obs.get("rsi_14", 50.0)
        result["rsi_14"] = float(rsi_val) if rsi_val is not None else 50.0

        # Stochastic (already 0-100) - convert Decimal to float
        stoch_k_val = obs.get("stoch_k", 50.0)
        result["stoch_k"] = float(stoch_k_val) if stoch_k_val is not None else 50.0

        stoch_d_val = obs.get("stoch_d", 50.0)
        result["stoch_d"] = float(stoch_d_val) if stoch_d_val is not None else 50.0

        return result

    def _normalize_distance(
        self, value: float | None, close: float | None
    ) -> float | None:
        """
        Helper: Convert absolute value to % distance from close (NULL-safe).

        Formula:
            distance_pct = (value - close) / close * 100

        Args:
            value: Value to normalize
            close: Current close price

        Returns:
            Percentage distance or 0.0 if NULL
        """
        if value is None or close is None or close == 0:
            return 0.0
        # Convert to float to handle Decimal types from database
        value_float = float(value)
        close_float = float(close)
        return (value_float - close_float) / close_float * 100
