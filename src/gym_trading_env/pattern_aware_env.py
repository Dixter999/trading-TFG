"""
Pattern-Aware Observation Space for RL Training.

This module implements PatternAwareObservationSpace that creates a 26-feature
observation space for reinforcement learning training:
- 16 technical indicators (normalized)
- 6 pattern detection flags (binary 0/1)
- 4 market context features

TDD Phase: GREEN - Implementing class to pass tests

The observation space is compatible with gym.spaces.Box and provides
normalized features suitable for neural network training.
"""


import numpy as np
import pandas as pd
from gymnasium import spaces


class PatternAwareObservationSpace:
    """
    Enhanced observation space with pattern detection features.

    This class creates a 26-dimensional observation space that combines:
    1. Technical Indicators (16 features): Normalized price-based and
       oscillator indicators
    2. Pattern Detections (6 features): Binary flags for detected patterns
       (OB, LS, CHOCH)
    3. Market Context (4 features): Current position state and metrics

    The observation space is gym.spaces.Box compatible with shape (26,)
    and dtype float32.

    Example:
        >>> obs_space = PatternAwareObservationSpace()
        >>> print(obs_space.observation_space.shape)
        (26,)
        >>> print(len(obs_space.feature_names))
        26
        >>> observation = obs_space.get_observation(df, idx, position_info)
        >>> print(observation.shape)
        (26,)
    """

    def __init__(self):
        """
        Initialize observation space with 26 features.

        Creates a gym.spaces.Box with shape (26,) and dtype float32.
        Defines all 26 feature names in order.
        """
        # Create gym observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
        )

        # Define all 26 feature names
        self.feature_names = [
            # Technical Indicators (16)
            "rsi_14",
            "macd_line",
            "macd_signal",
            "macd_histogram",
            "atr_14",
            "sma_20",
            "sma_50",
            "sma_200",
            "ema_12",
            "ema_26",
            "ema_50",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "stoch_k",
            "stoch_d",
            # Pattern Detections (6 binary)
            "ob_bullish",
            "ob_bearish",
            "ls_bullish",
            "ls_bearish",
            "choch_bullish",
            "choch_bearish",
            # Market Context (4)
            "current_position",
            "unrealized_pnl_pct",
            "equity_drawdown_pct",
            "candles_in_position",
        ]

    def get_observation(
        self, df: pd.DataFrame, current_idx: int, position_info: dict
    ) -> np.ndarray:
        """
        Build complete observation vector from data and position state.

        Extracts technical indicators, pattern flags, and market context
        to create a 26-feature observation vector. All features are
        normalized and validated (no NaN or Inf values).

        Args:
            df: DataFrame with price data and technical indicators
            current_idx: Current index in DataFrame
            position_info: Dictionary with position state:
                - current_position: -1 (SHORT), 0 (FLAT), 1 (LONG)
                - unrealized_pnl_pct: Unrealized PnL as percentage
                - equity_drawdown_pct: Current drawdown percentage
                - candles_in_position: Number of candles in current position

        Returns:
            numpy array of shape (26,) with dtype float32

        Example:
            >>> df = load_h1_data()
            >>> position_info = {
            ...     'current_position': 1,
            ...     'unrealized_pnl_pct': 0.5,
            ...     'equity_drawdown_pct': -0.2,
            ...     'candles_in_position': 5
            ... }
            >>> obs = obs_space.get_observation(df, 100, position_info)
            >>> print(obs.shape)
            (26,)
        """
        # Get current row
        row = df.iloc[current_idx]
        current_price = row["close"]

        # Extract technical indicators (16 features)
        indicators = self._extract_indicators(row)
        normalized_indicators = self.normalize_indicators(indicators, current_price)

        # Extract pattern flags (6 features)
        pattern_flags = self.get_pattern_flags(df, current_idx)

        # Extract market context (4 features)
        market_context = self._extract_market_context(position_info)

        # Combine all features into observation vector
        observation = np.concatenate(
            [
                list(normalized_indicators.values()),
                list(pattern_flags.values()),
                list(market_context.values()),
            ]
        )

        # Convert to float32 and validate
        observation = observation.astype(np.float32)

        # Replace any NaN or Inf with 0
        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)

        return observation

    def _extract_indicators(self, row: pd.Series) -> dict[str, float]:
        """
        Extract technical indicators from DataFrame row.

        Args:
            row: DataFrame row with indicator columns

        Returns:
            Dictionary with indicator values
        """
        indicators = {
            "rsi_14": row.get("rsi_14", 50.0),
            "macd_line": row.get("macd_line", 0.0),
            "macd_signal": row.get("macd_signal", 0.0),
            "atr_14": row.get("atr_14", 0.0),
            "sma_20": row.get("sma_20", row["close"]),
            "sma_50": row.get("sma_50", row["close"]),
            "sma_200": row.get("sma_200", row["close"]),
            "ema_12": row.get("ema_12", row["close"]),
            "ema_26": row.get("ema_26", row["close"]),
            "ema_50": row.get("ema_50", row["close"]),
            "bb_upper_20": row.get("bb_upper_20", row["close"]),
            "bb_middle_20": row.get("bb_middle_20", row["close"]),
            "bb_lower_20": row.get("bb_lower_20", row["close"]),
            "stoch_k": row.get("stoch_k", 50.0),
            "stoch_d": row.get("stoch_d", 50.0),
        }

        # Calculate MACD histogram
        indicators["macd_histogram"] = (
            indicators["macd_line"] - indicators["macd_signal"]
        )

        return indicators

    def normalize_indicators(
        self, indicators: dict[str, float], current_price: float
    ) -> dict[str, float]:
        """
        Normalize technical indicators for neural network input.

        Normalization rules:
        - RSI: Already [0, 100], normalize to [0, 1] by dividing by 100
        - Stochastic K/D: Already [0, 100], normalize to [0, 1] by dividing by 100
        - Price-based (MACD, ATR, SMAs, EMAs, BBs): Normalize by current_price

        Args:
            indicators: Dictionary with raw indicator values
            current_price: Current close price for normalization

        Returns:
            Dictionary with normalized indicator values

        Example:
            >>> indicators = {'rsi_14': 70.0, 'macd_line': 0.0011}
            >>> normalized = obs_space.normalize_indicators(indicators, 1.1000)
            >>> print(normalized['rsi_14'])  # 0.7
            >>> print(normalized['macd_line'])  # 0.001
        """
        normalized = {}

        # Handle zero or negative price
        if current_price <= 0:
            # Return all zeros for safety
            return dict.fromkeys(indicators, 0.0)

        # RSI: [0, 100] → [0, 1]
        normalized["rsi_14"] = indicators["rsi_14"] / 100.0

        # MACD indicators: normalize by price
        normalized["macd_line"] = indicators["macd_line"] / current_price
        normalized["macd_signal"] = indicators["macd_signal"] / current_price
        normalized["macd_histogram"] = indicators["macd_histogram"] / current_price

        # ATR: normalize by price
        normalized["atr_14"] = indicators["atr_14"] / current_price

        # SMAs: normalize by price
        normalized["sma_20"] = indicators["sma_20"] / current_price
        normalized["sma_50"] = indicators["sma_50"] / current_price
        normalized["sma_200"] = indicators["sma_200"] / current_price

        # EMAs: normalize by price
        normalized["ema_12"] = indicators["ema_12"] / current_price
        normalized["ema_26"] = indicators["ema_26"] / current_price
        normalized["ema_50"] = indicators["ema_50"] / current_price

        # Bollinger Bands: normalize by price
        normalized["bb_upper"] = indicators["bb_upper_20"] / current_price
        normalized["bb_middle"] = indicators["bb_middle_20"] / current_price
        normalized["bb_lower"] = indicators["bb_lower_20"] / current_price

        # Stochastic: [0, 100] → [0, 1]
        normalized["stoch_k"] = indicators["stoch_k"] / 100.0
        normalized["stoch_d"] = indicators["stoch_d"] / 100.0

        return normalized

    def get_pattern_flags(self, df: pd.DataFrame, current_idx: int) -> dict[str, int]:
        """
        Extract binary pattern detection flags from DataFrame.

        Checks for presence of patterns at current index:
        - Order Blocks (bullish/bearish): Check if OB levels exist
        - Liquidity Sweeps (bullish/bearish): Not yet implemented (returns 0)
        - CHOCH (bullish/bearish): Not yet implemented (returns 0)

        Args:
            df: DataFrame with pattern detection columns
            current_idx: Current index in DataFrame

        Returns:
            Dictionary with 6 binary flags (0 or 1)

        Example:
            >>> flags = obs_space.get_pattern_flags(df, 100)
            >>> print(flags['ob_bullish'])  # 0 or 1
            >>> print(len(flags))  # 6
        """
        row = df.iloc[current_idx]

        flags = {}

        # Order Block detection: Check if OB levels are present (not NaN)
        flags["ob_bullish"] = (
            1
            if pd.notna(row.get("ob_bullish_high"))
            and pd.notna(row.get("ob_bullish_low"))
            else 0
        )
        flags["ob_bearish"] = (
            1
            if pd.notna(row.get("ob_bearish_high"))
            and pd.notna(row.get("ob_bearish_low"))
            else 0
        )

        # Liquidity Sweep detection: Not yet implemented
        # TODO: Implement when liquidity sweep detection is available
        flags["ls_bullish"] = 0
        flags["ls_bearish"] = 0

        # CHOCH detection: Not yet implemented
        # TODO: Implement when CHOCH detection columns are available
        flags["choch_bullish"] = 0
        flags["choch_bearish"] = 0

        return flags

    def _extract_market_context(self, position_info: dict) -> dict[str, float]:
        """
        Extract market context features from position state.

        Args:
            position_info: Dictionary with position information

        Returns:
            Dictionary with 4 market context features
        """
        context = {
            "current_position": float(position_info.get("current_position", 0)),
            "unrealized_pnl_pct": float(position_info.get("unrealized_pnl_pct", 0.0)),
            "equity_drawdown_pct": float(position_info.get("equity_drawdown_pct", 0.0)),
            "candles_in_position": min(
                float(position_info.get("candles_in_position", 0)), 100.0
            ),
        }

        return context
