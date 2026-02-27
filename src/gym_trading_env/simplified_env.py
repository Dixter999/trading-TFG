"""
Simplified Trading Environment - NO Order Blocks

This environment removes Order Block features to simplify training:
- Removes 8 OB observation dimensions
- Removes OB feature extraction overhead
- Removes hybrid action complexity
- Uses simple 3-action space (hold/buy/sell)

Purpose: Prove concept works before adding complexity.

Designated for: Track 7C (baseline PPO training)
Order Blocks: Track 7D (future enhancement)
"""

import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from .datafeed import DataFeed
from .normalizer import Normalizer
from .data_splitter import DataSplitter

logger = logging.getLogger(__name__)


# Constants
VALID_SPLITS = ["train", "val", "eval"]
NUM_ACTIONS = 3  # Simple: hold (0), buy (1), sell (2)
OBS_SPACE_LOW = -1000.0
OBS_SPACE_HIGH = 1000.0
EURUSD_SPREAD_PIPS = 2
SLIPPAGE_PIPS_RANGE = (0, 1)
MIN_DATA_POINTS = 2
TIMESTAMP_COLUMN = "rate_time"
POSITION_SIZE = 100000


class SimplifiedTradingEnv(gym.Env):
    """
    Simplified EURUSD H1 Trading Environment.

    Removes Order Block features and complexity, using only:
    - OHLC data from PostgreSQL
    - Essential technical indicators from database
    - Simple buy/sell/hold actions
    - PnL-based rewards

    This is for Track 7C proof-of-concept training.
    Order Block features will be added in Track 7D.
    """

    def __init__(
        self,
        datafeed: DataFeed,
        normalizer: Normalizer,
        split: str = "train",
        reward_function: str = "pnl"
    ):
        """
        Initialize SimplifiedTradingEnv.

        Args:
            datafeed: DataFeed instance for loading market data
            normalizer: Normalizer instance for preprocessing
            split: Data split ('train', 'val', or 'eval')
            reward_function: Reward function type (only 'pnl' supported for now)
        """
        # Validate split
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got '{split}'")

        # Store parameters
        self.datafeed = datafeed
        self.normalizer = normalizer
        self.split = split
        self.logger = logging.getLogger(__name__)

        # Load and split data
        self._load_and_split_data()

        # Define action space: Discrete(3)
        # 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Build observation space dynamically
        self.observation_space = self._build_observation_space()

        # Trading parameters
        self.spread_pips = EURUSD_SPREAD_PIPS
        self.slippage_pips = SLIPPAGE_PIPS_RANGE

        logger.info(
            f"SimplifiedTradingEnv initialized: split={split}, "
            f"data_size={len(self.data)}, "
            f"obs_space_shape={self.observation_space.shape}"
        )

    def _load_and_split_data(self):
        """Load data from datafeed and apply train/val/eval split."""
        # Load all data
        full_data = self.datafeed.load_data()

        # Split data
        splitter = DataSplitter()
        train_df, val_df, eval_df = splitter.split(
            full_data,
            timestamp_col=TIMESTAMP_COLUMN,
            validate=False,
        )

        # Store all splits
        self.train_data = train_df.reset_index(drop=True)
        self.val_data = val_df.reset_index(drop=True)
        self.eval_data = eval_df.reset_index(drop=True)

        # Select appropriate split
        if self.split == "train":
            self.data = self.train_data
        elif self.split == "val":
            self.data = self.val_data
        else:  # eval
            self.data = self.eval_data

    def _build_observation_space(self) -> spaces.Box:
        """
        Build observation space dynamically based on first observation.

        SIMPLIFIED: No Order Block features included.
        Only uses OHLC + technical indicators from database.
        """
        if len(self.data) < MIN_DATA_POINTS:
            raise ValueError(
                f"Insufficient data: need at least {MIN_DATA_POINTS} observations"
            )

        # Get first observation
        first_obs_raw = self.data.iloc[0].to_dict()
        first_obs = {
            key: (None if pd.isna(value) else value)
            for key, value in first_obs_raw.items()
        }

        # NO ORDER BLOCK FEATURES ADDED HERE
        # This is the key difference from the original environment

        # Normalize to get indicator structure
        normalized_obs = self.normalizer.normalize_observation(first_obs, None)

        # Count non-NULL indicators
        non_null_indicators = [
            key for key, value in normalized_obs.items() if value is not None
        ]

        # Store indicator keys
        self.indicator_keys = non_null_indicators
        n_indicators = len(non_null_indicators)

        logger.info(
            f"Observation space: {n_indicators} indicators "
            f"(NO Order Blocks): {non_null_indicators}"
        )

        return spaces.Box(
            low=OBS_SPACE_LOW,
            high=OBS_SPACE_HIGH,
            shape=(n_indicators,),
            dtype=np.float32,
        )

    def _execute_trade(self, action: int, obs_raw: Dict[str, float]) -> float:
        """
        Execute trade with realistic spread and slippage.

        Simplified: No hybrid actions, no Order Block context.

        Args:
            action: Trading action (0=hold, 1=buy, 2=sell)
            obs_raw: Raw market observation

        Returns:
            reward: PnL from closing previous position
        """
        current_price = float(obs_raw["close"])

        # Close previous position and calculate PnL
        reward = self._close_position_if_exists(current_price)

        # Execute new position
        if action == 1:  # Buy
            entry_price = self._calculate_entry_price_buy(current_price)
            self._open_long_position(entry_price)
        elif action == 2:  # Sell
            entry_price = self._calculate_entry_price_sell(current_price)
            self._open_short_position(entry_price)
        # action == 0 (Hold) - no position change

        return reward

    def _close_position_if_exists(self, current_price: float) -> float:
        """Close existing position and calculate PnL."""
        if self.position != 0 and self.position_entry_price > 0:
            return self._calculate_reward(
                current_price=current_price,
                entry_price=self.position_entry_price,
                position=self.position,
                position_size=self.position_size
            )
        return 0.0

    def _open_long_position(self, entry_price: float) -> None:
        """Open LONG position."""
        self.position = 1
        self.position_entry_price = entry_price

    def _open_short_position(self, entry_price: float) -> None:
        """Open SHORT position."""
        self.position = -1
        self.position_entry_price = entry_price

    def _calculate_entry_price_buy(self, current_price: float) -> float:
        """Calculate entry price for buy order with spread and slippage."""
        import random
        current_price = float(current_price)
        spread_in_price = self._pips_to_price(self.spread_pips)
        slippage_in_price = self._pips_to_price(
            random.uniform(self.slippage_pips[0], self.slippage_pips[1])
        )
        return current_price + spread_in_price + slippage_in_price

    def _calculate_entry_price_sell(self, current_price: float) -> float:
        """Calculate entry price for sell order with spread and slippage."""
        import random
        current_price = float(current_price)
        spread_in_price = self._pips_to_price(self.spread_pips)
        slippage_in_price = self._pips_to_price(
            random.uniform(self.slippage_pips[0], self.slippage_pips[1])
        )
        return current_price - spread_in_price - slippage_in_price

    def _pips_to_price(self, pips: float) -> float:
        """Convert pips to price value (for EURUSD: 1 pip = 0.0001)."""
        return pips / 10000

    def _calculate_reward(
        self,
        current_price: float,
        entry_price: float,
        position: int,
        position_size: float
    ) -> float:
        """
        Calculate PnL-based reward.

        Simplified: No Order Block context, no SKIP bonuses.

        Args:
            current_price: Current market price
            entry_price: Position entry price
            position: Position direction (1=LONG, -1=SHORT, 0=FLAT)
            position_size: Position size in base currency

        Returns:
            reward: PnL in USD
        """
        if position == 0:
            return 0.0

        # Calculate price movement
        price_change = current_price - entry_price

        # Calculate PnL (LONG profits from rising prices, SHORT from falling)
        pnl = position * price_change * position_size

        return pnl

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode."""
        super().reset(seed=seed, options=options)

        # Load data for current split
        if self.split == "train":
            self.data = self.train_data
        elif self.split == "val":
            self.data = self.val_data
        elif self.split == "eval":
            self.data = self.eval_data

        # Initialize episode state
        self.current_step = 0
        self.position = 0  # FLAT
        self.position_entry_price = 0.0
        self.position_size = POSITION_SIZE
        self.prev_obs_raw = None

        # Get initial observation
        obs_raw = self._get_current_observation()

        # Normalize
        obs_normalized = self.normalizer.normalize_observation(obs_raw, prev_obs=None)

        # Convert to array
        obs_array = np.array(
            [obs_normalized.get(key) if obs_normalized.get(key) is not None else 0.0
             for key in self.indicator_keys],
            dtype=np.float32,
        )

        # Validate
        if np.isnan(obs_array).any():
            nan_indices = np.where(np.isnan(obs_array))[0]
            nan_keys = [self.indicator_keys[i] for i in nan_indices]
            raise ValueError(f"NaN values in reset observation: {nan_keys}")

        info = {"split": self.split, "step": self.current_step}
        return obs_array, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep.

        Simplified: No Order Block detection, no hybrid actions.

        Args:
            action: Simple action (0=hold, 1=buy, 2=sell)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get current observation
        obs_raw = self._get_current_observation()

        # Validate OHLC
        self._validate_ohlc(obs_raw)

        # Execute trade (simplified - no OB logic)
        reward = self._execute_trade(action, obs_raw)

        # Normalize observation
        obs_normalized = self.normalizer.normalize_observation(obs_raw, self.prev_obs_raw)

        # Convert to array
        obs_array = np.array(
            [obs_normalized.get(key) if obs_normalized.get(key) is not None else 0.0
             for key in self.indicator_keys],
            dtype=np.float32,
        )

        # Validate
        if np.isnan(obs_array).any():
            nan_indices = np.where(np.isnan(obs_array))[0]
            nan_keys = [self.indicator_keys[i] for i in nan_indices]
            raise ValueError(f"NaN values in step observation: {nan_keys}")

        # Update state
        self.prev_obs_raw = obs_raw
        self.current_step += 1

        # Check termination
        terminated = self.current_step >= len(self.data)
        truncated = False

        # CRITICAL FIX: Force-close open positions at episode end
        # Without this, models don't get reward feedback for their trades!
        if terminated and self.position != 0:
            current_price = float(obs_raw["close"])
            final_reward = self._close_position_if_exists(current_price)
            reward += final_reward
            info = {
                "step": self.current_step,
                "position": 0,  # Position is now closed
                "final_position_closed": True,
                "final_position_pnl": final_reward
            }
            # Reset position
            self.position = 0
            self.position_entry_price = 0.0
        else:
            info = {"step": self.current_step, "position": self.position}

        return obs_array, reward, terminated, truncated, info

    def _validate_ohlc(self, obs: Dict[str, float]) -> None:
        """Validate OHLC price relationships."""
        high = obs.get("high")
        low = obs.get("low")
        open_price = obs.get("open")
        close = obs.get("close")

        if high < low:
            raise ValueError(f"Invalid OHLC: high ({high}) < low ({low})")
        if high < close:
            raise ValueError(f"Invalid OHLC: high ({high}) < close ({close})")
        if high < open_price:
            raise ValueError(f"Invalid OHLC: high ({high}) < open ({open_price})")
        if low > close:
            raise ValueError(f"Invalid OHLC: low ({low}) > close ({close})")
        if low > open_price:
            raise ValueError(f"Invalid OHLC: low ({low}) > open ({open_price})")

    def _get_current_observation(self) -> Dict[str, Any]:
        """
        Get current observation at current timestep.

        SIMPLIFIED: NO Order Block features added.
        Only returns OHLC + technical indicators from database.
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call reset() first.")

        if self.current_step >= len(self.data):
            raise RuntimeError(
                f"Step {self.current_step} out of bounds (data length: {len(self.data)})"
            )

        # Get current row
        row = self.data.iloc[self.current_step]
        obs_dict = row.to_dict()

        # Replace NaN with None
        obs_dict = {
            key: (None if pd.isna(value) else value)
            for key, value in obs_dict.items()
        }

        # NO ORDER BLOCK FEATURES ADDED
        # This is the key simplification

        return obs_dict
