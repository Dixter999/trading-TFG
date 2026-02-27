"""
EURUSD H1 Trading Environment for Reinforcement Learning.

This module implements a custom Gymnasium environment for EURUSD trading
on the H1 timeframe, integrating with dual databases and providing
normalized observations.

TDD Phase: GREEN - Minimal implementation to pass tests.

Stream A (COMPLETE): __init__(), _build_observation_space(), action_space
Stream B (COMPLETE): _execute_trade(), _calculate_reward()
Stream C (COMPLETE): step(), reset(), _validate_ohlc(), _get_current_observation()
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
from .features.order_block_features import extract_order_block_features, OrderBlockSignal
from .features.market_phase import detect_market_phase, MarketPhase
from .actions.hybrid_actions import apply_hybrid_action, HybridAction


logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Data split options
VALID_SPLITS = ["train", "val", "eval"]

# Action space constants
NUM_ACTIONS = 5  # Hybrid actions: SKIP (0), ENTER (1), ENTER_2X (2), ADJUST_SL (3), ADJUST_TP (4)

# Observation space bounds (for percentage values)
OBS_SPACE_LOW = -1000.0  # Allow extreme percentage values
OBS_SPACE_HIGH = 1000.0

# Trading parameters
EURUSD_SPREAD_PIPS = 2  # Typical EURUSD spread
SLIPPAGE_PIPS_RANGE = (0, 1)  # Min and max slippage in pips

# Data requirements
MIN_DATA_POINTS = 2  # Minimum observations needed for building obs space

# DataSplitter configuration
TIMESTAMP_COLUMN = "rate_time"

# Position parameters
POSITION_SIZE = 100000  # Standard lot (100,000 units)

# Hybrid strategy baseline parameters
DEFAULT_STOP_LOSS_PIPS = 50  # Default stop-loss distance in pips
DEFAULT_TAKE_PROFIT_PIPS = 100  # Default take-profit distance in pips
OB_LOOKBACK_BARS = 50  # Order Block detection lookback window


class TradingEnv(gym.Env):
    """
    EURUSD H1 Trading Environment.

    A custom Gymnasium-compatible trading environment that:
    - Loads data from dual PostgreSQL databases
    - Normalizes observations to percentage-based representation
    - Provides discrete action space (hold/buy/sell)
    - Dynamically builds observation space excluding NULL indicators

    Attributes:
        datafeed: DataFeed instance for loading market data
        normalizer: Normalizer instance for preprocessing observations
        split: Data split to use ('train', 'val', or 'eval')
        action_space: Discrete(3) space for hold/buy/sell actions
        observation_space: Box space for normalized indicators (dynamic)
    """

    def __init__(
        self,
        datafeed: DataFeed,
        normalizer: Normalizer,
        split: str = "train",
        reward_function: str = "pnl"
    ):
        """
        Initialize TradingEnv.

        Args:
            datafeed: DataFeed instance for loading market data
            normalizer: Normalizer instance for preprocessing
            split: Data split ('train', 'val', or 'eval')
            reward_function: Reward function type ('pnl', 'sharpe', 'risk_adjusted', 'custom_pattern')

        Raises:
            ValueError: If split is not one of 'train', 'val', 'eval'
            ValueError: If reward_function is not a valid reward type
        """
        # Validate split parameter
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got '{split}'")

        # Validate and instantiate reward function
        from .rewards import reward_factory

        try:
            self.reward_fn = reward_factory.create(reward_function)
        except ValueError as e:
            raise ValueError(f"Invalid reward_function: {e}")

        # Store parameters
        self.datafeed = datafeed
        self.normalizer = normalizer
        self.split = split
        self.reward_function_name = reward_function
        self.logger = logging.getLogger(__name__)

        # Load and split data
        self._load_and_split_data()

        # Define action space: Discrete(NUM_ACTIONS)
        # 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Build observation space dynamically
        self.observation_space = self._build_observation_space()

        # Trading parameters
        self.spread_pips = EURUSD_SPREAD_PIPS
        self.slippage_pips = SLIPPAGE_PIPS_RANGE

        logger.info(
            f"TradingEnv initialized: split={split}, "
            f"reward_function={reward_function}, "
            f"data_size={len(self.data)}, "
            f"obs_space_shape={self.observation_space.shape}"
        )

    def _load_and_split_data(self):
        """
        Load data from datafeed and apply train/val/eval split.

        Uses DataSplitter to split data temporally with 50/25/25 ratio.
        """
        # Load all data
        full_data = self.datafeed.load_data()

        # Split data
        splitter = DataSplitter()
        train_df, val_df, eval_df = splitter.split(
            full_data,
            timestamp_col=TIMESTAMP_COLUMN,
            validate=False,  # Skip validation for now
        )

        # Store all splits for reset() to use
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

        The observation space excludes NULL indicators (e.g., order blocks
        that may not always be present). This makes the space shape match
        the actual observation dimensions.

        Returns:
            Box space with shape matching non-NULL indicators

        Algorithm:
            1. Get first raw observation from data
            2. Normalize it to get indicator keys
            3. Count non-NULL indicators
            4. Create Box space with shape (non_null_count,)
        """
        # Get first observation from data
        if len(self.data) < MIN_DATA_POINTS:
            raise ValueError(
                f"Insufficient data: need at least {MIN_DATA_POINTS} observations "
                f"to build observation space"
            )

        # Convert first two rows to dict for normalization
        first_obs_raw = self.data.iloc[0].to_dict()
        # Replace NaN with None for proper NULL handling
        first_obs = {key: (None if pd.isna(value) else value) for key, value in first_obs_raw.items()}
        prev_obs = None  # First observation has no previous

        # Add Order Block features (CRITICAL: must match _get_current_observation)
        ob_signal = extract_order_block_features(self.data, 0, lookback_bars=OB_LOOKBACK_BARS)
        first_obs['ob_direction'] = 1.0 if ob_signal.direction == "LONG" else (-1.0 if ob_signal.direction == "SHORT" else 0.0)
        first_obs['ob_strength'] = ob_signal.strength if ob_signal.is_valid else 0.0
        first_obs['ob_distance_pips'] = ob_signal.distance_pips if ob_signal.is_valid else 0.0
        first_obs['ob_age_bars'] = float(ob_signal.age_bars) if ob_signal.is_valid else 0.0
        first_obs['ob_is_valid'] = 1.0 if ob_signal.is_valid else 0.0

        # Normalize to get indicator structure
        normalized_obs = self.normalizer.normalize_observation(first_obs, prev_obs)

        # Count non-NULL indicators
        non_null_indicators = [
            key for key, value in normalized_obs.items() if value is not None
        ]

        # Store indicator keys for later use
        self.indicator_keys = non_null_indicators
        n_indicators = len(non_null_indicators)

        logger.info(
            f"Observation space: {n_indicators} indicators "
            f"(excluding NULLs): {non_null_indicators}"
        )

        # Create Box space with reasonable bounds for percentage values
        # Most percentage values will be in range [-100, +100]
        # but we allow [-1000, +1000] for extreme cases
        return spaces.Box(
            low=OBS_SPACE_LOW,
            high=OBS_SPACE_HIGH,
            shape=(n_indicators,),
            dtype=np.float32,
        )

    # ========================================================================
    # STREAM B METHODS (NOT IMPLEMENTED YET)
    # ========================================================================

    def _execute_trade(
        self,
        simple_action: int,
        obs_raw: Dict[str, float],
        hybrid_action: int = 0,
        ob_is_valid: bool = False
    ) -> float:
        """
        Execute trade with realistic spread and slippage.

        REFACTORED: Now passes action and ob_is_valid to reward calculation for
        realistic reward shaping (transaction costs, SKIP incentives, etc.)

        This method implements realistic forex trading mechanics:
        - Spread: Applied to account for bid-ask difference (2 pips for EURUSD)
        - Slippage: Random market friction (0-1 pip range)
        - Position tracking: Updates position state (LONG/SHORT/FLAT)
        - Reward calculation: Uses raw prices for accurate PnL with context

        Trading Flow:
        1. Calculate PnL from previous position (if any) with reward context
        2. Close previous position (if switching direction)
        3. Open new position with spread and slippage applied
        4. Return PnL from closed position

        Args:
            simple_action: Trading action
                - 0: Hold (maintain current position)
                - 1: Buy (go long or increase long position)
                - 2: Sell (go short or increase short position)
            obs_raw: Raw (non-normalized) market observation
                Required keys: 'close' (current price)
            hybrid_action: Hybrid action taken by agent (for reward calculation, 0-4)
            ob_is_valid: Whether Order Block pattern is valid (for reward calculation)

        Returns:
            reward: Profit/loss from closing previous position (may include costs/bonuses)
        """
        current_price = obs_raw["close"]

        # Step 1: Close previous position and calculate PnL with context
        reward = self._close_position_if_exists(current_price, hybrid_action=hybrid_action, ob_is_valid=ob_is_valid)

        # Step 2: Execute new position based on simple_action
        if simple_action == 1:  # Buy (go LONG)
            entry_price = self._calculate_entry_price_buy(current_price)
            self._open_long_position(entry_price)
        elif simple_action == 2:  # Sell (go SHORT)
            entry_price = self._calculate_entry_price_sell(current_price)
            self._open_short_position(entry_price)
        # simple_action == 0 (Hold) - no position change

        return reward

    def _close_position_if_exists(
        self,
        current_price: float,
        hybrid_action: int = 0,
        ob_is_valid: bool = False
    ) -> float:
        """
        Close existing position and calculate realized PnL using configured reward function.

        Now includes context (action, ob_is_valid) for realistic reward shaping.

        Args:
            current_price: Current market price for position closing
            hybrid_action: Hybrid action taken by agent (for reward context)
            ob_is_valid: Whether Order Block pattern is valid (for reward context)

        Returns:
            Realized reward from closed position (0.0 if no position exists)
        """
        if self.position != 0 and self.position_entry_price > 0:
            return self.reward_fn.calculate(
                entry_price=self.position_entry_price,
                exit_price=current_price,
                position=self.position,
                position_size=self.position_size,
                action=hybrid_action,
                ob_is_valid=ob_is_valid,
                trade_count=self.trade_count,
            )
        return 0.0

    def _open_long_position(self, entry_price: float) -> None:
        """
        Open a LONG position (buy order).

        Args:
            entry_price: Calculated entry price with spread/slippage
        """
        self.position = 1  # LONG
        self.position_entry_price = entry_price

    def _open_short_position(self, entry_price: float) -> None:
        """
        Open a SHORT position (sell order).

        Args:
            entry_price: Calculated entry price with spread/slippage
        """
        self.position = -1  # SHORT
        self.position_entry_price = entry_price

    def _calculate_entry_price_buy(self, current_price: float) -> float:
        """
        Calculate entry price for buy order with spread and slippage.

        Buy orders execute at ASK price (higher than current):
        Entry = Current + Spread + Slippage

        Args:
            current_price: Current market price

        Returns:
            Entry price with spread and slippage applied
        """
        import random

        # Convert Decimal to float (PostgreSQL returns Decimal for numeric columns)
        current_price = float(current_price)

        spread_in_price = self._pips_to_price(self.spread_pips)
        slippage_in_price = self._pips_to_price(
            random.uniform(self.slippage_pips[0], self.slippage_pips[1])
        )
        return current_price + spread_in_price + slippage_in_price

    def _calculate_entry_price_sell(self, current_price: float) -> float:
        """
        Calculate entry price for sell order with spread and slippage.

        Sell orders execute at BID price (lower than current):
        Entry = Current - Spread - Slippage

        Args:
            current_price: Current market price

        Returns:
            Entry price with spread and slippage applied
        """
        import random

        # Convert Decimal to float (PostgreSQL returns Decimal for numeric columns)
        current_price = float(current_price)

        spread_in_price = self._pips_to_price(self.spread_pips)
        slippage_in_price = self._pips_to_price(
            random.uniform(self.slippage_pips[0], self.slippage_pips[1])
        )
        return current_price - spread_in_price - slippage_in_price

    def _pips_to_price(self, pips: float) -> float:
        """
        Convert pips to price value.

        For EURUSD: 1 pip = 0.0001

        Args:
            pips: Number of pips

        Returns:
            Price value equivalent
        """
        return pips / 10000

    def _create_baseline_entry(
        self,
        ob_signal: OrderBlockSignal,
        current_price: float
    ) -> Dict[str, float]:
        """
        Create baseline entry parameters from Order Block signal.

        This method generates standard entry parameters based on the Order Block
        direction. The RL agent can then modify these parameters using hybrid actions.

        Args:
            ob_signal: Order Block signal containing direction and features
            current_price: Current market price

        Returns:
            Dictionary with baseline entry parameters:
                - direction: "BUY" for LONG signal, "SELL" for SHORT signal
                - entry_price: Current market price
                - stop_loss: SL price (DEFAULT_STOP_LOSS_PIPS from entry)
                - take_profit: TP price (DEFAULT_TAKE_PROFIT_PIPS from entry)
                - position_size: Standard position size (1.0)

        Example:
            >>> signal = OrderBlockSignal(direction="LONG", ...)
            >>> baseline = env._create_baseline_entry(signal, 1.1000)
            >>> baseline["direction"]
            'BUY'
            >>> baseline["stop_loss"]
            1.0950  # 50 pips below entry for LONG
        """
        # Convert Decimal to float (PostgreSQL returns Decimal for numeric columns)
        current_price = float(current_price)

        # Convert OB signal direction to trade direction
        direction = "BUY" if ob_signal.direction == "LONG" else "SELL"

        # Calculate SL/TP prices based on direction
        sl_distance = self._pips_to_price(DEFAULT_STOP_LOSS_PIPS)
        tp_distance = self._pips_to_price(DEFAULT_TAKE_PROFIT_PIPS)

        if direction == "BUY":
            stop_loss = current_price - sl_distance
            take_profit = current_price + tp_distance
        else:  # SELL
            stop_loss = current_price + sl_distance
            take_profit = current_price - tp_distance

        return {
            "direction": direction,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": 1.0,
        }

    def _execute_hybrid_trade(
        self,
        action: int,
        baseline_entry: Dict[str, float],
        obs_raw: Dict[str, float],
        ob_is_valid: bool,
        ob_signal=None
    ) -> float:
        """
        Execute trade using hybrid action applied to baseline entry.

        Args:
            action: Hybrid action (0=SKIP, 1=ENTER, 2=ENTER_2X, 3=ADJUST_SL, 4=ADJUST_TP)
            baseline_entry: Baseline entry parameters from Order Block signal
            obs_raw: Raw observation data
            ob_is_valid: Whether Order Block signal is valid

        Returns:
            Reward from executing the trade

        Note:
            This method applies the hybrid action to the baseline entry and
            converts the result to a simple action (hold/buy/sell) for execution.
            Reward calculation includes action context for realistic reward shaping.
        """
        # Check if OB signal is valid
        if not baseline_entry:
            # No valid OB: Reward SKIP, penalize other actions
            current_price = obs_raw["close"]
            return self._calculate_reward(
                current_price=current_price,
                entry_price=current_price,
                position=0,  # No position
                position_size=0,
                action=action,
                ob_is_valid=False,
                ob_signal=ob_signal,
            )

        # Apply hybrid action to baseline
        modified_entry = apply_hybrid_action(action, baseline_entry)

        if modified_entry is None:
            # SKIP action - calculate reward with SKIP context
            current_price = obs_raw["close"]
            return self._calculate_reward(
                current_price=current_price,
                entry_price=current_price,  # No position
                position=0,  # FLAT
                position_size=0,
                action=action,  # SKIP
                ob_is_valid=ob_is_valid,
                ob_signal=ob_signal,
            )

        # Execute trade and track it
        simple_action = 1 if modified_entry["direction"] == "BUY" else 2
        reward = self._execute_trade(simple_action, obs_raw, hybrid_action=action, ob_is_valid=ob_is_valid)

        # Increment trade count if a trade was executed (reward != 0 means position changed)
        if reward != 0:
            self.trade_count += 1

        return reward

    def _calculate_reward(
        self,
        current_price: float,
        entry_price: float,
        position: int,
        position_size: float,
        action: int = 0,
        ob_is_valid: bool = False,
        ob_signal=None,
    ) -> float:
        """
        Calculate reward using raw (non-normalized) prices.

        REFACTORED: Now passes action, ob_is_valid, and trade_count to reward function
        for realistic reward shaping (transaction costs, SKIP incentives, etc.)

        This method delegates to the configured reward function, passing all necessary
        context for sophisticated reward calculations.

        Args:
            current_price: Current market price (raw, e.g., 1.1050)
            entry_price: Position entry price (raw, e.g., 1.1000)
            position: Position direction
                - 1: LONG (profit when price rises)
                - 0: FLAT (no position, no PnL)
                - -1: SHORT (profit when price falls)
            position_size: Position size in base currency (e.g., 100000 for 1 standard lot)
            action: Action taken by agent (0=SKIP, 1=ENTER, 2=ENTER_2X, 3=ADJUST_SL, 4=ADJUST_TP)
            ob_is_valid: Whether a valid Order Block pattern exists at this step

        Returns:
            reward: Calculated reward (may include transaction costs, SKIP bonuses, etc.)

        Example:
            >>> # LONG position with valid OB pattern
            >>> reward = env._calculate_reward(1.1050, 1.1000, 1, 100000, action=1, ob_is_valid=True)
            >>> # Reward includes PnL minus transaction costs

            >>> # SKIP when no valid OB (good decision!)
            >>> reward = env._calculate_reward(1.1000, 1.1000, 0, 0, action=0, ob_is_valid=False)
            >>> # Reward = +$1 SKIP bonus
        """
        # Delegate to configured reward function with full context
        return self.reward_fn.calculate(
            entry_price=entry_price,
            exit_price=current_price,
            position=position,
            position_size=position_size,
            action=action,
            ob_is_valid=ob_is_valid,
            trade_count=self.trade_count,
            ob_signal=ob_signal,
        )

    # ========================================================================
    # STREAM C METHODS (NOT IMPLEMENTED YET)
    # ========================================================================

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility
            options: Additional options

        Returns:
            observation: Normalized observation array
            info: Dictionary with episode information
        """
        # CRITICAL: Must call super().reset() first for Gymnasium seeding
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
        self.prev_obs_raw = None  # For normalizer
        self.trade_count = 0  # Track total trades in episode
        self.current_market_phase = None  # Will be set during first step

        # Detect initial market phase
        self.current_market_phase = detect_market_phase(
            self.data,
            self.current_step,
            lookback_bars=30
        )

        # Get initial observation (raw)
        obs_raw = self._get_current_observation()

        # Normalize (note: normalizer expects prev_obs for some calculations)
        obs_normalized = self.normalizer.normalize_observation(obs_raw, prev_obs=None)

        # CRITICAL: Use only the indicator keys defined in observation space
        # This ensures consistent observation dimensions across all timesteps
        # Handle None values from normalizer (NULL indicators) by replacing with 0.0
        obs_array = np.array(
            [obs_normalized.get(key) if obs_normalized.get(key) is not None else 0.0
             for key in self.indicator_keys],
            dtype=np.float32,
        )

        # Debug: Check for NaN values
        if np.isnan(obs_array).any():
            nan_indices = np.where(np.isnan(obs_array))[0]
            nan_keys = [self.indicator_keys[i] for i in nan_indices]
            nan_values = {key: obs_normalized.get(key) for key in nan_keys}
            self.logger.error(f"NaN detected in reset observation at indices {nan_indices}")
            self.logger.error(f"NaN keys: {nan_keys}")
            self.logger.error(f"NaN raw values: {nan_values}")
            self.logger.error(f"obs_raw: {obs_raw}")
            raise ValueError(f"NaN values detected in observation: {nan_keys}")

        info = {"split": self.split, "step": self.current_step}
        return obs_array, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep in the environment with hybrid action support.

        This method implements the hybrid RL strategy where Order Block signals
        provide entry opportunities, and the RL agent chooses how to refine the
        execution using one of 5 hybrid actions.

        Workflow:
            1. Get current market observation
            2. Extract Order Block signal (if any)
            3. If OB signal exists:
                - Create baseline entry parameters
                - Apply hybrid action (SKIP/ENTER/ENTER_2X/ADJUST_SL/ADJUST_TP)
                - Execute trade (if not SKIP)
            4. If no OB signal:
                - FORCE CLOSE any open position (prevents "buy and hold")
                - Reward SKIP, penalize other actions
            5. Return normalized observation and reward

        Anti-Buy-and-Hold Protection:
            Positions are automatically closed when Order Block signals expire.
            This prevents the model from learning a simple "buy once and hold forever"
            strategy, ensuring it only holds positions during active OB signals.

        Args:
            action: Hybrid action (0-4):
                - 0: SKIP - Ignore this Order Block (no entry)
                - 1: ENTER - Accept baseline entry as-is
                - 2: ENTER_2X - Double position size
                - 3: ADJUST_SL - Tighten stop-loss by 30%
                - 4: ADJUST_TP - Extend take-profit by 50%

        Returns:
            Tuple containing:
                - observation: Normalized observation array (np.ndarray)
                - reward: Reward from this timestep (float)
                - terminated: Whether episode ended naturally (bool)
                - truncated: Whether episode hit time limit (bool)
                - info: Additional information dict

        Example:
            >>> obs, reward, terminated, truncated, info = env.step(action=1)
            >>> # ENTER action: accepts Order Block entry with baseline params
        """
        # Get current observation (raw)
        obs_raw = self._get_current_observation()

        # Validate OHLC prices
        self._validate_ohlc(obs_raw)

        # Detect current market phase
        self.current_market_phase = detect_market_phase(
            self.data,
            self.current_step,
            lookback_bars=30  # Use 30 bars for phase detection
        )

        # Extract Order Block features for hybrid action decision
        ob_signal = extract_order_block_features(
            self.data,
            self.current_step,
            lookback_bars=OB_LOOKBACK_BARS
        )

        # Execute trade based on OB signal validity
        if ob_signal.is_valid:
            # Valid OB signal: create baseline and apply hybrid action
            baseline_entry = self._create_baseline_entry(ob_signal, obs_raw["close"])
            reward = self._execute_hybrid_trade(action, baseline_entry, obs_raw, ob_is_valid=True, ob_signal=ob_signal)
        else:
            # No valid OB signal: Force close any open position to prevent "buy and hold"
            # This ensures the model only holds positions when OB signals are active
            if self.position != 0:
                # Close position and calculate PnL
                reward = self._close_position_if_exists(obs_raw["close"], hybrid_action=action, ob_is_valid=False)
                self.position = 0
                self.position_entry_price = 0.0
            else:
                # No position: SKIP is rewarded, other actions penalized
                reward = self._execute_hybrid_trade(action, {}, obs_raw, ob_is_valid=False, ob_signal=ob_signal)

        # Normalize observation for agent
        obs_normalized = self.normalizer.normalize_observation(obs_raw, self.prev_obs_raw)

        # CRITICAL: Use only the indicator keys defined in observation space
        # This ensures consistent observation dimensions across all timesteps
        # Handle None values from normalizer (NULL indicators) by replacing with 0.0
        obs_array = np.array(
            [obs_normalized.get(key) if obs_normalized.get(key) is not None else 0.0
             for key in self.indicator_keys],
            dtype=np.float32,
        )

        # Debug: Check for NaN values
        if np.isnan(obs_array).any():
            nan_indices = np.where(np.isnan(obs_array))[0]
            nan_keys = [self.indicator_keys[i] for i in nan_indices]
            nan_values = {key: obs_normalized.get(key) for key in nan_keys}
            self.logger.error(f"NaN detected in step observation at step {self.current_step}")
            self.logger.error(f"NaN indices: {nan_indices}")
            self.logger.error(f"NaN keys: {nan_keys}")
            self.logger.error(f"NaN normalized values: {nan_values}")
            self.logger.error(f"obs_raw: {obs_raw}")
            self.logger.error(f"prev_obs_raw: {self.prev_obs_raw}")
            raise ValueError(f"NaN values detected in observation: {nan_keys}")

        # Update state
        self.prev_obs_raw = obs_raw
        self.current_step += 1

        # Check termination
        terminated = self.current_step >= len(self.data)
        truncated = False

        info = {"step": self.current_step, "position": self.position}

        return obs_array, reward, terminated, truncated, info

    def _validate_ohlc(self, obs: Dict[str, float]) -> None:
        """
        Validate OHLC price relationships.

        Args:
            obs: Observation dictionary with OHLC data

        Raises:
            ValueError: If OHLC validation fails
        """
        high = obs.get("high")
        low = obs.get("low")
        open_price = obs.get("open")
        close = obs.get("close")

        # Check all OHLC relationships
        if high < low:
            raise ValueError(f"Invalid OHLC: high ({high}) < low ({low})")
        if high < close:
            raise ValueError(f"Invalid OHLC: high ({high}) < close ({close})")
        if high < open_price:
            raise ValueError(f"Invalid OHLC: high ({high}) < open ({open_price})")

        # For low violations, check which is more severe and report that one
        if low > close and low > open_price:
            # Both violated - report the larger violation
            if (low - open_price) > (low - close):
                raise ValueError(f"Invalid OHLC: low ({low}) > open ({open_price})")
            else:
                raise ValueError(f"Invalid OHLC: low ({low}) > close ({close})")
        elif low > close:
            raise ValueError(f"Invalid OHLC: low ({low}) > close ({close})")
        elif low > open_price:
            raise ValueError(f"Invalid OHLC: low ({low}) > open ({open_price})")

    def _get_current_observation(self) -> Dict[str, Any]:
        """
        Get current observation at current timestep.

        UPDATED: Now includes Order Block features in observations!

        Returns:
            Dictionary with raw (non-normalized) observation data including:
            - Technical indicators from CSV
            - Order Block features (direction, strength, distance, age)
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call reset() first.")

        if self.current_step >= len(self.data):
            raise RuntimeError(
                f"Step {self.current_step} out of bounds (data length: {len(self.data)})"
            )

        # Get current row and convert to dict
        row = self.data.iloc[self.current_step]
        obs_dict = row.to_dict()

        # Replace NaN with None for proper NULL handling
        # pandas.to_dict() converts None to NaN, but we need None for normalizer
        obs_dict = {key: (None if pd.isna(value) else value) for key, value in obs_dict.items()}

        # Add Order Block features to observation
        ob_signal = extract_order_block_features(
            self.data,
            self.current_step,
            lookback_bars=OB_LOOKBACK_BARS
        )

        # Add OB features as additional observation dimensions
        obs_dict['ob_direction'] = 1.0 if ob_signal.direction == "LONG" else (-1.0 if ob_signal.direction == "SHORT" else 0.0)
        obs_dict['ob_strength'] = ob_signal.strength if ob_signal.is_valid else 0.0
        obs_dict['ob_distance_pips'] = ob_signal.distance_pips if ob_signal.is_valid else 0.0
        obs_dict['ob_age_bars'] = float(ob_signal.age_bars) if ob_signal.is_valid else 0.0
        obs_dict['ob_is_valid'] = 1.0 if ob_signal.is_valid else 0.0

        return obs_dict

    def seed(self, seed: Optional[int] = None) -> None:
        """
        Set random seed for reproducibility.

        Args:
            seed: Random seed value
        """
        if seed is not None:
            np.random.seed(seed)
            self.logger.debug(f"Environment seed set to {seed}")
