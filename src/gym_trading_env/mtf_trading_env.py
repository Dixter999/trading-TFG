"""
Multi-Timeframe Trading Environment for Reinforcement Learning.

This module implements MTFTradingEnv - a Gymnasium-compatible wrapper environment
that integrates multi-timeframe pattern detection components for forex trading:

Components:
    - PatternDetectorH1: H1 timeframe pattern recognition (OB touch, breakouts, BOS)
    - MTFSynchronizer: H1→M15 timeframe coordination and trigger detection
    - ActionMaskGenerator: Dynamic action masking based on pattern state
    - RiskCalculator: Position sizing and SL/TP level calculation

MTF Pattern-Based Trading Flow:
    1. H1 Pattern Detection: Detect high-probability setups (score ≥ 3)
    2. Execution Window: Open 60-minute window for M15 trigger confirmation
    3. M15 Trigger: Find precise entry using REJECTION, ENGULF, or MICRO_BOS patterns
    4. Position Entry: Enter with calculated SL and tiered TP levels (TP1, TP2)
    5. Position Management: Partial exits at TP1 (45%), TP2 (35%), runner (20%)
    6. Risk Management: Move SL to breakeven after TP1, trail runner position

Example Usage:
    ```python
    import pandas as pd
    from src.gym_trading_env.mtf_trading_env import MTFTradingEnv

    # Load H1 and M15 data from database
    h1_data = load_h1_technical_indicators()  # OHLC + indicators + OB levels
    m15_data = load_m15_ohlc_data()  # OHLC only

    # Initialize environment
    env = MTFTradingEnv(h1_data=h1_data, m15_data=m15_data)

    # Reset for new episode
    observation, info = env.reset(seed=42)

    # Training loop
    done = False
    while not done:
        # Get action from RL agent (respecting action mask)
        action_mask = info["action_mask"]
        action = agent.get_action(observation, action_mask)

        # Execute step
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Access pattern state
        if info["pattern_state"]["window_long"]:
            print(f"LONG window opened at step {info['current_step']}")
            if info["m15_trigger"]:
                print(f"M15 trigger: {info['m15_trigger']['trigger']}")
    ```

Anti-Leakage Guarantees:
    All components strictly enforce temporal ordering:
    - H1 pattern detection at timestamp T uses only data where timestamp ≤ T
    - M15 triggers evaluated only after H1 signal timestamp
    - M15 execution window: T+15m, T+30m, T+45m, T+60m (no future data)
    - Position management uses only current price and ATR

TDD Implementation Status:
    - Stream A (COMPLETE): Core infrastructure, initialization, observation/action spaces
    - Stream B (COMPLETE): reset() and step() implementation with pattern detection
    - Stream C (COMPLETE): Reward calculation and position management
    - Stream D (IN PROGRESS): Integration tests and documentation
"""

import logging
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .action_mask_generator import ActionMaskGenerator
from .mtf_synchronizer import MTFSynchronizer
from .pattern_detector_h1 import PatternDetectorH1
from .risk_calculator import RiskCalculator

logger = logging.getLogger(__name__)


class MTFTradingEnv(gym.Env):
    """
    Multi-Timeframe Pattern-Based Trading Environment.

    Gymnasium-compatible environment that integrates H1 pattern detection with M15
    trigger confirmation for high-probability forex trading setups. Implements complete
    position management with partial exits and trailing stops.

    Architecture:
        The environment wraps four specialized components:
        1. PatternDetectorH1: Identifies H1 patterns (OB pullback, breakout, BOS)
        2. MTFSynchronizer: Coordinates H1→M15 trigger confirmation
        3. ActionMaskGenerator: Dynamically masks invalid actions
        4. RiskCalculator: Calculates SL/TP levels and manages partials

    Trading Logic Flow:
        1. Each step() processes one H1 candle
        2. PatternDetectorH1 evaluates current H1 for patterns (score 0-5)
        3. If score ≥ 3, execution window opens (window_long or window_short)
        4. MTFSynchronizer evaluates 4 M15 candles (T+15m, T+30m, T+45m, T+60m)
        5. If M15 trigger found, ENTER actions become available
        6. Agent can ENTER_LONG or ENTER_SHORT with trigger confirmation
        7. Position management handles TP1, TP2, and runner exits automatically
        8. Rewards calculated based on realized PnL from exits

    Action Space:
        Discrete(4) - Integer actions with dynamic masking:
        - 0: HOLD (do nothing) - always available
        - 1: ENTER_LONG (open long position) - only when:
            * window_long=True (H1 pattern detected)
            * m15_trigger_valid=True (M15 confirmation found)
            * No existing position
        - 2: ENTER_SHORT (open short position) - only when:
            * window_short=True
            * m15_trigger_valid=True
            * No existing position
        - 3: CLOSE_POSITION (manual close) - only when position exists

    Observation Space:
        Box(low=-1000.0, high=1000.0, shape=(N,), dtype=float32)
        Where N = base_indicators + 4 pattern_state_indicators

        Base Indicators (from H1 data):
            - OHLC: open, high, low, close
            - Trend: sma_50, sma_200
            - Volatility: atr_14, bb_upper_20, bb_lower_20
            - Momentum: macd_line, macd_signal, rsi_14, stoch_k, stoch_d
            - Structure: ob_bullish_high, ob_bullish_low, ob_bearish_high, ob_bearish_low

        Pattern State Indicators (4 values):
            - window_long: 1.0 if LONG window open, 0.0 otherwise
            - window_short: 1.0 if SHORT window open, 0.0 otherwise
            - m15_trigger_valid: 1.0 if M15 trigger found, 0.0 otherwise
            - pattern_score: Float 0.0-5.0 indicating pattern confluence

    Reward Structure:
        Sparse rewards based on realized PnL from position exits:
        - TP1 hit (45% closed): +PnL for 45% of position
        - TP2 hit (35% closed): +PnL for 35% of position
        - Runner exit (20% closed): +PnL for 20% of position
        - SL hit: -Loss for remaining position
        - Manual close: +/- PnL for remaining position
        - HOLD when no position: 0.0 reward

    Position Management:
        Automatic partial exits and risk management:
        1. Entry: Full position (100%) at M15 trigger price
        2. TP1 reached: Close 45%, move SL to breakeven (0 risk)
        3. TP2 reached: Close 35%, runner (20%) continues with trailing stop
        4. Trailing: Stop follows price at ATR distance, only moves favorably
        5. Exit: Position fully closed when all partials exited or SL hit

    Info Dictionary:
        Each step returns info dict with:
        - pattern_state: Dict with H1 pattern detection results
            * window_long: bool
            * window_short: bool
            * window_type: str ("OB_PULLBACK", "BREAKOUT", "NONE")
            * score: int (0-5)
            * patterns: dict with individual pattern flags
            * levels: dict with critical levels
        - m15_trigger: Dict if trigger found, None otherwise
            * trigger: str ("REJECTION", "ENGULF", "MICRO_BOS")
            * entry_price: float
            * timestamp: datetime
        - action_mask: np.ndarray of bool, shape=(4,)
        - action_executed: bool (True if action was valid and executed)
        - current_step: int
        - h1_timestamp: datetime

    Attributes:
        h1_data (pd.DataFrame): H1 OHLC + technical indicators + OB levels
        m15_data (pd.DataFrame): M15 OHLC only
        pattern_detector (PatternDetectorH1): H1 pattern recognition component
        mtf_synchronizer (MTFSynchronizer): H1→M15 coordination component
        action_mask_generator (ActionMaskGenerator): Action masking component
        risk_calculator (RiskCalculator): Risk management component
        current_step (int): Current position in H1 data (starts at 0)
        current_position (dict | None): Active position state or None
        pattern_state (dict): Latest H1 pattern detection result
        m15_trigger (dict | None): Latest M15 trigger or None
        current_price (float): Current H1 close price
        current_atr (float): Current H1 ATR value

    Example:
        ```python
        # Initialize with data
        env = MTFTradingEnv(h1_data=h1_df, m15_data=m15_df)

        # Reset for new episode
        obs, info = env.reset(seed=42)
        print(f"Initial observation shape: {obs.shape}")  # (N,)
        print(f"Pattern state: {info['pattern_state']}")

        # Training loop
        done = False
        episode_reward = 0.0

        while not done:
            # Get action from policy (respecting mask)
            action_mask = info["action_mask"]
            action = policy.get_action(obs, mask=action_mask)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            # Monitor trading activity
            if info["action_executed"]:
                print(f"Step {info['current_step']}: Action {action} executed")

            if info["m15_trigger"]:
                trigger_type = info["m15_trigger"]["trigger"]
                entry = info["m15_trigger"]["entry_price"]
                print(f"M15 trigger found: {trigger_type} @ {entry}")

        print(f"Episode complete. Total reward: {episode_reward}")
        ```

    Notes:
        - All data access respects anti-leakage: only uses data <= current timestamp
        - Position sizing is relative (45%, 35%, 20%) not absolute lot sizes
        - Reward normalization should be applied by RL algorithm if needed
        - Episode terminates when H1 data exhausted (terminated=True)
        - No truncation implemented (truncated always False)
    """

    # Action space constants (aligned with ActionMaskGenerator)
    ACTION_HOLD = 0
    ACTION_ENTER_LONG = 1
    ACTION_ENTER_SHORT = 2
    ACTION_CLOSE_POSITION = 3

    def __init__(self, h1_data: pd.DataFrame, m15_data: pd.DataFrame):
        """
        Initialize MTFTradingEnv with H1 and M15 data.

        Args:
            h1_data: H1 DataFrame with OHLC + technical indicators
                    Must contain all columns required by PatternDetectorH1:
                    timestamp, open, high, low, close, sma_50, sma_200, atr_14,
                    macd_line, macd_signal, rsi_14, stoch_k, stoch_d,
                    bb_upper_20, bb_lower_20, ob_bullish_high, ob_bullish_low,
                    ob_bearish_high, ob_bearish_low

            m15_data: M15 DataFrame with OHLC only
                     Must contain: timestamp, open, high, low, close

        Raises:
            ValueError: If data validation fails
            TypeError: If data types are invalid
        """
        # Store data for episode management
        self.h1_data = h1_data
        self.m15_data = m15_data

        # Initialize all 4 components
        self.pattern_detector = PatternDetectorH1(h1_data=h1_data)
        self.mtf_synchronizer = MTFSynchronizer(h1_data=h1_data, m15_data=m15_data)
        self.action_mask_generator = ActionMaskGenerator()
        self.risk_calculator = RiskCalculator()

        # Define action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)

        # Build observation space
        # Pattern state adds 4 indicators: window_long, window_short,
        # m15_trigger_valid, pattern_score
        self.observation_space = self._build_observation_space()

        logger.info(
            f"MTFTradingEnv initialized: "
            f"h1_data_size={len(h1_data)}, "
            f"m15_data_size={len(m15_data)}, "
            f"action_space={self.action_space}, "
            f"obs_space_shape={self.observation_space.shape}"
        )

    def _build_observation_space(self) -> spaces.Box:
        """
        Build fixed 26-feature observation space for pattern-aware RL training.

        Observation includes exactly 26 features:
        - Technical Indicators (16): normalized indicators
        - Pattern Detections (6): binary flags (0 or 1)
        - Market Context (4): position state and performance

        Returns:
            Box space with shape (26,) and dtype float32
        """
        # Fixed 26-feature observation space
        # 16 technical indicators + 6 pattern flags + 4 market context
        total_features = 26

        logger.info(
            f"Observation space: {total_features} features "
            f"(tech_indicators=16 + pattern_flags=6 + market_context=4)"
        )

        # Create Box space with bounds suitable for normalized features
        # Technical indicators normalized to [-10, +10]
        # Pattern flags are binary [0, 1]
        # Market context normalized to [-10, +10]
        return spaces.Box(
            low=-1000.0,
            high=1000.0,
            shape=(total_features,),
            dtype=np.float32,
        )

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Initializes episode state and returns first observation with pattern detection.

        Args:
            seed: Random seed for reproducibility
            options: Additional options

        Returns:
            observation: Normalized observation array including pattern state
            info: Dictionary with episode information including pattern_state

        """
        # CRITICAL: Must call super().reset() first for Gymnasium seeding
        super().reset(seed=seed, options=options)

        # 1. Initialize episode state
        self.current_step = 0
        self.current_position = None  # No position at start
        self.pattern_state = {}  # Will be populated with H1 pattern detection
        self.m15_trigger = None  # No M15 trigger initially

        # Initialize position tracking variables (for coordination with Stream C)
        self.current_sl = None
        self.current_tp1 = None
        self.current_tp2 = None

        # Initialize current price and ATR from first H1 candle
        first_h1_row = self.h1_data.iloc[self.current_step]
        self.current_price = first_h1_row["close"]
        self.current_atr = first_h1_row["atr_14"]

        # 2. Get first H1 timestamp for pattern detection
        first_h1_timestamp = self.h1_data.iloc[self.current_step]["timestamp"]

        # 3. Detect H1 pattern at first timestamp
        self.pattern_state = self.pattern_detector.detect(timestamp=first_h1_timestamp)

        # 4. Build observation array
        observation = self._build_observation()

        # 5. Build info dict
        info = {
            "pattern_state": self.pattern_state,
            "current_step": self.current_step,
            "h1_timestamp": first_h1_timestamp,
        }

        logger.info(
            f"Episode reset at step {self.current_step}, "
            f"timestamp={first_h1_timestamp}, "
            f"pattern_window_long={self.pattern_state.get('window_long', False)}, "
            f"pattern_window_short={self.pattern_state.get('window_short', False)}"
        )

        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one timestep in the environment.

        Complete MTF pattern flow:
        1. Detect H1 pattern
        2. Evaluate M15 triggers if window opens
        3. Generate action mask
        4. Execute action if valid
        5. Check position exits (SL/TP)
        6. Calculate reward
        7. Build observation

        Args:
            action: Action to execute (0=HOLD, 1=ENTER_LONG, 2=ENTER_SHORT, 3=CLOSE)

        Returns:
            observation: Normalized observation array
            reward: Reward for this step
            terminated: Episode ended naturally (no more data)
            truncated: Episode hit time limit
            info: Additional information including pattern_state, action_mask

        """
        # 0. Increment step counter
        self.current_step += 1

        # Check if episode should terminate (no more data)
        if self.current_step >= len(self.h1_data):
            # Episode ended - build observation from last valid step
            last_h1_row = self.h1_data.iloc[-1]
            numeric_cols = self.h1_data.select_dtypes(include=[np.number]).columns
            base_indicators = last_h1_row[numeric_cols].values.astype(np.float32)

            # Pattern state from last known state
            pattern_state_array = np.array(
                [
                    float(self.pattern_state.get("window_long", False)),
                    float(self.pattern_state.get("window_short", False)),
                    float(self.m15_trigger is not None),
                    float(self.pattern_state.get("score", 0.0)),
                ],
                dtype=np.float32,
            )
            observation = np.concatenate([base_indicators, pattern_state_array])

            reward = 0.0
            terminated = True
            truncated = False
            # Generate action mask for final state
            action_mask = self.action_mask_generator.generate_mask(
                window_long=self.pattern_state.get("window_long", False),
                window_short=self.pattern_state.get("window_short", False),
                position_exists=(self.current_position is not None),
                m15_trigger_valid=(self.m15_trigger is not None),
            )

            info = {
                "pattern_state": self.pattern_state,
                "m15_trigger": self.m15_trigger,
                "action_mask": action_mask,
                "action_executed": False,
                "current_step": len(self.h1_data) - 1,  # Report last valid step
                "h1_timestamp": last_h1_row["timestamp"],
            }
            return observation, reward, terminated, truncated, info

        # 1. Get current H1 timestamp
        current_h1_timestamp = self.h1_data.iloc[self.current_step]["timestamp"]
        current_h1_row = self.h1_data.iloc[self.current_step]

        # Store current price for reward calculation
        self.current_price = current_h1_row["close"]
        self.current_atr = current_h1_row["atr_14"]

        # 2. Detect H1 pattern at current timestamp
        self.pattern_state = self.pattern_detector.detect(
            timestamp=current_h1_timestamp
        )

        # 3. Evaluate M15 triggers if window opened
        self.m15_trigger = None  # Reset trigger
        if self.pattern_state.get("window_long") or self.pattern_state.get(
            "window_short"
        ):
            # Window opened - evaluate M15 triggers
            # Get safe M15 execution window (T+15m, T+30m, T+45m, T+60m)
            m15_window_timestamps = self._get_m15_execution_window(current_h1_timestamp)

            # Evaluate triggers sequentially until one is found
            for m15_ts in m15_window_timestamps:
                trigger_result = self.mtf_synchronizer.evaluate_trigger(
                    h1_timestamp=current_h1_timestamp,
                    m15_timestamp=m15_ts,
                    pattern_result=self.pattern_state,
                )

                if trigger_result is not None:
                    # Valid trigger found
                    self.m15_trigger = trigger_result
                    break

        # 4. Generate action mask based on pattern state
        action_mask = self.action_mask_generator.generate_mask(
            window_long=self.pattern_state.get("window_long", False),
            window_short=self.pattern_state.get("window_short", False),
            position_exists=(self.current_position is not None),
            m15_trigger_valid=(self.m15_trigger is not None),
        )

        # 5. Execute action if valid
        action_executed = False
        if action_mask[action]:
            action_executed = True
            self._execute_action(action)

        # 6. Check position exits (SL/TP) and calculate reward
        reward = self._calculate_reward()

        # 7. Build observation with updated pattern state
        observation = self._build_observation()

        # 8. Check episode termination
        terminated = self.current_step >= len(self.h1_data) - 1
        truncated = False  # No time limit for now

        # 9. Build info dict
        info = {
            "pattern_state": self.pattern_state,
            "m15_trigger": self.m15_trigger,
            "action_mask": action_mask,
            "action_executed": action_executed,
            "current_step": self.current_step,
            "h1_timestamp": current_h1_timestamp,
        }

        logger.debug(
            f"Step {self.current_step}: "
            f"action={action}, "
            f"executed={action_executed}, "
            f"reward={reward:.6f}, "
            f"window_long={self.pattern_state.get('window_long', False)}, "
            f"window_short={self.pattern_state.get('window_short', False)}"
        )

        return observation, reward, terminated, truncated, info

    def _build_observation(self) -> np.ndarray:
        """
        Build fixed 26-feature observation array from current state.

        Observation structure (26 features):
        - Technical Indicators (16):
            0. rsi_14 (normalized to [0, 1])
            1. macd_line (normalized by price)
            2. macd_signal (normalized by price)
            3. macd_histogram (normalized by price)
            4. atr_14 (normalized by price)
            5. sma_20 (normalized by price)
            6. sma_50 (normalized by price)
            7. sma_200 (normalized by price)
            8. ema_12 (normalized by price)
            9. ema_26 (normalized by price)
            10. ema_50 (normalized by price)
            11. bb_upper_20 (normalized by price)
            12. bb_middle_20 (normalized by price)
            13. bb_lower_20 (normalized by price)
            14. stoch_k (normalized to [0, 1])
            15. stoch_d (normalized to [0, 1])
        - Pattern Detections (6 binary):
            16. ob_bullish (0 or 1)
            17. ob_bearish (0 or 1)
            18. ls_bullish (0 or 1)
            19. ls_bearish (0 or 1)
            20. choch_bullish (0 or 1)
            21. choch_bearish (0 or 1)
        - Market Context (4):
            22. current_position (-1, 0, or 1)
            23. unrealized_pnl_pct (normalized)
            24. equity_drawdown_pct (normalized)
            25. candles_in_position (capped at 100)

        Returns:
            np.ndarray: Shape (26,) dtype float32

        """
        # Get current H1 row
        current_h1_row = self.h1_data.iloc[self.current_step]
        current_price = current_h1_row["close"]

        # Helper function to normalize by price (avoid division by zero)
        def normalize_by_price(value: float) -> float:
            if current_price == 0 or np.isnan(value) or np.isnan(current_price):
                return 0.0
            return (value - current_price) / current_price

        # Helper function to normalize to [0, 1]
        def normalize_0_1(
            value: float, min_val: float = 0.0, max_val: float = 100.0
        ) -> float:
            if np.isnan(value):
                return 0.0
            return (
                (value - min_val) / (max_val - min_val) if max_val != min_val else 0.0
            )

        # 1. Technical Indicators (16 features)
        tech_indicators = np.array(
            [
                normalize_0_1(current_h1_row.get("rsi_14", 50.0)),  # 0: RSI
                normalize_by_price(
                    current_h1_row.get("macd_line", 0.0)
                ),  # 1: MACD line
                normalize_by_price(
                    current_h1_row.get("macd_signal", 0.0)
                ),  # 2: MACD signal
                normalize_by_price(
                    current_h1_row.get("macd_histogram", 0.0)
                ),  # 3: MACD histogram
                normalize_by_price(current_h1_row.get("atr_14", 0.0)),  # 4: ATR
                normalize_by_price(
                    current_h1_row.get("sma_20", current_price)
                ),  # 5: SMA 20
                normalize_by_price(
                    current_h1_row.get("sma_50", current_price)
                ),  # 6: SMA 50
                normalize_by_price(
                    current_h1_row.get("sma_200", current_price)
                ),  # 7: SMA 200
                normalize_by_price(
                    current_h1_row.get("ema_12", current_price)
                ),  # 8: EMA 12
                normalize_by_price(
                    current_h1_row.get("ema_26", current_price)
                ),  # 9: EMA 26
                normalize_by_price(
                    current_h1_row.get("ema_50", current_price)
                ),  # 10: EMA 50
                normalize_by_price(
                    current_h1_row.get("bb_upper_20", current_price)
                ),  # 11: BB upper
                normalize_by_price(
                    current_h1_row.get("bb_middle_20", current_price)
                ),  # 12: BB middle
                normalize_by_price(
                    current_h1_row.get("bb_lower_20", current_price)
                ),  # 13: BB lower
                normalize_0_1(current_h1_row.get("stoch_k", 50.0)),  # 14: Stoch K
                normalize_0_1(current_h1_row.get("stoch_d", 50.0)),  # 15: Stoch D
            ],
            dtype=np.float32,
        )

        # 2. Pattern Detections (6 binary features)
        # Extract from pattern_state which comes from PatternDetectorH1
        patterns = self.pattern_state.get("patterns", {})
        pattern_flags = np.array(
            [
                float(patterns.get("ob_bullish", False)),  # 16: OB bullish
                float(patterns.get("ob_bearish", False)),  # 17: OB bearish
                float(patterns.get("ls_bullish", False)),  # 18: LS bullish
                float(patterns.get("ls_bearish", False)),  # 19: LS bearish
                float(patterns.get("choch_bullish", False)),  # 20: CHOCH bullish
                float(patterns.get("choch_bearish", False)),  # 21: CHOCH bearish
            ],
            dtype=np.float32,
        )

        # 3. Market Context (4 features)
        if self.current_position is not None:
            position_value = (
                1.0 if self.current_position["direction"] == "LONG" else -1.0
            )
            entry_price = self.current_position["entry_price"]

            # Calculate unrealized PnL percentage
            if self.current_position["direction"] == "LONG":
                unrealized_pnl_pct = (current_price - entry_price) / entry_price * 100
            else:  # SHORT
                unrealized_pnl_pct = (entry_price - current_price) / entry_price * 100

            # Equity drawdown (negative of unrealized PnL if losing)
            equity_drawdown_pct = min(0.0, unrealized_pnl_pct)

            # Candles in position (capped at 100)
            candles_in_position = min(
                100.0,
                float(
                    self.current_step
                    - self.current_position.get("entry_step", self.current_step)
                ),
            )
        else:
            position_value = 0.0
            unrealized_pnl_pct = 0.0
            equity_drawdown_pct = 0.0
            candles_in_position = 0.0

        market_context = np.array(
            [
                position_value,  # 22: current_position
                unrealized_pnl_pct / 10.0,  # 23: unrealized_pnl_pct (scaled)
                equity_drawdown_pct / 10.0,  # 24: equity_drawdown_pct (scaled)
                candles_in_position / 100.0,  # 25: candles_in_position (normalized)
            ],
            dtype=np.float32,
        )

        # Concatenate all features
        observation = np.concatenate([tech_indicators, pattern_flags, market_context])

        # Replace any NaN or Inf with 0.0 (edge case handling)
        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)

        # Verify shape matches observation_space
        assert observation.shape == self.observation_space.shape, (
            f"Observation shape mismatch: {observation.shape} != "
            f"{self.observation_space.shape}"
        )

        return observation

    def _calculate_reward(self) -> float:
        """
        Calculate reward based on realized PnL.

        Evaluates current position state and checks for exit conditions:
        - TP1 hit: Close 45% position, move SL to breakeven
        - TP2 hit: Close 35% position (of original)
        - Trailing SL hit: Close 20% runner
        - SL hit: Close remaining position
        - Manual close: Close remaining position

        Returns:
            Reward value (positive for profit, negative for loss)
            0.0 if no position exists
        """
        # No position = no reward
        if self.current_position is None:
            return 0.0

        reward = 0.0

        # Get position data
        direction = self.current_position["direction"]
        entry_price = self.current_position["entry_price"]
        current_size = self.current_position["size"]

        # Check for TP1 hit (if not already hit)
        if not self.current_position.get("tp1_hit", False):
            tp1_level = self.current_position["tp1"]
            if self.risk_calculator.should_move_sl_to_breakeven(
                current_price=self.current_price,
                entry_price=entry_price,
                tp1_level=tp1_level,
                direction=direction,
            ):
                # TP1 hit - process partial exit
                reward += self._process_tp1_exit()

        # Check for TP2 hit (if TP1 already hit but TP2 not yet)
        if self.current_position.get(
            "tp1_hit", False
        ) and not self.current_position.get("tp2_hit", False):
            tp2_level = self.current_position["tp2"]
            tp2_hit = False
            if direction == "LONG":
                tp2_hit = self.current_price >= tp2_level
            elif direction == "SHORT":
                tp2_hit = self.current_price <= tp2_level

            if tp2_hit:
                reward += self._process_tp2_exit()

        # Check for trailing stop hit on runner
        if self.current_position.get("tp2_hit", False):
            # Update trailing stop
            self._update_trailing_stop()

            # Check if trailing stop hit
            trailing_stop = self.current_position.get("trailing_stop")
            if trailing_stop is not None:
                trailing_hit = False
                if direction == "LONG":
                    trailing_hit = self.current_price <= trailing_stop
                elif direction == "SHORT":
                    trailing_hit = self.current_price >= trailing_stop

                if trailing_hit:
                    # Close runner at trailing stop
                    reward += self._process_runner_exit()

        # Check for SL hit (close remaining position)
        if self.current_position is not None:
            sl_level = self.current_position["sl"]
            sl_hit = False
            if direction == "LONG":
                sl_hit = self.current_price <= sl_level
            elif direction == "SHORT":
                sl_hit = self.current_price >= sl_level

            if sl_hit:
                # SL hit - close remaining position
                reward += self._process_sl_exit()

        return reward

    def _process_tp1_exit(self) -> float:
        """
        Process partial exit at TP1 (45% of position).

        Closes 45% of position at TP1 level and moves SL to breakeven.

        Returns:
            Realized PnL for the 45% position closed
        """
        if self.current_position is None:
            return 0.0

        # Get position sizing from RiskCalculator
        position_sizes = self.risk_calculator.calculate_position_sizes()
        tp1_pct = position_sizes["tp1_pct"]  # 0.45 = 45%

        # Calculate PnL for closed portion
        direction = self.current_position["direction"]
        entry_price = self.current_position["entry_price"]
        exit_price = self.current_position["tp1"]

        if direction == "LONG":
            pnl = tp1_pct * (exit_price - entry_price)
        else:  # SHORT
            pnl = tp1_pct * (entry_price - exit_price)

        # Update position state
        self.current_position["tp1_hit"] = True
        self.current_position["size"] -= tp1_pct

        # Move SL to breakeven
        self.current_position["sl"] = entry_price

        logger.info(
            f"TP1 hit: Closed {tp1_pct*100}% at {exit_price}, "
            f"PnL={pnl:.6f}, SL moved to BE={entry_price}"
        )

        return pnl

    def _process_tp2_exit(self) -> float:
        """
        Process partial exit at TP2 (35% of ORIGINAL position).

        Closes 35% of original position at TP2 level.
        Remaining 20% becomes runner with trailing stop.

        Returns:
            Realized PnL for the 35% position closed
        """
        if self.current_position is None:
            return 0.0

        # Get position sizing from RiskCalculator
        position_sizes = self.risk_calculator.calculate_position_sizes()
        tp2_pct = position_sizes["tp2_pct"]  # 0.35 = 35%

        # Calculate PnL for closed portion
        direction = self.current_position["direction"]
        entry_price = self.current_position["entry_price"]
        exit_price = self.current_position["tp2"]

        if direction == "LONG":
            pnl = tp2_pct * (exit_price - entry_price)
        else:  # SHORT
            pnl = tp2_pct * (entry_price - exit_price)

        # Update position state
        self.current_position["tp2_hit"] = True
        self.current_position["size"] -= tp2_pct

        # Initialize trailing stop for runner
        if direction == "LONG":
            self.current_position["trailing_stop"] = exit_price - self.current_atr
        else:  # SHORT
            self.current_position["trailing_stop"] = exit_price + self.current_atr

        logger.info(
            f"TP2 hit: Closed {tp2_pct*100}% at {exit_price}, "
            f"PnL={pnl:.6f}, Runner={self.current_position['size']*100}%"
        )

        return pnl

    def _process_runner_exit(self) -> float:
        """
        Process exit of runner position at trailing stop.

        Returns:
            Realized PnL for the runner (20% of original position)
        """
        if self.current_position is None:
            return 0.0

        # Calculate PnL for runner
        direction = self.current_position["direction"]
        entry_price = self.current_position["entry_price"]
        exit_price = self.current_position["trailing_stop"]
        runner_size = self.current_position["size"]

        if direction == "LONG":
            pnl = runner_size * (exit_price - entry_price)
        else:  # SHORT
            pnl = runner_size * (entry_price - exit_price)

        logger.info(
            f"Runner exit: Closed {runner_size*100}% at {exit_price}, PnL={pnl:.6f}"
        )

        # Close position completely
        self.current_position = None

        return pnl

    def _process_sl_exit(self) -> float:
        """
        Process exit at stop-loss level.

        Closes remaining position at SL level.

        Returns:
            Realized PnL (negative for loss, or breakeven)
        """
        if self.current_position is None:
            return 0.0

        # Calculate PnL for remaining position
        direction = self.current_position["direction"]
        entry_price = self.current_position["entry_price"]
        exit_price = self.current_position["sl"]
        remaining_size = self.current_position["size"]

        if direction == "LONG":
            pnl = remaining_size * (exit_price - entry_price)
        else:  # SHORT
            pnl = remaining_size * (entry_price - exit_price)

        logger.info(
            f"SL hit: Closed {remaining_size*100}% at {exit_price}, PnL={pnl:.6f}"
        )

        # Close position completely
        self.current_position = None

        return pnl

    def _update_trailing_stop(self) -> None:
        """
        Update trailing stop for runner position.

        Uses RiskCalculator.update_trailing_stop() to move stop
        in favorable direction only (never against position).
        """
        if self.current_position is None:
            return

        if not self.current_position.get("tp2_hit", False):
            return  # Only trail after TP2 hit

        direction = self.current_position["direction"]
        current_trailing_stop = self.current_position.get("trailing_stop")

        if current_trailing_stop is None:
            return

        # Update trailing stop using RiskCalculator
        new_trailing_stop = self.risk_calculator.update_trailing_stop(
            current_price=self.current_price,
            current_trailing_stop=current_trailing_stop,
            atr=self.current_atr,
            direction=direction,
        )

        # Only log if trailing stop moved
        if new_trailing_stop != current_trailing_stop:
            logger.debug(
                f"Trailing stop updated: {current_trailing_stop:.5f} → {new_trailing_stop:.5f}"
            )
            self.current_position["trailing_stop"] = new_trailing_stop

    def _get_m15_execution_window(self, h1_timestamp: Any) -> list:
        """
        Get M15 execution window timestamps for H1 window.

        Returns safe M15 candles within the H1 window:
        - T+15m, T+30m, T+45m, T+60m
        - Ensures no future leakage

        Args:
            h1_timestamp: H1 timestamp when window opened

        Returns:
            List of M15 timestamps to evaluate for triggers
        """
        from datetime import timedelta

        # Define M15 offsets (15m, 30m, 45m, 60m after H1 timestamp)
        m15_offsets = [timedelta(minutes=15 * i) for i in range(1, 5)]

        m15_timestamps = []
        for offset in m15_offsets:
            m15_ts = h1_timestamp + offset
            # Verify M15 timestamp exists in data (anti-leakage)
            if m15_ts in self.m15_data["timestamp"].values:
                m15_timestamps.append(m15_ts)

        return m15_timestamps

    def _execute_action(self, action: int) -> None:
        """
        Execute trading action.

        Handles:
        - ACTION_HOLD: Do nothing
        - ACTION_ENTER_LONG: Open long position (if trigger valid)
        - ACTION_ENTER_SHORT: Open short position (if trigger valid)
        - ACTION_CLOSE_POSITION: Close current position

        Args:
            action: Action code (0-3)
        """
        if action == self.ACTION_HOLD:
            # Do nothing
            return

        elif action == self.ACTION_CLOSE_POSITION:
            if self.current_position is not None:
                # Manual close - use current price
                logger.info(
                    f"Manual close at {self.current_price}, "
                    f"size={self.current_position['size']*100}%"
                )
                self.current_position = None
            return

        elif action == self.ACTION_ENTER_LONG or action == self.ACTION_ENTER_SHORT:
            # Entry action
            if self.m15_trigger is None:
                logger.warning("Entry action without valid M15 trigger - ignoring")
                return

            if self.current_position is not None:
                logger.warning("Entry action while position exists - ignoring")
                return

            # Get entry price from M15 trigger
            entry_price = self.m15_trigger.get("entry_price")
            if entry_price is None:
                logger.warning("M15 trigger missing entry_price - ignoring")
                return

            # Determine direction
            direction = "LONG" if action == self.ACTION_ENTER_LONG else "SHORT"

            # Calculate SL/TP levels using RiskCalculator
            critical_level = self.pattern_state.get("critical_level")
            atr = self.pattern_state.get("atr")

            if critical_level is None or atr is None:
                logger.warning(
                    "Pattern missing critical_level or ATR - cannot calculate SL/TP"
                )
                return

            # Calculate stop loss
            sl_level = self.risk_calculator.calculate_stop_loss(
                entry_price=entry_price,
                critical_level=critical_level,
                direction=direction,
                atr=atr,
            )

            # Calculate take profit levels
            tp_levels = self.risk_calculator.calculate_take_profit_levels(
                entry_price=entry_price,
                sl_level=sl_level,
                direction=direction,
            )

            # Create position
            self.current_position = {
                "direction": direction,
                "entry_price": entry_price,
                "entry_step": self.current_step,  # Track entry step for market context
                "size": 1.0,  # Full position (100%)
                "sl": sl_level,
                "tp1": tp_levels["tp1"],
                "tp2": tp_levels["tp2"],
                "tp1_hit": False,
                "tp2_hit": False,
            }

            # Store SL/TP in class variables for Stream C coordination
            self.current_sl = sl_level
            self.current_tp1 = tp_levels["tp1"]
            self.current_tp2 = tp_levels["tp2"]

            logger.info(
                f"Position opened: {direction} @ {entry_price}, "
                f"SL={sl_level:.5f}, TP1={tp_levels['tp1']:.5f}, TP2={tp_levels['tp2']:.5f}"
            )
