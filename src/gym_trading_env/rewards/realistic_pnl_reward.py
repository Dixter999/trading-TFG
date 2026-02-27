"""
Realistic PnL reward with transaction costs and SKIP incentives.

This reward function includes:
- Transaction costs (spread + commission)
- SKIP reward when no valid pattern
- Overtrading penalty
- Realistic forex trading conditions

Author: Issue #254 - Baseline RL training
"""

from .base import BaseReward


class RealisticPnLReward(BaseReward):
    """
    Realistic PnL reward function with transaction costs.

    Key Features:
    1. Transaction costs on every trade (spread + commission)
    2. SKIP reward when no valid Order Block pattern (+$1 bonus)
    3. Non-SKIP penalty when no valid OB (-$1 penalty) - prevents blind trading
    4. Overtrading penalty (discourages excessive trading)
    5. Realistic forex conditions (2 pip spread + $7 commission per lot)

    Args:
        spread_pips: Spread in pips (default: 2.0 for EURUSD)
        commission_per_side: Commission per standard lot PER SIDE (default: $3.5, round-trip = $7)
        pip_value: Value of 1 pip for standard lot (default: $10 for EURUSD)
        skip_reward: Reward for SKIP when no valid pattern (default: 1.0)
        overtrade_threshold: Max trades per episode before penalty (default: 100)
        overtrade_penalty: Penalty per trade over threshold (default: -5.0)

    Example:
        >>> reward_fn = RealisticPnLReward()
        >>>
        >>> # Profitable trade with costs
        >>> # Raw PnL: +50 pips = $500
        >>> # Costs: 2 pip spread ($20) + round-trip commission ($3.5×2 = $7) = -$27
        >>> # Net: $500 - $27 = $473
        >>> reward_fn.calculate(
        ...     entry_price=1.1000,
        ...     exit_price=1.1050,  # +50 pips profit
        ...     position=1,  # LONG
        ...     position_size=100000,  # 1 standard lot
        ...     action=1,  # ENTER
        ...     ob_is_valid=True,
        ...     trade_count=10
        ... )
        473.0
        >>>
        >>> # SKIP when no valid pattern (good skip!)
        >>> reward_fn.calculate(
        ...     entry_price=1.1000,
        ...     exit_price=1.1000,
        ...     position=0,
        ...     position_size=0,
        ...     action=0,  # SKIP
        ...     ob_is_valid=False,
        ...     trade_count=10
        ... )
        1.0
    """

    def __init__(
        self,
        spread_pips: float = 2.0,
        commission_per_side: float = 3.5,
        pip_value: float = 10.0,
        skip_reward: float = 1.0,
        overtrade_threshold: int = 100,
        overtrade_penalty: float = -5.0,
    ):
        """
        Initialize realistic reward function.

        Args:
            spread_pips: Spread in pips (2.0 = typical EURUSD spread)
            commission_per_side: Commission per standard lot per SIDE ($3.5 typical for MT5 Razor/ECN accounts)
                                Round trip = $3.5 open + $3.5 close = $7 total
            pip_value: Value of 1 pip per standard lot ($10 for EURUSD)
            skip_reward: Reward for SKIP when no valid pattern
            overtrade_threshold: Max trades before penalty kicks in
            overtrade_penalty: Penalty per trade over threshold
        """
        self.spread_pips = spread_pips
        self.commission_per_side = commission_per_side
        self.pip_value = pip_value
        self.skip_reward = skip_reward
        self.overtrade_threshold = overtrade_threshold
        self.overtrade_penalty = overtrade_penalty

    def calculate(
        self,
        entry_price: float,
        exit_price: float,
        position: int,
        position_size: float,
        action: int = 0,
        ob_is_valid: bool = False,
        trade_count: int = 0,
    ) -> float:
        """
        Calculate realistic PnL reward with costs.

        Args:
            entry_price: Position entry price
            exit_price: Position exit price
            position: Position direction (1=LONG, -1=SHORT, 0=FLAT)
            position_size: Position size in base currency (100000 = 1 lot)
            action: Action taken (0=SKIP, 1=ENTER, etc.)
            ob_is_valid: Whether Order Block pattern is valid
            trade_count: Total trades in episode so far

        Returns:
            Net reward after all costs and bonuses
        """
        # SKIP action handling (must check action FIRST, before position)
        if action == 0:
            # Reward SKIP when no valid pattern (good decision!)
            if not ob_is_valid:
                return self.skip_reward
            # Penalty for SKIP when valid pattern exists (missed opportunity)
            else:
                return -0.5

        # No position case (checking reward before trade execution)
        if position == 0:
            # Attempting to trade without valid OB signal - strong penalty!
            if not ob_is_valid:
                return -1.0  # Penalize trying to trade without valid OB
            # Valid OB but no position yet - neutral (about to enter position)
            return 0.0

        # Calculate raw PnL
        price_diff = exit_price - entry_price
        raw_pnl = price_diff * position * position_size

        # Calculate transaction costs
        # 1. Spread cost (paid on entry, in pips converted to dollars)
        spread_cost = self.spread_pips * self.pip_value * (position_size / 100000)

        # 2. Commission (paid per side: entry + exit)
        # Example: $3.5/side × 2 sides × (position_size/100000 lots) = $7 for 1 standard lot
        commission_cost = 2 * self.commission_per_side * (position_size / 100000)

        # Total transaction cost (spread + round-trip commission)
        transaction_cost = spread_cost + commission_cost

        # Net PnL after costs
        net_pnl = raw_pnl - transaction_cost

        # Overtrading penalty
        if trade_count > self.overtrade_threshold:
            overtrading_penalty = self.overtrade_penalty
        else:
            overtrading_penalty = 0.0

        # Final reward
        final_reward = net_pnl + overtrading_penalty

        return final_reward

    def get_info(self) -> dict:
        """
        Get reward function configuration.

        Returns:
            Dictionary with reward parameters
        """
        return {
            "type": "realistic_pnl",
            "spread_pips": self.spread_pips,
            "commission_per_side": self.commission_per_side,
            "pip_value": self.pip_value,
            "skip_reward": self.skip_reward,
            "overtrade_threshold": self.overtrade_threshold,
            "overtrade_penalty": self.overtrade_penalty,
        }
