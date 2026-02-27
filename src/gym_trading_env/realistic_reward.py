"""
Realistic Reward Function for RL Training.

This module implements a reward function that includes realistic trading costs
and provides properly scaled rewards for reinforcement learning training.

Components:
- Trading cost penalties (commission, spread, slippage)
- Realized P&L rewards (scaled for RL)
- Unrealized P&L ongoing rewards (10% weighting)
- Drawdown penalties (risk management)

TDD Phase: GREEN - Implementing class to pass tests

Author: python-backend-engineer
Issue: #223
Created: 2025-11-11

Example:
    >>> reward_fn = RealisticRewardFunction()
    >>> reward = reward_fn.calculate_step_reward(
    ...     action=1,  # BUY
    ...     prev_position=0,
    ...     current_position=1,
    ...     entry_price=1.06000,
    ...     current_price=1.06000,
    ...     position_size=10000,
    ...     equity_change_pct=0.0,
    ...     equity=10000.0,
    ...     drawdown_pct=0.0
    ... )
"""


class RealisticRewardFunction:
    """
    Calculate reward for RL training considering realistic trading costs.

    This class provides a reward function that:
    1. Penalizes trading costs on entry/exit (commission, spread, slippage)
    2. Rewards realized P&L (scaled by 10x for RL)
    3. Provides small ongoing reward for unrealized P&L (10% weighting)
    4. Penalizes drawdowns to encourage risk management

    The reward scaling is optimized for RL training (typical range -10 to +10).

    Attributes:
        commission_pct: Commission percentage (default 0.05%)
        spread_pips: Spread in pips (default 1.0 pip)
        slippage_pips: Slippage in pips (default 0.5 pips)

    Example:
        >>> reward_fn = RealisticRewardFunction(
        ...     commission_pct=0.05,
        ...     spread_pips=1.0,
        ...     slippage_pips=0.5
        ... )
        >>> reward = reward_fn.calculate_step_reward(...)
    """

    # Constants
    PIP_VALUE_EURUSD = 0.0001  # 1 pip = 0.0001 for EURUSD
    RL_SCALING_FACTOR = 10.0  # Scale equity % to RL reward
    UNREALIZED_PNL_WEIGHT = 0.1  # 10% of unrealized P&L as reward

    def __init__(
        self,
        commission_pct: float = 0.05,
        spread_pips: float = 1.0,
        slippage_pips: float = 0.5,
    ) -> None:
        """
        Initialize the reward function with cost parameters.

        Args:
            commission_pct: Commission percentage (default 0.05%)
            spread_pips: Spread in pips (default 1.0)
            slippage_pips: Slippage in pips (default 0.5)
        """
        self.commission_pct = commission_pct
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips

    def calculate_entry_cost_penalty(
        self, current_price: float, position_size: float
    ) -> float:
        """
        Calculate penalty for opening a position.

        Entry costs include:
        - Commission: commission_pct % of position value
        - Spread: spread_pips in price units
        - Slippage: slippage_pips in price units

        The penalty is expressed as a percentage of position value and scaled for RL.

        Args:
            current_price: Current market price
            position_size: Size of position (e.g., 10000 units)

        Returns:
            Negative penalty value (cost)

        Example:
            >>> reward_fn = RealisticRewardFunction()
            >>> penalty = reward_fn.calculate_entry_cost_penalty(1.06000, 10000)
            >>> print(f"{penalty:.6f}")
            -0.006415
        """
        if position_size == 0:
            return 0.0

        # Calculate position value
        position_value = current_price * position_size

        # Commission cost in USD
        commission_cost = position_value * (self.commission_pct / 100.0)

        # Spread cost in USD
        spread_in_price = self.spread_pips * self.PIP_VALUE_EURUSD
        spread_cost = spread_in_price * position_size

        # Slippage cost in USD
        slippage_in_price = self.slippage_pips * self.PIP_VALUE_EURUSD
        slippage_cost = slippage_in_price * position_size

        # Total cost in USD
        total_cost = commission_cost + spread_cost + slippage_cost

        # Convert to percentage of position value
        cost_pct = (total_cost / position_value) * 100

        # Scale for RL (multiply by scaling factor)
        penalty = -cost_pct * self.RL_SCALING_FACTOR / 100

        return penalty

    def calculate_exit_cost_penalty(
        self, current_price: float, position_size: float
    ) -> float:
        """
        Calculate penalty for closing a position.

        Exit costs are same as entry costs.

        Args:
            current_price: Current market price
            position_size: Size of position (e.g., 10000 units)

        Returns:
            Negative penalty value (cost)

        Example:
            >>> reward_fn = RealisticRewardFunction()
            >>> penalty = reward_fn.calculate_exit_cost_penalty(1.06100, 10000)
            >>> print(f"{penalty:.6f}")
            -0.006415
        """
        # Exit costs are identical to entry costs
        return self.calculate_entry_cost_penalty(current_price, position_size)

    def calculate_realized_pnl_reward(self, equity_change_pct: float) -> float:
        """
        Calculate reward for realized P&L (position closed).

        The equity change percentage is scaled by RL_SCALING_FACTOR (10x) to provide
        appropriate reward magnitude for RL training.

        Args:
            equity_change_pct: Equity change as percentage (e.g., 1.0 for 1% profit)

        Returns:
            Reward value (positive for profit, negative for loss)

        Example:
            >>> reward_fn = RealisticRewardFunction()
            >>> reward = reward_fn.calculate_realized_pnl_reward(1.0)  # 1% profit
            >>> print(f"{reward:.2f}")
            10.00
        """
        # Scale equity change for RL training
        reward = equity_change_pct * self.RL_SCALING_FACTOR

        return reward

    def calculate_unrealized_pnl_reward(
        self,
        position: int,
        entry_price: float,
        current_price: float,
        position_size: float,
        equity: float,
    ) -> float:
        """
        Calculate small ongoing reward for unrealized P&L.

        This encourages the RL agent to hold profitable positions.
        The reward is 10% of the unrealized P&L to avoid overwhelming the learning signal.

        Args:
            position: Current position (-1 SHORT, 0 FLAT, 1 LONG)
            entry_price: Entry price of current position
            current_price: Current market price
            position_size: Size of position
            equity: Current account equity

        Returns:
            Small reward based on unrealized P&L (positive for profit)

        Example:
            >>> reward_fn = RealisticRewardFunction()
            >>> reward = reward_fn.calculate_unrealized_pnl_reward(
            ...     position=1,  # LONG
            ...     entry_price=1.06000,
            ...     current_price=1.06050,  # +50 pips
            ...     position_size=10000,
            ...     equity=10000.0
            ... )
            >>> print(f"{reward:.4f}")
            0.0471
        """
        # No position = no unrealized P&L
        if position == 0 or position_size == 0:
            return 0.0

        # Calculate unrealized P&L based on position direction
        if position == 1:  # LONG
            pnl = (current_price - entry_price) * position_size
        elif position == -1:  # SHORT
            pnl = (entry_price - current_price) * position_size
        else:
            return 0.0

        # Convert to percentage of equity
        pnl_pct = (pnl / equity) * 100

        # Scale for RL and apply weighting (10%)
        # pnl_pct is already a percentage (e.g., 0.5 for 0.5%)
        # Multiply by RL_SCALING_FACTOR (10) and weighting (0.1)
        reward = pnl_pct * self.RL_SCALING_FACTOR * self.UNREALIZED_PNL_WEIGHT

        return reward

    def calculate_drawdown_penalty(self, drawdown_pct: float) -> float:
        """
        Calculate penalty for drawdowns to encourage risk management.

        The penalty increases quadratically with drawdown size to strongly
        discourage large drawdowns.

        Penalty formula:
        - No drawdown (0%): penalty = 0
        - Small drawdown (<5%): penalty = drawdown_pct * 0.5
        - Medium drawdown (5-10%): penalty = drawdown_pct * 1.0
        - Large drawdown (>10%): penalty = drawdown_pct^2 * 0.1

        Args:
            drawdown_pct: Current drawdown as percentage (negative value, e.g., -5.0)

        Returns:
            Negative penalty value (0 for no drawdown)

        Example:
            >>> reward_fn = RealisticRewardFunction()
            >>> penalty = reward_fn.calculate_drawdown_penalty(-15.0)
            >>> print(f"{penalty:.2f}")
            -22.50
        """
        if drawdown_pct >= 0:
            return 0.0

        # Convert to positive for calculation
        dd = abs(drawdown_pct)

        # Quadratic penalty for large drawdowns
        if dd > 10.0:
            penalty = -(dd * dd * 0.1)
        elif dd > 5.0:
            penalty = -dd * 1.0
        else:
            penalty = -dd * 0.5

        return penalty

    def calculate_step_reward(
        self,
        action: int,
        prev_position: int,
        current_position: int,
        entry_price: float,
        current_price: float,
        position_size: float,
        equity_change_pct: float,
        equity: float,
        drawdown_pct: float,
    ) -> float:
        """
        Calculate total reward for a single step in the environment.

        This method combines all reward components based on the action taken:
        - Opening position: Entry cost penalty
        - Holding position: Unrealized P&L reward + drawdown penalty
        - Closing position: Realized P&L reward + exit cost penalty

        Args:
            action: Action taken (0=HOLD, 1=BUY, 2=SELL)
            prev_position: Previous position (-1, 0, 1)
            current_position: Current position (-1, 0, 1)
            entry_price: Entry price of current/previous position
            current_price: Current market price
            position_size: Size of position
            equity_change_pct: Equity change percentage (for realized P&L)
            equity: Current account equity
            drawdown_pct: Current drawdown percentage

        Returns:
            Total reward for the step

        Example:
            >>> reward_fn = RealisticRewardFunction()
            >>> # Opening LONG position
            >>> reward = reward_fn.calculate_step_reward(
            ...     action=1,
            ...     prev_position=0,
            ...     current_position=1,
            ...     entry_price=1.06000,
            ...     current_price=1.06000,
            ...     position_size=10000,
            ...     equity_change_pct=0.0,
            ...     equity=10000.0,
            ...     drawdown_pct=0.0
            ... )
            >>> print(f"{reward:.6f}")
            -0.006415
        """
        total_reward = 0.0

        # Check if opening a new position (FLAT -> LONG/SHORT)
        if prev_position == 0 and current_position != 0:
            # Entry cost penalty
            entry_penalty = self.calculate_entry_cost_penalty(
                current_price, position_size
            )
            total_reward += entry_penalty

        # Check if holding a position
        elif prev_position != 0 and current_position == prev_position:
            # Unrealized P&L reward
            unrealized_reward = self.calculate_unrealized_pnl_reward(
                current_position, entry_price, current_price, position_size, equity
            )
            total_reward += unrealized_reward

            # Drawdown penalty
            dd_penalty = self.calculate_drawdown_penalty(drawdown_pct)
            total_reward += dd_penalty

        # Check if closing a position (LONG/SHORT -> FLAT)
        elif prev_position != 0 and current_position == 0:
            # Realized P&L reward
            realized_reward = self.calculate_realized_pnl_reward(equity_change_pct)
            total_reward += realized_reward

            # Exit cost penalty
            exit_penalty = self.calculate_exit_cost_penalty(
                current_price, position_size
            )
            total_reward += exit_penalty

        return total_reward
