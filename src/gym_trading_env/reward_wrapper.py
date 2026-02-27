"""
Reward Function Wrapper for Gym Environment Integration.

This module provides RewardFunctionWrapper - a helper class that integrates
RealisticRewardFunction into gym trading environments while tracking statistics
and providing analytics.

TDD Phase: GREEN - Implementing wrapper to pass integration tests

Author: python-backend-engineer
Issue: #223 Phase 2
Created: 2025-11-11

Example:
    >>> from src.gym_trading_env.reward_wrapper import RewardFunctionWrapper
    >>> wrapper = RewardFunctionWrapper(commission_pct=0.05)
    >>> reward = wrapper.calculate_step_reward(
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
    >>> stats = wrapper.get_statistics()
    >>> print(stats['mean_reward'])
"""

import numpy as np
from typing import Dict, List

from .realistic_reward import RealisticRewardFunction


class RewardFunctionWrapper:
    """
    Wrapper for RealisticRewardFunction with statistics tracking.

    This class wraps RealisticRewardFunction to provide:
    1. Simplified interface for gym environment integration
    2. Automatic tracking of reward statistics
    3. Correlation analysis between rewards and equity changes
    4. Episode-level analytics

    The wrapper maintains internal history of rewards and equity changes
    to compute correlation and distribution statistics.

    Attributes:
        reward_function: Instance of RealisticRewardFunction
        reward_history: List of all rewards calculated
        equity_change_history: List of corresponding equity changes

    Example:
        >>> wrapper = RewardFunctionWrapper()
        >>> reward = wrapper.calculate_step_reward(...)
        >>> stats = wrapper.get_statistics()
        >>> print(f"Correlation: {stats['correlation']:.3f}")
    """

    def __init__(
        self,
        commission_pct: float = 0.05,
        spread_pips: float = 1.0,
        slippage_pips: float = 0.5,
    ) -> None:
        """
        Initialize reward function wrapper.

        Args:
            commission_pct: Commission percentage (default 0.05%)
            spread_pips: Spread in pips (default 1.0)
            slippage_pips: Slippage in pips (default 0.5)
        """
        self.reward_function = RealisticRewardFunction(
            commission_pct=commission_pct,
            spread_pips=spread_pips,
            slippage_pips=slippage_pips,
        )

        # Track reward history for statistics
        self.reward_history: List[float] = []
        self.equity_change_history: List[float] = []

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
        Calculate reward for current step and update history.

        This is a pass-through to RealisticRewardFunction.calculate_step_reward
        with automatic history tracking.

        Args:
            action: Action taken (0=HOLD, 1=BUY, 2=SELL)
            prev_position: Previous position (-1, 0, 1)
            current_position: Current position (-1, 0, 1)
            entry_price: Entry price of position
            current_price: Current market price
            position_size: Position size
            equity_change_pct: Equity change percentage
            equity: Current equity
            drawdown_pct: Current drawdown percentage

        Returns:
            Reward for this step

        Example:
            >>> wrapper = RewardFunctionWrapper()
            >>> reward = wrapper.calculate_step_reward(
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
        """
        # Calculate reward using underlying reward function
        reward = self.reward_function.calculate_step_reward(
            action=action,
            prev_position=prev_position,
            current_position=current_position,
            entry_price=entry_price,
            current_price=current_price,
            position_size=position_size,
            equity_change_pct=equity_change_pct,
            equity=equity,
            drawdown_pct=drawdown_pct,
        )

        # Track history for statistics
        self.reward_history.append(reward)
        self.equity_change_history.append(equity_change_pct)

        return reward

    def get_statistics(self) -> Dict[str, float]:
        """
        Get reward statistics from history.

        Calculates and returns:
        - mean_reward: Average reward per step
        - std_reward: Standard deviation of rewards
        - min_reward: Minimum reward observed
        - max_reward: Maximum reward observed
        - reward_count: Number of rewards calculated
        - correlation: Correlation between rewards and equity changes (if >= 2 samples)

        Returns:
            Dictionary with reward statistics

        Example:
            >>> wrapper = RewardFunctionWrapper()
            >>> # ... calculate some rewards ...
            >>> stats = wrapper.get_statistics()
            >>> print(f"Mean: {stats['mean_reward']:.3f}")
            >>> print(f"Correlation: {stats['correlation']:.3f}")
        """
        if len(self.reward_history) == 0:
            return {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "reward_count": 0,
            }

        stats = {
            "mean_reward": float(np.mean(self.reward_history)),
            "std_reward": float(np.std(self.reward_history)),
            "min_reward": float(np.min(self.reward_history)),
            "max_reward": float(np.max(self.reward_history)),
            "reward_count": len(self.reward_history),
        }

        # Calculate correlation if we have at least 2 samples
        if len(self.reward_history) >= 2:
            # Only calculate correlation for non-zero equity changes
            # (to avoid correlation issues with HOLD actions)
            non_zero_indices = [
                i
                for i, eq_change in enumerate(self.equity_change_history)
                if abs(eq_change) > 0.001
            ]

            if len(non_zero_indices) >= 2:
                rewards_nz = [self.reward_history[i] for i in non_zero_indices]
                equity_changes_nz = [
                    self.equity_change_history[i] for i in non_zero_indices
                ]

                correlation = np.corrcoef(equity_changes_nz, rewards_nz)[0, 1]
                stats["correlation"] = float(correlation)

        return stats

    def reset_statistics(self) -> None:
        """
        Reset reward and equity change history.

        Useful for starting a new episode or training run.

        Example:
            >>> wrapper = RewardFunctionWrapper()
            >>> # ... train for episode ...
            >>> wrapper.reset_statistics()  # Clear for next episode
        """
        self.reward_history = []
        self.equity_change_history = []
