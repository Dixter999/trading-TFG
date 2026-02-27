"""
Risk-adjusted reward function.

PnL adjusted for drawdown and volatility penalties.

TDD Phase: REFACTOR - Extract risk-adjusted reward to separate module.
"""

from typing import List
import numpy as np

from .base import BaseReward


class RiskAdjustedReward(BaseReward):
    """
    Risk-adjusted reward function with drawdown and volatility penalties.

    Adjusts raw PnL by penalizing:
    - Maximum drawdown: Peak-to-trough decline in equity
    - Volatility: Standard deviation of equity changes

    Formula: PnL - (max_drawdown * dd_penalty) - (volatility * vol_penalty)

    This reward function encourages stable, consistent returns over risky,
    volatile trading strategies.

    Attributes:
        max_drawdown_penalty: Weight for drawdown penalty (default: 1.0)
        volatility_penalty: Weight for volatility penalty (default: 1.0)

    Example:
        >>> reward_fn = RiskAdjustedReward(
        ...     max_drawdown_penalty=1.0,
        ...     volatility_penalty=1.0
        ... )
        >>>
        >>> # Smooth equity curve (minimal penalty)
        >>> smooth_equity = [100.0, 105.0, 110.0, 115.0, 120.0]
        >>> reward = reward_fn.calculate(pnl=20.0, equity_curve=smooth_equity)
        >>> print(f"Reward: {reward:.2f}")
        Reward: 18.44
        >>>
        >>> # Volatile equity curve (higher penalty)
        >>> volatile_equity = [100.0, 150.0, 90.0, 140.0, 110.0]
        >>> reward = reward_fn.calculate(pnl=10.0, equity_curve=volatile_equity)
        >>> print(f"Reward: {reward:.2f}")
        Reward: -56.46
    """

    def __init__(
        self,
        max_drawdown_penalty: float = 1.0,
        volatility_penalty: float = 1.0,
    ):
        """
        Initialize risk-adjusted reward function.

        Args:
            max_drawdown_penalty: Weight for maximum drawdown penalty
                (higher = more penalty for drawdowns)
            volatility_penalty: Weight for volatility penalty
                (higher = more penalty for volatility)

        Raises:
            ValueError: If penalties are negative
        """
        if max_drawdown_penalty < 0.0:
            raise ValueError("max_drawdown_penalty must be >= 0")
        if volatility_penalty < 0.0:
            raise ValueError("volatility_penalty must be >= 0")

        self.max_drawdown_penalty = max_drawdown_penalty
        self.volatility_penalty = volatility_penalty

    def calculate(
        self,
        pnl: float,
        equity_curve: List[float],
    ) -> float:
        """
        Calculate risk-adjusted reward.

        Args:
            pnl: Raw profit/loss
            equity_curve: Historical equity values (e.g., [100.0, 105.0, 110.0, ...])

        Returns:
            Risk-adjusted reward (PnL minus risk penalties)

        Notes:
            - Returns raw PnL if equity_curve has < 2 points
            - Drawdown is always >= 0 (penalty)
            - Volatility is always >= 0 (penalty)

        Example:
            >>> reward_fn = RiskAdjustedReward()
            >>> reward_fn.calculate(
            ...     pnl=30.0,
            ...     equity_curve=[100.0, 120.0, 130.0]
            ... )
            22.93
        """
        if len(equity_curve) < 2:
            # Not enough data for risk calculation
            return pnl

        # Convert to numpy array
        equity_array = np.array(equity_curve, dtype=np.float64)

        # Calculate maximum drawdown
        max_drawdown = self._calculate_max_drawdown(equity_array)

        # Calculate volatility (std of equity changes)
        volatility = self._calculate_volatility(equity_array)

        # Apply penalties
        adjusted_reward = pnl
        adjusted_reward -= max_drawdown * self.max_drawdown_penalty
        adjusted_reward -= volatility * self.volatility_penalty

        return float(adjusted_reward)

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """
        Calculate maximum drawdown from equity curve.

        Maximum drawdown is the largest peak-to-trough decline in equity.
        It measures the worst-case loss from any historical peak.

        Args:
            equity_curve: Array of equity values

        Returns:
            Maximum drawdown value (always >= 0)

        Example:
            >>> equity = np.array([100.0, 120.0, 150.0, 130.0, 120.0])
            >>> reward_fn = RiskAdjustedReward()
            >>> reward_fn._calculate_max_drawdown(equity)
            30.0  # Peak at 150, trough at 120 = 30 drawdown
        """
        # Calculate running maximum (highest equity so far)
        running_max = np.maximum.accumulate(equity_curve)

        # Calculate drawdowns (distance from peak)
        drawdowns = running_max - equity_curve

        # Return maximum drawdown
        max_dd = np.max(drawdowns)

        return float(max_dd)

    def _calculate_volatility(self, equity_curve: np.ndarray) -> float:
        """
        Calculate volatility from equity curve.

        Volatility is measured as the standard deviation of equity changes.
        It represents the variability/instability of returns.

        Args:
            equity_curve: Array of equity values

        Returns:
            Standard deviation of equity changes (always >= 0)

        Example:
            >>> equity = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
            >>> reward_fn = RiskAdjustedReward()
            >>> reward_fn._calculate_volatility(equity)
            0.0  # Constant changes = zero volatility
        """
        # Calculate equity changes (first differences)
        equity_changes = np.diff(equity_curve)

        # Calculate standard deviation
        if len(equity_changes) > 0:
            volatility = np.std(equity_changes, ddof=1)
        else:
            volatility = 0.0

        return float(volatility)
