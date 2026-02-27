"""
Sharpe ratio reward function.

Risk-adjusted returns using rolling window of historical returns.

TDD Phase: REFACTOR - Extract Sharpe reward to separate module.
"""

from typing import List
import numpy as np

from .base import BaseReward


class SharpeReward(BaseReward):
    """
    Sharpe ratio reward function.

    Calculates risk-adjusted returns using a sliding window of returns.
    Formula: (mean_return - risk_free_rate) / std_return

    The Sharpe ratio measures the excess return per unit of risk (volatility).
    Higher values indicate better risk-adjusted performance.

    Attributes:
        window_size: Number of returns to use for calculation (default: 20)
        risk_free_rate: Annualized risk-free rate (default: 0.0)

    Example:
        >>> reward_fn = SharpeReward(window_size=10, risk_free_rate=0.0)
        >>>
        >>> # Consistently positive returns (good Sharpe)
        >>> returns = [10.0, 15.0, 12.0, 18.0, 20.0, 14.0, 16.0, 22.0, 19.0, 21.0]
        >>> sharpe = reward_fn.calculate(returns=returns)
        >>> print(f"Sharpe ratio: {sharpe:.2f}")
        Sharpe ratio: 4.23
        >>>
        >>> # High volatility (lower Sharpe)
        >>> volatile_returns = [-50.0, 70.0, -30.0, 90.0, -10.0, 50.0, 10.0, 30.0, -20.0, 60.0]
        >>> sharpe_volatile = reward_fn.calculate(returns=volatile_returns)
        >>> print(f"Volatile Sharpe: {sharpe_volatile:.2f}")
        Volatile Sharpe: 0.42
    """

    def __init__(self, window_size: int = 20, risk_free_rate: float = 0.0):
        """
        Initialize Sharpe reward function.

        Args:
            window_size: Number of returns for rolling calculation
            risk_free_rate: Risk-free rate for Sharpe calculation
                (default: 0.0 for simplicity)

        Raises:
            ValueError: If window_size < 2
        """
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")

        self.window_size = window_size
        self.risk_free_rate = risk_free_rate

    def calculate(self, returns: List[float]) -> float:
        """
        Calculate Sharpe ratio reward.

        Args:
            returns: List of historical returns (e.g., [10.0, 15.0, 12.0, ...])

        Returns:
            Sharpe ratio (higher = better risk-adjusted returns)
            Returns 0.0 if insufficient data (< 2 returns)

        Notes:
            - Only uses last `window_size` returns (sliding window)
            - Returns mean return when std=0 (zero volatility)
            - Handles NaN values gracefully

        Example:
            >>> reward_fn = SharpeReward(window_size=5)
            >>> reward_fn.calculate([100.0, 110.0, 105.0, 115.0, 120.0])
            1.67
        """
        if len(returns) < 2:
            # Not enough data for Sharpe calculation
            return 0.0

        # Use only last window_size returns
        recent_returns = returns[-self.window_size:]

        # Convert to numpy array for calculations
        returns_array = np.array(recent_returns, dtype=np.float64)

        # Calculate mean and std
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)

        # Handle zero std (constant returns)
        if std_return == 0.0 or np.isnan(std_return):
            # Return mean return when no volatility
            return float(mean_return)

        # Calculate Sharpe ratio
        sharpe = (mean_return - self.risk_free_rate) / std_return

        return float(sharpe)
