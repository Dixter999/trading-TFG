"""
Base reward class for trading environment.

This module defines the abstract interface that all reward functions must implement.

TDD Phase: REFACTOR - Extract base class for better organization.
"""

from abc import ABC, abstractmethod


class BaseReward(ABC):
    """
    Abstract base class for reward functions.

    All reward implementations must inherit from this class and implement
    the calculate() method.

    Example:
        >>> class MyReward(BaseReward):
        ...     def calculate(self, **kwargs) -> float:
        ...         return kwargs.get('pnl', 0.0) * 1.5
        ...
        >>> reward_fn = MyReward()
        >>> reward_fn.calculate(pnl=100.0)
        150.0
    """

    @abstractmethod
    def calculate(self, **kwargs) -> float:
        """
        Calculate reward based on trading metrics.

        Args:
            **kwargs: Reward-specific parameters

        Returns:
            Calculated reward value

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        pass
