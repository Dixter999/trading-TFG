"""
Reward function factory/registry.

Provides a registry pattern for creating reward function instances.

TDD Phase: REFACTOR - Extract factory to separate module.
"""

from typing import Dict, Type, List

from .base import BaseReward


class RewardFactory:
    """
    Factory for creating reward function instances.

    Provides a registry pattern for reward functions, allowing:
    - Registration of custom reward types
    - Creation of reward instances by name
    - Listing available reward types

    Example:
        >>> factory = RewardFactory()
        >>> factory.register("pnl", PnLReward)
        >>> factory.register("sharpe", SharpeReward)
        >>>
        >>> # Create reward function by name
        >>> reward_fn = factory.create("pnl")
        >>> isinstance(reward_fn, PnLReward)
        True
        >>>
        >>> # Create with custom parameters
        >>> sharpe_fn = factory.create("sharpe", window_size=30)
        >>> sharpe_fn.window_size
        30
        >>>
        >>> # List available rewards
        >>> factory.list_rewards()
        ['pnl', 'sharpe']
    """

    def __init__(self):
        """Initialize empty reward registry."""
        self._registry: Dict[str, Type[BaseReward]] = {}

    def register(self, name: str, reward_class: Type[BaseReward]) -> None:
        """
        Register a reward function type.

        Args:
            name: Name for this reward type (e.g., "pnl", "sharpe")
            reward_class: Reward class to register (must inherit from BaseReward)

        Raises:
            TypeError: If reward_class doesn't inherit from BaseReward

        Example:
            >>> factory = RewardFactory()
            >>> factory.register("custom", MyCustomReward)
        """
        if not issubclass(reward_class, BaseReward):
            raise TypeError(
                f"reward_class must inherit from BaseReward, "
                f"got {reward_class.__name__}"
            )

        self._registry[name] = reward_class

    def create(self, name: str, **kwargs) -> BaseReward:
        """
        Create a reward function instance.

        Args:
            name: Name of registered reward type
            **kwargs: Arguments for reward constructor

        Returns:
            Reward function instance

        Raises:
            ValueError: If reward type not registered

        Example:
            >>> factory = RewardFactory()
            >>> factory.register("sharpe", SharpeReward)
            >>> reward_fn = factory.create("sharpe", window_size=50)
            >>> reward_fn.window_size
            50
        """
        if name not in self._registry:
            raise ValueError(
                f"Unknown reward type: {name}. "
                f"Available: {list(self._registry.keys())}"
            )

        reward_class = self._registry[name]
        return reward_class(**kwargs)

    def list_rewards(self) -> List[str]:
        """
        List all registered reward types.

        Returns:
            List of reward type names

        Example:
            >>> factory = RewardFactory()
            >>> factory.register("pnl", PnLReward)
            >>> factory.register("sharpe", SharpeReward)
            >>> factory.list_rewards()
            ['pnl', 'sharpe']
        """
        return list(self._registry.keys())
