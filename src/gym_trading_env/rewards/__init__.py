"""
Reward function implementations for trading environment.

This module provides various reward calculation strategies:
- PnL-based: Simple profit/loss calculation
- Sharpe ratio: Risk-adjusted returns over a window
- Risk-adjusted: PnL with drawdown and volatility penalties
- Pattern behavior: Rewards correct trading decisions based on market phase,
  pattern recognition, and risk management quality (NOT based on PnL outcome)

TDD Phase: REFACTOR - Organized into modular structure.

Example:
    >>> from gym_trading_env.rewards import PnLReward, reward_factory
    >>>
    >>> # Direct instantiation
    >>> reward_fn = PnLReward()
    >>> reward = reward_fn.calculate(
    ...     entry_price=1.1000,
    ...     exit_price=1.1050,
    ...     position=1,
    ...     position_size=100000
    ... )
    >>> print(reward)
    500.0
    >>>
    >>> # Factory pattern
    >>> sharpe_fn = reward_factory.create("sharpe", window_size=30)
    >>> print(sharpe_fn.window_size)
    30
"""

from .base import BaseReward
from .pnl_reward import PnLReward
from .sharpe_reward import SharpeReward
from .risk_adjusted_reward import RiskAdjustedReward
from .custom_pattern_reward import CustomPatternReward
from .realistic_pnl_reward import RealisticPnLReward
from .pattern_behavior_reward import PatternBehaviorReward, MarketPhase, PositionInfo
from .expert_aligned_reward import ExpertAlignedReward
from .factory import RewardFactory


# Global reward factory instance with built-in rewards pre-registered
reward_factory = RewardFactory()
reward_factory.register("pnl", PnLReward)
reward_factory.register("sharpe", SharpeReward)
reward_factory.register("risk_adjusted", RiskAdjustedReward)
reward_factory.register("custom_pattern", CustomPatternReward)
reward_factory.register("realistic_pnl", RealisticPnLReward)
reward_factory.register("pattern_behavior", PatternBehaviorReward)
reward_factory.register("expert_aligned", ExpertAlignedReward)


# Export all public classes and instances
__all__ = [
    "BaseReward",
    "PnLReward",
    "SharpeReward",
    "RiskAdjustedReward",
    "CustomPatternReward",
    "RealisticPnLReward",
    "PatternBehaviorReward",
    "ExpertAlignedReward",
    "MarketPhase",
    "PositionInfo",
    "RewardFactory",
    "reward_factory",
]
