"""
Expert-aligned reward function for imitation learning.

Extends PatternBehaviorReward with expert alignment bonus to encourage
the RL agent to learn from successful QuantConnect strategies.
"""

from typing import Optional
from .pattern_behavior_reward import PatternBehaviorReward, MarketPhase, PositionInfo


class ExpertAlignedReward(PatternBehaviorReward):
    """
    Reward function that combines pattern-based behavior rewards with
    expert imitation learning bonuses.

    The agent receives:
    1. Base pattern behavior reward (phase + pattern + risk + outcome)
    2. Expert alignment bonus (reward for matching expert actions)

    This enables the RL agent to learn from successful QC strategies while
    still being able to discover novel profitable behaviors.
    """

    def __init__(self, expert_alignment_weight: float = 0.5):
        """
        Initialize expert-aligned reward function.

        Args:
            expert_alignment_weight: Weight for expert alignment bonus (0.0-1.0)
                - 0.0 = Pure pattern behavior (no imitation)
                - 0.5 = Balanced between behavior and imitation
                - 1.0 = Pure imitation learning
        """
        super().__init__()
        self.expert_alignment_weight = expert_alignment_weight

    def calculate(
        self,
        entry_price: float = 0.0,
        exit_price: float = 0.0,
        position: int = 0,
        position_size: float = 0.0,
        action: int = 0,
        ob_is_valid: bool = False,
        trade_count: int = 0,
        ob_signal=None,
        expert_action: Optional[int] = None,
        market_phase: Optional[MarketPhase] = None,
        **kwargs
    ) -> float:
        """
        Calculate reward with expert alignment.

        This method provides compatibility with TradingEnv's raw parameter interface
        while also supporting PatternBehaviorReward's typed object interface.

        Args:
            entry_price: Position entry price
            exit_price: Position exit price (current price)
            position: Position direction (1=LONG, -1=SHORT, 0=FLAT)
            position_size: Position size in base currency
            action: Agent's action (0=SKIP, 1=ENTER, etc.)
            ob_is_valid: Whether order block signal is valid
            trade_count: Number of trades executed so far
            ob_signal: Order block signal (can be None, dict, or OrderBlockSignal)
            expert_action: Expert's action at this timestep (optional)
            market_phase: Current market phase
            **kwargs: Other reward parameters

        Returns:
            Combined reward = base_reward + alignment_bonus

        Example:
            >>> reward_fn = ExpertAlignedReward(expert_alignment_weight=0.5)
            >>> reward = reward_fn.calculate(
            ...     entry_price=1.1000,
            ...     exit_price=1.1050,
            ...     position=1,
            ...     position_size=100000,
            ...     action=1,
            ...     expert_action=1
            ... )
        """
        # For now, use simple PnL-based reward as base
        # PatternBehaviorReward requires complex typed objects we don't have
        # TODO: In future, create typed objects from raw parameters

        # Calculate simple PnL as base reward
        if position == 0:
            base_reward = 0.0
        else:
            price_diff = exit_price - entry_price
            base_reward = price_diff * position * position_size

        # If no expert action provided, return base reward only
        if expert_action is None:
            return base_reward

        # Calculate expert alignment bonus
        alignment_bonus = self._evaluate_expert_alignment(action, expert_action)

        # Combine rewards with weighting
        # expert_weight = 0.5 means: 50% base + 50% expert
        # This allows agent to still discover novel strategies while learning from expert
        total_reward = (
            base_reward * (1.0 - self.expert_alignment_weight) +
            alignment_bonus * self.expert_alignment_weight
        )

        return total_reward

    def _evaluate_expert_alignment(self, action: int, expert_action: int) -> float:
        """
        Evaluate how well agent's action matches expert's action.

        Reward Structure:
        - Match: +10 (strong positive signal)
        - Mismatch: -5 (moderate penalty to discourage random exploration)

        The asymmetry (bigger reward than penalty) encourages imitation
        while not being overly punitive for exploration.

        Args:
            action: Agent's action
            expert_action: Expert's recommended action

        Returns:
            Alignment reward (positive for match, negative for mismatch)
        """
        if action == expert_action:
            # Agent matches expert - strong positive reinforcement
            return 10.0
        else:
            # Agent diverges from expert - moderate penalty
            # Not too harsh to allow beneficial exploration
            return -5.0
