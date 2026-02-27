"""
Custom pattern-based reward function.

Multiplies PnL by pattern confidence to encourage high-quality trades.

TDD Phase: GREEN - Minimal implementation to pass tests.
"""

from .base import BaseReward


class CustomPatternReward(BaseReward):
    """
    Custom pattern confidence reward function.

    Rewards trades based on pattern confidence multiplier:
    - High confidence (>0.7): 1.5x multiplier
    - Medium confidence (0.4-0.7): 1.0x multiplier
    - Low confidence (<0.4): 0.5x multiplier

    Formula: reward = pnl * confidence_multiplier

    This encourages the agent to:
    1. Wait for high-confidence patterns
    2. Trade less frequently but with higher quality
    3. Learn pattern recognition alongside trading

    Example:
        >>> reward_fn = CustomPatternReward()
        >>>
        >>> # High confidence profitable trade
        >>> reward_fn.calculate(pnl=100.0, pattern_confidence=0.8)
        150.0
        >>>
        >>> # Low confidence profitable trade (penalized)
        >>> reward_fn.calculate(pnl=100.0, pattern_confidence=0.2)
        50.0
    """

    def __init__(
        self,
        high_confidence_multiplier: float = 1.5,
        medium_confidence_multiplier: float = 1.0,
        low_confidence_multiplier: float = 0.5,
        high_confidence_threshold: float = 0.7,
        low_confidence_threshold: float = 0.4,
    ):
        """
        Initialize CustomPatternReward.

        Args:
            high_confidence_multiplier: Multiplier for confidence > 0.7 (default: 1.5)
            medium_confidence_multiplier: Multiplier for confidence 0.4-0.7 (default: 1.0)
            low_confidence_multiplier: Multiplier for confidence < 0.4 (default: 0.5)
            high_confidence_threshold: Threshold for high confidence (default: 0.7)
            low_confidence_threshold: Threshold for low confidence (default: 0.4)
        """
        self.high_confidence_multiplier = high_confidence_multiplier
        self.medium_confidence_multiplier = medium_confidence_multiplier
        self.low_confidence_multiplier = low_confidence_multiplier
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold

    def calculate(
        self,
        pnl: float = None,
        pattern_confidence: float = 0.5,  # Default to medium confidence
        entry_price: float = None,
        exit_price: float = None,
        position: int = None,
        position_size: float = None,
        **kwargs  # Accept additional parameters but ignore them
    ) -> float:
        """
        Calculate pattern-based reward.

        Args:
            pnl: Profit/loss from the trade (if None, calculated from prices)
            pattern_confidence: Pattern confidence score [0.0, 1.0]
                              (default: 0.5 for medium confidence)
            entry_price: Position entry price (used to calculate PnL if pnl is None)
            exit_price: Position exit price (used to calculate PnL if pnl is None)
            position: Position direction (used to calculate PnL if pnl is None)
            position_size: Position size (used to calculate PnL if pnl is None)
            **kwargs: Additional parameters (ignored by this reward function)

        Returns:
            PnL multiplied by confidence-based multiplier

        Example:
            >>> reward_fn = CustomPatternReward()
            >>> reward_fn.calculate(pnl=100.0, pattern_confidence=0.8)
            150.0
        """
        # Calculate PnL if not provided
        if pnl is None and all(v is not None for v in [entry_price, exit_price, position, position_size]):
            if position == 0:
                pnl = 0.0
            else:
                price_diff = exit_price - entry_price
                pnl = price_diff * position * position_size
        elif pnl is None:
            pnl = 0.0

        # Clamp pattern confidence to valid range [0.0, 1.0]
        pattern_confidence = max(0.0, min(1.0, pattern_confidence))

        # Determine multiplier based on confidence level
        if pattern_confidence > self.high_confidence_threshold:
            multiplier = self.high_confidence_multiplier
        elif pattern_confidence >= self.low_confidence_threshold:
            multiplier = self.medium_confidence_multiplier
        else:
            multiplier = self.low_confidence_multiplier

        # Apply multiplier to PnL
        return pnl * multiplier
