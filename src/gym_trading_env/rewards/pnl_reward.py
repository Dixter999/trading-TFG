"""
PnL-based reward function.

Simple profit/loss calculation for trading positions.

TDD Phase: REFACTOR - Extract PnL reward to separate module.
"""

from .base import BaseReward


class PnLReward(BaseReward):
    """
    Simple profit/loss reward function.

    Calculates raw PnL from position entry and exit prices.
    Formula: (exit_price - entry_price) * position * position_size

    Where:
    - position: 1 (LONG), -1 (SHORT), 0 (FLAT)
    - position_size: Size in base currency units

    Example:
        >>> reward_fn = PnLReward()
        >>>
        >>> # Profitable long position
        >>> reward_fn.calculate(
        ...     entry_price=1.1000,
        ...     exit_price=1.1050,
        ...     position=1,
        ...     position_size=100000
        ... )
        500.0
        >>>
        >>> # Losing short position
        >>> reward_fn.calculate(
        ...     entry_price=1.1000,
        ...     exit_price=1.1050,
        ...     position=-1,
        ...     position_size=100000
        ... )
        -500.0
    """

    def calculate(
        self,
        entry_price: float,
        exit_price: float,
        position: int,
        position_size: float,
        **kwargs  # Accept additional parameters but ignore them
    ) -> float:
        """
        Calculate PnL reward.

        Args:
            entry_price: Position entry price (e.g., 1.1000)
            exit_price: Position exit price (e.g., 1.1050)
            position: Position direction (1=LONG, -1=SHORT, 0=FLAT)
            position_size: Position size in base currency (e.g., 100000)
            **kwargs: Additional parameters (ignored by this reward function)

        Returns:
            PnL value (positive = profit, negative = loss)

        Example:
            >>> reward_fn = PnLReward()
            >>> reward_fn.calculate(1.1000, 1.1050, 1, 100000)
            500.0
        """
        if position == 0:
            return 0.0

        # Convert Decimal to float (PostgreSQL returns Decimal for numeric columns)
        entry_price = float(entry_price)
        exit_price = float(exit_price)

        # Calculate price difference
        price_diff = exit_price - entry_price

        # Apply position direction:
        # LONG (position=1): profit when exit > entry
        # SHORT (position=-1): profit when entry > exit
        pnl = price_diff * position * position_size

        return pnl
