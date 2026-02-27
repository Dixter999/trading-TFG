"""
Leverage Calculator for paper trading position sizing.

Issue #332: Comprehensive Decision Logging & Full Trading Pipeline Integration
Stream B: Leverage Calculator

This module provides the LeverageCalculator class for:
- Position sizing based on risk management
- Leverage calculations for different currency pairs
- Margin requirement calculations
- P&L per pip calculations
"""

from typing import ClassVar


class LeverageCalculator:
    """
    Calculator for leverage-based position sizing in forex trading.

    Handles position sizing calculations that account for:
    - Risk-based position sizing (% of account at risk)
    - Leverage limits per currency pair
    - Margin requirements
    - Pip value calculations

    Attributes:
        LEVERAGE_MAP: Default leverage values for different currency pairs.
        STANDARD_LOT_SIZE: Units in one standard lot (100,000).
        default_leverage: Default leverage for unknown symbols.
    """

    # Major pairs get 30x, minor/exotic pairs get 20x or less
    LEVERAGE_MAP: ClassVar[dict[str, int]] = {
        # Major pairs - 30x leverage
        "EURUSD": 30,
        "GBPUSD": 30,
        "USDJPY": 30,
        "USDCHF": 30,
        "EURGBP": 30,
        "EURJPY": 30,
        "GBPJPY": 30,
        # Minor pairs - 20x leverage
        "AUDUSD": 20,
        "USDCAD": 20,
        "NZDUSD": 20,
        "AUDCAD": 20,
        "AUDNZD": 20,
        "CADJPY": 20,
        "NZDJPY": 20,
        # Exotic pairs - 10x leverage (default)
    }

    # JPY pairs have different pip size
    JPY_PAIRS: ClassVar[set[str]] = {
        "USDJPY",
        "EURJPY",
        "GBPJPY",
        "AUDJPY",
        "CADJPY",
        "NZDJPY",
        "CHFJPY",
    }

    # Lot sizes
    STANDARD_LOT_SIZE: ClassVar[int] = 100_000  # 1 standard lot = 100,000 units

    def __init__(self, default_leverage: int = 30) -> None:
        """
        Initialize the LeverageCalculator.

        Args:
            default_leverage: Default leverage for unknown symbols (default: 30)
        """
        self.default_leverage = default_leverage

    def get_leverage(self, symbol: str) -> int:
        """
        Get the leverage for a trading symbol.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")

        Returns:
            Leverage ratio (e.g., 30 for 30x leverage)
        """
        normalized_symbol = symbol.upper()
        return self.LEVERAGE_MAP.get(normalized_symbol, self.default_leverage)

    def get_pip_value(self, symbol: str, lot_size: float = 1.0) -> float:
        """
        Calculate pip value for a symbol.

        For most pairs: 1 pip = 0.0001
        For JPY pairs: 1 pip = 0.01

        Pip value = pip_size * lot_size * STANDARD_LOT_SIZE

        For EURUSD: pip value = 0.0001 * 1.0 * 100,000 = $10 per standard lot
        For USDJPY: pip value = 0.01 * 1.0 * 100,000 / rate = varies (~$6.67 at 150)

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            lot_size: Position size in lots (default: 1.0 = standard lot)

        Returns:
            Pip value in account currency (typically USD)
        """
        normalized_symbol = symbol.upper()

        if normalized_symbol in self.JPY_PAIRS:
            # JPY pairs: pip = 0.01, but need to convert to USD
            # Simplified: assume ~$6.67 per pip per standard lot
            # This is an approximation; real implementation would use live rates
            pip_value_per_lot = 6.67
        else:
            # Standard pairs: 0.0001 * 100,000 = $10 per pip per standard lot
            pip_value_per_lot = 10.0

        return pip_value_per_lot * lot_size

    def get_contract_value(self, symbol: str, price: float) -> float:
        """
        Get full contract value for 1 standard lot.

        Contract value = lot_size * STANDARD_LOT_SIZE * price
        For 1 lot: 100,000 * price

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            price: Current market price

        Returns:
            Contract value in account currency (typically USD)
        """
        # For pairs where USD is the quote currency (EURUSD, GBPUSD, etc.)
        # Contract value = 100,000 * price
        return self.STANDARD_LOT_SIZE * price

    def calculate_position_size(
        self,
        symbol: str,
        balance: float,
        risk_percent: float,
        stop_loss_pips: float,
        current_price: float,
    ) -> float:
        """
        Calculate position size in lots based on risk management.

        The position size is the minimum of:
        1. Risk-based size: risk_amount / (stop_loss_pips * pip_value)
        2. Leverage-based max: (balance * leverage) / contract_value

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            balance: Account balance in USD
            risk_percent: Risk per trade as percentage (e.g., 1.0 for 1%)
            stop_loss_pips: Stop loss distance in pips
            current_price: Current market price

        Returns:
            Position size in lots (can be fractional for micro lots)

        Raises:
            ValueError: If balance, stop_loss_pips, or risk_percent are invalid
        """
        # Validate inputs
        if balance <= 0:
            raise ValueError("balance must be positive")
        if stop_loss_pips <= 0:
            raise ValueError("stop_loss_pips must be positive")
        if risk_percent < 0:
            raise ValueError("risk_percent cannot be negative")

        # Zero risk means zero position
        if risk_percent == 0:
            return 0.0

        leverage = self.get_leverage(symbol)
        pip_value_per_lot = self.get_pip_value(symbol, lot_size=1.0)

        # Calculate risk amount in account currency
        risk_amount = balance * (risk_percent / 100.0)

        # Position size based on risk
        # position_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
        position_size_risk = risk_amount / (stop_loss_pips * pip_value_per_lot)

        # Maximum position size based on leverage
        contract_value = self.get_contract_value(symbol, current_price)
        max_position_leverage = (balance * leverage) / contract_value

        # Return the smaller of the two (risk management)
        return min(position_size_risk, max_position_leverage)

    def calculate_margin_required(
        self,
        symbol: str,
        lot_size: float,
        current_price: float,
    ) -> float:
        """
        Calculate margin required for a position.

        Margin = (lot_size * contract_value) / leverage

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            lot_size: Position size in lots
            current_price: Current market price

        Returns:
            Margin required in account currency (typically USD)
        """
        leverage = self.get_leverage(symbol)
        contract_value = self.get_contract_value(symbol, current_price)

        # Margin = position_value / leverage
        position_value = lot_size * contract_value
        return position_value / leverage

    def calculate_pnl_per_pip(
        self,
        symbol: str,
        lot_size: float,
    ) -> float:
        """
        Calculate P&L per pip movement for a given lot size.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            lot_size: Position size in lots

        Returns:
            P&L per pip in account currency (typically USD)
        """
        return self.get_pip_value(symbol, lot_size)
