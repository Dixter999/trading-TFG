"""
Risk Manager for Paper Trading Engine (Issue #432).

This module provides the RiskManager class for enforcing risk management rules:
- Position limits (max 4 concurrent, 1 per symbol)
- Daily trade limits (max 20 trades per day)
- Daily loss limits (max 100 pips loss)
- Correlation rules (no same-direction correlated pairs)
- Currency exposure limits (max 2 USD/EUR positions)
- Position sizing (PF-based volume tiers, Issue #627)
- Dynamic balance-based position sizing (Issue #630)
- Market hours validation (block weekend trading)
- Rejection cooldown (prevent spam after risk rejection)
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import yaml

from src.paper_trading.balance_allocator import BalanceAllocator, SignalBudgetAllocator
from src.paper_trading.models import SL_PIPS, PositionDirection
from src.paper_trading.position_manager import PositionManager

# FX market hours (UTC)
# Market closes Friday ~22:00 UTC and opens Sunday ~22:00 UTC
# For simplicity, we block all of Saturday and Sunday
WEEKEND_DAYS = {5, 6}  # Saturday=5, Sunday=6

# Default cooldown period after rejection (in seconds)
DEFAULT_REJECTION_COOLDOWN_SECONDS = 300  # 5 minutes

# Correlation matrix for currency pairs (Issue #432)
# Key: tuple of symbols (sorted alphabetically)
# Value: correlation coefficient (0.6+ is considered correlated)
CORRELATIONS: dict[tuple[str, str], float] = {
    ("EURJPY", "EURUSD"): 0.7,  # High correlation (both EUR pairs)
    ("EURUSD", "GBPUSD"): 0.6,  # Moderate correlation (both USD pairs)
}


def get_correlation(symbol1: str, symbol2: str) -> float:
    """Get correlation between two symbols.

    Args:
        symbol1: First trading symbol
        symbol2: Second trading symbol

    Returns:
        Correlation coefficient, 0 if no correlation defined
    """
    # Sort symbols to ensure consistent key lookup
    key = tuple(sorted([symbol1, symbol2]))
    return CORRELATIONS.get(key, 0.0)


@dataclass
class RiskConfig:
    """Configuration for risk management parameters.

    Attributes:
        max_concurrent_positions: Maximum number of open positions (dynamic, derived from signal count)
        max_positions_per_symbol: Maximum positions per symbol (default: 8)
        max_daily_trades: Maximum trades allowed per day (default: 30)
        max_daily_loss_pips: Maximum allowed daily loss in pips (default: 150)
        risk_per_trade_pct: Risk percentage per trade (default: 0.01 = 1%)
        rejection_cooldown_seconds: Cooldown after rejection in seconds (default: 300)
        max_position_size: Maximum position size in lots (default: 1.0)
            Note: Actual position sizes are controlled by PF-scaled caps (Issue #635).
            This value acts as an absolute upper bound.
    """

    max_concurrent_positions: int = 25
    max_positions_per_symbol: int = 8
    max_daily_trades: int = 30
    max_daily_loss_pips: int = 150
    risk_per_trade_pct: float = 0.01
    rejection_cooldown_seconds: int = DEFAULT_REJECTION_COOLDOWN_SECONDS
    max_position_size: float = 1.0  # Let PF-scaled caps control sizing (Issue #635)
    lookahead_candles: int = 6  # Hours of future signal estimation window
    signal_hit_rate: float = 0.04  # Expected signal fire rate per candle close


class RiskManager:
    """Enforces risk management rules for paper trading.

    This class validates trading signals against configured risk limits
    before allowing positions to be opened.

    Attributes:
        config: RiskConfig with risk parameters (dict or RiskConfig)
        positions: PositionManager instance for position tracking
        daily_trades: Count of trades executed today
        daily_pnl_pips: Cumulative P&L in pips for today
        rejection_times: Dict mapping symbol to last rejection timestamp
        _approved_models: List of approved models for PF lookup
    """

    # Class-level logger for easy mocking in tests
    logger = logging.getLogger(__name__)

    def _get_config_value(self, key: str, default=None):
        """Get config value from RiskConfig.

        Args:
            key: Configuration key name
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return getattr(self.config, key, default)

    def __init__(self, config: RiskConfig | dict, position_manager: PositionManager) -> None:
        """Initialize RiskManager.

        Args:
            config: Risk configuration parameters (RiskConfig or dict)
            position_manager: PositionManager instance for position tracking
        """
        # Handle both dict and RiskConfig types - convert dict to RiskConfig
        if isinstance(config, dict):
            self.config = RiskConfig(**{k: v for k, v in config.items() if k in RiskConfig.__dataclass_fields__})
        else:
            self.config = config

        self.positions = position_manager
        self.daily_trades: int = 0
        self.daily_pnl_pips: float = 0
        self.rejection_times: dict[str, datetime] = {}

        # Live performance tracker for adaptive sizing (replaces PF-based sizing)
        self.live_tracker = None  # Set externally after construction

        # Load approved models for PF lookup (Issue #627)
        self._approved_models = self._load_approved_models()

        # Validate max_position_size configuration
        max_size = self._get_config_value('max_position_size')
        if max_size is None:
            self.logger.warning(
                "max_position_size not configured - using default 1.0 lots (PF-scaled caps apply)"
            )
            self.config.max_position_size = 1.0

        # Note: max_position_size of 1.0 is now expected as PF-scaled caps control actual sizing
        if max_size and max_size > 1.0:
            self.logger.warning(
                f"max_position_size {max_size} exceeds maximum 1.0 lots"
            )

    def _load_approved_models(self) -> list:
        """Load approved models from local file for PF lookup.

        Returns:
            List of approved model dictionaries, or empty list on failure.
        """
        # TFG: Local file only (no GCS)
        local_paths = [
            "config/approved_models.yaml",
        ]
        for path in local_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        data = yaml.safe_load(f)
                    models = data.get('approved_models', [])
                    self.logger.info(f"Loaded {len(models)} approved models for PF lookup (from {path})")
                    return models
            except Exception as e:
                self.logger.warning(f"Failed to load from {path}: {e}")

        self.logger.error("Failed to load approved models from any source")
        return []

    def _get_volume_from_pf(self, pf: float) -> float:
        """Get lot size based on Phase 5 Profit Factor (Issue #631).

        PF-Based Volume Tiers:
        - PF >= 3.0: 0.5 lots (Excellent - highest confidence)
        - PF >= 2.0: 0.3 lots (Very good)
        - PF >= 1.5: 0.2 lots (Good)
        - PF < 1.5:  0.1 lots (Conservative base tier)

        Args:
            pf: Phase 5 Profit Factor value

        Returns:
            Lot size based on PF tier (0.1, 0.2, 0.3, or 0.5)
        """
        if pf >= 3.0:
            return 0.5
        elif pf >= 2.0:
            return 0.3
        elif pf >= 1.5:
            return 0.2
        return 0.1

    def _get_pf_scaled_cap(self, pf: float) -> float:
        """Get position size cap based on Profit Factor tier (Issue #635).

        This replaces the hardcoded 0.30 lot cap with dynamic PF-based caps
        that scale with signal quality. Higher PF signals can use larger sizes.

        PF-Based Position Size Caps:
        - PF >= 3.0: 1.00 lots (Elite - highest confidence)
        - PF >= 2.5: 0.75 lots (Strong)
        - PF >= 2.0: 0.50 lots (Good)
        - PF >= 1.5: 0.35 lots (Moderate)
        - PF < 1.5:  0.20 lots (Baseline - conservative)

        Args:
            pf: Phase 5 Profit Factor value

        Returns:
            Maximum position size cap in lots based on PF tier
        """
        if pf >= 3.0:
            return 1.0   # Elite: up to 1.0 lots
        elif pf >= 2.5:
            return 0.75  # Strong: up to 0.75 lots
        elif pf >= 2.0:
            return 0.50  # Good: up to 0.50 lots
        elif pf >= 1.5:
            return 0.35  # Moderate: up to 0.35 lots
        else:
            return 0.20  # Baseline: up to 0.20 lots

    def _get_phase5_pf(self, symbol: str, signal_name: str, timeframe: str) -> float:
        """Get Phase 5 PF for signal from approved models.

        Args:
            symbol: Trading symbol (will be lowercased for matching)
            signal_name: Signal name from entry evaluator
            timeframe: Signal timeframe (e.g., 'H4')

        Returns:
            Phase 5 PF, or 1.0 if not found (conservative tier).
        """
        symbol_lower = symbol.lower()

        for model in self._approved_models:
            if (model.get('symbol') == symbol_lower and
                model.get('signal_name') == signal_name and
                model.get('timeframe') == timeframe):
                return float(model.get('phase5_pf', 1.0))

        self.logger.warning(
            f"Phase 5 PF not found for {symbol} {signal_name} {timeframe} - "
            f"using conservative 1.0"
        )
        return 1.0

    def can_open_position(
        self, symbol: str, direction: PositionDirection
    ) -> tuple[bool, str]:
        """Check if a new position is allowed by risk rules.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            direction: Position direction (LONG or SHORT)

        Returns:
            Tuple of (can_open: bool, reason: str)
            If can_open is False, reason contains the blocking rule
        """
        # Check market hours first (block weekend trading)
        if not self.is_market_open():
            return (False, "Market is closed (weekend)")

        # Check if symbol is on cooldown after recent rejection
        if self.is_on_cooldown(symbol):
            return (False, f"{symbol} is on cooldown after rejection")

        # Check max_positions_per_symbol limit (counted per direction)
        dir_str = direction.value if hasattr(direction, 'value') else str(direction).upper()
        symbol_dir_positions = sum(
            1 for pos in self.positions._positions.values()
            if pos.symbol == symbol and pos.direction.value == dir_str
        )
        if symbol_dir_positions >= self.config.max_positions_per_symbol:
            return (
                False,
                f"Max positions per symbol+direction reached for {symbol} {dir_str} "
                f"({symbol_dir_positions}/{self.config.max_positions_per_symbol})"
            )

        # Check position limit
        if not self.check_position_limit():
            self.record_rejection(symbol)
            return (False, f"Maximum position limit reached ({self.config.max_concurrent_positions})")

        # Check daily trade limit
        if not self.check_daily_trade_limit():
            self.record_rejection(symbol)
            return (False, f"Daily trade limit reached ({self.config.max_daily_trades})")

        # Check daily loss limit
        if not self.check_daily_loss_limit():
            self.record_rejection(symbol)
            return (False, f"Daily loss limit reached ({self.config.max_daily_loss_pips} pips)")

        # Check correlation rules
        if not self.check_correlation_rules(symbol, direction):
            self.record_rejection(symbol)
            return (
                False,
                f"Correlation rule violation: {symbol} correlates with existing position",
            )

        return (True, "")

    async def can_open_position_async(
        self, symbol: str, direction: PositionDirection
    ) -> tuple[bool, str]:
        """Async version of can_open_position that uses MT5 for market hours check.

        This method queries MT5 for fresh market data to determine if the market
        is actually open, rather than relying on simple calendar-based weekend checks.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            direction: Position direction (LONG or SHORT)

        Returns:
            Tuple of (can_open: bool, reason: str)
            If can_open is False, reason contains the blocking rule
        """
        # Check market hours using MT5 data freshness check
        is_open, market_reason = await self.is_market_open_mt5(symbol)
        if not is_open:
            return (False, market_reason)

        # Check if symbol is on cooldown after recent rejection
        if self.is_on_cooldown(symbol):
            return (False, f"{symbol} is on cooldown after rejection")

        # Check max_positions_per_symbol limit (counted per direction)
        dir_str = direction.value if hasattr(direction, 'value') else str(direction).upper()
        symbol_dir_positions = sum(
            1 for pos in self.positions._positions.values()
            if pos.symbol == symbol and pos.direction.value == dir_str
        )
        if symbol_dir_positions >= self.config.max_positions_per_symbol:
            return (
                False,
                f"Max positions per symbol+direction reached for {symbol} {dir_str} "
                f"({symbol_dir_positions}/{self.config.max_positions_per_symbol})"
            )

        # Check position limit
        if not self.check_position_limit():
            self.record_rejection(symbol)
            return (False, f"Maximum position limit reached ({self.config.max_concurrent_positions})")

        # Check daily trade limit
        if not self.check_daily_trade_limit():
            self.record_rejection(symbol)
            return (False, f"Daily trade limit reached ({self.config.max_daily_trades})")

        # Check daily loss limit
        if not self.check_daily_loss_limit():
            self.record_rejection(symbol)
            return (False, f"Daily loss limit reached ({self.config.max_daily_loss_pips} pips)")

        # Check correlation rules
        if not self.check_correlation_rules(symbol, direction):
            self.record_rejection(symbol)
            return (
                False,
                f"Correlation rule violation: {symbol} correlates with existing position",
            )

        return (True, "")

    def check_position_limit(self) -> bool:
        """Check if under maximum concurrent positions.

        Returns:
            True if can open more positions, False if at limit
        """
        current_positions = len(self.positions.get_all_positions())
        return current_positions < self._get_config_value('max_concurrent_positions', 4)

    def check_daily_trade_limit(self) -> bool:
        """Check if under daily trade limit.

        Returns:
            True if can make more trades, False if at limit
        """
        return self.daily_trades < self._get_config_value('max_daily_trades', 20)

    def check_daily_loss_limit(self) -> bool:
        """Check if under daily loss limit.

        Returns:
            True if can continue trading, False if loss limit hit
        """
        return self.daily_pnl_pips > -self._get_config_value('max_daily_loss_pips', 100)

    def check_correlation_rules(
        self, symbol: str, direction: PositionDirection
    ) -> bool:
        """Check if position violates correlation rules.

        Blocks same-direction positions on highly correlated pairs
        (correlation >= 0.6).

        Args:
            symbol: Symbol to check
            direction: Direction of proposed position

        Returns:
            True if position is allowed, False if blocked by correlation rules
        """
        open_positions = self.positions.get_all_positions()

        for _, position in open_positions.items():
            # Use position.symbol, not dictionary key (which may include direction suffix)
            correlation = get_correlation(symbol, position.symbol)

            # If correlation is high (>= 0.6)
            if correlation >= 0.6:
                # Block same direction (concentrated risk)
                if position.direction == direction:
                    return False

        return True

    def calculate_position_size(
        self,
        symbol: str,
        account_balance: Decimal,
        signal_name: str = None,
        timeframe: str = None
    ) -> Decimal:
        """Calculate position size based on Phase 5 PF and risk limits.

        PF-Based Volume Tiers (Issue #627):
        - PF >= 3.0: 0.5 lots (Excellent - high confidence)
        - PF >= 2.0: 0.3 lots (Very good)
        - PF >= 1.5: 0.2 lots (Good)
        - PF >= 1.2: 0.1 lots (Conservative - minimum profitable)

        Args:
            symbol: Trading symbol
            account_balance: Current account balance in USD
            signal_name: Signal name for PF lookup (optional)
            timeframe: Signal timeframe for PF lookup (optional)

        Returns:
            Position size in lots (capped by max_position_size)
        """
        # Get Phase 5 PF if signal info provided
        if signal_name and timeframe:
            phase5_pf = self._get_phase5_pf(symbol, signal_name, timeframe)
        else:
            phase5_pf = 1.0  # Conservative default

        # Determine tier based on PF
        if phase5_pf >= 3.0:
            volume = Decimal("0.5")  # Tier 1: Excellent
            risk_multiplier = Decimal("3.0")
        elif phase5_pf >= 2.0:
            volume = Decimal("0.3")  # Tier 2: Very good
            risk_multiplier = Decimal("2.0")
        elif phase5_pf >= 1.5:
            volume = Decimal("0.2")  # Tier 3: Good
            risk_multiplier = Decimal("1.5")
        else:
            volume = Decimal("0.1")  # Tier 4: Conservative
            risk_multiplier = Decimal("1.0")

        # Apply max_position_size cap from config
        max_size = Decimal(str(self._get_config_value('max_position_size', 0.3)))

        if volume > max_size:
            self.logger.info(
                f"Position size capped: {volume} -> {max_size} lots "
                f"(PF={phase5_pf:.2f}, tier would be {volume})"
            )
            volume = max_size

        # Log volume decision
        self.logger.info(
            f"Position size for {symbol}: {volume} lots "
            f"(PF={phase5_pf:.2f}, risk_multiplier={risk_multiplier})"
        )

        return volume

    def calculate_position_size_dynamic(
        self,
        symbol: str,
        account_balance: Decimal,
        direction: str = "LONG",
        signal_name: str = None,
        timeframe: str = None,
        use_dynamic_sizing: bool = True
    ) -> Decimal:
        """Calculate position size using dynamic balance-based sizing (Issue #630).

        This method provides intelligent position sizing that adapts to:
        1. Current available balance
        2. Per-symbol margin requirements
        3. Maximum concurrent position limits
        4. Diversification preferences
        5. Profit factor bonus multiplier

        When use_dynamic_sizing=False, falls back to PF-based tiers (Issue #627).

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            account_balance: Current account balance in USD
            direction: Trade direction ("LONG" or "SHORT")
            signal_name: Signal name for PF lookup (optional)
            timeframe: Signal timeframe for PF lookup (optional)
            use_dynamic_sizing: Use dynamic sizing (True) or PF tiers (False)

        Returns:
            Position size in lots
        """
        if not use_dynamic_sizing:
            # Fallback to legacy PF-based tiers
            return self.calculate_position_size(
                symbol=symbol,
                account_balance=account_balance,
                signal_name=signal_name,
                timeframe=timeframe
            )

        # Get Phase 5 PF if signal info provided
        if signal_name and timeframe:
            phase5_pf = self._get_phase5_pf(symbol, signal_name, timeframe)
        else:
            phase5_pf = 1.0  # Conservative default

        # Get max concurrent positions from config
        max_concurrent = self._get_config_value('max_concurrent_positions', 12)

        # Create balance allocator with current balance
        allocator = BalanceAllocator(
            total_balance=float(account_balance),
            max_concurrent_positions=max_concurrent
        )

        # Get active positions for allocation calculation
        active_positions = {}
        for pos_id, pos in self.positions._positions.items():
            # Calculate margin used per position (simplified estimate)
            # In practice, this should use actual entry prices
            margin_estimate = float(pos.size) * 3500  # ~$3500 per lot average
            active_positions[pos_id] = {
                "symbol": pos.symbol,
                "direction": pos.direction.value,
                "size": pos.size,
                "margin_used": margin_estimate
            }

        # Get dynamic position size
        dynamic_size = allocator.get_position_size(
            symbol=symbol,
            direction=direction,
            profit_factor=phase5_pf,
            active_positions=active_positions
        )

        # Issue #635: Use balance-based lot size directly (matches dashboard)
        # Only apply config max_position_size as absolute upper bound (default 1.0)
        config_max = float(self._get_config_value('max_position_size', 1.0))

        if dynamic_size > config_max:
            self.logger.info(
                f"Position size capped at config max: {dynamic_size:.2f} -> {config_max:.2f} lots"
            )
            dynamic_size = config_max

        self.logger.info(
            f"Dynamic position size for {symbol}: {dynamic_size:.2f} lots "
            f"(balance=${float(account_balance):.0f}, PF={phase5_pf:.2f}, "
            f"active_positions={len(active_positions)})"
        )

        return Decimal(str(dynamic_size))

    def calculate_batch_position_sizes(
        self,
        ready_signals: list[dict],
        account_balance: Decimal,
        peak_balance: Decimal = Decimal("0"),
        signals_by_tf: dict[str, int] | None = None,
    ) -> dict[str, Decimal]:
        """Calculate position sizes for a batch of ready signals using SignalBudgetAllocator.

        Distributes available balance across all ready signals, weighted by PF.
        This replaces per-symbol sizing with cross-symbol budget allocation.

        Args:
            ready_signals: List of dicts with keys:
                - key: unique signal identifier
                - symbol: trading symbol
                - profit_factor: Phase 5 PF
            account_balance: Current account equity in USD
            signals_by_tf: Pre-built dict mapping timeframe to signal count.
                If None, falls back to building from _approved_models.

        Returns:
            Dict mapping signal key to Decimal lot size.
        """
        max_concurrent = self._get_config_value('max_concurrent_positions', 12)
        lookahead = self._get_config_value('lookahead_candles', 8)
        hit_rate = self._get_config_value('signal_hit_rate', 0.04)
        max_lot = float(self._get_config_value('max_position_size', 1.0))

        allocator = SignalBudgetAllocator(
            equity=float(account_balance),
            peak_balance=float(peak_balance),
            max_concurrent=max_concurrent,
            lookahead_candles=lookahead,
            signal_hit_rate=hit_rate,
            max_lot=max_lot,
            live_tracker=self.live_tracker,
        )

        # Build open_positions list from position manager
        open_positions = []
        for pos_id, pos in self.positions._positions.items():
            open_positions.append({
                "symbol": pos.symbol,
                "size": float(pos.size),
            })

        # Use caller-provided signals_by_tf, or build from approved models
        if signals_by_tf is None:
            signals_by_tf = {}
            for model in self._approved_models:
                tf = model.get("timeframe", "H4")
                signals_by_tf[tf] = signals_by_tf.get(tf, 0) + 1

        raw_sizes = allocator.calculate_batch_sizes(
            ready_signals=ready_signals,
            open_positions=open_positions,
            configured_signals_by_timeframe=signals_by_tf,
        )

        return {key: Decimal(str(size)) for key, size in raw_sizes.items()}

    def get_allocation_summary(self, account_balance: Decimal) -> dict:
        """Get allocation summary for visualization (Issue #630).

        Returns allocation details for all symbols based on current balance
        and open positions.

        Args:
            account_balance: Current account balance in USD

        Returns:
            Dict with allocation summary including:
            - total_balance
            - available_balance
            - tradeable_symbols
            - diversification_score
            - per-symbol allocations
        """
        max_concurrent = self._get_config_value('max_concurrent_positions', 12)

        allocator = BalanceAllocator(
            total_balance=float(account_balance),
            max_concurrent_positions=max_concurrent
        )

        # Get active positions
        active_positions = {}
        for pos_id, pos in self.positions._positions.items():
            margin_estimate = float(pos.size) * 3500
            active_positions[pos_id] = {
                "symbol": pos.symbol,
                "direction": pos.direction.value,
                "size": pos.size,
                "margin_used": margin_estimate
            }

        return allocator.get_allocation_summary(active_positions)

    def reset_daily_counters(self) -> None:
        """Reset daily trade count and P&L.

        Should be called at market open each day.
        """
        self.daily_trades = 0
        self.daily_pnl_pips = 0

    def record_trade(self, pnl_pips: Decimal) -> None:
        """Record a completed trade and update daily counters.

        Args:
            pnl_pips: P&L of the trade in pips
        """
        self.daily_trades += 1
        self.daily_pnl_pips += float(pnl_pips)

    def is_market_open(self) -> bool:
        """Check if FX market is open.

        FX market hours (UTC):
        - Closes Friday at 22:00 UTC
        - Reopens Sunday at 22:00 UTC
        - Open Monday-Friday 22:00 UTC (previous day) to 22:00 UTC

        Returns:
            True if market is open, False if closed
        """
        now = datetime.now(timezone.utc)
        weekday = now.weekday()
        hour = now.hour

        # Saturday is always closed (weekday=5)
        if weekday == 5:
            return False

        # Sunday: market opens at 22:00 UTC (weekday=6)
        if weekday == 6:
            return hour >= 22

        # Friday: market closes at 22:00 UTC (weekday=4)
        if weekday == 4:
            return hour < 22

        # Monday-Thursday: market is open all day (weekday 0-3)
        return True

    def is_on_cooldown(self, symbol: str) -> bool:
        """Check if symbol is on cooldown after recent rejection.

        Prevents spam signals from the same symbol after a risk rejection.

        Args:
            symbol: Trading symbol to check

        Returns:
            True if symbol is still on cooldown, False if can retry
        """
        if symbol not in self.rejection_times:
            return False
        elapsed = datetime.now(timezone.utc) - self.rejection_times[symbol]
        return elapsed < timedelta(seconds=self._get_config_value('rejection_cooldown_seconds', DEFAULT_REJECTION_COOLDOWN_SECONDS))

    def record_rejection(self, symbol: str) -> None:
        """Record rejection timestamp for cooldown tracking.

        Args:
            symbol: Trading symbol that was rejected
        """
        self.rejection_times[symbol] = datetime.now(timezone.utc)

    async def is_market_open_mt5(self, symbol: str = "EURUSD") -> tuple[bool, str]:
        """Check if market is open using calendar-based check.

        TFG: No MT5 gateway available, uses calendar-based weekend detection.

        Args:
            symbol: Trading symbol to check (default: "EURUSD")

        Returns:
            Tuple of (is_open: bool, reason: str)
        """
        if self.is_market_open():
            return (True, "Market open (calendar-based check)")
        return (False, "Market closed (weekend)")
