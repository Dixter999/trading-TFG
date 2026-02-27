"""
Balance Allocator for Dynamic Position Sizing (Issue #630).

This module provides intelligent balance allocation that dynamically calculates
position sizes based on:
- Current available balance
- Per-symbol margin requirements (contract_value / leverage)
- Maximum concurrent position limits
- Diversification preferences

The BalanceAllocator replaces the hardcoded PF-based tiers (0.1L, 0.2L, 0.3L, 0.5L)
with dynamic sizing that adapts to changing account balance.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from src.paper_trading.live_performance_tracker import LivePerformanceTracker

logger = logging.getLogger(__name__)

# Standard lot size in forex
STANDARD_LOT_SIZE: int = 100_000  # 1 standard lot = 100,000 units

# Symbol configurations with leverage and pip values
SYMBOL_CONFIGS: dict[str, dict] = {
    # Major pairs - 30x leverage
    "EURUSD": {"leverage": 30, "pip_value": 10.0, "pip_size": 0.0001, "base_currency": "EUR"},
    "GBPUSD": {"leverage": 30, "pip_value": 10.0, "pip_size": 0.0001, "base_currency": "GBP"},
    "USDJPY": {"leverage": 30, "pip_value": 6.67, "pip_size": 0.01, "base_currency": "USD"},
    "USDCHF": {"leverage": 30, "pip_value": 10.0, "pip_size": 0.0001, "base_currency": "USD"},
    # Minor pairs - 20x leverage
    "EURJPY": {"leverage": 20, "pip_value": 6.67, "pip_size": 0.01, "base_currency": "EUR"},
    "EURGBP": {"leverage": 20, "pip_value": 10.0, "pip_size": 0.0001, "base_currency": "EUR"},
    "EURCAD": {"leverage": 20, "pip_value": 7.50, "pip_size": 0.0001, "base_currency": "EUR"},
    "USDCAD": {"leverage": 20, "pip_value": 7.50, "pip_size": 0.0001, "base_currency": "USD"},
    "GBPJPY": {"leverage": 20, "pip_value": 6.67, "pip_size": 0.01, "base_currency": "GBP"},
    "AUDUSD": {"leverage": 20, "pip_value": 10.0, "pip_size": 0.0001, "base_currency": "AUD"},
    "NZDUSD": {"leverage": 20, "pip_value": 10.0, "pip_size": 0.0001, "base_currency": "NZD"},
    # Cross pairs - 20x leverage
    "AUDCAD": {"leverage": 20, "pip_value": 7.50, "pip_size": 0.0001, "base_currency": "AUD"},
    "AUDNZD": {"leverage": 20, "pip_value": 6.00, "pip_size": 0.0001, "base_currency": "AUD"},
    "CADJPY": {"leverage": 20, "pip_value": 6.67, "pip_size": 0.01, "base_currency": "CAD"},
    "NZDJPY": {"leverage": 20, "pip_value": 6.67, "pip_size": 0.01, "base_currency": "NZD"},
}

# Default leverage for unknown symbols
DEFAULT_LEVERAGE: int = 10

# Reference leverage for normalization (average of 30x majors + 20x minors)
# Ensures PF drives lot sizing, not symbol leverage differences
REFERENCE_LEVERAGE: int = 25


def calculate_margin_per_lot(
    symbol: str,
    price: float = 1.0,
    base_rate: float = 1.0
) -> float:
    """
    Calculate margin required for 1 standard lot.

    For pairs where USD is the quote currency (EURUSD, GBPUSD):
        margin = (100,000 * base_rate) / leverage

    For pairs where USD is the base currency (USDJPY, USDCHF):
        margin = 100,000 / leverage  (price doesn't affect much)

    For cross pairs (EURJPY, GBPJPY):
        margin = (100,000 * base_to_usd_rate) / leverage

    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        price: Current market price (used for non-USD pairs)
        base_rate: Base currency to USD rate (for cross pairs like EURJPY)

    Returns:
        Margin required in USD for 1 standard lot
    """
    config = SYMBOL_CONFIGS.get(symbol.upper(), {"leverage": DEFAULT_LEVERAGE})
    leverage = config["leverage"]

    # Determine contract value in USD
    base_currency = config.get("base_currency", "")

    if base_currency == "USD":
        # USD-based pairs: contract value is simply 100,000 USD
        contract_value = STANDARD_LOT_SIZE
    elif symbol.upper().endswith("USD"):
        # Pairs where USD is quote (EURUSD, GBPUSD): use base_rate or price
        # Price represents base/quote, so for EURUSD, price=1.10 means 1 EUR = 1.10 USD
        effective_rate = base_rate if base_rate != 1.0 else price
        contract_value = STANDARD_LOT_SIZE * effective_rate
    else:
        # Cross pairs (EURJPY, etc.): need to convert to USD
        # Use base_rate as the base currency to USD rate
        contract_value = STANDARD_LOT_SIZE * base_rate

    margin = contract_value / leverage
    return round(margin, 2)


async def persist_config_params(db_pool) -> None:
    """UPSERT margin/leverage/pip configs for all symbols to trading_config_params.

    Called at pod startup so TypeScript/frontend can read from DB
    instead of maintaining hardcoded copies.
    """
    upsert = """
        INSERT INTO trading_config_params
            (param_key, param_value, category, symbol, metadata, updated_at)
        VALUES ($1, $2, $3, $4, $5, NOW())
        ON CONFLICT (param_key) DO UPDATE SET
            param_value=$2, metadata=$5, updated_at=NOW()
    """
    import json
    count = 0
    try:
        async with db_pool.acquire() as conn:
            for symbol, config in SYMBOL_CONFIGS.items():
                leverage = config["leverage"]
                pip_value = config["pip_value"]
                pip_size = config["pip_size"]
                base_currency = config.get("base_currency", "")

                # Calculate margin_per_lot using same logic as runtime
                if base_currency == "USD":
                    margin_per_lot = STANDARD_LOT_SIZE / leverage
                elif symbol.upper().endswith("USD"):
                    margin_per_lot = (STANDARD_LOT_SIZE * 1.10) / leverage
                else:
                    margin_per_lot = (STANDARD_LOT_SIZE * 1.10) / leverage

                margin_per_lot = round(margin_per_lot, 2)

                metadata = json.dumps({
                    "leverage": leverage,
                    "pip_value": pip_value,
                    "pip_size": pip_size,
                    "base_currency": base_currency,
                })

                await conn.execute(
                    upsert,
                    f"margin:{symbol}", margin_per_lot, "margin", symbol, metadata,
                )
                count += 1

            # Also persist PF multiplier tiers
            pf_tiers = [
                ("pf_multiplier:tier_below_1.2", 0.50, {"pf_range": "<1.2"}),
                ("pf_multiplier:tier_1.2_2.0", 0.50, {"pf_range": "1.2-2.0", "formula": "0.5 + (pf-1.2)*0.3125"}),
                ("pf_multiplier:tier_2.0_3.0", 0.75, {"pf_range": "2.0-3.0", "formula": "0.75 + (pf-2.0)*0.25"}),
                ("pf_multiplier:tier_3.0_plus", 1.00, {"pf_range": ">=3.0"}),
            ]
            for key, value, meta in pf_tiers:
                await conn.execute(
                    upsert,
                    key, value, "pf_multiplier", None, json.dumps(meta),
                )
                count += 1

            # Diversification rules
            div_rules = [
                ("diversification:max_single_symbol", 0.25),
                ("diversification:max_single_direction", 0.60),
                ("diversification:margin_reserve_ratio", 0.10),
                ("diversification:min_position_size", 0.01),
            ]
            for key, value in div_rules:
                await conn.execute(
                    upsert,
                    key, value, "diversification", None, None,
                )
                count += 1

        logger.info(f"Persisted {count} config params to trading_config_params table")
    except Exception as e:
        logger.error(f"Failed to persist config params: {e}")


class DiversificationRules:
    """
    Rules to ensure balanced portfolio allocation.

    These rules prevent over-concentration in any single symbol or direction,
    promoting diversification across the portfolio.
    """

    # Max 25% of balance per symbol
    MAX_SINGLE_SYMBOL_ALLOCATION: ClassVar[float] = 0.25

    # Max 60% in LONG or SHORT direction
    MAX_SINGLE_DIRECTION_ALLOCATION: ClassVar[float] = 0.60

    # Minimum 0.01 lots (micro lot)
    MIN_POSITION_SIZE: ClassVar[float] = 0.01

    # Prefer 1 position per symbol for diversity
    PREFERRED_POSITIONS_PER_SYMBOL: ClassVar[int] = 1

    # Reserve 10% of balance for margin buffer
    MARGIN_RESERVE_RATIO: ClassVar[float] = 0.10

    @staticmethod
    def calculate_max_lots_for_symbol(
        symbol: str,
        total_balance: float,
        margin_per_lot: float,
        existing_symbol_positions: int
    ) -> float:
        """
        Calculate maximum lots allowed for a symbol considering:
        1. Maximum allocation percentage
        2. Existing positions in same symbol
        3. Minimum viable position size

        Args:
            symbol: Trading symbol
            total_balance: Total account balance in USD
            margin_per_lot: Margin required for 1 lot
            existing_symbol_positions: Number of existing positions in this symbol

        Returns:
            Maximum lots allowed (0.0 if can't afford minimum)
        """
        if total_balance <= 0 or margin_per_lot <= 0:
            return 0.0

        # Max allocation for this symbol
        max_allocation = total_balance * DiversificationRules.MAX_SINGLE_SYMBOL_ALLOCATION

        # Reduce allocation if already have position in this symbol
        if existing_symbol_positions > 0:
            # Halve allocation for each additional position
            reduction_factor = 0.5 ** existing_symbol_positions
            max_allocation *= reduction_factor

        # Calculate max lots from allocation
        max_lots = max_allocation / margin_per_lot

        # Round down to 2 decimal places (0.01 lot increments)
        max_lots = float(int(max_lots * 100) / 100)

        # Ensure minimum viable size
        if max_lots < DiversificationRules.MIN_POSITION_SIZE:
            return 0.0

        return max_lots


@dataclass
class SymbolAllocation:
    """Allocation details for a single symbol."""

    max_lots: float
    margin_per_lot: float
    available_balance: float
    leverage: int


class BalanceAllocator:
    """
    Intelligent balance allocation with diversification preference.

    Key principle: Allocate balance across symbols proportionally,
    preferring more smaller positions over fewer larger ones.

    Attributes:
        total_balance: Total account balance in USD
        max_concurrent: Maximum concurrent positions allowed
        symbol_configs: Symbol configuration data
    """

    def __init__(
        self,
        total_balance: float,
        max_concurrent_positions: int = 12
    ) -> None:
        """
        Initialize BalanceAllocator.

        Args:
            total_balance: Total account balance in USD
            max_concurrent_positions: Maximum number of concurrent positions (default: 12)
        """
        self.total_balance = total_balance
        self.max_concurrent = max_concurrent_positions
        self.symbol_configs = SYMBOL_CONFIGS

    def calculate_allocation(
        self,
        active_positions: dict
    ) -> dict[str, dict]:
        """
        Calculate how much balance is available for each symbol.

        Args:
            active_positions: Dict of active positions with margin_used info
                Format: {position_id: {"margin_used": float, ...}}

        Returns:
            Dict mapping symbol to allocation details:
            {symbol: {max_lots: float, margin_per_lot: float, available_balance: float}}
        """
        # Calculate margin already used
        margin_used = sum(
            pos.get("margin_used", 0.0)
            for pos in active_positions.values()
        )

        # Available balance after margin reserve and used margin
        available = self.total_balance - margin_used
        available *= (1 - DiversificationRules.MARGIN_RESERVE_RATIO)
        available = max(0, available)

        # Count existing positions per symbol
        positions_per_symbol: dict[str, int] = {}
        for pos_id, pos in active_positions.items():
            symbol = pos.get("symbol", pos_id.split("_")[0] if "_" in pos_id else "UNKNOWN")
            positions_per_symbol[symbol] = positions_per_symbol.get(symbol, 0) + 1

        allocation: dict[str, dict] = {}

        for symbol, config in self.symbol_configs.items():
            # Calculate margin per lot for this symbol
            # Use default price estimates for now
            if symbol.endswith("USD"):
                # Major pairs: estimate price around 1.0-1.5
                margin_per_lot = calculate_margin_per_lot(symbol, price=1.10)
            elif "JPY" in symbol:
                # JPY pairs: use estimated base rate
                margin_per_lot = calculate_margin_per_lot(symbol, base_rate=1.10)
            else:
                margin_per_lot = calculate_margin_per_lot(symbol, price=1.0)

            existing_count = positions_per_symbol.get(symbol, 0)

            max_lots = DiversificationRules.calculate_max_lots_for_symbol(
                symbol=symbol,
                total_balance=available,
                margin_per_lot=margin_per_lot,
                existing_symbol_positions=existing_count
            )

            allocation[symbol] = {
                "max_lots": max_lots,
                "margin_per_lot": margin_per_lot,
                "available_balance": available,
                "leverage": config["leverage"]
            }

        return allocation

    def get_position_size(
        self,
        symbol: str,
        direction: str,
        profit_factor: float,
        active_positions: dict,
        symbol_price: float = 1.10
    ) -> float:
        """
        Get dynamic lot size based on:
        1. Available balance after existing positions
        2. Symbol's margin requirement
        3. Diversification rules (don't over-allocate to single symbol)
        4. Performance factor (PF bonus multiplier, not fixed tiers)

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            direction: Trade direction ("LONG" or "SHORT")
            profit_factor: Phase 5 profit factor for the signal
            active_positions: Dict of current open positions
            symbol_price: Current symbol price (default: 1.10)

        Returns:
            Lot size (0.01 to max allowed), or 0.0 if insufficient margin
        """
        # Handle edge cases
        if self.total_balance <= 0:
            return 0.0

        # Calculate current allocation
        allocation = self.calculate_allocation(active_positions)

        # Get symbol allocation (or calculate for unknown symbols)
        if symbol.upper() in allocation:
            symbol_alloc = allocation[symbol.upper()]
            max_lots = symbol_alloc["max_lots"]
            margin_per_lot = symbol_alloc["margin_per_lot"]
        else:
            # Unknown symbol - use conservative defaults
            margin_per_lot = calculate_margin_per_lot(symbol, price=symbol_price)
            existing_count = sum(
                1 for pos in active_positions.values()
                if pos.get("symbol", "").upper() == symbol.upper()
            )
            available = self.total_balance * (1 - DiversificationRules.MARGIN_RESERVE_RATIO)
            max_lots = DiversificationRules.calculate_max_lots_for_symbol(
                symbol=symbol,
                total_balance=available,
                margin_per_lot=margin_per_lot,
                existing_symbol_positions=existing_count
            )

        if max_lots <= 0:
            logger.info(
                f"Position size for {symbol}: 0.0 lots (insufficient margin, "
                f"need ${margin_per_lot:.2f} per lot)"
            )
            return 0.0

        # Apply PF-based scaling (not fixed tiers, but gradual scaling)
        # PF 1.0 = 50% of max, PF 2.0 = 75% of max, PF 3.0+ = 100% of max
        if profit_factor >= 3.0:
            pf_multiplier = 1.0
        elif profit_factor >= 2.0:
            # Scale from 0.75 to 1.0 between PF 2.0 and 3.0
            pf_multiplier = 0.75 + (profit_factor - 2.0) * 0.25
        elif profit_factor >= 1.2:
            # Scale from 0.5 to 0.75 between PF 1.2 and 2.0
            pf_multiplier = 0.5 + (profit_factor - 1.2) * 0.3125
        else:
            # Below 1.2 PF: 50% of max
            pf_multiplier = 0.5

        # Normalize: convert symbol-specific max_lots to reference-leverage base
        # This ensures PF drives lot sizing, not symbol leverage
        reference_margin = STANDARD_LOT_SIZE / REFERENCE_LEVERAGE
        dollar_allocation = max_lots * margin_per_lot
        max_lots_normalized = dollar_allocation / reference_margin

        # Apply PF multiplier on normalized (leverage-neutral) base
        position_size = max_lots_normalized * pf_multiplier

        # Safety: actual lots cannot exceed what the symbol's real margin allows
        if position_size > max_lots:
            position_size = max_lots

        # Round down to 0.01 lot increments
        position_size = float(int(position_size * 100) / 100)

        # Ensure minimum lot size if position is viable
        if position_size > 0 and position_size < DiversificationRules.MIN_POSITION_SIZE:
            position_size = DiversificationRules.MIN_POSITION_SIZE

        logger.info(
            f"Position size for {symbol}: {position_size:.2f} lots "
            f"(max={max_lots:.2f}, normalized={max_lots_normalized:.2f}, "
            f"PF={profit_factor:.2f}, multiplier={pf_multiplier:.2f})"
        )

        return position_size

    def get_diversification_score(self, active_positions: dict) -> float:
        """
        Calculate diversification score (0-100%).

        Score is based on:
        - Number of different symbols (more = better)
        - Balance across symbols (even = better)
        - Direction balance (mixed = better)

        Args:
            active_positions: Dict of current open positions

        Returns:
            Diversification score as percentage (0-100)
        """
        if not active_positions:
            return 100.0  # Empty portfolio is "perfectly diversified"

        # Count unique symbols
        symbols = set()
        long_count = 0
        short_count = 0
        total_margin = 0.0
        margin_per_symbol: dict[str, float] = {}

        for pos_id, pos in active_positions.items():
            symbol = pos.get("symbol", pos_id.split("_")[0])
            direction = pos.get("direction", "LONG")
            margin = pos.get("margin_used", 0.0)

            symbols.add(symbol)
            if direction.upper() == "LONG":
                long_count += 1
            else:
                short_count += 1
            total_margin += margin
            margin_per_symbol[symbol] = margin_per_symbol.get(symbol, 0.0) + margin

        # Symbol diversity score (more symbols = higher score)
        max_symbols = min(len(self.symbol_configs), self.max_concurrent)
        symbol_score = min(100, (len(symbols) / max_symbols) * 100) if max_symbols > 0 else 0

        # Direction balance score (50/50 = 100%, all one way = 50%)
        total_positions = long_count + short_count
        if total_positions > 0:
            direction_ratio = min(long_count, short_count) / max(long_count, short_count) if max(long_count, short_count) > 0 else 1.0
            direction_score = 50 + (direction_ratio * 50)
        else:
            direction_score = 100

        # Concentration score (even spread = 100%, concentrated = lower)
        if total_margin > 0 and len(symbols) > 1:
            expected_per_symbol = total_margin / len(symbols)
            variance = sum(
                ((m - expected_per_symbol) / expected_per_symbol) ** 2
                for m in margin_per_symbol.values()
            ) / len(symbols)
            concentration_score = max(0, 100 - (variance * 100))
        else:
            concentration_score = 100

        # Weighted average
        final_score = (symbol_score * 0.4) + (direction_score * 0.3) + (concentration_score * 0.3)

        return round(final_score, 1)

    def get_allocation_summary(self, active_positions: dict) -> dict:
        """
        Get a summary of current allocation for visualization.

        Args:
            active_positions: Dict of current open positions

        Returns:
            Dict with summary information for UI display
        """
        allocation = self.calculate_allocation(active_positions)

        # Calculate totals
        margin_used = sum(
            pos.get("margin_used", 0.0)
            for pos in active_positions.values()
        )
        available = self.total_balance - margin_used
        tradeable_symbols = [
            s for s, a in allocation.items()
            if a["max_lots"] >= DiversificationRules.MIN_POSITION_SIZE
        ]

        return {
            "total_balance": self.total_balance,
            "margin_used": margin_used,
            "available_balance": available,
            "margin_reserve": self.total_balance * DiversificationRules.MARGIN_RESERVE_RATIO,
            "tradeable_symbols": len(tradeable_symbols),
            "total_symbols": len(self.symbol_configs),
            "max_concurrent_positions": self.max_concurrent,
            "current_positions": len(active_positions),
            "diversification_score": self.get_diversification_score(active_positions),
            "allocations": allocation,
        }


# Timeframe durations in minutes for lookahead estimation
TIMEFRAME_MINUTES: dict[str, int] = {
    "M30": 30, "H1": 60, "H2": 120, "H3": 180, "H4": 240,
    "H6": 360, "H8": 480, "H12": 720, "D1": 1440,
}

# Margin reserve ratio for signal budget allocator (10% buffer)
MARGIN_RESERVE_RATIO: float = 0.10

# Lot increment and limits
MIN_LOT: float = 0.01
DEFAULT_MAX_LOT: float = 1.0


@dataclass
class SignalBudgetAllocator:
    """Signal-based position sizing allocator.

    Distributes available balance across READY signals, weighted by
    live performance (via LivePerformanceTracker) or equal if no tracker.

    Algorithm:
        1. available = equity - margin_used - 10% reserve
        2. remaining_slots = max_concurrent - open_positions
        3. future_estimated = lookahead estimation (signals × hit_rate × candle_closes)
        4. total_slots = min(ready + future, remaining_slots)
        5. budget_per_slot = available / total_slots
        6. candle_budget = ready_count × budget_per_slot
        7. weight-based: lots_i = (w_i / sum_w) × candle_budget / margin_per_lot(symbol_i)
    """

    equity: float
    peak_balance: float = 0.0
    max_concurrent: int = 12
    lookahead_candles: int = 8
    signal_hit_rate: float = 0.04
    max_lot: float = DEFAULT_MAX_LOT
    live_tracker: LivePerformanceTracker | None = None

    def calculate_batch_sizes(
        self,
        ready_signals: list[dict[str, Any]],
        open_positions: list[dict[str, Any]],
        configured_signals_by_timeframe: dict[str, int],
    ) -> dict[str, float]:
        """Calculate lot sizes for a batch of ready signals.

        Args:
            ready_signals: List of dicts with keys:
                - key: unique signal identifier (e.g. "EURUSD:long:SMA_cross:H2")
                - symbol: trading symbol
                - profit_factor: Phase 5 PF (float)
            open_positions: List of dicts with keys:
                - symbol: trading symbol
                - size: lot size (float)
            configured_signals_by_timeframe: Dict mapping timeframe to signal count
                e.g. {"H1": 37, "H2": 25, "H4": 20, "D1": 15}

        Returns:
            Dict mapping signal key to lot size (float).
            Empty dict if no slots available or no ready signals.
        """
        if not ready_signals:
            return {}

        remaining_slots = self.max_concurrent - len(open_positions)
        if remaining_slots <= 0:
            logger.info(
                "SignalBudgetAllocator: No slots available "
                f"({len(open_positions)}/{self.max_concurrent} positions open)"
            )
            return {}

        # Step 1: Calculate available balance
        margin_used = self._estimate_margin_used(open_positions)
        available = (self.equity - margin_used) * (1 - MARGIN_RESERVE_RATIO)
        available = max(0.0, available)

        if available <= 0:
            logger.info("SignalBudgetAllocator: No available balance after margin + reserve")
            return {}

        # Step 2: Estimate future signal count in lookahead window
        future_estimated = self._estimate_future_signals(configured_signals_by_timeframe)

        # Step 3: Calculate total slots and budget
        ready_count = len(ready_signals)
        total_slots = min(ready_count + future_estimated, remaining_slots)
        total_slots = max(total_slots, 1)  # Avoid division by zero

        budget_per_slot = available / total_slots
        candle_budget = ready_count * budget_per_slot

        logger.info(
            f"SignalBudgetAllocator: equity=${self.equity:.0f}, margin_used=${margin_used:.0f}, "
            f"available=${available:.0f}, open={len(open_positions)}, "
            f"remaining_slots={remaining_slots}, ready={ready_count}, "
            f"future_est={future_estimated}, total_slots={total_slots}, "
            f"budget_per_slot=${budget_per_slot:.0f}, candle_budget=${candle_budget:.0f}"
        )

        # Step 4: Distribute by weight (live performance or equal)
        return self._distribute_by_weight(ready_signals, candle_budget)

    def _estimate_margin_used(self, open_positions: list[dict[str, Any]]) -> float:
        """Estimate total margin used by open positions.

        Uses per-symbol margin calculation rather than flat estimate.
        """
        total = 0.0
        for pos in open_positions:
            symbol = pos.get("symbol", "EURUSD")
            size = float(pos.get("size", 0.01))
            margin = calculate_margin_per_lot(symbol, price=1.10, base_rate=1.10)
            total += size * margin
        return total

    def _estimate_future_signals(
        self, configured_signals_by_timeframe: dict[str, int]
    ) -> int:
        """Estimate how many signals might fire in the lookahead window.

        For each timeframe, calculates how many candle closes occur in
        lookahead_candles hours, multiplied by signal count and hit rate.
        """
        lookahead_minutes = self.lookahead_candles * 60
        total = 0.0

        for tf, signal_count in configured_signals_by_timeframe.items():
            tf_minutes = TIMEFRAME_MINUTES.get(tf, 60)
            candle_closes = lookahead_minutes / tf_minutes
            expected = candle_closes * signal_count * self.signal_hit_rate
            total += expected

        return max(0, int(math.ceil(total)))

    def _calculate_drawdown_throttle(self) -> float:
        """Portfolio-level drawdown throttle (Kelly v3 S3.3 Step 6).

        Returns multiplier in [0.0, 1.0] applied to candle budget.
        """
        if self.peak_balance <= 0:
            return 1.0

        cfg_start = 0.03
        cfg_end = 0.10
        if self.live_tracker is not None:
            cfg_start = self.live_tracker.config.dd_throttle_start
            cfg_end = self.live_tracker.config.dd_throttle_end

        current_dd = (self.peak_balance - self.equity) / self.peak_balance
        if current_dd <= 0:
            return 1.0

        if current_dd < cfg_start:
            return 1.0
        elif current_dd < cfg_end:
            return 1.0 - (current_dd - cfg_start) / (cfg_end - cfg_start)
        else:
            return 0.0

    def _distribute_by_weight(
        self,
        ready_signals: list[dict[str, Any]],
        candle_budget: float,
    ) -> dict[str, float]:
        """Distribute candle budget across signals weighted by live performance.

        Uses LivePerformanceTracker weights when available, otherwise equal (1.0).
        Applies portfolio-level drawdown throttle to candle_budget.
        """
        from src.paper_trading.live_performance_tracker import LivePerformanceTracker

        # Apply drawdown throttle
        throttle = self._calculate_drawdown_throttle()
        if throttle <= 0.0:
            logger.warning(
                f"Drawdown throttle=0.0 (close-only mode): "
                f"equity=${self.equity:.0f}, peak=${self.peak_balance:.0f}, "
                f"DD={(self.peak_balance - self.equity) / self.peak_balance:.1%}"
            )
            return {}
        if throttle < 1.0:
            dd_pct = (self.peak_balance - self.equity) / self.peak_balance
            logger.info(
                f"Drawdown throttle={throttle:.3f}: "
                f"equity=${self.equity:.0f}, peak=${self.peak_balance:.0f}, "
                f"DD={dd_pct:.1%}"
            )
            candle_budget *= throttle

        # Build weight map
        weights: dict[str, float] = {}
        for s in ready_signals:
            key = s["key"]
            if self.live_tracker is not None:
                # Build tracker-compatible 3-part key from 4-part signal key
                # signal key: "EURUSD:LONG:MACD_Stoch_long:H1"
                # tracker key: "EURUSD:LONG:MACD_Stoch_long"
                parts = key.split(":")
                if len(parts) >= 3:
                    tracker_key = ":".join(parts[:3])
                else:
                    tracker_key = key
                weights[key] = self.live_tracker.get_weight(tracker_key)
            else:
                weights[key] = 1.0

        weight_sum = sum(weights.values())
        if weight_sum <= 0:
            return {}

        result: dict[str, float] = {}

        for signal in ready_signals:
            key = signal["key"]
            symbol = signal["symbol"]
            w = weights[key]

            # Proportional share of candle budget (in USD)
            # w/weight_sum = relative split between concurrent signals
            # * w = absolute Kelly scaling (low-confidence → smaller lots even solo)
            share = w / weight_sum
            dollar_allocation = share * candle_budget * w

            # Convert to lots using symbol-specific margin
            margin_per_lot = calculate_margin_per_lot(symbol, price=1.10, base_rate=1.10)
            if margin_per_lot <= 0:
                continue

            lots = dollar_allocation / margin_per_lot

            # Enforce min/max lot constraints
            lots = min(lots, self.max_lot)
            # Round down to 0.01 increments
            lots = float(int(lots * 100) / 100)

            if lots < MIN_LOT:
                lots = 0.0  # Below minimum, skip

            result[key] = lots

            logger.info(
                f"  {key}: weight={w:.3f}, share={share:.1%}, "
                f"${dollar_allocation:.0f} / ${margin_per_lot:.0f} margin = "
                f"{lots:.2f} lots"
            )

        return result
