"""
Order Block Backtest Engine.

Issue #416: Rule-based OB backtest engine for simulating trades based on OB touch events.

Trade Strategy (v1):
- Entry: First touch at ob_mid price (midpoint of OB zone)
- SL: Opposite side of OB zone + 1 pip buffer
- TP: Based on R multiples (1R, 2R, 3R where R = risk = SL distance)

State Transitions:
- Entry on FIRST_TOUCH event
- Exit when price hits TP (WIN) or SL (LOSS)
- Trade remains OPEN if neither is hit

Pip Values:
- JPY pairs: 0.01 per pip
- All others: 0.0001 per pip
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Tuple

# Import pip value function from ob_lifecycle
from data_syncer.ob_lifecycle import get_pip_value

# Public API
__all__ = [
    "TradeSetup",
    "TradeResult",
    "BacktestMetrics",
    "calculate_entry_price",
    "calculate_sl_price",
    "calculate_tp_price",
    "determine_trade_outcome",
    "calculate_pnl_pips",
    "aggregate_metrics",
    "execute_ob_trade",
]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TradeSetup:
    """
    Trade setup parameters for OB-based entry.

    Attributes:
        order_block_id: Database ID of the order block
        symbol: Trading pair symbol (e.g., 'EURUSD')
        timeframe: Candle timeframe (e.g., 'H1')
        direction: Trade direction ('BULLISH' for long, 'BEARISH' for short)
        entry_price: Entry price at ob_mid
        sl_price: Stop loss price (opposite side of OB + buffer)
        tp_price: Take profit price based on R multiple
        tp_multiplier: R multiple for TP (1.0, 2.0, 3.0)
        entry_ts: Timestamp of entry (touch event timestamp)
    """

    order_block_id: int
    symbol: str
    timeframe: str
    direction: str
    entry_price: float
    sl_price: float
    tp_price: float
    tp_multiplier: float
    entry_ts: int


@dataclass
class TradeResult:
    """
    Result of a completed or open trade.

    Attributes:
        outcome: Trade outcome ('WIN', 'LOSS', 'OPEN')
        pnl_pips: Profit/loss in pips (0 for OPEN trades)
        exit_ts: Timestamp of exit (None for OPEN trades)
        exit_price: Exit price (None for OPEN trades)
    """

    outcome: str
    pnl_pips: float
    exit_ts: Optional[int] = None
    exit_price: Optional[float] = None


@dataclass
class BacktestMetrics:
    """
    Aggregated backtest metrics for a group of trades.

    Attributes:
        total_trades: Total number of closed trades (excludes OPEN)
        wins: Number of winning trades
        losses: Number of losing trades
        win_rate: Percentage of winning trades (0.0-1.0)
        profit_factor: Gross profit / gross loss
        expectancy_pips: Average profit per trade in pips
    """

    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    expectancy_pips: float


@dataclass
class FullTradeResult:
    """
    Complete trade result including setup and result.

    Combines TradeSetup parameters with TradeResult for CSV output.
    """

    order_block_id: int
    symbol: str
    timeframe: str
    ob_type: str  # 'BULLISH' or 'BEARISH'
    entry_ts: int
    entry_price: float
    sl_price: float
    tp_price: float
    tp_multiplier: float
    outcome: str
    pnl_pips: float
    exit_ts: Optional[int] = None


# =============================================================================
# Protocol for Candle-like objects
# =============================================================================


class CandleLike(Protocol):
    """Protocol for objects that have OHLC data and timestamp."""

    ts: int
    open: float
    high: float
    low: float
    close: float


class OrderBlockLike(Protocol):
    """Protocol for objects that represent an order block."""

    id: int
    symbol: str
    timeframe: str
    direction: str
    ob_low: float
    ob_high: float
    ts: int


# =============================================================================
# Core Calculation Functions
# =============================================================================


def calculate_entry_price(ob_low: float, ob_high: float) -> float:
    """
    Calculate entry price as midpoint of OB zone (ob_mid).

    Entry at ob_mid provides better risk:reward compared to zone edge entry.

    Args:
        ob_low: Lower boundary of the OB zone
        ob_high: Upper boundary of the OB zone

    Returns:
        Entry price (midpoint of the zone)

    Examples:
        >>> calculate_entry_price(1.0800, 1.0850)
        1.0825
    """
    return (ob_low + ob_high) / 2


def calculate_sl_price(
    direction: str,
    ob_low: float,
    ob_high: float,
    symbol: str,
    buffer_pips: float = 1.0,
) -> float:
    """
    Calculate stop loss price based on OB direction with pip buffer.

    For bullish OB (long trade): SL below ob_low - buffer
    For bearish OB (short trade): SL above ob_high + buffer

    Args:
        direction: Trade direction ('BULLISH' or 'BEARISH')
        ob_low: Lower boundary of the OB zone
        ob_high: Upper boundary of the OB zone
        symbol: Trading pair symbol for pip value calculation
        buffer_pips: Number of pips for the buffer (default 1.0)

    Returns:
        Stop loss price

    Examples:
        >>> calculate_sl_price('BULLISH', 1.0800, 1.0850, 'EURUSD', 1.0)
        1.0799
        >>> calculate_sl_price('BEARISH', 1.1000, 1.1050, 'EURUSD', 1.0)
        1.1051
    """
    pip_value = get_pip_value(symbol)
    buffer = buffer_pips * pip_value

    if direction == "BULLISH":
        return ob_low - buffer
    else:  # BEARISH
        return ob_high + buffer


def calculate_tp_price(
    direction: str,
    entry_price: float,
    sl_price: float,
    tp_multiplier: float,
) -> float:
    """
    Calculate take profit price based on R multiple.

    R = risk = distance from entry to SL
    TP = entry +/- (R * multiplier) depending on direction

    Args:
        direction: Trade direction ('BULLISH' or 'BEARISH')
        entry_price: Entry price at ob_mid
        sl_price: Stop loss price
        tp_multiplier: R multiple (1.0, 2.0, 3.0)

    Returns:
        Take profit price

    Examples:
        >>> calculate_tp_price('BULLISH', 1.0825, 1.0799, 1.0)
        1.0851  # Risk = 0.0026, TP = 1.0825 + 0.0026
        >>> calculate_tp_price('BEARISH', 1.1025, 1.1051, 2.0)
        1.0973  # Risk = 0.0026, TP = 1.1025 - (0.0026 * 2)
    """
    if direction == "BULLISH":
        risk = entry_price - sl_price
        return entry_price + (risk * tp_multiplier)
    else:  # BEARISH
        risk = sl_price - entry_price
        return entry_price - (risk * tp_multiplier)


def determine_trade_outcome(
    direction: str,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    candles: List[Any],
) -> Tuple[str, Optional[int]]:
    """
    Determine if trade hit TP or SL first by scanning candles.

    Scans candles chronologically and checks if price touched TP or SL.
    For bullish trades: TP hit if high >= tp_price, SL hit if low <= sl_price
    For bearish trades: TP hit if low <= tp_price, SL hit if high >= sl_price

    When both TP and SL are hit on the same candle:
    - Use open price direction to determine which was likely hit first
    - If open is favorable (closer to TP direction), assume TP hit first
    - Otherwise, assume SL hit first

    Args:
        direction: Trade direction ('BULLISH' or 'BEARISH')
        entry_price: Entry price at ob_mid
        sl_price: Stop loss price
        tp_price: Take profit price
        candles: List of subsequent candles to scan

    Returns:
        Tuple of (outcome, exit_ts) where:
        - outcome: 'WIN' if TP hit first, 'LOSS' if SL hit first, 'OPEN' if neither
        - exit_ts: Timestamp of exit candle, or None if OPEN
    """
    if not candles:
        return "OPEN", None

    for candle in candles:
        if direction == "BULLISH":
            tp_hit = candle.high >= tp_price
            sl_hit = candle.low <= sl_price

            if tp_hit and sl_hit:
                # Both hit same candle - check open direction
                # If open > entry, price was moving up, likely hit TP first
                if candle.open > entry_price:
                    return "WIN", candle.ts
                else:
                    return "LOSS", candle.ts
            elif tp_hit:
                return "WIN", candle.ts
            elif sl_hit:
                return "LOSS", candle.ts
        else:  # BEARISH
            tp_hit = candle.low <= tp_price
            sl_hit = candle.high >= sl_price

            if tp_hit and sl_hit:
                # Both hit same candle - check open direction
                # If open < entry, price was moving down, likely hit TP first
                if candle.open < entry_price:
                    return "WIN", candle.ts
                else:
                    return "LOSS", candle.ts
            elif tp_hit:
                return "WIN", candle.ts
            elif sl_hit:
                return "LOSS", candle.ts

    return "OPEN", None


def calculate_pnl_pips(
    direction: str,
    entry_price: float,
    exit_price: float,
    symbol: str,
) -> float:
    """
    Calculate profit/loss in pips.

    For bullish (long): pnl = (exit - entry) / pip_value
    For bearish (short): pnl = (entry - exit) / pip_value

    Args:
        direction: Trade direction ('BULLISH' or 'BEARISH')
        entry_price: Entry price
        exit_price: Exit price (TP or SL)
        symbol: Trading pair symbol for pip value calculation

    Returns:
        PnL in pips (positive for profit, negative for loss)

    Examples:
        >>> calculate_pnl_pips('BULLISH', 1.0825, 1.0851, 'EURUSD')
        26.0  # (1.0851 - 1.0825) / 0.0001
        >>> calculate_pnl_pips('BEARISH', 1.1025, 1.1051, 'EURUSD')
        -26.0  # (1.1025 - 1.1051) / 0.0001
    """
    pip_value = get_pip_value(symbol)

    if direction == "BULLISH":
        return (exit_price - entry_price) / pip_value
    else:  # BEARISH
        return (entry_price - exit_price) / pip_value


# =============================================================================
# Metrics Aggregation
# =============================================================================


def aggregate_metrics(results: List[TradeResult]) -> BacktestMetrics:
    """
    Aggregate trade results into backtest metrics.

    Calculates win rate, profit factor, and expectancy from a list of trades.
    Only includes closed trades (excludes OPEN).

    Args:
        results: List of TradeResult objects

    Returns:
        BacktestMetrics with aggregated statistics

    Examples:
        >>> results = [
        ...     TradeResult(outcome='WIN', pnl_pips=26.0),
        ...     TradeResult(outcome='LOSS', pnl_pips=-26.0),
        ... ]
        >>> metrics = aggregate_metrics(results)
        >>> metrics.win_rate
        0.5
    """
    # Filter out OPEN trades
    closed_trades = [r for r in results if r.outcome != "OPEN"]

    if not closed_trades:
        return BacktestMetrics(
            total_trades=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy_pips=0.0,
        )

    wins = [r for r in closed_trades if r.outcome == "WIN"]
    losses = [r for r in closed_trades if r.outcome == "LOSS"]

    total_trades = len(closed_trades)
    win_count = len(wins)
    loss_count = len(losses)

    # Win rate
    win_rate = win_count / total_trades if total_trades > 0 else 0.0

    # Profit factor (gross profit / gross loss)
    gross_profit = sum(r.pnl_pips for r in wins) if wins else 0.0
    gross_loss = abs(sum(r.pnl_pips for r in losses)) if losses else 0.0

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    # Expectancy (average pnl per trade)
    total_pnl = sum(r.pnl_pips for r in closed_trades)
    expectancy_pips = total_pnl / total_trades if total_trades > 0 else 0.0

    return BacktestMetrics(
        total_trades=total_trades,
        wins=win_count,
        losses=loss_count,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy_pips=expectancy_pips,
    )


# =============================================================================
# Full Trade Execution
# =============================================================================


def execute_ob_trade(
    ob: Any,
    touch_candle: Any,
    subsequent_candles: List[Any],
    tp_multiplier: float,
    buffer_pips: float = 1.0,
) -> FullTradeResult:
    """
    Execute a complete OB trade from setup to outcome.

    This is the main entry point for simulating a single OB trade:
    1. Calculate entry price (ob_mid)
    2. Calculate SL price (opposite side + buffer)
    3. Calculate TP price (based on R multiple)
    4. Scan subsequent candles for outcome
    5. Calculate PnL if trade closed

    Args:
        ob: Order block object with id, symbol, timeframe, direction, ob_low, ob_high, ts
        touch_candle: Candle that triggered the touch event
        subsequent_candles: Candles after touch for outcome determination
        tp_multiplier: R multiple for TP (1.0, 2.0, 3.0)
        buffer_pips: Number of pips for SL buffer (default 1.0)

    Returns:
        FullTradeResult with complete trade information

    Examples:
        >>> result = execute_ob_trade(ob, touch, candles, tp_multiplier=2.0)
        >>> print(result.outcome)  # 'WIN', 'LOSS', or 'OPEN'
        >>> print(result.pnl_pips)  # e.g., 52.0 for 2R win
    """
    # Calculate trade parameters
    entry_price = calculate_entry_price(ob.ob_low, ob.ob_high)
    sl_price = calculate_sl_price(
        direction=ob.direction,
        ob_low=ob.ob_low,
        ob_high=ob.ob_high,
        symbol=ob.symbol,
        buffer_pips=buffer_pips,
    )
    tp_price = calculate_tp_price(
        direction=ob.direction,
        entry_price=entry_price,
        sl_price=sl_price,
        tp_multiplier=tp_multiplier,
    )

    # Determine outcome
    outcome, exit_ts = determine_trade_outcome(
        direction=ob.direction,
        entry_price=entry_price,
        sl_price=sl_price,
        tp_price=tp_price,
        candles=subsequent_candles,
    )

    # Calculate PnL
    if outcome == "WIN":
        exit_price = tp_price
        pnl_pips = calculate_pnl_pips(ob.direction, entry_price, exit_price, ob.symbol)
    elif outcome == "LOSS":
        exit_price = sl_price
        pnl_pips = calculate_pnl_pips(ob.direction, entry_price, exit_price, ob.symbol)
    else:  # OPEN
        pnl_pips = 0.0

    return FullTradeResult(
        order_block_id=ob.id,
        symbol=ob.symbol,
        timeframe=ob.timeframe,
        ob_type=ob.direction,
        entry_ts=touch_candle.ts,
        entry_price=entry_price,
        sl_price=sl_price,
        tp_price=tp_price,
        tp_multiplier=tp_multiplier,
        outcome=outcome,
        pnl_pips=pnl_pips,
        exit_ts=exit_ts,
    )
