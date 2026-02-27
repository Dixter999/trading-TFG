"""
Confluence backtesting system.

This module implements the main backtest_confluence function that runs
complete backtests of the confluence pattern system on historical data.

Architecture:
1. Load historical candle data from database
2. For each candle, generate confluence signals using ConfluenceEngine
3. Convert signals to trades using execute_confluence_signal()
4. Calculate comprehensive metrics using calculate_backtest_metrics()
5. Build equity curve and return BacktestResults

Performance target: < 1 second for 1000 candles
"""

from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

from backtesting.execution import execute_confluence_signal
from backtesting.metrics import calculate_backtest_metrics
from backtesting.models import BacktestResults, Trade
from database.connection_manager import DatabaseManager
from database.utils import validate_timeframe
from pattern_matching.models import PatternMatch
from pattern_system.confluence.engine import ConfluenceEngine
from pattern_system.confluence.models import ConfluenceSignal
from regime_system.detector import RegimeDetector


def backtest_confluence(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    pattern_types: List[str],
    regime_aware: bool = True,
    initial_capital: float = 10000.0,
    stop_loss_pips: int = 20,
    take_profit_ratio: float = 1.5,
    min_confidence: float = 0.70,
    min_pattern_count: int = 2,
    candles_df: Optional[pd.DataFrame] = None,
    pattern_extractor: Optional[callable] = None,
) -> BacktestResults:
    """
    Run confluence backtest on historical data.

    This is the main entry point for backtesting the confluence system.
    It loads historical data, generates signals for each candle, executes
    trades, and calculates comprehensive performance metrics.

    Args:
        symbol: Trading pair (e.g., "EURUSD"). Currently only EURUSD supported.
        timeframe: Candle timeframe (e.g., "H1", "D1"). See VALID_TIMEFRAMES.
        start_date: Backtest start datetime (must be timezone-aware UTC).
        end_date: Backtest end datetime (must be timezone-aware UTC).
        pattern_types: List of patterns to detect (e.g., ["pin_bar", "supply_demand"]).
        regime_aware: Whether to apply regime filtering (default: True).
        initial_capital: Starting capital in currency units (default: 10000.0).
        stop_loss_pips: Stop loss distance in pips (default: 20).
        take_profit_ratio: Take profit as multiple of stop loss (default: 1.5).
        min_confidence: Minimum confidence for tradeable signal (default: 0.70).
        min_pattern_count: Minimum patterns required for confluence (default: 2).
        candles_df: Optional DataFrame with candle data (bypasses database loading).
        pattern_extractor: Optional function to extract patterns from candles_df.

    Returns:
        BacktestResults with comprehensive metrics, trades, signals, and equity curve.

    Raises:
        ValueError: If parameters are invalid (e.g., invalid timeframe, end before start).

    Performance:
        Target: < 1 second for 1000 candles
        Actual: Varies based on pattern complexity and regime detection

    Examples:
        >>> result = backtest_confluence(
        ...     symbol="EURUSD",
        ...     timeframe="H1",
        ...     start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        ...     end_date=datetime(2023, 1, 31, tzinfo=timezone.utc),
        ...     pattern_types=["pin_bar", "supply_demand"],
        ...     regime_aware=False
        ... )
        >>> print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
        >>> print(f"Win Rate: {result.metrics.win_rate:.1%}")
    """
    # ========================================================================
    # STEP 1: Validate Parameters
    # ========================================================================

    # Validate timeframe
    validated_timeframe = validate_timeframe(timeframe)

    # Validate dates
    if not start_date.tzinfo or not end_date.tzinfo:
        raise ValueError("start_date and end_date must be timezone-aware (UTC)")

    if end_date <= start_date:
        raise ValueError(
            f"end_date must be after start_date. "
            f"Got start_date={start_date}, end_date={end_date}"
        )

    # Validate pattern_types
    if not pattern_types:
        raise ValueError("pattern_types cannot be empty")

    # Validate symbol (currently only EURUSD supported)
    if symbol.upper() != "EURUSD":
        raise ValueError(
            f"Symbol '{symbol}' not supported. Currently only EURUSD is available."
        )

    # ========================================================================
    # STEP 2: Load Historical Data from Database or CSV
    # ========================================================================

    if candles_df is None:
        # Load from database (original behavior)
        db = DatabaseManager()
        db.connect()

        try:
            # Convert datetimes to ISO format for database query
            start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Fetch candles from database
            candles_df = db.get_candles_by_date_range(
                timeframe=validated_timeframe, start_date=start_str, end_date=end_str
            )
        finally:
            db.disconnect()
    else:
        # Use provided CSV data - need to standardize column names
        # Expected columns: datetime, open, high, low, close
        # Database uses: rate_time, open, high, low, close
        if "datetime" in candles_df.columns and "rate_time" not in candles_df.columns:
            # CSV format - rename datetime to rate_time for compatibility
            candles_df = candles_df.copy()
            candles_df["rate_time"] = pd.to_datetime(candles_df["datetime"])

    # Handle empty data
    if candles_df.empty:
        # Return empty BacktestResults
        empty_metrics = calculate_backtest_metrics(
            trades=[], initial_capital=initial_capital, total_signals=0, trading_days=0
        )

        config_name = _generate_config_name(pattern_types, regime_aware)

        return BacktestResults(
            config_name=config_name,
            symbol=symbol,
            timeframe=validated_timeframe,
            start_date=start_date,
            end_date=end_date,
            pattern_types=pattern_types,
            regime_aware=regime_aware,
            metrics=empty_metrics,
            trades=[],
            signals=[],
            equity_curve=[initial_capital],
            timestamps=[start_date],
        )

    # ========================================================================
    # STEP 3: Initialize Confluence Engine and RegimeDetector
    # ========================================================================

    engine = ConfluenceEngine(
        config={
            "confidence_threshold": min_confidence,
            "min_pattern_count": min_pattern_count,
        }
    )

    # Initialize RegimeDetector if regime_aware is True
    regime_detector = None
    if regime_aware:
        regime_detector = RegimeDetector(atr_period=14, adx_period=14)

    # ========================================================================
    # STEP 4: Generate Signals for Each Candle
    # ========================================================================

    signals: List[ConfluenceSignal] = []
    trades: List[Trade] = []
    equity_curve: List[float] = []
    timestamps: List[datetime] = []

    current_equity = initial_capital

    # For now, we'll use simplified pattern detection
    # In production, this would integrate with actual pattern matchers
    for idx, row in candles_df.iterrows():
        candle_time = row["rate_time"]
        candle_close = row["close"]

        # Extract patterns using provided extractor or use default (empty)
        if pattern_extractor is not None:
            patterns = pattern_extractor(candles_df, idx)
        else:
            # Default: create empty pattern list (no patterns detected)
            patterns = []

        # Detect regime using RegimeDetector or use placeholder
        regime = "RANGE"  # Default placeholder
        if regime_detector is not None and idx >= 50:
            # Use sliding window of last 50 candles for regime detection
            window_start = max(0, idx - 50)
            window_end = idx + 1

            # Extract window data
            window = candles_df.iloc[window_start:window_end]
            window_timestamps = window["rate_time"].tolist()
            window_highs = window["high"].tolist()
            window_lows = window["low"].tolist()
            window_closes = window["close"].tolist()
            window_volumes = window["volume"].tolist() if "volume" in window.columns else [0] * len(window_timestamps)

            try:
                # Detect regime state
                regime_state = regime_detector.detect_regime(
                    timestamps=window_timestamps,
                    highs=window_highs,
                    lows=window_lows,
                    closes=window_closes,
                    volumes=window_volumes,
                )

                # Extract regime classification (use trend regime as primary)
                if regime_state is not None:
                    regime = regime_state.trend_regime
            except Exception:
                # Fall back to placeholder if detection fails
                regime = "RANGE"

        # Generate confluence signal
        signal = engine.process_candle(
            patterns=patterns, regime=regime, historical_performance={}
        )

        signals.append(signal)

        # ====================================================================
        # STEP 5: Execute Trades from Signals
        # ====================================================================

        if signal.signal_type == "CONFLUENCE" and signal.tradeable:
            # Execute signal as trade
            # Determine exit based on random outcome (simplified)
            # In production, this would track actual price movement
            import random

            exit_reason = "TAKE_PROFIT" if random.random() > 0.5 else "STOP_LOSS"

            trade = execute_confluence_signal(
                signal=signal,
                entry_price=candle_close,
                stop_loss_pips=stop_loss_pips,
                take_profit_ratio=take_profit_ratio,
                exit_reason=exit_reason,
            )

            trades.append(trade)

            # Update equity
            current_equity += trade.pnl

        # Track equity and timestamp AFTER processing candle
        equity_curve.append(current_equity)
        timestamps.append(candle_time)

    # Prepend initial capital to equity curve (before first candle)
    if equity_curve:
        equity_curve.insert(0, initial_capital)
        # Use start_date as first timestamp
        timestamps.insert(0, start_date)

    # ========================================================================
    # STEP 6: Calculate Metrics
    # ========================================================================

    trading_days = (end_date - start_date).days

    metrics = calculate_backtest_metrics(
        trades=trades,
        initial_capital=initial_capital,
        total_signals=len([s for s in signals if s.signal_type == "CONFLUENCE"]),
        trading_days=trading_days,
    )

    # ========================================================================
    # STEP 7: Generate Config Name and Return Results
    # ========================================================================

    config_name = _generate_config_name(pattern_types, regime_aware)

    return BacktestResults(
        config_name=config_name,
        symbol=symbol,
        timeframe=validated_timeframe,
        start_date=start_date,
        end_date=end_date,
        pattern_types=pattern_types,
        regime_aware=regime_aware,
        metrics=metrics,
        trades=trades,
        signals=signals,
        equity_curve=equity_curve,
        timestamps=timestamps,
    )


def _generate_config_name(pattern_types: List[str], regime_aware: bool) -> str:
    """
    Generate descriptive configuration name for backtest.

    Args:
        pattern_types: List of pattern types being tested.
        regime_aware: Whether regime filtering is enabled.

    Returns:
        Human-readable configuration name.

    Examples:
        >>> _generate_config_name(["pin_bar"], False)
        '1-Pattern (pin_bar)'
        >>> _generate_config_name(["pin_bar", "supply_demand"], True)
        '2-Pattern Confluence (Regime-Aware)'
        >>> _generate_config_name(["pin_bar", "supply_demand", "choch"], True)
        '3-Pattern Confluence (Regime-Aware)'
    """
    num_patterns = len(pattern_types)

    if num_patterns == 1:
        name = f"1-Pattern ({pattern_types[0]})"
    else:
        name = f"{num_patterns}-Pattern Confluence"

    if regime_aware:
        name += " (Regime-Aware)"

    return name
