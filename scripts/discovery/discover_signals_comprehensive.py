#!/usr/bin/env python3
"""
Comprehensive Multi-Timeframe Signal Discovery (Issue #516).

This script implements the comprehensive discovery strategy documented in
docs/SIGNAL_DISCOVERY_METHODOLOGY.md Section 3.7.

Key Features:
- Tests ALL existing indicators (no Docker changes needed)
- Tests ALL 9 timeframes: M30, H1, H2, H3, H4, H6, H8, H12, D1
- Tests ALL reasonable indicator combinations
- Outputs ranked signal database in JSON format
- One-time investment → permanent signal database → fast future deployment

Usage:
    # Single symbol, all timeframes
    python scripts/discover_signals_comprehensive.py --symbol EURUSD

    # Custom timeframes
    python scripts/discover_signals_comprehensive.py \
        --symbol EURUSD \
        --timeframes M30,H1,H2,H3,H4,H6,H8,H12,D1

    # All symbols (parallelized)
    for symbol in EURUSD GBPUSD USDJPY EURJPY XAGUSD USDCAD; do
        python scripts/discover_signals_comprehensive.py --symbol $symbol &
    done
"""

import argparse
import json
import logging
import math
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from joblib import Parallel, delayed

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default data path - UPDATED to use backed up technical indicators
DEFAULT_DATA_PATH = Path("data/indicators")

# All supported timeframes
ALL_TIMEFRAMES = ["M30", "H1", "H2", "H3", "H4", "H6", "H8", "H12", "D1"]

# Symbol-specific pip values
PIP_VALUES = {
    "eurusd": 0.0001,
    "gbpusd": 0.0001,
    "audusd": 0.0001,
    "nzdusd": 0.0001,
    "usdchf": 0.0001,
    "usdcad": 0.0001,
    "usdjpy": 0.01,
    "eurjpy": 0.01,
    "gbpjpy": 0.01,
    "audjpy": 0.01,
    "xauusd": 0.01,
    "xagusd": 0.01,
}


def get_pip_value(symbol: str) -> float:
    """Get the pip value for a symbol."""
    symbol_lower = symbol.lower()
    if symbol_lower in PIP_VALUES:
        return PIP_VALUES[symbol_lower]
    if "jpy" in symbol_lower:
        return 0.01
    return 0.0001


@dataclass
class SignalResult:
    """Result of testing a single signal on a single timeframe."""

    rank: int
    signal_name: str
    timeframe: str
    direction: str
    win_rate: float
    trades: int
    p_value: float
    edge_pct: float
    expectancy_pips: float
    frequency_per_year: int
    degradation: float
    status: str
    signal_definition: str
    category: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


def load_m30_data(symbol: str, data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load M30 data with technical indicators from backed up CSV files.

    CRITICAL: Uses correct naming pattern:
    - EURUSD: technical_indicators.csv (plural, no suffix)
    - Others: technical_indicator_{symbol}.csv (singular, with suffix)
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_PATH

    # Handle EURUSD special naming convention
    symbol_upper = symbol.upper()
    if symbol_upper == "EURUSD":
        # EURUSD uses plural "technical_indicators.csv" (NO symbol suffix)
        csv_path = data_dir / "technical_indicators.csv"
    else:
        # All other symbols use singular "technical_indicator_{symbol}.csv"
        symbol_lower = symbol.lower()
        csv_path = data_dir / f"technical_indicator_{symbol_lower}.csv"

    logger.info(f"Loading data from {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {csv_path}\n"
            f"Expected naming:\n"
            f"  - EURUSD: technical_indicators.csv (plural, no suffix)\n"
            f"  - Others: technical_indicator_{{symbol}}.csv (singular, with suffix)"
        )

    df = pd.read_csv(csv_path)

    # Parse datetime - handle multiple possible column names
    if "readable_date" in df.columns and df["readable_date"].notna().mean() > 0.5:
        df["datetime"] = pd.to_datetime(df["readable_date"])
    elif "timestamp" in df.columns:
        # Technical indicator backup files use 'timestamp' column
        df["datetime"] = pd.to_datetime(df["timestamp"])
    elif "rate_time" in df.columns:
        df["datetime"] = pd.to_datetime(df["rate_time"], unit="s")
    else:
        raise ValueError(
            f"No valid datetime column found. Available columns: {', '.join(df.columns[:10])}"
        )

    # Filter for M30 timeframe if timeframe column exists
    if "timeframe" in df.columns:
        df = df[df["timeframe"] == "M30"].copy()
        logger.info(f"Filtered to M30 timeframe: {len(df):,} bars")

    df = df.sort_values("datetime").reset_index(drop=True)

    # Indicators already computed in backed up file - just verify they exist
    required_indicators = ["sma_20", "ema_12", "rsi_14", "macd_line", "bb_upper_20"]
    missing = [ind for ind in required_indicators if ind not in df.columns]
    if missing:
        logger.warning(f"Missing expected indicators: {missing}")
        logger.warning("Computing missing indicators on the fly...")
        df = compute_indicators(df)

    logger.info(
        f"Loaded {len(df):,} bars from {df['datetime'].min()} to {df['datetime'].max()}"
    )

    # DON'T set datetime as index here - let run_discovery_for_timeframe handle it
    # Each parallel worker will set it as index when needed

    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators if not present."""
    df = df.copy()

    # Ensure numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # SMA
    if "sma_20" not in df.columns:
        df["sma_20"] = df["close"].rolling(20).mean()
    if "sma_50" not in df.columns:
        df["sma_50"] = df["close"].rolling(50).mean()
    if "sma_200" not in df.columns:
        df["sma_200"] = df["close"].rolling(200).mean()

    # EMA
    if "ema_12" not in df.columns:
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    if "ema_26" not in df.columns:
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    if "ema_50" not in df.columns:
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    # RSI
    if "rsi_14" not in df.columns:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

    # ATR
    if "atr_14" not in df.columns:
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["atr_14"] = true_range.rolling(14).mean()

    # Bollinger Bands
    if "bb_upper_20" not in df.columns:
        bb_std = df["close"].rolling(20).std()
        df["bb_middle_20"] = df["sma_20"]
        df["bb_upper_20"] = df["bb_middle_20"] + 2 * bb_std
        df["bb_lower_20"] = df["bb_middle_20"] - 2 * bb_std

    # MACD
    if "macd_line" not in df.columns:
        df["macd_line"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

    # Stochastic
    if "stoch_k" not in df.columns:
        low_14 = df["low"].rolling(14).min()
        high_14 = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    return df


def aggregate_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Aggregate M30 data to higher timeframes.

    Supports: M30, H1, H2, H3, H4, H6, H8, H12, D1
    """
    timeframe_upper = timeframe.upper()

    if timeframe_upper == "M30":
        return df

    resample_map = {
        "H1": "1h",
        "H2": "2h",
        "H3": "3h",
        "H4": "4h",
        "H6": "6h",
        "H8": "8h",
        "H12": "12h",
        "D1": "1D",
    }

    rule = resample_map.get(timeframe_upper)
    if rule is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # Only aggregate columns that exist
    available_agg = {k: v for k, v in agg_dict.items() if k in df.columns}

    result = df.resample(rule).agg(available_agg).dropna()

    return result


def get_all_signal_combinations() -> dict[str, list[tuple[str, Callable, str, str]]]:
    """
    Get ALL signal combinations to test.

    Returns:
        Dictionary mapping category to (name, condition, direction, definition) tuples
    """
    signals = {
        # Single indicator signals
        "trend_single": [
            # SMA crossovers
            (
                "SMA20_cross_SMA50_long",
                lambda df: (df["sma_20"].shift(1) <= df["sma_50"].shift(1))
                & (df["sma_20"] > df["sma_50"]),
                "long",
                "sma_20 crosses above sma_50",
            ),
            (
                "SMA20_cross_SMA50_short",
                lambda df: (df["sma_20"].shift(1) >= df["sma_50"].shift(1))
                & (df["sma_20"] < df["sma_50"]),
                "short",
                "sma_20 crosses below sma_50",
            ),
            # EMA crossovers
            (
                "EMA12_cross_EMA26_long",
                lambda df: (df["ema_12"].shift(1) <= df["ema_26"].shift(1))
                & (df["ema_12"] > df["ema_26"]),
                "long",
                "ema_12 crosses above ema_26",
            ),
            (
                "EMA12_cross_EMA26_short",
                lambda df: (df["ema_12"].shift(1) >= df["ema_26"].shift(1))
                & (df["ema_12"] < df["ema_26"]),
                "short",
                "ema_12 crosses below ema_26",
            ),
        ],
        # Momentum signals
        "momentum_single": [
            # RSI oversold/overbought
            (
                "RSI14_oversold_long",
                lambda df: df["rsi_14"] < 30,
                "long",
                "rsi_14 < 30",
            ),
            (
                "RSI14_overbought_short",
                lambda df: df["rsi_14"] > 70,
                "short",
                "rsi_14 > 70",
            ),
            (
                "RSI14_extreme_oversold_long",
                lambda df: df["rsi_14"] < 20,
                "long",
                "rsi_14 < 20",
            ),
            (
                "RSI14_extreme_overbought_short",
                lambda df: df["rsi_14"] > 80,
                "short",
                "rsi_14 > 80",
            ),
            # MACD crossovers
            (
                "MACD_cross_signal_long",
                lambda df: (df["macd_line"].shift(1) <= df["macd_signal"].shift(1))
                & (df["macd_line"] > df["macd_signal"]),
                "long",
                "macd_line crosses above macd_signal",
            ),
            (
                "MACD_cross_signal_short",
                lambda df: (df["macd_line"].shift(1) >= df["macd_signal"].shift(1))
                & (df["macd_line"] < df["macd_signal"]),
                "short",
                "macd_line crosses below macd_signal",
            ),
            # Stochastic oversold/overbought
            (
                "Stoch_K_oversold_long",
                lambda df: df["stoch_k"] < 20,
                "long",
                "stoch_k < 20",
            ),
            (
                "Stoch_K_overbought_short",
                lambda df: df["stoch_k"] > 80,
                "short",
                "stoch_k > 80",
            ),
        ],
        # Volatility signals
        "volatility_single": [
            # Bollinger Bands
            (
                "BB_lower_touch_long",
                lambda df: df["close"] < df["bb_lower_20"],
                "long",
                "close < bb_lower_20",
            ),
            (
                "BB_upper_touch_short",
                lambda df: df["close"] > df["bb_upper_20"],
                "short",
                "close > bb_upper_20",
            ),
            # BB squeeze (volatility compression)
            (
                "BB_squeeze_breakout_long",
                lambda df: (
                    (
                        (df["bb_upper_20"] - df["bb_lower_20"]) / df["sma_20"]
                    ).rolling(20).quantile(0.2)
                    > ((df["bb_upper_20"] - df["bb_lower_20"]) / df["sma_20"]).shift(1)
                )
                & (df["close"] > df["bb_upper_20"]),
                "long",
                "bb_width < quantile(0.2) & close > bb_upper",
            ),
            (
                "BB_squeeze_breakout_short",
                lambda df: (
                    (
                        (df["bb_upper_20"] - df["bb_lower_20"]) / df["sma_20"]
                    ).rolling(20).quantile(0.2)
                    > ((df["bb_upper_20"] - df["bb_lower_20"]) / df["sma_20"]).shift(1)
                )
                & (df["close"] < df["bb_lower_20"]),
                "short",
                "bb_width < quantile(0.2) & close < bb_lower",
            ),
        ],
        # Volume signals
        "volume_single": [
            (
                "Volume_spike_breakout_long",
                lambda df: (df["volume"] > df["volume"].rolling(20).mean() * 2.0)
                & (df["close"] > df["sma_50"]),
                "long",
                "volume > 2*ma20 & close > sma_50",
            ),
            (
                "Volume_spike_breakout_short",
                lambda df: (df["volume"] > df["volume"].rolling(20).mean() * 2.0)
                & (df["close"] < df["sma_50"]),
                "short",
                "volume > 2*ma20 & close < sma_50",
            ),
        ],
        # Two-indicator combinations
        "combo_two": [
            # RSI + BB
            (
                "RSI_BB_confluence_long",
                lambda df: (df["rsi_14"] < 30) & (df["close"] < df["bb_lower_20"]),
                "long",
                "rsi < 30 & close < bb_lower",
            ),
            (
                "RSI_BB_confluence_short",
                lambda df: (df["rsi_14"] > 70) & (df["close"] > df["bb_upper_20"]),
                "short",
                "rsi > 70 & close > bb_upper",
            ),
            # EMA + RSI
            (
                "EMA_RSI_long",
                lambda df: (df["ema_12"] > df["ema_26"])
                & (df["rsi_14"] > 40)
                & (df["rsi_14"] < 70),
                "long",
                "ema_12 > ema_26 & rsi healthy",
            ),
            (
                "EMA_RSI_short",
                lambda df: (df["ema_12"] < df["ema_26"])
                & (df["rsi_14"] > 30)
                & (df["rsi_14"] < 60),
                "short",
                "ema_12 < ema_26 & rsi healthy",
            ),
            # Volume + EMA
            (
                "Volume_EMA_long",
                lambda df: (df["volume"] > df["volume"].rolling(20).mean() * 1.5)
                & (df["ema_12"] > df["ema_26"]),
                "long",
                "volume > 1.5*ma20 & ema_12 > ema_26",
            ),
            (
                "Volume_EMA_short",
                lambda df: (df["volume"] > df["volume"].rolling(20).mean() * 1.5)
                & (df["ema_12"] < df["ema_26"]),
                "short",
                "volume > 1.5*ma20 & ema_12 < ema_26",
            ),
        ],
        # Three-indicator combinations
        "combo_three": [
            # Volume + EMA + RSI
            (
                "Volume_EMA_RSI_long",
                lambda df: (df["volume"] > df["volume"].rolling(20).mean() * 1.5)
                & (df["ema_12"] > df["ema_26"])
                & (df["rsi_14"] > 40)
                & (df["rsi_14"] < 70),
                "long",
                "volume > 1.5*ma20 & ema_cross & rsi healthy",
            ),
            (
                "Volume_EMA_RSI_short",
                lambda df: (df["volume"] > df["volume"].rolling(20).mean() * 1.5)
                & (df["ema_12"] < df["ema_26"])
                & (df["rsi_14"] > 30)
                & (df["rsi_14"] < 60),
                "short",
                "volume > 1.5*ma20 & ema_cross & rsi healthy",
            ),
            # BB + RSI + Volume
            (
                "BB_RSI_Volume_long",
                lambda df: (df["close"] < df["bb_lower_20"])
                & (df["rsi_14"] < 30)
                & (df["volume"] > df["volume"].rolling(20).mean()),
                "long",
                "bb_lower & rsi < 30 & volume > avg",
            ),
            (
                "BB_RSI_Volume_short",
                lambda df: (df["close"] > df["bb_upper_20"])
                & (df["rsi_14"] > 70)
                & (df["volume"] > df["volume"].rolling(20).mean()),
                "short",
                "bb_upper & rsi > 70 & volume > avg",
            ),
        ],
    }

    return signals


def test_signal(
    df: pd.DataFrame,
    signal_func: Callable,
    direction: str,
    symbol: str,
    sl_pips: int = 30,
    tp_pips: int = 30,
    horizon_bars: int = 48,
) -> dict:
    """
    Test a single signal on given data.

    Returns:
        Dictionary with results: trades, wins, losses, win_rate, expectancy, etc.
    """
    try:
        signal = signal_func(df)
        if signal is None or signal.sum() == 0:
            return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0}

        pip_value = get_pip_value(symbol)
        signal_bars = df[signal].index.tolist()

        wins = 0
        losses = 0
        total_pips = 0
        last_trade_close_idx = -1  # Track when previous trade closed (prevents overlapping)

        for entry_idx in signal_bars:
            if entry_idx >= len(df) - horizon_bars:
                continue

            # CRITICAL: Skip if previous trade hasn't closed yet (prevents overlapping trades)
            if entry_idx <= last_trade_close_idx:
                continue

            entry_price = df.loc[entry_idx, "close"]

            # Calculate SL/TP prices
            if direction.lower() == "long":
                tp_price = entry_price + (tp_pips * pip_value)
                sl_price = entry_price - (sl_pips * pip_value)
            else:  # short
                tp_price = entry_price - (tp_pips * pip_value)
                sl_price = entry_price + (sl_pips * pip_value)

            # Check next N bars for TP/SL hit
            hit_tp = False
            hit_sl = False
            close_bar = entry_idx + horizon_bars  # Default: trade expires at horizon

            for i in range(entry_idx + 1, min(entry_idx + horizon_bars + 1, len(df))):
                high = df.loc[i, "high"]
                low = df.loc[i, "low"]

                if direction.lower() == "long":
                    if high >= tp_price:
                        hit_tp = True
                        close_bar = i
                        break
                    if low <= sl_price:
                        hit_sl = True
                        close_bar = i
                        break
                else:  # short
                    if low <= tp_price:
                        hit_tp = True
                        close_bar = i
                        break
                    if high >= sl_price:
                        hit_sl = True
                        close_bar = i
                        break

            # Update last trade close index to prevent overlapping trades
            last_trade_close_idx = close_bar

            if hit_tp:
                wins += 1
                total_pips += tp_pips
            elif hit_sl:
                losses += 1
                total_pips -= sl_pips

        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        expectancy = total_pips / total_trades if total_trades > 0 else 0.0

        return {
            "trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pips": total_pips,
            "expectancy": expectancy,
        }

    except Exception as e:
        logger.warning(f"Error testing signal: {e}")
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0}


def calculate_significance(wins: int, total: int, null_prob: float = 0.5) -> float:
    """Calculate statistical significance using binomial test."""
    if total < 30:
        return 1.0

    result = stats.binomtest(wins, total, null_prob, alternative="greater")
    return result.pvalue


def run_discovery_for_timeframe(
    df_m30: pd.DataFrame,
    symbol: str,
    timeframe: str,
    min_trades: int = 50,
    p_value_threshold: float = 0.05,
) -> list[SignalResult]:
    """
    Run discovery for a single timeframe.

    Returns:
        List of SignalResult objects for this timeframe
    """
    logger.info(f"  Timeframe {timeframe}: Aggregating data...")

    # Set datetime as index for resampling
    df_m30_indexed = df_m30.set_index("datetime")
    df_tf = aggregate_to_timeframe(df_m30_indexed, timeframe)

    # Recompute indicators for this timeframe
    df_tf = df_tf.reset_index()
    df_tf.rename(columns={"index": "datetime"}, inplace=True)
    df_tf = compute_indicators(df_tf)

    logger.info(f"  Timeframe {timeframe}: Testing {len(df_tf):,} bars")

    # Get all signal combinations
    all_signals = get_all_signal_combinations()

    results = []
    total_signals = sum(len(signals) for signals in all_signals.values())
    tested = 0

    for category, signals in all_signals.items():
        for signal_name, signal_func, direction, definition in signals:
            tested += 1

            # Test the signal
            result = test_signal(
                df_tf,
                signal_func,
                direction,
                symbol,
                sl_pips=30,
                tp_pips=30,
                horizon_bars=48,
            )

            if result["trades"] < min_trades:
                continue

            # Calculate statistics
            p_value = calculate_significance(
                result["wins"], result["trades"], null_prob=0.5
            )

            if p_value > p_value_threshold:
                continue

            # Estimate annual frequency (bars per year depends on timeframe)
            bars_per_year = {
                "M30": 17520,
                "H1": 8760,
                "H2": 4380,
                "H3": 2920,
                "H4": 2190,
                "H6": 1460,
                "H8": 1095,
                "H12": 730,
                "D1": 365,
            }

            total_bars = len(df_tf)
            frequency_per_year = int(
                result["trades"] / total_bars * bars_per_year.get(timeframe, 8760)
            )

            # Determine status
            if result["win_rate"] >= 65 and p_value < 0.01:
                status = "excellent"
            elif result["win_rate"] >= 55 and p_value < 0.05:
                status = "valid"
            else:
                status = "marginal"

            signal_result = SignalResult(
                rank=0,  # Will be set later
                signal_name=signal_name,
                timeframe=timeframe,
                direction=direction,
                win_rate=round(result["win_rate"], 2),
                trades=result["trades"],
                p_value=round(p_value, 4),
                edge_pct=round(result["win_rate"] - 50, 2),
                expectancy_pips=round(result["expectancy"], 2),
                frequency_per_year=frequency_per_year,
                degradation=0.0,  # Unknown until OOS validation
                status=status,
                signal_definition=definition,
                category=category,
            )

            results.append(signal_result)

    logger.info(
        f"  Timeframe {timeframe}: Found {len(results)} significant signals "
        f"(tested {tested}/{total_signals})"
    )

    return results


def run_comprehensive_discovery(
    symbol: str,
    timeframes: list[str],
    data_dir: Path | None = None,
    min_trades: int = 50,
    p_value_threshold: float = 0.05,
    n_jobs: int = -1,
) -> dict:
    """
    Run comprehensive discovery across all timeframes.

    Returns:
        Dictionary with all results ready for JSON export
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"COMPREHENSIVE MULTI-TIMEFRAME DISCOVERY: {symbol.upper()}")
    logger.info(f"{'='*80}")
    logger.info(f"Timeframes: {', '.join(timeframes)}")
    logger.info(f"Min trades: {min_trades}")
    logger.info(f"P-value threshold: {p_value_threshold}")
    logger.info(f"Parallel jobs: {n_jobs if n_jobs > 0 else cpu_count()}")
    logger.info(f"{'='*80}\n")

    # Load M30 base data
    df_m30 = load_m30_data(symbol, data_dir)

    # Run discovery for each timeframe (parallelizable)
    if HAS_JOBLIB and n_jobs != 1:
        n_jobs_actual = cpu_count() if n_jobs < 0 else n_jobs
        logger.info(f"Running parallel discovery across {len(timeframes)} timeframes...")

        timeframe_results = Parallel(n_jobs=n_jobs_actual)(
            delayed(run_discovery_for_timeframe)(
                df_m30, symbol, tf, min_trades, p_value_threshold
            )
            for tf in timeframes
        )
    else:
        logger.info("Running sequential discovery...")
        timeframe_results = [
            run_discovery_for_timeframe(
                df_m30, symbol, tf, min_trades, p_value_threshold
            )
            for tf in timeframes
        ]

    # Flatten all results
    all_signals = []
    for results in timeframe_results:
        all_signals.extend(results)

    # Sort by composite score: edge × (1 - p_value) × log(frequency + 1)
    for signal in all_signals:
        signal.rank = 0  # Temporary
        edge = signal.edge_pct / 100
        score = edge * (1 - signal.p_value) * math.log(signal.frequency_per_year + 1)
        # Add score as attribute for sorting
        signal._score = score

    all_signals.sort(key=lambda x: x._score, reverse=True)

    # Assign ranks
    for i, signal in enumerate(all_signals, 1):
        signal.rank = i
        delattr(signal, "_score")  # Remove temporary score

    # Build summary by timeframe
    summary_by_timeframe = {}
    for tf in timeframes:
        tf_signals = [s for s in all_signals if s.timeframe == tf]
        if tf_signals:
            summary_by_timeframe[tf] = {
                "signals_found": len(tf_signals),
                "avg_wr": round(sum(s.win_rate for s in tf_signals) / len(tf_signals), 2),
                "best_signal": tf_signals[0].signal_name,
            }
        else:
            summary_by_timeframe[tf] = {
                "signals_found": 0,
                "avg_wr": 0.0,
                "best_signal": None,
            }

    # Build summary by direction
    summary_by_direction = {}
    for direction in ["long", "short"]:
        dir_signals = [s for s in all_signals if s.direction == direction]
        if dir_signals:
            summary_by_direction[direction] = {
                "signals_found": len(dir_signals),
                "avg_wr": round(sum(s.win_rate for s in dir_signals) / len(dir_signals), 2),
                "best_signal": dir_signals[0].signal_name,
            }
        else:
            summary_by_direction[direction] = {
                "signals_found": 0,
                "avg_wr": 0.0,
                "best_signal": None,
            }

    # Build final output
    output = {
        "symbol": symbol.upper(),
        "discovery_date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_combinations_tested": len(all_signals),
        "timeframes_tested": timeframes,
        "top_signals": [s.to_dict() for s in all_signals],
        "summary_by_timeframe": summary_by_timeframe,
        "summary_by_direction": summary_by_direction,
    }

    return output


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Multi-Timeframe Signal Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help="Symbol to analyze (e.g., EURUSD)",
    )
    parser.add_argument(
        "--timeframes",
        default=",".join(ALL_TIMEFRAMES),
        help=f"Comma-separated timeframes (default: all 9 timeframes)",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=50,
        help="Minimum trades for validity (default: 50)",
    )
    parser.add_argument(
        "--p-value",
        type=float,
        default=0.05,
        help="Significance threshold (default: 0.05)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers (-1 for all cores)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"Data directory path (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (default: results/comprehensive_discovery/{symbol}_all_signals.json)",
    )

    args = parser.parse_args()

    # Parse timeframes
    timeframes = [tf.strip().upper() for tf in args.timeframes.split(",")]

    # Validate timeframes
    for tf in timeframes:
        if tf not in ALL_TIMEFRAMES:
            logger.error(f"Invalid timeframe: {tf}. Must be one of: {', '.join(ALL_TIMEFRAMES)}")
            sys.exit(1)

    # Run discovery
    try:
        results = run_comprehensive_discovery(
            symbol=args.symbol,
            timeframes=timeframes,
            data_dir=args.data_dir,
            min_trades=args.min_trades,
            p_value_threshold=args.p_value,
            n_jobs=args.n_jobs,
        )

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = Path("results/comprehensive_discovery")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{args.symbol.lower()}_all_signals.json"

        # Save results
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n{'='*80}")
        logger.info(f"DISCOVERY COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total signals found: {results['total_combinations_tested']}")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"\nTop 10 signals:")
        for i, signal in enumerate(results["top_signals"][:10], 1):
            logger.info(
                f"  {i}. {signal['signal_name']} ({signal['timeframe']} {signal['direction'].upper()}) - "
                f"WR: {signal['win_rate']}%, Trades: {signal['trades']}, p={signal['p_value']}"
            )

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during discovery: {e}")
        raise


if __name__ == "__main__":
    main()
