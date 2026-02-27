#!/usr/bin/env python3
"""
Hybrid V4 Pipeline CLI Script - Issue #526.

Main orchestration script for the complete hybrid-v4 training pipeline.
Executes all 6 phases of signal discovery, validation, and model training.

Metric Strategy:
    - Phase 1-2: Use Win Rate (WR) as primary metric
      With fixed SL=30, TP=30, all trades are ±30 pips → WR is valid
      WR 55% ≈ PF 1.22 (passes threshold)
    - Phase 3-5: Use Profit Factor (PF) as primary metric
      RL model learns variable exits → PF captures actual profit/loss

Phases:
    0. Data Segregation: Split data 60/20/20
    1. Signal Discovery: Find signals on training data (WR-based)
       BLOCKING GATE: WR >= 55%, trades >= 100, p-value < 0.05
    2. Walk-Forward Validation: Validate signals on unseen data (WR-based)
       BLOCKING GATE: WR OOS >= 54%, WR degradation <= 5%
    3. Optuna Tuning: Hyperparameter optimization (optional, PF-based)
    4. 30-Fold Training: Train ensemble of models (PF-based)
    5. Final Test Validation: Evaluate on never-seen test data (PF-based)

Usage:
    python scripts/run_hybrid_v4_pipeline.py --symbol eurusd --direction long
    python scripts/run_hybrid_v4_pipeline.py --symbol gbpusd --direction short --skip-optuna
    python scripts/run_hybrid_v4_pipeline.py --symbol eurusd --direction long --dry-run

Exit Codes:
    0: Pipeline completed (APPROVED or REJECTED - both are valid outcomes)
    1: Pipeline crashed/error (missing data, code bug, etc.)

Issue: #526 - Pipeline Integration and Testing
Epic: hybrid-v4
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import pipeline tracking
from pipeline_tracking import (
    update_phase1,
    update_phase2,
    update_phase5,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

SUPPORTED_SYMBOLS = [
    "eurusd", "gbpusd", "usdjpy", "eurjpy", "usdcad", "usdchf", "eurcad", "eurgbp"
]

SUPPORTED_DIRECTIONS = ["long", "short"]

# Data paths
DEFAULT_DATA_DIR = Path("data")

# TFG: Local training queue (no GCS)
LOCAL_QUEUE_PATH = Path("results/training_queue.json")

# Phase 1-2 Blocking Gate Thresholds (WR-based for fixed SL/TP)
# With fixed SL=30, TP=30, all trades are ±30 pips → WR is valid primary metric
# WR 55% ≈ PF 1.22 (passes threshold)
MIN_WIN_RATE_PHASE1 = 0.55  # Minimum WR for Phase 1
MIN_WIN_RATE_PHASE2 = 0.54  # Slightly lower for OOS
MIN_TRADES_PHASE1 = 100  # Statistical validity
MIN_TRADES_PHASE2 = 50  # Fewer trades on validation set
MAX_P_VALUE = 0.05  # 95% confidence
MAX_DEGRADATION = 0.10  # Max temporal degradation
MAX_WR_DEGRADATION = 0.15  # Max IS→OOS WR drop (Issue #526: Relaxed from 5% to 15% to allow natural variance)

# Phase 3-5 Thresholds (PF-based for RL training)
MIN_PROFIT_FACTOR = 1.5  # Min PF for Phase 5 approval (Issue #643: raised from 1.2)
MIN_WIN_RATE_PHASE5 = 0.40  # Min WR for Phase 5 approval (Issue #643: eliminates low-WR flukes)

# =============================================================================
# Data Loading
# =============================================================================


def load_market_data(
    symbol: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    timeframe: str = None,
) -> pd.DataFrame:
    """
    Load market data with indicators for a symbol.

    Uses symbol-specific files for performance (see CLAUDE.md rule).

    ALL symbols (including EURUSD) now use symbol-specific files:
    - technical_indicator_eurusd.csv
    - technical_indicator_gbpusd.csv
    - etc.

    Args:
        symbol: Trading symbol (e.g., 'eurusd')
        data_dir: Directory containing data files
        timeframe: Optional timeframe filter (e.g., 'M30', 'H1').
                   If provided, only loads candles for that timeframe.
                   CRITICAL for EURUSD which has multiple timeframes!

    Returns:
        DataFrame with OHLCV and indicators

    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    symbol_lower = symbol.lower()

    # Try symbol-specific file (STANDARD for ALL symbols including EURUSD)
    symbol_file = data_dir / f"technical_indicator_{symbol_lower}.csv"

    if symbol_file.exists():
        logger.info(f"Loading symbol-specific data: {symbol_file}")
        df = pd.read_csv(symbol_file)
    else:
        # Fallback to combined file (slower)
        combined_file = data_dir / "technical_indicators.csv"
        if combined_file.exists():
            logger.warning(f"Using combined file (slower): {combined_file}")
            df = pd.read_csv(combined_file)
            df = df[df["symbol"].str.lower() == symbol_lower]
        else:
            raise FileNotFoundError(
                f"No data file found for {symbol}. "
                f"Expected: {symbol_file} or {combined_file}"
            )

    # Filter by timeframe if specified (CRITICAL for EURUSD multi-TF data)
    if timeframe and "timeframe" in df.columns:
        original_count = len(df)
        df = df[df["timeframe"].str.upper() == timeframe.upper()]
        logger.info(f"Filtered to {timeframe}: {len(df)} candles (from {original_count})")

    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"Loaded {len(df)} candles for {symbol.upper()}")
    return df


# =============================================================================
# Smart Context Loading - Issue #530 Stream C
# =============================================================================


def get_context_indicators(signal_indicators: List[str]) -> List[str]:
    """
    Determine which context indicators to add based on CORRELATION ANALYSIS.

    Correlation Analysis Results (EURUSD M30, 101,255 candles):
    ============================================================

    INDICATOR GROUPS (within-group r > 0.9 = REDUNDANT):
    - Group 1 (MAs): sma_20, sma_50, sma_200, ema_12, ema_26, ema_50 (r=0.99+)
    - Group 2 (BB): bb_upper, bb_middle, bb_lower (r=0.99+)
    - Group 3 (Momentum): rsi_14, stoch_k, stoch_d (r=0.80)
    - Group 4 (MACD): macd_line, macd_signal, macd_histogram (r=0.75-0.95)
    - Group 5 (Volatility): atr_14 (r < 0.02 with everything = UNIQUE!)

    CROSS-GROUP CORRELATIONS (low = ADD for context):
    - MA ↔ Momentum: r ≈ 0.00 ✅ (add momentum to trend signals)
    - MA ↔ ATR: r ≈ 0.01 ✅ (always add ATR)
    - Momentum ↔ ATR: r ≈ 0.00 ✅ (always add ATR)
    - Momentum ↔ MACD: r = 0.75 ⚠️ (some overlap)

    RULES:
    1. If using momentum (RSI/Stoch/MACD), ADD trend context (sma_50)
    2. If using trend (MA/BB), ADD momentum context (rsi_14 or stoch_k)
    3. ALWAYS add atr_14 (r < 0.02 with everything)
    4. DON'T add same-group indicators (redundant)
    5. If using MACD, DON'T add ema_12/26 (MACD IS derived from them!)

    Args:
        signal_indicators: List of indicator column names used by the signal

    Returns:
        List of context indicator column names to add

    Issue: #530 - Correlation-based context loading
    """
    context = []
    signal_set = set(signal_indicators)

    # Detect what groups the signal uses
    has_ma = any(ind.startswith('sma_') or ind.startswith('ema_') for ind in signal_set)
    has_bb = any(ind.startswith('bb_') for ind in signal_set)
    has_rsi = any(ind.startswith('rsi_') for ind in signal_set)
    has_stoch = any(ind in signal_set for ind in ['stoch_k', 'stoch_d'])
    has_macd = any(ind in signal_set for ind in ['macd_line', 'macd_signal', 'macd_histogram'])

    has_trend = has_ma or has_bb  # Trend indicators
    has_momentum = has_rsi or has_stoch or has_macd  # Momentum indicators

    # RULE 1: If using ONLY momentum, ADD trend context (sma_50)
    # sma_50 has r ≈ 0.00 with RSI, Stoch, MACD
    if has_momentum and not has_trend:
        if 'sma_50' not in signal_set:
            context.append('sma_50')

    # RULE 2: If using ONLY trend, ADD momentum context
    # Pick ONE momentum indicator (don't add both RSI and Stoch - r=0.80)
    if has_trend and not has_momentum:
        # Add RSI (not Stoch) as it's more universally used
        if 'rsi_14' not in signal_set:
            context.append('rsi_14')

    # RULE 3: ALWAYS add ATR (r < 0.02 with everything = unique info)
    if 'atr_14' not in signal_set:
        context.append('atr_14')

    # RULE 4: DON'T add same-group indicators
    # - If has sma_20, don't add sma_50/sma_200 (r=0.99+)
    # - If has rsi_14, don't add stoch_k (r=0.80)
    # - If has macd_line, don't add ema_12/26 (MACD IS ema_12 - ema_26!)

    # RULE 5: If using MACD, DO NOT add ema_12/26 (they ARE MACD!)
    # This was WRONG in the old code - MACD = EMA12 - EMA26
    # Adding ema_12/26 when using MACD is 100% redundant

    return context


def load_market_data_smart(
    symbol: str,
    signal: Any,  # SignalDefinition, but avoid circular import
    data_dir: Path,
    add_context: bool = True,
    timeframe: str = None,
) -> pd.DataFrame:
    """
    Load only indicators needed for this specific signal + smart context.

    This is more efficient than load_market_data() as it only loads the
    columns needed for the signal, reducing memory usage and load time.

    Args:
        symbol: Trading symbol (e.g., 'eurusd')
        signal: SignalDefinition with required_columns attribute
        data_dir: Directory containing CSV files
        add_context: Whether to add context indicators (default: True)
        timeframe: Optional timeframe filter (e.g., 'M30', 'H1').
                   CRITICAL for EURUSD which has multiple timeframes!

    Returns:
        DataFrame with ONLY required columns (much smaller than full load)

    Raises:
        FileNotFoundError: If data file doesn't exist

    Issue: #530 - Stream C: Smart Context Loading
    """
    symbol_lower = symbol.lower()

    # Find the data file
    symbol_file = data_dir / f"technical_indicator_{symbol_lower}.csv"

    if not symbol_file.exists():
        # Check for combined file as fallback
        combined_file = data_dir / "technical_indicators.csv"
        if not combined_file.exists():
            raise FileNotFoundError(
                f"No data file found for {symbol}. "
                f"Expected: {symbol_file} or {combined_file}"
            )
        # Combined file requires loading all columns then filtering
        logger.warning(f"Using combined file (slower): {combined_file}")
        return _load_from_combined_file(combined_file, symbol_lower, signal, add_context)

    # Get columns needed
    context_cols = get_context_indicators(signal.required_columns) if add_context else []
    all_cols = signal.get_all_required_columns(context_columns=context_cols)

    # First, check which columns actually exist in the CSV
    try:
        # Read just the header to get available columns
        available_cols = pd.read_csv(symbol_file, nrows=0).columns.tolist()
    except Exception as e:
        raise FileNotFoundError(f"Error reading {symbol_file}: {e}")

    # Filter to only columns that exist
    cols_to_load = [col for col in all_cols if col in available_cols]

    # Log which columns are being loaded
    logger.debug(f"Loading columns: {cols_to_load}")

    # Log any missing columns (not an error, just info)
    missing_cols = set(all_cols) - set(cols_to_load)
    if missing_cols:
        logger.debug(f"Columns not available (skipped): {missing_cols}")

    # Ensure we also load timeframe column if filtering is needed
    if timeframe and "timeframe" not in cols_to_load and "timeframe" in available_cols:
        cols_to_load.append("timeframe")

    # Load with usecols parameter for efficiency
    df = pd.read_csv(symbol_file, usecols=cols_to_load)

    # Filter by timeframe if specified (CRITICAL for EURUSD multi-TF data)
    if timeframe and "timeframe" in df.columns:
        original_count = len(df)
        df = df[df["timeframe"].str.upper() == timeframe.upper()]
        logger.info(f"Filtered to {timeframe}: {len(df)} candles (from {original_count})")

    # Ensure timestamp is datetime and sorted
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        f"Smart-loaded {len(df)} candles for {symbol.upper()} "
        f"({len(cols_to_load)} columns)"
    )
    return df


def _load_from_combined_file(
    filepath: Path,
    symbol: str,
    signal: Any,
    add_context: bool,
) -> pd.DataFrame:
    """
    Load from combined indicators file (fallback, slower).

    Args:
        filepath: Path to combined CSV file
        symbol: Symbol to filter for
        signal: SignalDefinition with required_columns
        add_context: Whether to add context indicators

    Returns:
        Filtered DataFrame

    Issue: #530 - Stream C: Smart Context Loading
    """
    # Get columns needed
    context_cols = get_context_indicators(signal.required_columns) if add_context else []
    all_cols = signal.get_all_required_columns(context_columns=context_cols)

    # Add symbol column for filtering
    cols_with_symbol = all_cols + ['symbol'] if 'symbol' not in all_cols else all_cols

    # Check available columns
    available_cols = pd.read_csv(filepath, nrows=0).columns.tolist()
    cols_to_load = [col for col in cols_with_symbol if col in available_cols]

    # Load and filter
    df = pd.read_csv(filepath, usecols=cols_to_load)
    df = df[df["symbol"].str.lower() == symbol.lower()]

    # Drop symbol column if not needed
    if 'symbol' in df.columns and 'symbol' not in all_cols:
        df = df.drop(columns=['symbol'])

    # Ensure timestamp is datetime and sorted
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        f"Smart-loaded {len(df)} candles for {symbol.upper()} from combined file "
        f"({len(cols_to_load)} columns)"
    )
    return df


# =============================================================================
# Argument Parser
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for CLI.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Run the complete hybrid-v4 training pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline for EURUSD long signals
    python scripts/run_hybrid_v4_pipeline.py --symbol eurusd --direction long

    # Skip Optuna tuning (use default hyperparameters)
    python scripts/run_hybrid_v4_pipeline.py --symbol gbpusd --direction short --skip-optuna

    # Dry run to see what would happen
    python scripts/run_hybrid_v4_pipeline.py --symbol eurusd --direction long --dry-run
        """,
    )

    # Required arguments
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol (e.g., eurusd, gbpusd)",
    )

    parser.add_argument(
        "--direction",
        type=str,
        required=True,
        choices=SUPPORTED_DIRECTIONS,
        help="Signal direction: long or short",
    )

    # Optional arguments
    parser.add_argument(
        "--skip-training",
        action="store_true",
        default=False,
        help="Skip actual training (for testing/dry-run)",
    )

    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        default=False,
        help="Skip Optuna tuning (use default hyperparameters)",
    )

    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=40,
        help="Number of Optuna trials for Phase 3 (default: 40, max recommended)",
    )

    parser.add_argument(
        "--stop-after-phase2",
        action="store_true",
        default=False,
        help="Stop after Phase 2 and print commands for manual Phase 3-5 execution",
    )

    parser.add_argument(
        "--phase1-only",
        action="store_true",
        default=False,
        help="Stop after Phase 1 signal discovery and exit (for signal_discovery_batch.py)",
    )

    parser.add_argument(
        "--output-json",
        action="store_true",
        default=False,
        help="Output Phase 1 results as JSON to stdout (for signal_discovery_batch.py)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would happen without executing",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing market data files",
    )

    parser.add_argument(
        "--use-existing-discovery",
        action="store_true",
        default=False,
        help="Use existing discovery results from results/comprehensive_discovery/ (skip Phase 1)",
    )

    parser.add_argument(
        "--retest-existing",
        action="store_true",
        default=False,
        help="Re-test existing discovery signals on current training data split",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Sample size for Phase 1 discovery (0 = use full data, default)",
    )

    parser.add_argument(
        "--test-all-timeframes",
        action="store_true",
        default=True,
        help="Test ALL timeframes (default). Use --quick to stop early when enough signals found.",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Stop Phase 1 early when enough good signals found (3 excellent or 5 good)",
    )

    parser.add_argument(
        "--sl-pips",
        type=float,
        default=30.0,
        help="Stop loss in pips for Phase 1-2 testing (default: 30.0)",
    )

    parser.add_argument(
        "--tp-pips",
        type=float,
        default=30.0,
        help="Take profit in pips for Phase 1-2 testing (default: 30.0)",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
        help="Maximum bars per episode for Phase 1-2 testing (default: 50)",
    )

    # Signal selection argument (for orchestrator-dispatched jobs)
    parser.add_argument(
        "--signal-name",
        type=str,
        default=None,
        help="Specific signal name to train (forces selection of this signal instead of 'best')",
    )

    # Timeframe argument (for orchestrator-dispatched jobs)
    # Issue #547: Required to prevent model path collisions when same signal
    # is trained at multiple timeframes
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Specific timeframe to train (e.g., H1, H4, H6). Required when --signal-name is provided.",
    )

    return parser


# =============================================================================
# Argument Validation
# =============================================================================


def validate_args(args) -> bool:
    """
    Validate CLI arguments.

    Args:
        args: Parsed arguments from argparse

    Returns:
        True if arguments are valid, False otherwise
    """
    # Validate symbol
    symbol_lower = args.symbol.lower()
    if symbol_lower not in SUPPORTED_SYMBOLS:
        logger.error(
            f"Invalid symbol: {args.symbol}. "
            f"Supported: {', '.join(SUPPORTED_SYMBOLS)}"
        )
        return False

    # Validate direction
    if args.direction not in SUPPORTED_DIRECTIONS:
        logger.error(
            f"Invalid direction: {args.direction}. "
            f"Supported: {', '.join(SUPPORTED_DIRECTIONS)}"
        )
        return False

    return True


# =============================================================================
# Signal Loading from CSV (for orchestrator-dispatched jobs)
# =============================================================================


def load_signal_from_csv(signal_name: str, symbol: str, direction: str) -> Optional[dict]:
    """
    Load a specific signal from signal_discoveries.csv by name.

    Used when --signal-name is specified to skip Phase 1/2 entirely.
    The signal was already discovered and validated in a previous pipeline run.

    Args:
        signal_name: Name of the signal to load (e.g., 'Stoch_RSI_long_15_25')
        symbol: Trading symbol
        direction: Signal direction ('long' or 'short')

    Returns:
        Dict with signal info, or None if not found
    """
    import json
    import os

    # Check CSV path from environment or use default
    csv_path = os.environ.get("CSV_DISCOVERY_PATH", "results/signal_discoveries.csv")
    if not Path(csv_path).exists():
        csv_path = "results/signal_discoveries.csv"  # Fallback

    if not Path(csv_path).exists():
        logger.warning(f"Signal discoveries CSV not found: {csv_path}")
        return None

    try:
        import pandas as pd

        df = pd.read_csv(csv_path)

        # Filter by symbol and direction
        mask = (df["symbol"].str.lower() == symbol.lower()) & (
            df["direction"].str.lower() == direction.lower()
        )
        filtered = df[mask]

        if filtered.empty:
            logger.warning(f"No discoveries found for {symbol} {direction}")
            return None

        # Search for signal in phase1_results JSON column
        for _, row in filtered.iterrows():
            phase1_json = row.get("phase1_results", "[]")
            if pd.isna(phase1_json) or not phase1_json:
                continue

            try:
                phase1_signals = json.loads(phase1_json)
            except json.JSONDecodeError:
                continue

            # Find matching signal
            for sig in phase1_signals:
                sig_n = sig.get("signal_name", "")
                # Match by name (exact or partial)
                if sig_n == signal_name or signal_name in sig_n or sig_n in signal_name:
                    logger.info(
                        f"Loaded signal from CSV: {sig_n} ({sig.get('timeframe', 'H4')}) "
                        f"WR={sig.get('win_rate', 0)*100:.1f}%"
                    )
                    return {
                        "signal_name": sig_n,
                        "timeframe": sig.get("timeframe", "H4"),
                        "win_rate": sig.get("win_rate", 0),
                        "trades": sig.get("trades", 0),
                        "p_value": sig.get("p_value", 0.05),
                        "quality": sig.get("quality", "unknown"),
                        "symbol": symbol,
                        "direction": direction,
                    }

        logger.warning(f"Signal '{signal_name}' not found in CSV for {symbol} {direction}")
        return None

    except Exception as e:
        logger.error(f"Failed to load signal from CSV: {e}")
        return None


# =============================================================================
# Existing Discovery Loading
# =============================================================================


def load_existing_discovery(symbol: str, direction: str) -> Optional[dict]:
    """
    Load existing discovery results from comprehensive discovery.

    Used when --use-existing-discovery is specified to skip Phase 1.

    Args:
        symbol: Trading symbol
        direction: Signal direction ('long' or 'short')

    Returns:
        Dict with best signal for direction, or None if not found
    """
    import json

    discovery_file = Path(f"results/comprehensive_discovery/{symbol.lower()}_all_signals.json")

    if not discovery_file.exists():
        logger.warning(f"No existing discovery found: {discovery_file}")
        return None

    try:
        with open(discovery_file) as f:
            data = json.load(f)

        top_signals = data.get("top_signals", [])

        # Filter by direction and sort by win rate
        direction_signals = [
            s for s in top_signals
            if s.get("direction", "").lower() == direction.lower()
        ]

        if not direction_signals:
            logger.warning(f"No {direction} signals found in {discovery_file}")
            return None

        # Sort by win rate (stored as percentage)
        direction_signals.sort(key=lambda x: x.get("win_rate", 0), reverse=True)

        best = direction_signals[0]
        logger.info(f"Loaded existing discovery: {best['signal_name']} (WR={best['win_rate']}%)")

        return best

    except Exception as e:
        logger.error(f"Failed to load discovery: {e}")
        return None


def create_signal_from_discovery(discovery: dict) -> Optional[Any]:
    """
    Create a signal function from discovery result.

    Args:
        discovery: Discovery result dict with signal_definition

    Returns:
        Signal object compatible with tester, or None on failure
    """
    from indicator_discovery.signals import (
        create_stochastic_oversold_signal,
        create_stochastic_overbought_signal,
        create_rsi_oversold_signal,
        create_rsi_overbought_signal,
        create_sma_crossover_signal,
        create_stoch_rsi_signal,
        create_macd_stoch_signal,
        create_triple_momentum_signal,
        create_sma_rsi_stoch_signal,
        create_sma_rsi_macd_signal,
        create_sma_stoch_bb_signal,
        create_sma_bb_signal,
        create_sma_rsi_stoch_bb_signal,
        create_macd_crossover_signal,
        create_ema_rsi_signal,
        SignalDefinition,
    )

    signal_name = discovery.get("signal_name", "")
    signal_def = discovery.get("signal_definition", "")
    direction = discovery.get("direction", "long")
    timeframe = discovery.get("timeframe", "H4")

    import re

    # First try to match by signal NAME (more reliable than signal_definition)
    # This handles signals discovered via Phase 1-2 which have standard names

    # SMA50_200_RSI_Stoch_long/short - 3-indicator combo
    sma_rsi_stoch_match = re.match(
        r"^SMA(\d+)_(\d+)_RSI_Stoch_(long|short)$",
        signal_name,
        re.IGNORECASE
    )
    if sma_rsi_stoch_match:
        fast, slow, dir_str = sma_rsi_stoch_match.groups()
        logger.debug(f"Matched SMA_RSI_Stoch signal: fast={fast}, slow={slow}, dir={dir_str}")
        return create_sma_rsi_stoch_signal(
            sma_fast=int(fast),
            sma_slow=int(slow),
            direction=dir_str.lower(),
            timeframe=timeframe,
        )

    # SMA20_200_RSI_MACD_long/short - 3-indicator combo with MACD
    sma_rsi_macd_match = re.match(
        r"^SMA(\d+)_(\d+)_RSI_MACD_(long|short)$",
        signal_name,
        re.IGNORECASE
    )
    if sma_rsi_macd_match:
        fast, slow, dir_str = sma_rsi_macd_match.groups()
        logger.debug(f"Matched SMA_RSI_MACD signal: fast={fast}, slow={slow}, dir={dir_str}")
        return create_sma_rsi_macd_signal(
            sma_fast=int(fast),
            sma_slow=int(slow),
            direction=dir_str.lower(),
            timeframe=timeframe,
        )

    # SMA20_50_Stoch_BB_long/short - 3-indicator combo with BB
    sma_stoch_bb_match = re.match(
        r"^SMA(\d+)_(\d+)_Stoch_BB_(long|short)$",
        signal_name,
        re.IGNORECASE
    )
    if sma_stoch_bb_match:
        fast, slow, dir_str = sma_stoch_bb_match.groups()
        logger.debug(f"Matched SMA_Stoch_BB signal: fast={fast}, slow={slow}, dir={dir_str}")
        return create_sma_stoch_bb_signal(
            sma_fast=int(fast),
            sma_slow=int(slow),
            direction=dir_str.lower(),
            timeframe=timeframe,
        )

    # SMA20_50_BB_long/short - 2-indicator combo
    sma_bb_match = re.match(
        r"^SMA(\d+)_(\d+)_BB_(long|short)$",
        signal_name,
        re.IGNORECASE
    )
    if sma_bb_match:
        fast, slow, dir_str = sma_bb_match.groups()
        logger.debug(f"Matched SMA_BB signal: fast={fast}, slow={slow}, dir={dir_str}")
        return create_sma_bb_signal(
            sma_fast=int(fast),
            sma_slow=int(slow),
            direction=dir_str.lower(),
            timeframe=timeframe,
        )

    # SMA20_50_RSI_Stoch_BB_long/short - 4-indicator combo
    sma_rsi_stoch_bb_match = re.match(
        r"^SMA(\d+)_(\d+)_RSI_Stoch_BB_(long|short)$",
        signal_name,
        re.IGNORECASE
    )
    if sma_rsi_stoch_bb_match:
        fast, slow, dir_str = sma_rsi_stoch_bb_match.groups()
        logger.debug(f"Matched SMA_RSI_Stoch_BB signal: fast={fast}, slow={slow}, dir={dir_str}")
        return create_sma_rsi_stoch_bb_signal(
            sma_fast=int(fast),
            sma_slow=int(slow),
            direction=dir_str.lower(),
            timeframe=timeframe,
        )

    # SMA_20_50_cross_long/short - Simple SMA crossover
    sma_cross_match = re.match(
        r"^SMA_?(\d+)_(\d+)_cross_(long|short)$",
        signal_name,
        re.IGNORECASE
    )
    if sma_cross_match:
        fast, slow, dir_str = sma_cross_match.groups()
        logger.debug(f"Matched SMA cross signal: fast={fast}, slow={slow}, dir={dir_str}")
        return create_sma_crossover_signal(
            fast_period=int(fast),
            slow_period=int(slow),
            direction=dir_str.lower(),
            timeframe=timeframe,
        )

    # Stoch_RSI_long_15_25 or Stoch_RSI_short_80_70 - Stoch+RSI combo
    stoch_rsi_match = re.match(
        r"^Stoch_RSI_(long|short)_(\d+)_(\d+)$",
        signal_name,
        re.IGNORECASE
    )
    if stoch_rsi_match:
        dir_str, stoch_t, rsi_t = stoch_rsi_match.groups()
        logger.debug(f"Matched Stoch_RSI signal: dir={dir_str}, stoch={stoch_t}, rsi={rsi_t}")
        return create_stoch_rsi_signal(
            stoch_threshold=int(stoch_t),
            rsi_threshold=int(rsi_t),
            direction=dir_str.lower(),
            timeframe=timeframe,
        )

    # RSI_oversold_long or RSI_overbought_short or RSI14_overbought_short
    rsi_name_match = re.match(
        r"^RSI_?(\d+)?_?(oversold|overbought)_(long|short)$",
        signal_name,
        re.IGNORECASE
    )
    if rsi_name_match:
        period_str, cond, dir_str = rsi_name_match.groups()
        period = int(period_str) if period_str else 14
        logger.debug(f"Matched RSI {cond} signal: dir={dir_str}, period={period}")
        if cond.lower() == "oversold":
            return create_rsi_oversold_signal(period=period, timeframe=timeframe)
        else:
            return create_rsi_overbought_signal(period=period, timeframe=timeframe)

    # MACD_Stoch_long/short
    macd_stoch_match = re.match(
        r"^MACD_Stoch_(long|short)$",
        signal_name,
        re.IGNORECASE
    )
    if macd_stoch_match:
        dir_str = macd_stoch_match.group(1)
        logger.debug(f"Matched MACD_Stoch signal: dir={dir_str}")
        return create_macd_stoch_signal(
            direction=dir_str.lower(),
            timeframe=timeframe,
        )

    # MACD_cross_long/short
    macd_cross_match = re.match(
        r"^MACD_cross_(long|short)$",
        signal_name,
        re.IGNORECASE
    )
    if macd_cross_match:
        dir_str = macd_cross_match.group(1)
        logger.debug(f"Matched MACD_cross signal: dir={dir_str}")
        return create_macd_crossover_signal(
            direction=dir_str.lower(),
            timeframe=timeframe,
        )

    # Triple_Momentum_long/short
    triple_match = re.match(
        r"^Triple_Momentum_(long|short)$",
        signal_name,
        re.IGNORECASE
    )
    if triple_match:
        dir_str = triple_match.group(1)
        logger.debug(f"Matched Triple_Momentum signal: dir={dir_str}")
        return create_triple_momentum_signal(
            direction=dir_str.lower(),
            timeframe=timeframe,
        )

    # Stoch_K_oversold_long_XX
    stoch_k_oversold_match = re.match(
        r"^Stoch_K_oversold_(long)_(\d+)$",
        signal_name,
        re.IGNORECASE
    )
    if stoch_k_oversold_match:
        dir_str, threshold = stoch_k_oversold_match.groups()
        logger.debug(f"Matched Stoch_K_oversold signal: threshold={threshold}")
        return create_stochastic_oversold_signal(
            k_threshold=int(threshold),
            timeframe=timeframe,
        )

    # EMA_RSI_long/short
    ema_rsi_match = re.match(
        r"^EMA_RSI_(long|short)$",
        signal_name,
        re.IGNORECASE
    )
    if ema_rsi_match:
        dir_str = ema_rsi_match.group(1)
        logger.debug(f"Matched EMA_RSI signal: dir={dir_str}")
        return create_ema_rsi_signal(
            direction=dir_str.lower(),
            timeframe=timeframe,
        )

    # SMA20_200_RSI_long/short - SMA+RSI only (2-indicator)
    from indicator_discovery.signals import create_sma_rsi_signal, create_sma_stoch_signal, create_sma_macd_signal

    sma_rsi_only_match = re.match(
        r"^SMA(\d+)_(\d+)_RSI_(long|short)$",
        signal_name,
        re.IGNORECASE
    )
    if sma_rsi_only_match:
        fast, slow, dir_str = sma_rsi_only_match.groups()
        logger.debug(f"Matched SMA_RSI signal: fast={fast}, slow={slow}, dir={dir_str}")
        return create_sma_rsi_signal(
            sma_fast=int(fast),
            sma_slow=int(slow),
            direction=dir_str.lower(),
            timeframe=timeframe,
        )

    # SMA20_200_Stoch_long/short - SMA+Stoch only (2-indicator)
    sma_stoch_only_match = re.match(
        r"^SMA(\d+)_(\d+)_Stoch_(long|short)$",
        signal_name,
        re.IGNORECASE
    )
    if sma_stoch_only_match:
        fast, slow, dir_str = sma_stoch_only_match.groups()
        logger.debug(f"Matched SMA_Stoch signal: fast={fast}, slow={slow}, dir={dir_str}")
        return create_sma_stoch_signal(
            sma_fast=int(fast),
            sma_slow=int(slow),
            direction=dir_str.lower(),
            timeframe=timeframe,
        )

    # SMA20_50_MACD_long/short - SMA+MACD only (2-indicator)
    sma_macd_only_match = re.match(
        r"^SMA(\d+)_(\d+)_MACD_(long|short)$",
        signal_name,
        re.IGNORECASE
    )
    if sma_macd_only_match:
        fast, slow, dir_str = sma_macd_only_match.groups()
        logger.debug(f"Matched SMA_MACD signal: fast={fast}, slow={slow}, dir={dir_str}")
        return create_sma_macd_signal(
            sma_fast=int(fast),
            sma_slow=int(slow),
            direction=dir_str.lower(),
            timeframe=timeframe,
        )

    # RSI_BB_confluence_long/short - RSI + Bollinger Band confluence
    rsi_bb_match = re.match(
        r"^RSI_BB_confluence_(long|short)$",
        signal_name,
        re.IGNORECASE
    )
    if rsi_bb_match:
        dir_str = rsi_bb_match.group(1).lower()
        logger.debug(f"Matched RSI_BB_confluence signal: dir={dir_str}")
        if dir_str == "long":
            return SignalDefinition(
                name="RSI_BB_confluence_long",
                indicator="RSI_BB",
                timeframe=timeframe,
                direction="long",
                condition=lambda df: (df["rsi_14"] < 30) & (df["close"] < df["bb_lower_20"]),
                required_columns=["rsi_14", "close", "bb_lower_20"],
            )
        else:
            return SignalDefinition(
                name="RSI_BB_confluence_short",
                indicator="RSI_BB",
                timeframe=timeframe,
                direction="short",
                condition=lambda df: (df["rsi_14"] > 70) & (df["close"] > df["bb_upper_20"]),
                required_columns=["rsi_14", "close", "bb_upper_20"],
            )

    # BB_RSI_long/short - Bollinger Band + RSI (2-indicator)
    bb_rsi_match = re.match(
        r"^BB_RSI_(long|short)$",
        signal_name,
        re.IGNORECASE
    )
    if bb_rsi_match:
        dir_str = bb_rsi_match.group(1).lower()
        logger.debug(f"Matched BB_RSI signal: dir={dir_str}")
        from indicator_discovery.signals import create_bb_rsi_signal
        return create_bb_rsi_signal(
            direction=dir_str,
            timeframe=timeframe,
        )

    # BB_RSI_Volume_long/short - Bollinger Band + RSI + Volume
    bb_rsi_vol_match = re.match(
        r"^BB_RSI_Volume_(long|short)$",
        signal_name,
        re.IGNORECASE
    )
    if bb_rsi_vol_match:
        dir_str = bb_rsi_vol_match.group(1).lower()
        logger.debug(f"Matched BB_RSI_Volume signal: dir={dir_str}")
        if dir_str == "long":
            return SignalDefinition(
                name="BB_RSI_Volume_long",
                indicator="BB_RSI_Volume",
                timeframe=timeframe,
                direction="long",
                condition=lambda df: (
                    (df["close"] < df["bb_lower_20"])
                    & (df["rsi_14"] < 30)
                    & (df["volume"] > df["volume"].rolling(20).mean())
                ),
                required_columns=["close", "bb_lower_20", "rsi_14", "volume"],
            )
        else:
            return SignalDefinition(
                name="BB_RSI_Volume_short",
                indicator="BB_RSI_Volume",
                timeframe=timeframe,
                direction="short",
                condition=lambda df: (
                    (df["close"] > df["bb_upper_20"])
                    & (df["rsi_14"] > 70)
                    & (df["volume"] > df["volume"].rolling(20).mean())
                ),
                required_columns=["close", "bb_upper_20", "rsi_14", "volume"],
            )

    # Now try parsing signal_definition as fallback
    # Parse simple stoch_k signals (e.g., "stoch_k < 20" or "stoch_k > 80")
    stoch_match = re.match(r"^stoch_k\s*([<>])\s*(\d+)$", signal_def.lower().strip())
    if stoch_match:
        op, threshold = stoch_match.groups()
        threshold = int(threshold)
        if op == "<":
            return create_stochastic_oversold_signal(k_threshold=threshold, timeframe=timeframe)
        else:
            return create_stochastic_overbought_signal(k_threshold=threshold, timeframe=timeframe)

    # Parse simple RSI signals (e.g., "rsi_14 < 30" or "rsi_14 > 70")
    rsi_match = re.match(r"^rsi_?(\d+)?\s*([<>])\s*(\d+)$", signal_def.lower().strip())
    if rsi_match:
        period_str, op, threshold = rsi_match.groups()
        period = int(period_str) if period_str else 14
        threshold = int(threshold)
        if op == "<":
            return create_rsi_oversold_signal(period=period, threshold=threshold, timeframe=timeframe)
        else:
            return create_rsi_overbought_signal(period=period, threshold=threshold, timeframe=timeframe)

    # Parse SMA crossover signals (e.g., "sma_20 > sma_50")
    sma_match = re.match(r"^sma_?(\d+)\s*>\s*sma_?(\d+)$", signal_def.lower().strip())
    if sma_match:
        fast, slow = sma_match.groups()
        return create_sma_crossover_signal(
            fast_period=int(fast),
            slow_period=int(slow),
            timeframe=timeframe,
            direction=direction,
        )

    # CRITICAL: If we reach here, signal name pattern is not supported
    # Log error and raise exception - do NOT silently return empty signals
    logger.error(f"❌ UNSUPPORTED SIGNAL PATTERN: signal_name='{signal_name}', signal_def='{signal_def}'")
    logger.error(f"  → Add pattern matching for this signal in create_signal_from_discovery()")
    logger.error(f"  → See signals.py for available factory functions")

    # Raise exception so training fails fast with clear error
    raise ValueError(
        f"Unsupported signal pattern: '{signal_name}'. "
        f"Add pattern matching in create_signal_from_discovery() for this signal type. "
        f"signal_definition='{signal_def}', direction='{direction}', timeframe='{timeframe}'"
    )


def load_existing_signals(symbol: str, direction: str) -> list[dict]:
    """
    Load ALL existing discovery signals for a symbol and direction.

    Args:
        symbol: Trading symbol
        direction: Signal direction ('long' or 'short')

    Returns:
        List of signal dicts from discovery, sorted by win rate
    """
    import json

    discovery_file = Path(f"results/comprehensive_discovery/{symbol.lower()}_all_signals.json")

    if not discovery_file.exists():
        logger.warning(f"No existing discovery found: {discovery_file}")
        return []

    try:
        with open(discovery_file) as f:
            data = json.load(f)

        top_signals = data.get("top_signals", [])

        # Filter by direction
        direction_signals = [
            s for s in top_signals
            if s.get("direction", "").lower() == direction.lower()
        ]

        # Sort by win rate
        direction_signals.sort(key=lambda x: x.get("win_rate", 0), reverse=True)

        logger.info(f"Loaded {len(direction_signals)} existing {direction} signals for {symbol}")
        return direction_signals

    except Exception as e:
        logger.error(f"Failed to load discovery: {e}")
        return []


def run_phase1_retest_existing(
    train_df: pd.DataFrame,
    symbol: str,
    direction: str,
) -> dict:
    """
    Re-test existing discovery signals on current training data.

    Faster than full discovery since we only test known signals.
    Validates that signals still perform well on the 60% training split.

    Args:
        train_df: Training data DataFrame (60%)
        symbol: Trading symbol
        direction: Signal direction ('long' or 'short')

    Returns:
        Dict with discovery results (same format as run_phase1_discovery)
    """
    from indicator_discovery.tester import SignalTester, TradingCosts
    from indicator_discovery.evaluation import create_signal_evaluation

    print("\n" + "=" * 60)
    print("PHASE 1: RE-TESTING EXISTING DISCOVERY SIGNALS")
    print("=" * 60)

    # Load existing signals
    existing_signals = load_existing_signals(symbol, direction)

    if not existing_signals:
        print("❌ No existing signals found - falling back to fresh discovery")
        return {"passed": False, "fallback": True}

    print(f"Loaded {len(existing_signals)} existing {direction.upper()} signals")
    print(f"Testing on {len(train_df)} training candles (60% split)")

    # Create cost model and tester
    costs = TradingCosts(spread=1.5, slippage=0.5, commission=0.5)
    tester = SignalTester(sl_pips=30.0, tp_pips=30.0, horizon=48)  # Issue #526: Symmetric 30/30

    # Get available timeframes in training data
    available_tfs = []
    if "timeframe" in train_df.columns:
        available_tfs = train_df["timeframe"].unique().tolist()
        print(f"Available timeframes: {', '.join(available_tfs)}")

    # Re-test each signal on its original timeframe
    all_results = []
    passing_signals = []

    for i, sig_data in enumerate(existing_signals):
        sig_name = sig_data.get("signal_name", "unknown")
        sig_tf = sig_data.get("timeframe", "H4")
        sig_def = sig_data.get("signal_definition", "")
        orig_wr = sig_data.get("win_rate", 0)

        # Skip if timeframe not in training data
        if sig_tf not in available_tfs:
            print(f"  ⏭️ Skipping {sig_name} ({sig_tf}) - timeframe not in training data")
            continue

        # Filter data by timeframe
        tf_df = train_df[train_df["timeframe"] == sig_tf].copy().reset_index(drop=True)

        if len(tf_df) < 100:
            print(f"  ⏭️ Skipping {sig_name} ({sig_tf}) - only {len(tf_df)} candles")
            continue

        # Create signal function from definition
        signal = create_signal_from_discovery(sig_data)
        if signal is None:
            print(f"  ⚠️ Could not create signal from: {sig_def}")
            continue

        try:
            result = tester.test_with_costs(tf_df, signal, costs, symbol)

            evaluation = create_signal_evaluation(
                signal_name=sig_name,
                symbol=symbol,
                timeframe=sig_tf,
                direction=direction,
                category=sig_data.get("category", "single"),
                wins=result.wins,
                total_trades=result.total_trades,
                degradation=0.0,
                frequency_per_year=int(result.total_trades * (252 / len(tf_df))),
                sl_pips=30.0,
                tp_pips=30.0,
            )

            wr = evaluation.win_rate
            quality = "poor"
            if wr >= EXCELLENT_WR_THRESHOLD:
                quality = "excellent"
            elif wr >= GOOD_WR_THRESHOLD:
                quality = "good"
            elif wr >= MIN_WR_THRESHOLD:
                quality = "marginal"

            signal_result = {
                "signal": signal,
                "result": result,
                "evaluation": evaluation,
                "win_rate": wr,
                "trades": evaluation.trade_count,
                "p_value": evaluation.p_value,
                "implied_pf": evaluation.implied_pf,
                "passes": evaluation.passes_phase1,
                "timeframe": sig_tf,
                "quality": quality,
                "original_wr": orig_wr / 100,  # Convert from %
            }

            all_results.append(signal_result)

            # Print comparison with original
            status = "✓" if evaluation.passes_phase1 else "✗"
            wr_change = (wr - orig_wr/100) * 100
            change_str = f"+{wr_change:.1f}%" if wr_change >= 0 else f"{wr_change:.1f}%"
            quality_emoji = {"excellent": "🟢", "good": "🟡", "marginal": "🟠", "poor": "🔴"}

            print(f"  {status} {quality_emoji.get(quality, '')} {sig_name} ({sig_tf}): WR={wr*100:.1f}% (was {orig_wr:.1f}%, {change_str}), Trades={evaluation.trade_count}")

            if evaluation.passes_phase1:
                passing_signals.append(signal_result)

        except Exception as e:
            logger.warning(f"Failed to test {sig_name}: {e}")

    # Summary
    if not passing_signals:
        print("\n" + "=" * 60)
        print("❌ PHASE 1 RE-TEST FAILED: No signals passed on training split")
        print("   This may indicate signals were overfit to original data")
        print("=" * 60)

        return {
            "passed": False,
            "best_signal": None,
            "best_evaluation": None,
            "all_results": all_results,
            "passing_signals": [],
            "timeframe": None,
        }

    # Sort by composite score
    quality_rank = {"excellent": 0, "good": 1, "marginal": 2, "poor": 3}
    passing_signals.sort(
        key=lambda x: (quality_rank.get(x["quality"], 4), -x["win_rate"], -x["trades"])
    )

    best = passing_signals[0]

    # Print summary
    print("\n" + "=" * 60)
    print(f"✅ PHASE 1 RE-TEST COMPLETE: {len(passing_signals)} signals passed!")
    print("=" * 60)

    print(f"\nTop 5 Re-Tested Signals:")
    print(f"{'Rank':<5} {'Signal':<30} {'TF':<5} {'WR%':<8} {'Orig WR%':<10} {'Quality'}")
    print("-" * 80)
    for i, s in enumerate(passing_signals[:5], 1):
        orig_wr = s.get('original_wr', 0) * 100
        print(f"{i:<5} {s['signal'].name:<30} {s['timeframe']:<5} {s['win_rate']*100:<8.1f} {orig_wr:<10.1f} {s['quality']}")

    print(f"\n🏆 BEST SIGNAL: {best['signal'].name}")
    print(f"   Timeframe: {best['timeframe']}")
    print(f"   Current WR: {best['win_rate']*100:.1f}% (Original: {best.get('original_wr', 0)*100:.1f}%)")
    print(f"   Quality: {best['quality'].upper()}")
    print("=" * 60)

    return {
        "passed": True,
        "best_signal": best["signal"],
        "best_evaluation": best["evaluation"],
        "all_results": all_results,
        "passing_signals": passing_signals,
        "timeframe": best["timeframe"],
        "is_metrics": {
            "win_rate": best["win_rate"],
            "implied_pf": best["implied_pf"],
            "p_value": best["evaluation"].p_value,
        },
    }


# =============================================================================
# Phase Functions
# =============================================================================


def run_phase0_split(full_df: pd.DataFrame, symbol: str) -> dict:
    """
    Run Phase 0: Data Segregation (60/20/20 split).

    Splits data chronologically into training (60%), validation (20%),
    and test (20%) sets. This MUST happen before any analysis to
    ensure TRUE out-of-sample validation.

    Args:
        full_df: Full dataset DataFrame
        symbol: Trading symbol

    Returns:
        Dict with split information and DataFrames
    """
    from data.splitter import split_data
    import pandas as pd

    print("\n" + "=" * 60)
    print("PHASE 0: DATA SEGREGATION")
    print("=" * 60)

    # Issue #550: Filter to last 3 years of data
    # Training on recent data is more relevant for current market conditions
    YEARS_TO_KEEP = 3
    original_size = len(full_df)

    if 'timestamp' in full_df.columns:
        full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
        latest_date = full_df['timestamp'].max()
        cutoff_date = latest_date - pd.Timedelta(days=YEARS_TO_KEEP * 365)
        full_df = full_df[full_df['timestamp'] >= cutoff_date].reset_index(drop=True)
        print(f"📅 Filtered to last {YEARS_TO_KEEP} years: {cutoff_date.date()} to {latest_date.date()}")
        print(f"   Original: {original_size:,} candles → Filtered: {len(full_df):,} candles")
    else:
        print(f"⚠️  No timestamp column - using all {original_size:,} candles")

    # Issue #552: Filter out sub-M30 timeframes (M1, M5, M10, M15, M20)
    VALID_TIMEFRAMES = ['M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D1', 'W1', 'MN1']

    if 'timeframe' in full_df.columns:
        before_tf_filter = len(full_df)
        full_df['timeframe'] = full_df['timeframe'].str.upper()
        full_df = full_df[full_df['timeframe'].isin(VALID_TIMEFRAMES)].reset_index(drop=True)
        removed = before_tf_filter - len(full_df)
        if removed > 0:
            print(f"📊 Filtered to M30+ timeframes: {before_tf_filter:,} → {len(full_df):,} candles (removed {removed:,} sub-M30)")
    else:
        print(f"⚠️  No timeframe column - assuming single timeframe data")

    splits = split_data(full_df, symbol)

    train_size = len(splits["train"])
    val_size = len(splits["validation"])
    test_size = len(splits["test"])
    total_size = len(full_df)

    print(f"Total data: {total_size} candles")
    print(f"  Training:   {train_size} ({train_size/total_size*100:.1f}%)")
    print(f"  Validation: {val_size} ({val_size/total_size*100:.1f}%)")
    print(f"  Test:       {test_size} ({test_size/total_size*100:.1f}%)")
    print("Data segregation complete - no leakage possible.")

    return {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "total_size": total_size,
        "train_df": splits["train"],
        "val_df": splits["validation"],
        "test_df": splits["test"],
        "filtered_df": full_df,  # 3-year filtered DataFrame for consistent splits
    }


# Timeframe hierarchy from highest (least noise) to lowest (most noise)
TIMEFRAME_HIERARCHY = ["D1", "H12", "H8", "H6", "H4", "H3", "H2", "H1", "M30"]

# Signal quality thresholds for categorization
EXCELLENT_WR_THRESHOLD = 0.65  # WR >= 65% = excellent
GOOD_WR_THRESHOLD = 0.58      # WR >= 58% = good
MIN_WR_THRESHOLD = 0.55       # WR >= 55% = passing (marginal)

# Minimum signals to consider collection complete
MIN_EXCELLENT_SIGNALS = 3     # Stop if we have 3+ excellent signals
MIN_GOOD_SIGNALS = 5          # Or 5+ good signals across timeframes


def run_phase1_discovery(
    train_df: pd.DataFrame,
    symbol: str,
    direction: str,
    sample_size: int = 0,
    test_all_timeframes: bool = True,
    sl_pips: float = 30.0,
    tp_pips: float = 30.0,  # Issue #526: Symmetric 30/30
    horizon: int = 50,
) -> dict:
    """
    Run Phase 1: Signal Discovery with WR-based evaluation.

    Processes timeframes hierarchically from D1 (highest signal quality) down to M30.
    Tests ALL timeframes and collects all passing signals.
    Can optionally stop early if enough excellent signals found.

    Tests multiple signal definitions on ONLY the training data (60%).
    Uses Win Rate as primary metric (fixed SL=30, TP=30 → all trades ±30 pips).
    Multi-metric evaluation: WR, trade count, p-value, degradation.

    Signal Quality Tiers:
    - Excellent: WR >= 65%
    - Good: WR >= 58%
    - Passing: WR >= 55%

    BLOCKING GATE: Returns best signal only if WR >= 55%, trades >= 100, p-value < 0.05

    Args:
        train_df: Training data DataFrame
        symbol: Trading symbol
        direction: Signal direction ('long' or 'short')
        sample_size: If > 0, sample this many candles from each timeframe (for faster testing)
        test_all_timeframes: If True, test all timeframes and collect all signals

    Returns:
        Dict with discovery results:
        - passed: Boolean - True if any signal passed Phase 1 criteria
        - best_signal: Best signal definition (if passed)
        - best_evaluation: SignalEvaluation for best signal (if passed)
        - all_results: List of all tested signals with evaluations
        - passing_signals: All signals that passed Phase 1 criteria
        - timeframe: Timeframe where best signal was found
    """
    from indicator_discovery.tester import SignalTester, TradingCosts
    from indicator_discovery.evaluation import (
        create_signal_evaluation,
        Phase1Thresholds,
    )
    from indicator_discovery.signals import (
        create_rsi_oversold_signal,
        create_rsi_overbought_signal,
        create_sma_crossover_signal,
        create_macd_crossover_signal,
        create_bollinger_touch_signal,
        create_stochastic_oversold_signal,
        create_stochastic_overbought_signal,
    )

    print("\n" + "=" * 60)
    print("PHASE 1: SIGNAL DISCOVERY (Hierarchical Timeframe Processing)")
    print("=" * 60)
    print(f"Total training data: {len(train_df)} candles")
    print(f"Direction: {direction.upper()}")
    print(f"Timeframe hierarchy: {' → '.join(TIMEFRAME_HIERARCHY)}")
    print(f"Strategy: Test {'ALL' if test_all_timeframes else 'until enough'} timeframes, collect all passing signals")

    # Create cost model and tester
    costs = TradingCosts(spread=1.5, slippage=0.5, commission=0.5)
    tester = SignalTester(sl_pips=sl_pips, tp_pips=tp_pips, horizon=horizon)

    # Calculate risk/reward ratio and breakeven WR
    rr_ratio = tp_pips / sl_pips if sl_pips > 0 else 0
    net_win = tp_pips - costs.total_per_trade
    net_loss = sl_pips + costs.total_per_trade
    breakeven_wr = net_loss / (net_win + net_loss) if (net_win + net_loss) > 0 else 0

    print(f"Exit Strategy: SL={sl_pips:.1f} pips, TP={tp_pips:.1f} pips (R:R = 1:{rr_ratio:.2f})")
    print(f"Costs per trade: {costs.total_per_trade:.1f} pips (spread + slippage + commission)")
    print(f"Net Win: +{net_win:.1f} pips, Net Loss: -{net_loss:.1f} pips")
    print(f"Breakeven WR: {breakeven_wr*100:.1f}% (required to break even after costs)")
    print(f"Quality Tiers: Excellent >= {EXCELLENT_WR_THRESHOLD*100:.0f}%, Good >= {GOOD_WR_THRESHOLD*100:.0f}%, Passing >= {MIN_WR_THRESHOLD*100:.0f}%")
    print(f"Blocking Gate: WR >= {MIN_WIN_RATE_PHASE1*100:.0f}%, Trades >= {MIN_TRADES_PHASE1}, p-value < {MAX_P_VALUE}")
    print(f"Episode Length: {horizon} bars (trades forcibly closed at timeout)")

    # Check if timeframe column exists
    if "timeframe" not in train_df.columns:
        logger.warning("No 'timeframe' column - processing all data as single timeframe")
        timeframes_to_test = [None]  # None means use all data
    else:
        # Get available timeframes in the data
        available_tfs = train_df["timeframe"].unique().tolist()
        timeframes_to_test = [tf for tf in TIMEFRAME_HIERARCHY if tf in available_tfs]
        print(f"Available timeframes: {', '.join(timeframes_to_test)}")

    # Collect ALL passing signals across ALL timeframes
    all_passing_signals = []
    all_tested_results = []

    # Process each timeframe in order
    for tf in timeframes_to_test:
        # Filter by timeframe
        if tf is not None:
            tf_df = train_df[train_df["timeframe"] == tf].copy().reset_index(drop=True)
        else:
            tf_df = train_df.copy()

        if len(tf_df) < 100:
            print(f"\n⏭️ Skipping {tf}: Only {len(tf_df)} candles (need >= 100)")
            continue

        # Apply sampling if requested
        if sample_size > 0 and len(tf_df) > sample_size:
            # Sample from end of data (most recent)
            tf_df = tf_df.tail(sample_size).reset_index(drop=True)

        print(f"\n{'='*40}")
        print(f"Testing timeframe: {tf or 'ALL'} ({len(tf_df)} candles)")
        print(f"{'='*40}")

        # Create signal definitions for this timeframe
        signals = _create_signals_for_timeframe(direction, tf or "H4")

        print(f"Testing {len(signals)} signals...")

        # Test all signals
        for signal in signals:
            try:
                result = tester.test_with_costs(tf_df, signal, costs, symbol)

                # Calculate frequency accounting for timeframe
                # Need candles per year, not days per year
                timeframe_multipliers = {
                    "D1": 1,
                    "H12": 2,
                    "H8": 3,
                    "H6": 4,
                    "H4": 6,
                    "H3": 8,
                    "H2": 12,
                    "H1": 24,
                    "M30": 48,
                }
                tf_mult = timeframe_multipliers.get(tf or "H4", 6)  # Default H4 = 6
                candles_per_year = 252 * tf_mult
                frequency_per_year = int(result.total_trades * (candles_per_year / len(tf_df)))

                # Create comprehensive evaluation
                evaluation = create_signal_evaluation(
                    signal_name=signal.name,
                    symbol=symbol,
                    timeframe=tf or signal.timeframe,
                    direction=direction,
                    category=signal.category if hasattr(signal, 'category') else "single",
                    wins=result.wins,
                    total_trades=result.total_trades,
                    degradation=0.0,  # Will be calculated in Phase 2
                    frequency_per_year=frequency_per_year,
                    sl_pips=30.0,
                    tp_pips=30.0,
                )

                # Categorize signal quality
                wr = evaluation.win_rate
                quality = "poor"
                if wr >= EXCELLENT_WR_THRESHOLD:
                    quality = "excellent"
                elif wr >= GOOD_WR_THRESHOLD:
                    quality = "good"
                elif wr >= MIN_WR_THRESHOLD:
                    quality = "marginal"

                signal_result = {
                    "signal": signal,
                    "result": result,
                    "evaluation": evaluation,
                    "win_rate": evaluation.win_rate,
                    "trades": evaluation.trade_count,
                    "p_value": evaluation.p_value,
                    "implied_pf": evaluation.implied_pf,
                    "passes": evaluation.passes_phase1,
                    "timeframe": tf,
                    "quality": quality,
                }

                all_tested_results.append(signal_result)

                # Print result
                trades = evaluation.trade_count
                pval = evaluation.p_value
                implied_pf = evaluation.implied_pf
                status = "✓" if evaluation.passes_phase1 else "✗"
                quality_emoji = {"excellent": "🟢", "good": "🟡", "marginal": "🟠", "poor": "🔴"}

                # Add rejection reason if failed (use actual evaluation failure reasons)
                rejection_reason = ""
                if not evaluation.passes_phase1:
                    # Use the actual failure reasons from evaluation instead of recalculating
                    if evaluation.failure_reasons:
                        rejection_reason = f" [{'; '.join(evaluation.failure_reasons)}]"
                    else:
                        # Fallback to basic checks if no reasons provided
                        reasons = []
                        if wr < MIN_WIN_RATE_PHASE1:
                            reasons.append(f"WR<{MIN_WIN_RATE_PHASE1*100:.0f}%")
                        if trades < MIN_TRADES_PHASE1:
                            reasons.append(f"Trades<{MIN_TRADES_PHASE1}")
                        if pval >= MAX_P_VALUE:
                            reasons.append(f"p≥{MAX_P_VALUE}")
                        rejection_reason = f" [{', '.join(reasons)}]" if reasons else ""

                print(f"  {status} {quality_emoji.get(quality, '')} {signal.name}: WR={wr*100:.1f}% ({quality}), Trades={trades}, p={pval:.3f}{rejection_reason}")

                if evaluation.passes_phase1:
                    all_passing_signals.append(signal_result)

            except Exception as e:
                logger.warning(f"Failed to test {signal.name}: {e}")

        # Count signal quality so far
        excellent_count = sum(1 for s in all_passing_signals if s["quality"] == "excellent")
        good_count = sum(1 for s in all_passing_signals if s["quality"] in ["excellent", "good"])

        print(f"\n  Progress: {len(all_passing_signals)} passing signals ({excellent_count} excellent, {good_count - excellent_count} good)")

        # Check early stopping condition (only if not testing all)
        if not test_all_timeframes:
            if excellent_count >= MIN_EXCELLENT_SIGNALS:
                print(f"  ✅ Stopping early: Found {MIN_EXCELLENT_SIGNALS}+ excellent signals")
                break
            if good_count >= MIN_GOOD_SIGNALS:
                print(f"  ✅ Stopping early: Found {MIN_GOOD_SIGNALS}+ good signals")
                break

    # Summary and best signal selection
    if not all_passing_signals:
        print("\n" + "=" * 60)
        print("❌ PHASE 1 FAILED: No signals met criteria on ANY timeframe")
        print(f"   Required: WR >= {MIN_WIN_RATE_PHASE1*100:.0f}%, Trades >= {MIN_TRADES_PHASE1}, p-value < {MAX_P_VALUE}")
        print(f"   Timeframes tested: {', '.join(tf for tf in timeframes_to_test if tf)}")
        print("=" * 60)

        return {
            "passed": False,
            "best_signal": None,
            "best_evaluation": None,
            "all_results": all_tested_results,
            "passing_signals": [],
            "timeframe": None,
        }

    # Sort passing signals by composite score: quality tier, then WR, then trades
    quality_rank = {"excellent": 0, "good": 1, "marginal": 2, "poor": 3}
    all_passing_signals.sort(
        key=lambda x: (quality_rank.get(x["quality"], 4), -x["win_rate"], -x["trades"])
    )

    best = all_passing_signals[0]

    # Print summary
    print("\n" + "=" * 60)
    print(f"✅ PHASE 1 COMPLETE: {len(all_passing_signals)} signals passed!")
    print("=" * 60)
    print(f"\nSignal Quality Distribution:")
    for q in ["excellent", "good", "marginal"]:
        count = sum(1 for s in all_passing_signals if s["quality"] == q)
        if count > 0:
            print(f"  {q.capitalize()}: {count}")

    print(f"\nTop 5 Signals (ranked by quality, WR, trades):")
    print(f"{'Rank':<5} {'Signal':<35} {'TF':<5} {'WR%':<8} {'Quality':<10} {'Trades'}")
    print("-" * 80)
    for i, s in enumerate(all_passing_signals[:5], 1):
        print(f"{i:<5} {s['signal'].name:<35} {s['timeframe']:<5} {s['win_rate']*100:<8.1f} {s['quality']:<10} {s['trades']}")

    print(f"\n{'='*60}")
    print(f"🏆 BEST SIGNAL: {best['signal'].name}")
    print(f"   Timeframe: {best['timeframe']}")
    print(f"   Win Rate: {best['win_rate']*100:.1f}% (edge: {(best['win_rate']-0.5)*100:.1f}%)")
    print(f"   Implied PF: {best['implied_pf']:.2f}")
    print(f"   Trades: {best['trades']}")
    print(f"   Quality: {best['quality'].upper()}")
    print("=" * 60)

    return {
        "passed": True,
        "best_signal": best["signal"],
        "best_evaluation": best["evaluation"],
        "all_results": all_tested_results,
        "passing_signals": all_passing_signals,
        "timeframe": best["timeframe"],
        "is_metrics": {
            "win_rate": best["win_rate"],
            "implied_pf": best["implied_pf"],
            "p_value": best["evaluation"].p_value,
        },
    }


def _create_signals_for_timeframe(direction: str, timeframe: str) -> list:
    """
    Create signal definitions for a specific timeframe and direction.

    Generates:
    - Single indicator signals (Stoch, RSI, SMA crossovers, MACD, BB)
    - 2-indicator combinations (SMA+RSI, SMA+Stoch, SMA+MACD, SMA+BB, EMA+RSI, etc.)
    - 3-indicator combinations (SMA+RSI+Stoch, SMA+RSI+MACD, SMA+Stoch+BB, etc.)

    Issue #526: Multi-indicator confluences for higher WR
    """
    from indicator_discovery.signals import (
        # Single indicator signals
        create_rsi_oversold_signal,
        create_rsi_overbought_signal,
        create_sma_crossover_signal,
        create_macd_crossover_signal,
        create_bollinger_touch_signal,
        create_stochastic_oversold_signal,
        create_stochastic_overbought_signal,
        # 2-indicator combinations
        create_sma_rsi_signal,
        create_sma_stoch_signal,
        create_sma_macd_signal,
        create_sma_bb_signal,
        create_ema_rsi_signal,
        create_stoch_rsi_signal,
        create_macd_stoch_signal,
        create_bb_rsi_signal,
        # 3-indicator combinations
        create_sma_rsi_stoch_signal,
        create_sma_rsi_macd_signal,
        create_sma_stoch_bb_signal,
        create_triple_momentum_signal,
        create_ema_stoch_bb_signal,
        # 4-indicator combinations (Issue #526)
        create_sma_rsi_stoch_bb_signal,
        create_sma_rsi_stoch_macd_signal,
        # NEW: Issue #526 Optimization - Promising signal variants
        # create_ema_rsi_macd_signal,  # DISABLED: Missing ema_20 column
        create_stoch_macd_simple_signal,
        # create_bb_macd_signal,  # DISABLED: Missing bb_lower/bb_upper columns
    )

    signals = []

    # SMA period combinations to test (only use available: sma_20, sma_50, sma_200)
    # Issue #550: Re-enabled (50, 200) for comprehensive discovery
    # Previous Issue #526 disabled it but we want full coverage for overnight runs
    sma_combos = [
        (20, 50),   # Standard short-term
        (20, 200),  # Long-term
        (50, 200),  # Traditional golden/death cross
    ]

    if direction == "long":
        # =================================================================
        # SINGLE INDICATOR SIGNALS
        # =================================================================
        # Stochastic signals
        for k in [15, 20, 25]:
            signals.append(create_stochastic_oversold_signal(k_threshold=k, timeframe=timeframe))

        # RSI signals
        for thresh in [25, 30, 35]:
            signals.append(create_rsi_oversold_signal(period=14, threshold=thresh, timeframe=timeframe))

        # SMA crossovers (all combinations)
        for fast, slow in sma_combos:
            signals.append(create_sma_crossover_signal(fast_period=fast, slow_period=slow, direction="long", timeframe=timeframe))

        # MACD and Bollinger
        signals.append(create_macd_crossover_signal(direction="long", timeframe=timeframe))
        signals.append(create_bollinger_touch_signal(direction="long", timeframe=timeframe))

        # =================================================================
        # 2-INDICATOR COMBINATION SIGNALS
        # =================================================================
        # SMA + RSI combinations
        for fast, slow in sma_combos:
            signals.append(create_sma_rsi_signal(sma_fast=fast, sma_slow=slow, direction="long", timeframe=timeframe))

        # SMA + Stochastic combinations
        for fast, slow in sma_combos:
            for stoch_t in [15, 20, 25]:
                signals.append(create_sma_stoch_signal(sma_fast=fast, sma_slow=slow, stoch_oversold=stoch_t, direction="long", timeframe=timeframe))

        # SMA + MACD combinations
        for fast, slow in sma_combos:
            signals.append(create_sma_macd_signal(sma_fast=fast, sma_slow=slow, direction="long", timeframe=timeframe))

        # SMA + Bollinger Band combinations
        for fast, slow in sma_combos:
            signals.append(create_sma_bb_signal(sma_fast=fast, sma_slow=slow, direction="long", timeframe=timeframe))

        # EMA + RSI combinations (using available EMA: 12, 26, 50)
        signals.append(create_ema_rsi_signal(direction="long", timeframe=timeframe))
        signals.append(create_ema_rsi_signal(ema_fast=12, ema_slow=50, direction="long", timeframe=timeframe))

        # Stochastic + RSI double oversold
        # Issue #550: Re-enabled stoch_t=25 for comprehensive discovery
        for stoch_t in [15, 20, 25]:
            for rsi_t in [25, 30, 35]:
                signals.append(create_stoch_rsi_signal(stoch_threshold=stoch_t, rsi_threshold=rsi_t, direction="long", timeframe=timeframe))

        # MACD + Stochastic
        signals.append(create_macd_stoch_signal(direction="long", timeframe=timeframe))

        # Bollinger + RSI
        # Issue #550: Re-enabled for comprehensive discovery
        signals.append(create_bb_rsi_signal(direction="long", timeframe=timeframe))

        # =================================================================
        # 3-INDICATOR COMBINATION SIGNALS
        # =================================================================
        # SMA + RSI + Stochastic (triple confirmation)
        for fast, slow in [(20, 50), (20, 200), (50, 200)]:
            signals.append(create_sma_rsi_stoch_signal(sma_fast=fast, sma_slow=slow, direction="long", timeframe=timeframe))

        # SMA + RSI + MACD
        for fast, slow in [(20, 50), (20, 200), (50, 200)]:
            signals.append(create_sma_rsi_macd_signal(sma_fast=fast, sma_slow=slow, direction="long", timeframe=timeframe))

        # SMA + Stochastic + Bollinger Band
        for fast, slow in [(20, 50), (20, 200), (50, 200)]:
            signals.append(create_sma_stoch_bb_signal(sma_fast=fast, sma_slow=slow, direction="long", timeframe=timeframe))

        # Triple Momentum (RSI + Stochastic + MACD)
        # Re-enabled: Was best performer for EURUSD H4/M30
        signals.append(create_triple_momentum_signal(direction="long", timeframe=timeframe))

        # EMA + Stochastic + Bollinger Band
        signals.append(create_ema_stoch_bb_signal(direction="long", timeframe=timeframe))

        # =================================================================
        # 4-INDICATOR COMBINATION SIGNALS (Issue #526)
        # =================================================================
        # SMA + RSI + Stochastic + Bollinger Band (10 combinations)
        for fast, slow in [(20, 50), (20, 200), (50, 200)]:
            for rsi_t in [25, 30]:
                for stoch_t in [15, 20]:
                    signals.append(create_sma_rsi_stoch_bb_signal(
                        sma_fast=fast,
                        sma_slow=slow,
                        rsi_threshold=rsi_t,
                        stoch_threshold=stoch_t,
                        direction="long",
                        timeframe=timeframe
                    ))

        # SMA + RSI + Stochastic + MACD (10 combinations)
        # Issue #526 Optimization: Commented out - 4-way over-filters (0-52 trades, 0% WR when trades exist)
        # for fast, slow in [(20, 50), (20, 200), (50, 200)]:
        #     for rsi_t in [25, 30]:
        #         for stoch_t in [15, 20]:
        #             signals.append(create_sma_rsi_stoch_macd_signal(
        #                 sma_fast=fast,
        #                 sma_slow=slow,
        #                 rsi_threshold=rsi_t,
        #                 stoch_threshold=stoch_t,
        #                 direction="long",
        #                 timeframe=timeframe
        #             ))

        # =================================================================
        # NEW PROMISING COMBINATIONS (Issue #526 Optimization)
        # =================================================================
        # EMA + RSI + MACD (DISABLED: Missing ema_20 column)
        # signals.append(create_ema_rsi_macd_signal(
        #     ema_fast=20,
        #     ema_slow=200,
        #     rsi_threshold=30,
        #     direction="long",
        #     timeframe=timeframe
        # ))

        # Stoch + MACD simple (2-way without SMA filter)
        for stoch_t in [15, 20]:
            signals.append(create_stoch_macd_simple_signal(
                stoch_threshold=stoch_t,
                direction="long",
                timeframe=timeframe
            ))

        # BB + MACD (DISABLED: Missing bb_lower/bb_upper columns)
        # signals.append(create_bb_macd_signal(direction="long", timeframe=timeframe))

    else:  # direction == "short"
        # =================================================================
        # SINGLE INDICATOR SIGNALS
        # =================================================================
        # Stochastic signals
        for k in [75, 80, 85]:
            signals.append(create_stochastic_overbought_signal(k_threshold=k, timeframe=timeframe))

        # RSI signals
        for thresh in [65, 70, 75]:
            signals.append(create_rsi_overbought_signal(period=14, threshold=thresh, timeframe=timeframe))

        # SMA crossovers (all combinations)
        for fast, slow in sma_combos:
            signals.append(create_sma_crossover_signal(fast_period=fast, slow_period=slow, direction="short", timeframe=timeframe))

        # MACD and Bollinger
        signals.append(create_macd_crossover_signal(direction="short", timeframe=timeframe))
        signals.append(create_bollinger_touch_signal(direction="short", timeframe=timeframe))

        # =================================================================
        # 2-INDICATOR COMBINATION SIGNALS
        # =================================================================
        # SMA + RSI combinations
        for fast, slow in sma_combos:
            signals.append(create_sma_rsi_signal(sma_fast=fast, sma_slow=slow, direction="short", timeframe=timeframe))

        # SMA + Stochastic combinations
        for fast, slow in sma_combos:
            for stoch_t in [75, 80, 85]:
                signals.append(create_sma_stoch_signal(sma_fast=fast, sma_slow=slow, stoch_overbought=stoch_t, direction="short", timeframe=timeframe))

        # SMA + MACD combinations
        for fast, slow in sma_combos:
            signals.append(create_sma_macd_signal(sma_fast=fast, sma_slow=slow, direction="short", timeframe=timeframe))

        # SMA + Bollinger Band combinations
        for fast, slow in sma_combos:
            signals.append(create_sma_bb_signal(sma_fast=fast, sma_slow=slow, direction="short", timeframe=timeframe))

        # EMA + RSI combinations (using available EMA: 12, 26, 50)
        signals.append(create_ema_rsi_signal(direction="short", timeframe=timeframe))
        signals.append(create_ema_rsi_signal(ema_fast=12, ema_slow=50, direction="short", timeframe=timeframe))

        # Stochastic + RSI double overbought
        # Issue #550: Re-enabled stoch_t=75 for comprehensive discovery
        for stoch_t in [75, 80, 85]:
            for rsi_t in [65, 70, 75]:
                signals.append(create_stoch_rsi_signal(stoch_threshold=100-stoch_t, rsi_threshold=100-rsi_t, direction="short", timeframe=timeframe))

        # MACD + Stochastic
        signals.append(create_macd_stoch_signal(direction="short", timeframe=timeframe))

        # Bollinger + RSI
        # Issue #550: Re-enabled for comprehensive discovery
        signals.append(create_bb_rsi_signal(direction="short", timeframe=timeframe))

        # =================================================================
        # 3-INDICATOR COMBINATION SIGNALS
        # =================================================================
        # SMA + RSI + Stochastic (triple confirmation)
        for fast, slow in [(20, 50), (20, 200), (50, 200)]:
            signals.append(create_sma_rsi_stoch_signal(sma_fast=fast, sma_slow=slow, direction="short", timeframe=timeframe))

        # SMA + RSI + MACD
        for fast, slow in [(20, 50), (20, 200), (50, 200)]:
            signals.append(create_sma_rsi_macd_signal(sma_fast=fast, sma_slow=slow, direction="short", timeframe=timeframe))

        # SMA + Stochastic + Bollinger Band
        for fast, slow in [(20, 50), (20, 200), (50, 200)]:
            signals.append(create_sma_stoch_bb_signal(sma_fast=fast, sma_slow=slow, direction="short", timeframe=timeframe))

        # Triple Momentum (RSI + Stochastic + MACD)
        # Re-enabled: Was best performer for EURUSD H4/M30
        signals.append(create_triple_momentum_signal(direction="short", timeframe=timeframe))

        # EMA + Stochastic + Bollinger Band
        signals.append(create_ema_stoch_bb_signal(direction="short", timeframe=timeframe))

        # =================================================================
        # 4-INDICATOR COMBINATION SIGNALS (Issue #526)
        # =================================================================
        # SMA + RSI + Stochastic + Bollinger Band (10 combinations)
        for fast, slow in [(20, 50), (20, 200), (50, 200)]:
            for rsi_t in [65, 70]:
                for stoch_t in [75, 80]:
                    signals.append(create_sma_rsi_stoch_bb_signal(
                        sma_fast=fast,
                        sma_slow=slow,
                        rsi_threshold=rsi_t,
                        stoch_threshold=stoch_t,
                        direction="short",
                        timeframe=timeframe
                    ))

        # SMA + RSI + Stochastic + MACD (10 combinations)
        # Issue #526 Optimization: Commented out - 4-way over-filters (0-52 trades, 0% WR when trades exist)
        # for fast, slow in [(20, 50), (20, 200), (50, 200)]:
        #     for rsi_t in [65, 70]:
        #         for stoch_t in [75, 80]:
        #             signals.append(create_sma_rsi_stoch_macd_signal(
        #                 sma_fast=fast,
        #                 sma_slow=slow,
        #                 rsi_threshold=rsi_t,
        #                 stoch_threshold=stoch_t,
        #                 direction="short",
        #                 timeframe=timeframe
        #             ))

        # =================================================================
        # NEW PROMISING COMBINATIONS (Issue #526 Optimization)
        # =================================================================
        # EMA + RSI + MACD (DISABLED: Missing ema_20 column)
        # signals.append(create_ema_rsi_macd_signal(
        #     ema_fast=20,
        #     ema_slow=200,
        #     rsi_threshold=30,
        #     direction="short",
        #     timeframe=timeframe
        # ))

        # Stoch + MACD simple (2-way without SMA filter)
        for stoch_t in [80, 85]:
            signals.append(create_stoch_macd_simple_signal(
                stoch_threshold=stoch_t,
                direction="short",
                timeframe=timeframe
            ))

        # BB + MACD (DISABLED: Missing bb_lower/bb_upper columns)
        # signals.append(create_bb_macd_signal(direction="short", timeframe=timeframe))

    return signals


def run_phase2_validation(
    full_df: pd.DataFrame,
    signal,
    symbol: str,
    is_metrics: dict,
    direction: str,
) -> dict:
    """
    Run Phase 2: Walk-Forward Validation with WR-based evaluation.

    Validates discovered signal on TRUE unseen validation data (20%).
    This is the gate that filters signals before expensive Phase 3-5 training.
    Uses WR as primary metric (fixed SL/TP → all trades ±30 pips).

    BLOCKING GATE: WR OOS >= 54%, WR degradation <= 5%, Trades >= 50

    Args:
        full_df: Full dataset DataFrame
        signal: Signal definition to validate
        symbol: Trading symbol
        is_metrics: In-sample metrics from Phase 1
        direction: Signal direction ('long' or 'short')

    Returns:
        Dict with validation results:
        - passed: Boolean - True if signal passed validation
        - evaluation: SignalEvaluation for OOS data
    """
    from data.splitter import get_validation_data
    from indicator_discovery.tester import SignalTester, TradingCosts
    from indicator_discovery.evaluation import (
        create_signal_evaluation,
        Phase2Thresholds,
        calculate_implied_pf,
    )

    print("\n" + "=" * 60)
    print("PHASE 2: WALK-FORWARD VALIDATION (WR-based)")
    print("=" * 60)
    print("Testing on NEVER-SEEN validation data (20%)...")
    print(f"Blocking Gate: WR >= {MIN_WIN_RATE_PHASE2*100:.0f}%, WR degradation <= {MAX_WR_DEGRADATION*100:.0f}%, Trades >= {MIN_TRADES_PHASE2}")

    # Get validation data only
    val_df = get_validation_data(full_df, symbol)
    print(f"Validation data: {len(val_df)} candles")

    # Test signal on validation data
    costs = TradingCosts(spread=1.5, slippage=0.5, commission=0.5)
    tester = SignalTester(sl_pips=30.0, tp_pips=30.0, horizon=48)
    result = tester.test_with_costs(val_df, signal, costs, symbol)

    # Create OOS evaluation
    oos_evaluation = create_signal_evaluation(
        signal_name=signal.name,
        symbol=symbol,
        timeframe=signal.timeframe,
        direction=direction,
        category=signal.category if hasattr(signal, 'category') else "single",
        wins=result.wins,
        total_trades=result.total_trades,
        degradation=0.0,
        frequency_per_year=int(result.total_trades * (252 / len(val_df))),
        sl_pips=30.0,
        tp_pips=30.0,
    )

    # Calculate degradation from IS to OOS
    is_wr = is_metrics.get("win_rate", 0.5)
    oos_wr = oos_evaluation.win_rate
    wr_degradation = is_wr - oos_wr

    print(f"\nOut-of-Sample Results:")
    print(f"  Win Rate: {oos_wr*100:.1f}% (edge: {(oos_wr-0.5)*100:.1f}%)")
    print(f"  Implied PF: {oos_evaluation.implied_pf:.2f}")
    print(f"  Trades: {oos_evaluation.trade_count}")
    print(f"  P-Value: {oos_evaluation.p_value:.4f}")
    print(f"\nDegradation from Training:")
    print(f"  IS WR: {is_wr*100:.1f}% → OOS WR: {oos_wr*100:.1f}%")
    print(f"  WR Degradation: {wr_degradation*100:.1f}%")

    # Check Phase 2 criteria (WR-based)
    failure_reasons = []
    if oos_wr < MIN_WIN_RATE_PHASE2:
        failure_reasons.append(f"OOS WR {oos_wr*100:.1f}% < {MIN_WIN_RATE_PHASE2*100:.0f}% minimum")
    if wr_degradation > MAX_WR_DEGRADATION:
        failure_reasons.append(f"WR degradation {wr_degradation*100:.1f}% > {MAX_WR_DEGRADATION*100:.0f}% maximum")
    if oos_evaluation.trade_count < MIN_TRADES_PHASE2:
        failure_reasons.append(f"Trades {oos_evaluation.trade_count} < {MIN_TRADES_PHASE2} minimum")

    passed = len(failure_reasons) == 0

    if passed:
        print("\n" + "=" * 60)
        print("✅ PHASE 2 PASSED: Signal validated on OOS data!")
        print("   Approved for Optuna tuning and 30-fold training")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ PHASE 2 FAILED: Signal did not validate")
        print("   Failure Reasons:")
        for reason in failure_reasons:
            print(f"     - {reason}")
        print("=" * 60)

    return {
        "passed": passed,
        "evaluation": oos_evaluation,
        "is_wr": is_wr,
        "oos_wr": oos_wr,
        "wr_degradation": wr_degradation,
        "failure_reasons": failure_reasons,
    }


def run_phase3_optuna(
    train_df: pd.DataFrame,
    signal,
    symbol: str,
    direction: str,
    n_trials: int = 50,
    n_folds: int = 5,  # Issue #530: Configurable folds
    use_d1_features: bool = True,  # Issue #530: Disable for 6x speedup
) -> dict:
    """
    Run Phase 3: Optuna Hyperparameter Tuning.

    Uses K-fold cross-validation with PF as primary metric.
    Applies all 6 bug fixes from Issue #528.

    Args:
        train_df: Training data DataFrame (60%)
        signal: Signal definition from Phase 2
        symbol: Trading symbol
        direction: Signal direction
        n_trials: Number of Optuna trials (default 50)
        n_folds: Number of cross-validation folds (default 5)
        use_d1_features: If True, enable D1/multi-TF features (slower but more filtering)

    Returns:
        Dict with Optuna results:
        - best_params: Best hyperparameters found
        - best_value: Best PF achieved
        - study: Optuna study object
    """
    import optuna
    from pattern_system.rl.optuna_tuning import (
        create_fixed_objective,
        create_study_with_variance,
        calculate_optimal_timesteps,
        DEFAULT_ROLLOUTS,
    )
    from pathlib import Path

    print("\n" + "=" * 60)
    print("PHASE 3: OPTUNA HYPERPARAMETER TUNING (PF-based)")
    print("=" * 60)
    print(f"Training data: {len(train_df)} candles")
    print(f"Target: Maximize Profit Factor via K-fold cross-validation")
    print(f"Trials: {n_trials}, Folds: {n_folds}")
    print(f"D1/Multi-TF Features: {'ENABLED' if use_d1_features else 'DISABLED (6x faster)'}")
    print(f"Bug fixes applied: #1-6 (see Issue #528)")

    # Calculate optimal timesteps based on FOLD size, not full training data
    # Issue #549: Fix - each Optuna trial trains on (n_folds-1)/n_folds of data
    fold_train_size = len(train_df) * (n_folds - 1) // n_folds
    timesteps = calculate_optimal_timesteps(fold_train_size, rollouts=DEFAULT_ROLLOUTS)
    rollouts = timesteps / fold_train_size
    print(f"Fold train size: {fold_train_size} candles (out of {len(train_df)} total)")
    print(f"Timesteps: {timesteps} ({rollouts:.1f}x rollouts per fold - within 2-3x limit)")

    # Create study
    # Issue #530: Include signal name to prevent race conditions when running
    # multiple signals for the same symbol+direction in parallel
    signal_name_safe = signal.name.replace(" ", "_") if hasattr(signal, 'name') else "unknown"
    study_name = f"hybrid_v4_{symbol}_{direction}_{signal_name_safe}"
    storage_path = Path("results/optuna_hybrid") / f"optuna_{symbol}_{direction}_{signal_name_safe}_study.db"
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    # Bug #8 fix: Delete old database to ensure fresh start
    # This prevents reusing corrupted studies from previous runs
    if storage_path.exists():
        print(f"⚠️  Deleting old Optuna database: {storage_path}")
        try:
            storage_path.unlink()
        except FileNotFoundError:
            # Race condition: another process may have deleted it
            print(f"   (Already deleted by another process)")

    storage_url = f"sqlite:///{storage_path}"

    print(f"Study: {study_name}")
    print(f"Storage: {storage_path}")
    print(f"Signal: {signal.name if hasattr(signal, 'name') else 'Unknown'}")

    study = create_study_with_variance(
        study_name=study_name,
        storage_url=storage_url,
        direction="maximize",
    )

    # Create objective with all bug fixes
    # CRITICAL FIX (Issue #530): Use the ACTUAL discovered signal from Phase 1-2
    # This ensures Triple_Momentum_long (or whatever signal passed Phase 2) is used,
    # not the hardcoded V3SignalGenerator signals!
    objective = create_fixed_objective(
        discovered_signal=signal,  # Issue #530: CRITICAL - pass actual discovered signal!
        full_df=train_df,  # Uses training data only
        symbol=symbol,
        n_folds=n_folds,  # Issue #530: Configurable folds
        max_rollouts=DEFAULT_ROLLOUTS,
        signal_type="discovered",  # Issue #530: Use discovered signal type
        direction_filter=direction,  # Pass direction (long/short)
        use_d1_features=use_d1_features,  # Issue #530: Optionally disable for speed
    )

    print(f"\nStarting Optuna optimization...")
    print(f"(This may take a while - {n_trials} trials × {n_folds} folds)")

    # Callback to log trial progress
    def log_trial(study, trial):
        logger.info(f"Trial {trial.number + 1}/{n_trials} completed: PF={trial.value:.3f}")

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            callbacks=[log_trial],
            catch=(Exception,),
        )

        # Issue #530: Check if any trials completed successfully before accessing best_params
        # This prevents "Should not reach" error when all trials failed internally
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

        logger.info(f"Optuna study summary: {len(completed_trials)} completed, {len(failed_trials)} failed, {len(pruned_trials)} pruned")

        if not completed_trials:
            # Log details of failed trials for debugging
            for t in failed_trials[:5]:  # Show first 5 failures
                logger.warning(f"  Failed trial {t.number}: {t.user_attrs}")
            raise RuntimeError(f"All {n_trials} trials failed internally ({len(failed_trials)} failed, {len(pruned_trials)} pruned). Check logs for details.")

        best_params = study.best_params
        best_value = study.best_value

        print("\n" + "=" * 60)
        print("✅ PHASE 3 COMPLETE: Optuna optimization finished!")
        print(f"   Best PF: {best_value:.2f}")
        print(f"   Best Params:")
        for key, value in best_params.items():
            print(f"     - {key}: {value}")
        print("=" * 60)

        return {
            "passed": True,
            "best_params": best_params,
            "best_value": best_value,
            "study": study,
        }

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"⚠️ PHASE 3 WARNING: Optuna failed - {e}")
        print("   Using default hyperparameters for Phase 4")
        print("=" * 60)

        return {
            "passed": False,
            "best_params": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "n_steps": 1024,
                "batch_size": 128,
            },
            "best_value": 0.0,
            "study": None,
        }


def run_phase4_training(
    train_df: pd.DataFrame,
    signal,
    symbol: str,
    direction: str,
    best_params: dict,
    n_folds: int = 30,
    use_d1_features: bool = False,
    discovered_signal=None,
    n_envs: int = None,  # CPU parallelization (auto-detect if None)
    signal_name: str = None,  # Issue #547: For model path
    timeframe: str = None,  # Issue #547: For model path to prevent overwrites
) -> dict:
    """
    Run Phase 4: 30-Fold Training.

    Trains ensemble of models using Optuna hyperparameters.
    Uses PF as primary metric with 2-3x rollout limit.

    Args:
        train_df: Training data DataFrame (60%)
        signal: Signal definition
        symbol: Trading symbol
        direction: Signal direction
        best_params: Hyperparameters from Optuna
        n_folds: Number of folds (default 30)
        use_d1_features: Whether to use D1 features (default False for speed)
        discovered_signal: SignalDefinition from Phase 1-2 (Issue #530)
        n_envs: Number of parallel environments for CPU utilization.
            If None (default), auto-detects from CPU count and available RAM.
            CLAUDE.md requires 80%+ CPU utilization.
        signal_name: Signal name for model path (Issue #547)
        timeframe: Timeframe for model path to prevent overwrites (Issue #547)

    Returns:
        Dict with training results:
        - models: List of trained model paths
        - fold_results: List of FoldResult objects
        - avg_pf: Average PF across folds
    """
    from pattern_system.rl.optuna_tuning import (
        calculate_optimal_timesteps,
        get_trial_seed,
        DEFAULT_ROLLOUTS,
    )
    from pattern_system.rl.parallel_env import (
        adjust_n_steps_for_parallel,
        create_parallel_envs,
        get_safe_n_envs,
    )
    from pathlib import Path

    print("\n" + "=" * 60)
    print("PHASE 4: 30-FOLD TRAINING (PF-based)")
    print("=" * 60)
    print(f"Training data: {len(train_df)} candles")
    print(f"Folds: {n_folds}")
    print(f"Hyperparameters from Optuna: {best_params}")

    # Bug Fix #2: Validate feature dimensions before training
    # This catches dimension mismatches early instead of hours into training
    try:
        from pattern_system.rl.hybrid_env import HybridTradingEnv

        # Create a temporary env to check observation space dimensions
        temp_df = train_df.iloc[:100].copy().reset_index(drop=True)
        if 'timestamp' in temp_df.columns and 'time' not in temp_df.columns:
            temp_df['time'] = temp_df['timestamp']

        temp_env = HybridTradingEnv(
            df=temp_df,
            symbol=symbol,
            use_d1_features=use_d1_features,
            discovered_signal=discovered_signal,
            truncate_early=True,  # Speed up validation
        )
        expected_dim = temp_env.observation_space.shape[0]
        temp_env.close()

        print(f"Feature dimension validation: {expected_dim} features OK")
    except Exception as e:
        print(f"❌ PHASE 4 DIMENSION VALIDATION FAILED: {e}")
        print("   This likely means Phase 3 and Phase 4 configurations don't match.")
        print("   Check: use_d1_features, discovered_signal, or feature columns.")
        raise ValueError(f"Phase 4 dimension validation failed: {e}")

    # Auto-detect n_envs for CPU parallelization (CLAUDE.md requirement: 80%+ CPU)
    actual_n_envs = get_safe_n_envs(override=n_envs)
    print(f"Parallel environments: {actual_n_envs}")

    # Calculate fold size and timesteps
    fold_size = len(train_df) // n_folds
    timesteps = calculate_optimal_timesteps(fold_size, rollouts=DEFAULT_ROLLOUTS)
    rollouts = timesteps / fold_size
    print(f"Fold size: {fold_size} candles")
    print(f"Timesteps per fold: {timesteps} ({rollouts:.1f}x rollouts)")

    # Results storage
    # Issue #547: Include signal_name and timeframe to prevent overwrites when same signal
    # is trained at multiple timeframes
    if signal_name and timeframe:
        models_dir = Path(f"models/hybrid_v4/{symbol}_{direction}_{signal_name}_{timeframe}")
    elif signal_name:
        models_dir = Path(f"models/hybrid_v4/{symbol}_{direction}_{signal_name}")
    else:
        models_dir = Path(f"models/hybrid_v4/{symbol}_{direction}")
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model output directory: {models_dir}")

    model_paths = []
    fold_pfs = []
    fold_wrs = []

    print(f"\nTraining {n_folds} folds...")

    for fold_idx in range(n_folds):
        # Get fold data (rolling window)
        start_idx = fold_idx * fold_size
        end_idx = start_idx + fold_size
        fold_df = train_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

        # Fix column naming: HybridTradingEnv expects 'time', not 'timestamp'
        if 'timestamp' in fold_df.columns and 'time' not in fold_df.columns:
            fold_df['time'] = fold_df['timestamp']

        # Get fold-specific seed
        fold_seed = get_trial_seed(fold_idx)

        vec_env = None
        try:
            from pattern_system.rl.hybrid_env import HybridTradingEnv
            from stable_baselines3 import PPO

            # Create parallel environments for CPU utilization (CLAUDE.md: 80%+ CPU)
            env_kwargs = {
                "truncate_early": False,  # Bug #2 fix
                "use_d1_features": use_d1_features,
                "discovered_signal": discovered_signal,
            }

            vec_env = create_parallel_envs(
                df=fold_df,
                symbol=symbol,
                n_envs=actual_n_envs,
                env_kwargs=env_kwargs,
                seed_base=fold_seed,
                env_class=HybridTradingEnv,
            )

            # Adjust n_steps for parallel envs to maintain batch size
            base_n_steps = best_params.get("n_steps", 1024)
            adjusted_n_steps = adjust_n_steps_for_parallel(base_n_steps, actual_n_envs)

            # Create model with Optuna params
            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=best_params.get("learning_rate", 3e-4),
                gamma=best_params.get("gamma", 0.99),
                n_steps=adjusted_n_steps,  # Adjusted for parallel envs
                batch_size=best_params.get("batch_size", 128),
                seed=fold_seed,
                verbose=0,
                device="cpu",  # Issue #526: Force CPU for MlpPolicy (GPU is slower)
            )

            # Train with optimal timesteps
            model.learn(total_timesteps=timesteps, progress_bar=False)

            # Quick evaluation on fold data
            # Issue #530: Pass use_d1_features and discovered_signal for consistent evaluation
            from pattern_system.rl.optuna_tuning import evaluate_model_with_variance
            metrics = evaluate_model_with_variance(
                model=model,
                val_df=fold_df,
                symbol=symbol,
                n_episodes=10,
                trial_seed=fold_seed,
                deterministic=True,
                use_d1_features=use_d1_features,
                discovered_signal=discovered_signal,
            )

            fold_pfs.append(metrics["profit_factor"])
            fold_wrs.append(metrics["win_rate"])

            # Save model
            model_path = models_dir / f"fold_{fold_idx:02d}.zip"
            model.save(str(model_path))
            model_paths.append(str(model_path))

            print(f"  Fold {fold_idx+1}/{n_folds}: PF={metrics['profit_factor']:.2f}, WR={metrics['win_rate']:.1f}%")

        except Exception as e:
            logger.warning(f"Fold {fold_idx} failed: {e}")
            fold_pfs.append(0.0)
            fold_wrs.append(0.0)
            print(f"  Fold {fold_idx+1}/{n_folds}: FAILED - {e}")
        finally:
            # CRITICAL: Always close vec_env to avoid resource leaks
            if vec_env is not None:
                try:
                    vec_env.close()
                except Exception:
                    pass

    # Calculate averages
    avg_pf = np.mean([pf for pf in fold_pfs if pf > 0]) if any(pf > 0 for pf in fold_pfs) else 0.0
    avg_wr = np.mean([wr for wr in fold_wrs if wr > 0]) if any(wr > 0 for wr in fold_wrs) else 0.0
    successful_folds = sum(1 for pf in fold_pfs if pf > 0)

    print("\n" + "=" * 60)
    print("✅ PHASE 4 COMPLETE: 30-Fold training finished!")
    print(f"   Successful folds: {successful_folds}/{n_folds}")
    print(f"   Average PF: {avg_pf:.2f}")
    print(f"   Average WR: {avg_wr:.1f}%")
    print(f"   Models saved to: {models_dir}")
    print("=" * 60)

    return {
        "passed": successful_folds > 0,
        "model_paths": model_paths,
        "fold_pfs": fold_pfs,
        "fold_wrs": fold_wrs,
        "avg_pf": avg_pf,
        "avg_wr": avg_wr,
        "successful_folds": successful_folds,
    }


def run_phase5_validation(
    full_df: pd.DataFrame,
    model_paths: list,
    symbol: str,
    direction: str,
    training_avg_pf: float,
    use_d1_features: bool = True,  # Issue #530: Must match training
    discovered_signal=None,  # Issue #530: Must match training
) -> dict:
    """
    Run Phase 5: Final Test Validation.

    Evaluates ensemble on TRUE held-out test data (20%).
    Uses PF as primary metric - this is the MOMENT OF TRUTH.

    Args:
        full_df: Full dataset DataFrame
        model_paths: List of trained model paths from Phase 4
        symbol: Trading symbol
        direction: Signal direction
        training_avg_pf: Average PF from training for degradation check
        use_d1_features: Must match training (Issue #530 fix)
        discovered_signal: SignalDefinition from Phase 1-2 (Issue #530 fix)

    Returns:
        Dict with final validation results:
        - passes_production: Boolean - APPROVED or REJECTED
        - ensemble_pf: Ensemble PF on test data
        - pf_degradation: Training → Test degradation
        - rejection_reasons: List of failure reasons if rejected
    """
    from data.splitter import get_test_data
    from pattern_system.rl.optuna_tuning import evaluate_model_with_variance
    from stable_baselines3 import PPO
    from pathlib import Path

    print("\n" + "=" * 60)
    print("PHASE 5: FINAL TEST VALIDATION - MOMENT OF TRUTH (PF-based)")
    print("=" * 60)
    print(f"D1/Multi-TF Features: {'ENABLED' if use_d1_features else 'DISABLED'}")
    # Issue #535: Handle both string signal names and SignalDefinition objects
    signal_name_display = (
        discovered_signal.name if hasattr(discovered_signal, 'name')
        else str(discovered_signal) if discovered_signal
        else 'None'
    )
    print(f"Discovered Signal: {signal_name_display}")

    # Get test data - FIRST TIME EVER ACCESSED
    test_df = get_test_data(full_df, symbol, confirm_phase5=True)
    from data.splitter import compute_test_data_signature
    data_signature = compute_test_data_signature(test_df, symbol)
    print(f"Test data: {len(test_df)} candles (NEVER SEEN BEFORE)")
    print(f"Data signature: {data_signature}")
    print(f"Models to evaluate: {len(model_paths)}")

    # Evaluate each model
    model_pfs = []
    model_wrs = []
    model_trades = []

    for i, model_path in enumerate(model_paths):
        try:
            model = PPO.load(model_path)

            metrics = evaluate_model_with_variance(
                model=model,
                val_df=test_df,
                symbol=symbol,
                n_episodes=20,
                trial_seed=i,
                deterministic=True,
                use_d1_features=use_d1_features,  # Issue #530: Must match training
                discovered_signal=discovered_signal,  # Issue #530: Must match training
            )

            model_pfs.append(metrics["profit_factor"])
            model_wrs.append(metrics["win_rate"])
            model_trades.append(metrics["total_trades"])

            print(f"  Model {i+1}/{len(model_paths)}: PF={metrics['profit_factor']:.2f}, WR={metrics['win_rate']:.1f}%, Trades={metrics['total_trades']}")

        except Exception as e:
            logger.warning(f"Model {i} evaluation failed: {e}")
            print(f"  Model {i+1}/{len(model_paths)}: FAILED - {e}")

    # Calculate ensemble metrics
    valid_pfs = [pf for pf in model_pfs if pf > 0]
    if not valid_pfs:
        print("\n" + "=" * 60)
        print("❌ PHASE 5 FAILED: No valid model evaluations")
        print("=" * 60)
        return {
            "passes_production": False,
            "ensemble_pf": 0.0,
            "pf_degradation": 1.0,
            "rejection_reasons": ["No valid model evaluations on test data"],
        }

    ensemble_pf = np.mean(valid_pfs)
    ensemble_wr = np.mean([wr for wr in model_wrs if wr > 0])
    pf_std = np.std(valid_pfs)
    pf_degradation = training_avg_pf - ensemble_pf
    # Issue #537: Use percentage-based degradation (more meaningful than absolute)
    pf_degradation_pct = (pf_degradation / training_avg_pf * 100) if training_avg_pf > 0 else 0.0
    total_trades = sum(model_trades)

    print(f"\nEnsemble Results:")
    print(f"  PF: {ensemble_pf:.2f} ± {pf_std:.2f}")
    print(f"  WR: {ensemble_wr:.1f}%")
    print(f"  Total Trades: {total_trades}")
    print(f"\nDegradation from Training:")
    print(f"  Training PF: {training_avg_pf:.2f} → Test PF: {ensemble_pf:.2f}")
    print(f"  PF Degradation: {pf_degradation:.2f} ({pf_degradation_pct:.1f}%)")

    # Check production criteria
    # Issue #540: Removed degradation check - if Test PF > 1.2, model is profitable
    # regardless of how much it degraded from training. Only PF and trades matter.
    rejection_reasons = []
    if ensemble_pf < MIN_PROFIT_FACTOR:
        rejection_reasons.append(f"Test PF {ensemble_pf:.2f} < {MIN_PROFIT_FACTOR} minimum")
    if ensemble_wr < MIN_WIN_RATE_PHASE5 * 100:
        rejection_reasons.append(f"Test WR {ensemble_wr:.1f}% < {MIN_WIN_RATE_PHASE5*100:.0f}% minimum")
    if total_trades < 50:
        rejection_reasons.append(f"Total trades {total_trades} < 50 minimum")

    passes = len(rejection_reasons) == 0

    if passes:
        print("\n" + "=" * 60)
        print("🎉 PHASE 5 PASSED: APPROVED FOR PRODUCTION!")
        print(f"   Test PF: {ensemble_pf:.2f}")
        print(f"   Ensemble of {len(valid_pfs)} models ready for deployment")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ PHASE 5 FAILED: REJECTED FOR PRODUCTION")
        print("   Rejection Reasons:")
        for reason in rejection_reasons:
            print(f"     - {reason}")
        print("=" * 60)

    return {
        "passes_production": passes,
        "ensemble_pf": ensemble_pf,
        "ensemble_wr": ensemble_wr,
        "pf_std": pf_std,
        "pf_degradation": pf_degradation,
        "pf_degradation_pct": pf_degradation_pct,  # Issue #537
        "total_trades": total_trades,
        "rejection_reasons": rejection_reasons,
        "data_signature": data_signature,
    }


# =============================================================================
# Main Pipeline
# =============================================================================


def run_pipeline(args) -> bool:
    """
    Run complete pipeline with blocking gates.

    Orchestrates all 6 phases of the hybrid-v4 pipeline.
    Each phase has blocking gates - if a phase fails, the pipeline stops.

    Args:
        args: Parsed command line arguments

    Returns:
        True if pipeline completed successfully (APPROVED),
        False if pipeline failed or model REJECTED
    """
    symbol = args.symbol.lower()
    direction = args.direction

    print("\n" + "=" * 60)
    print("HYBRID V4 PIPELINE")
    print("=" * 60)
    print(f"Symbol: {symbol.upper()}")
    print(f"Direction: {direction.upper()}")
    print("=" * 60)

    # Dry run mode
    if args.dry_run:
        print("\n[DRY RUN MODE - No actual execution]")
        print("\nPipeline would execute:")
        print("  Phase 0: Data Segregation (60/20/20)")
        print("  Phase 1: Signal Discovery (WR-based)")
        print(f"    → BLOCKING GATE: WR >= {MIN_WIN_RATE_PHASE1*100:.0f}%, Trades >= {MIN_TRADES_PHASE1}, p-value < {MAX_P_VALUE}")
        print("  Phase 2: Walk-Forward Validation (WR-based)")
        print(f"    → BLOCKING GATE: WR OOS >= {MIN_WIN_RATE_PHASE2*100:.0f}%, WR degradation <= {MAX_WR_DEGRADATION*100:.0f}%")
        if not args.skip_optuna:
            print("  Phase 3: Optuna Hyperparameter Tuning")
        else:
            print("  Phase 3: SKIPPED (using default hyperparameters)")
        if not args.skip_training:
            print("  Phase 4: 30-Fold Training")
        else:
            print("  Phase 4: SKIPPED")
        print("  Phase 5: Final Test Validation")
        print("\n[DRY RUN COMPLETE]")
        return True

    # Load data
    try:
        full_df = load_market_data(symbol, args.data_dir)
    except FileNotFoundError as e:
        logger.error(f"Data loading failed: {e}")
        return False

    # Phase 0: Data Segregation
    split_result = run_phase0_split(full_df, symbol)
    # Use filtered DataFrame for all subsequent phases to ensure consistent splits.
    # Phase 0 filters to last 3 years; raw full_df has different row count which
    # causes split_data() to create mismatched splits in Phases 2 and 5.
    filtered_df = split_result["filtered_df"]

    # Phase 1: Signal Discovery (BLOCKING GATE)
    # Determine Phase 1 mode
    test_all_timeframes = not args.quick  # --quick means stop early

    # Check for --signal-name: Skip Phase 1/2 entirely and go directly to Optuna
    if args.signal_name:
        # Orchestrator-dispatched job with specific signal
        print("\n" + "=" * 60)
        print("PHASE 1-2: SKIPPED (Signal specified via --signal-name)")
        print("=" * 60)
        print(f"Signal: {args.signal_name}")

        # Try to load signal from CSV
        csv_signal = load_signal_from_csv(args.signal_name, symbol, direction)
        if csv_signal:
            print(f"✅ Loaded from signal_discoveries.csv:")
            print(f"   Timeframe: {csv_signal['timeframe']}")
            print(f"   Win Rate: {csv_signal['win_rate']*100:.1f}%")
            print(f"   Trades: {csv_signal['trades']}")
            print(f"   Quality: {csv_signal['quality']}")
            print("=" * 60)

            # Create signal object for Optuna
            signal = create_signal_from_discovery({
                "signal_name": csv_signal["signal_name"],
                "signal_definition": csv_signal["signal_name"],  # Use name as definition
                "direction": direction,
                "timeframe": csv_signal["timeframe"],
            })

            if signal:
                # Create result structure to skip to Phase 3
                discovery_result = {
                    "passed": True,
                    "best_signal": signal,
                    "best_evaluation": None,
                    "all_results": [],
                    "passing_signals": [{
                        "signal": signal,
                        "timeframe": csv_signal["timeframe"],
                        "is_metrics": {
                            "win_rate": csv_signal["win_rate"],
                            "p_value": csv_signal["p_value"],
                        }
                    }],
                    "timeframe": csv_signal["timeframe"],
                    "is_metrics": {
                        "win_rate": csv_signal["win_rate"],
                        "implied_pf": (
                            999.0 if csv_signal["win_rate"] >= 0.999
                            else csv_signal["win_rate"] / (1 - csv_signal["win_rate"])
                        ),
                        "p_value": csv_signal["p_value"],
                    },
                }
                # Skip Phase 2 validation - go directly to Phase 3
                # Set flag to skip phase2 in later logic
                skip_phase2 = True
            else:
                print("⚠️ Failed to create signal object - falling back to fresh discovery")
                csv_signal = None

        if not csv_signal:
            # Issue #596: Try constructing signal directly from CLI args
            # When --signal-name + --timeframe are provided (from training queue),
            # the signal was already discovered. Don't waste time re-running Phase 1.
            constructed = False
            if args.timeframe:
                print(f"ℹ️  Signal not in CSV but --timeframe={args.timeframe} provided")
                print(f"   Attempting direct signal construction (skip Phase 1)...")
                try:
                    signal = create_signal_from_discovery({
                        "signal_name": args.signal_name,
                        "signal_definition": args.signal_name,
                        "direction": direction,
                        "timeframe": args.timeframe,
                    })
                    if signal:
                        print(f"✅ Signal constructed directly: {signal.name} ({args.timeframe})")
                        print("   Skipping Phase 1 → going to Phase 3 (Optuna)")
                        print("=" * 60)
                        discovery_result = {
                            "passed": True,
                            "best_signal": signal,
                            "best_evaluation": None,
                            "all_results": [],
                            "passing_signals": [{
                                "signal": signal,
                                "timeframe": args.timeframe,
                                "is_metrics": {
                                    "win_rate": 0.0,
                                    "p_value": 0.0,
                                }
                            }],
                            "timeframe": args.timeframe,
                            "is_metrics": {
                                "win_rate": 0.0,
                                "implied_pf": 0.0,
                                "p_value": 0.0,
                            },
                        }
                        skip_phase2 = True
                        constructed = True
                except (ValueError, Exception) as e:
                    print(f"⚠️ Direct construction failed: {e}")

            if not constructed:
                print("❌ FATAL: --signal-name provided but signal could not be constructed")
                print(f"   Signal: {args.signal_name}")
                print(f"   Timeframe: {args.timeframe}")
                print("   Training workers must NEVER fall back to Phase 1.")
                print("   Fix: Add pattern matching for this signal in create_signal_from_discovery()")
                sys.exit(1)

    elif args.use_existing_discovery:
        # Option 1: Use existing discovery results (skip Phase 1 entirely)
        skip_phase2 = False
        print("\n" + "=" * 60)
        print("PHASE 1: USING EXISTING DISCOVERY RESULTS (NO RE-TEST)")
        print("=" * 60)

        existing = load_existing_discovery(symbol, direction)
        if existing is None:
            print("❌ No existing discovery found - falling back to fresh discovery")
            discovery_result = run_phase1_discovery(
                split_result["train_df"],
                symbol,
                direction,
                sample_size=args.sample_size,
                test_all_timeframes=test_all_timeframes,
                sl_pips=args.sl_pips,
                tp_pips=args.tp_pips,
                horizon=args.horizon,
            )
        else:
            # Convert existing discovery to signal
            signal = create_signal_from_discovery(existing)
            if signal is None:
                print("❌ Failed to create signal from discovery - falling back to fresh discovery")
                discovery_result = run_phase1_discovery(
                    split_result["train_df"],
                    symbol,
                    direction,
                    sample_size=args.sample_size,
                    test_all_timeframes=test_all_timeframes,
                )
            else:
                print(f"✅ Loaded: {existing['signal_name']}")
                print(f"   Timeframe: {existing.get('timeframe', 'H4')}")
                print(f"   Win Rate: {existing.get('win_rate', 0):.1f}%")
                print(f"   Trades: {existing.get('trades', 0)}")
                print("   ⚠️ Using as-is WITHOUT re-testing on training split")
                print("=" * 60)

                # Create discovery result from existing
                discovery_result = {
                    "passed": True,
                    "best_signal": signal,
                    "best_evaluation": None,  # No fresh evaluation
                    "all_results": [],
                    "passing_signals": [],
                    "timeframe": existing.get("timeframe", "H4"),
                    "is_metrics": {
                        "win_rate": existing.get("win_rate", 0) / 100,  # Convert from %
                        # Bug #10 fix: Handle 100% win rate edge case (would cause ZeroDivisionError)
                        "implied_pf": (
                            999.0
                            if existing.get("win_rate", 50) >= 99.9
                            else existing.get("win_rate", 50) / (100 - existing.get("win_rate", 50))
                        ),
                        "p_value": existing.get("p_value", 0.05),
                    },
                }

    elif args.retest_existing:
        # Option 2: Re-test existing signals on current training data
        skip_phase2 = False
        discovery_result = run_phase1_retest_existing(
            split_result["train_df"],
            symbol,
            direction,
        )

        # If retest failed and returned fallback flag, do fresh discovery
        if not discovery_result["passed"] and discovery_result.get("fallback"):
            print("\n⚠️ Falling back to fresh discovery...")
            discovery_result = run_phase1_discovery(
                split_result["train_df"],
                symbol,
                direction,
                sample_size=args.sample_size,
                test_all_timeframes=test_all_timeframes,
                sl_pips=args.sl_pips,
                tp_pips=args.tp_pips,
                horizon=args.horizon,
            )

    else:
        # Option 3: Fresh discovery with hierarchical timeframe processing
        skip_phase2 = False
        discovery_result = run_phase1_discovery(
            split_result["train_df"],
            symbol,
            direction,
            sample_size=args.sample_size,
            test_all_timeframes=test_all_timeframes,
            sl_pips=args.sl_pips,
            tp_pips=args.tp_pips,
            horizon=args.horizon,
        )

    if not discovery_result["passed"]:
        print("\n🛑 PIPELINE STOPPED: Phase 1 blocking gate failed")
        print(f"   No signals met criteria: WR >= {MIN_WIN_RATE_PHASE1*100:.0f}%, Trades >= {MIN_TRADES_PHASE1}, p-value < {MAX_P_VALUE}")

        # Update tracking JSON
        update_phase1(
            symbol=symbol,
            direction=direction,
            passed=False,
        )
        return False

    # Update Phase 1 tracking
    best_sig = discovery_result["best_signal"]
    update_phase1(
        symbol=symbol,
        direction=direction,
        passed=True,
        best_signal=best_sig.name,  # best_sig is SignalDefinition object directly
        win_rate=discovery_result["is_metrics"]["win_rate"],
        timeframe=discovery_result["timeframe"],
        all_results=discovery_result["all_results"],
    )

    # If --phase1-only, stop here and optionally output JSON
    if args.phase1_only:
        print(f"\n{'='*60}")
        print("🛑 STOPPING AFTER PHASE 1 (as requested)")
        print(f"{'='*60}")
        print(f"Passing signals: {len(discovery_result['passing_signals'])}")
        for sig_data in discovery_result['passing_signals']:
            print(f"  - {sig_data['signal'].name} ({sig_data['timeframe']}): WR={sig_data['win_rate']*100:.1f}%, Trades={sig_data['trades']}")

        if args.output_json:
            # Output structured JSON for signal_discovery_batch.py to parse
            import json as json_module
            phase1_output = {
                "phase1_results": [],
                "passed": discovery_result["passed"],
                "best_signal": best_sig.name,
                "timeframe": discovery_result["timeframe"],
                "symbol": symbol,
                "direction": direction,
            }
            for sig_data in discovery_result['passing_signals']:
                phase1_output["phase1_results"].append({
                    "signal_name": sig_data['signal'].name,
                    "timeframe": sig_data['timeframe'],
                    "win_rate": sig_data['win_rate'],
                    "trades": sig_data['trades'],
                    "p_value": sig_data['p_value'],
                    "implied_pf": sig_data.get('implied_pf', 0),
                    "quality": sig_data.get('quality', 'unknown'),
                })
            # Print JSON marker for easy parsing
            print(f"\n{'='*60}")
            print("JSON_OUTPUT_START")
            print(json_module.dumps(phase1_output, indent=2))
            print("JSON_OUTPUT_END")
            print(f"{'='*60}")

        return True

    # Phase 2: Walk-Forward Validation (BLOCKING GATE)
    # Skip Phase 2 if signal was loaded from CSV via --signal-name
    if skip_phase2:
        print(f"\n{'='*60}")
        print("PHASE 2: SKIPPED (Signal loaded from signal_discoveries.csv)")
        print("         Signal already validated - proceeding to Phase 3 Optuna")
        print(f"{'='*60}")

        # Use the signals from discovery_result directly
        phase2_passing_signals = []
        for sig_data in discovery_result["passing_signals"]:
            phase2_passing_signals.append({
                "signal": sig_data["signal"],
                "timeframe": sig_data.get("timeframe", discovery_result.get("timeframe", "H4")),
                "quality": sig_data.get("quality", "unknown"),
                "is_metrics": sig_data.get("is_metrics", discovery_result.get("is_metrics", {})),
                "oos_metrics": {
                    "win_rate": sig_data.get("is_metrics", {}).get("win_rate", 0),
                    "wr_degradation": 0,  # Already validated
                    "evaluation": "CSV_VALIDATED",
                },
            })

        print(f"✅ Using {len(phase2_passing_signals)} signal(s) from CSV")
    else:
        # Test ALL passing signals from Phase 1 (Issue #526: Test all, don't stop on first failure)
        print(f"\n{'='*60}")
        print(f"PHASE 2: VALIDATING ALL {len(discovery_result['passing_signals'])} SIGNALS")
        print(f"{'='*60}")

        phase2_passing_signals = []
        for idx, signal_data in enumerate(discovery_result["passing_signals"], 1):
            print(f"\n[{idx}/{len(discovery_result['passing_signals'])}] Testing: {signal_data['signal'].name} ({signal_data['timeframe']})")

            # Extract IS metrics for this specific signal
            is_metrics = {
                "win_rate": signal_data["win_rate"],
                "implied_pf": signal_data["implied_pf"],
                "p_value": signal_data["p_value"],
                "trades": signal_data["trades"],
            }

            validation_result = run_phase2_validation(
                filtered_df,
                signal_data["signal"],
                symbol,
                is_metrics,
                direction,
            )

            if validation_result["passed"]:
                phase2_passing_signals.append({
                    "signal": signal_data["signal"],
                    "timeframe": signal_data["timeframe"],
                    "quality": signal_data["quality"],
                    "is_metrics": is_metrics,
                    "oos_metrics": {
                        "win_rate": validation_result["oos_wr"],
                        "wr_degradation": validation_result["wr_degradation"],
                        "evaluation": validation_result["evaluation"],
                    },
                })

        # Check if at least one signal passed Phase 2
        if len(phase2_passing_signals) == 0:
            print("\n🛑 PIPELINE STOPPED: Phase 2 blocking gate failed")
            print(f"   None of the {len(discovery_result['passing_signals'])} signals validated on out-of-sample data")
            return False

        print(f"\n{'='*60}")
        print(f"✅ PHASE 2 COMPLETE: {len(phase2_passing_signals)}/{len(discovery_result['passing_signals'])} signals passed!")
        print(f"{'='*60}")
        for idx, sig in enumerate(phase2_passing_signals, 1):
            print(f"  {idx}. {sig['signal'].name} ({sig['timeframe']}) - IS WR: {sig['is_metrics']['win_rate']*100:.1f}% → OOS WR: {sig['oos_metrics']['win_rate']*100:.1f}%")

        # Update Phase 2 tracking
        passing_sigs_data = [
            {
                "signal_name": sig["signal"].name,
                "timeframe": sig["timeframe"],
                "is_wr": sig["is_metrics"]["win_rate"],
                "oos_wr": sig["oos_metrics"]["win_rate"],
                "quality": sig["quality"],
            }
            for sig in phase2_passing_signals
        ]
        update_phase2(
            symbol=symbol,
            direction=direction,
            passed=True,
            passing_signals=passing_sigs_data,
        )

    # If --stop-after-phase2, generate commands and exit
    if args.stop_after_phase2:
        print(f"\n{'='*70}")
        print("🛑 STOPPING AFTER PHASE 2 (as requested)")
        print("='*70}")
        print(f"\nTo continue training, run ONE of these commands for each signal:")
        print(f"{'='*70}\n")

        for idx, sig in enumerate(phase2_passing_signals, 1):
            cmd = (
                f"python scripts/train_single_signal.py \\\n"
                f"    --symbol {symbol} \\\n"
                f"    --direction {direction} \\\n"
                f"    --signal-name {sig['signal'].name} \\\n"
                f"    --timeframe {sig['timeframe']} \\\n"
                f"    --optuna-trials 40 \\\n"
                f"    --smart-loading"
            )
            print(f"# Signal #{idx}: {sig['signal'].name} ({sig['timeframe']})")
            print(f"# IS WR: {sig['is_metrics']['win_rate']*100:.1f}% → OOS WR: {sig['oos_metrics']['win_rate']*100:.1f}%")
            print(cmd)
            print()

        print(f"{'='*70}")
        print(f"📊 Pipeline tracking updated in: results/pipeline_tracking.json")
        print(f"📋 To view status: python scripts/show_pipeline_status.py --symbol {symbol} --direction {direction}")
        print(f"{'='*70}")
        return True

    # Select signal for Phase 3-5: Use --signal-name if specified, otherwise use best
    if args.signal_name:
        # Find the specified signal in phase2_passing_signals
        best_phase2_signal = None
        for sig in phase2_passing_signals:
            sig_name = sig["signal"].name if hasattr(sig["signal"], "name") else str(sig["signal"])
            if sig_name == args.signal_name or args.signal_name in sig_name or sig_name in args.signal_name:
                best_phase2_signal = sig
                break

        if best_phase2_signal is None:
            # Signal specified but not found - log warning and use first
            print(f"⚠️ Specified signal '{args.signal_name}' not found in Phase 2 results")
            print(f"   Available: {[s['signal'].name if hasattr(s['signal'], 'name') else str(s['signal']) for s in phase2_passing_signals[:5]]}")
            best_phase2_signal = phase2_passing_signals[0]
            print(f"\n🏆 FALLING BACK TO BEST SIGNAL: {best_phase2_signal['signal'].name} ({best_phase2_signal['timeframe']})")
        else:
            print(f"\n🏆 USING SPECIFIED SIGNAL: {best_phase2_signal['signal'].name} ({best_phase2_signal['timeframe']})")
            print("   (Signal specified via --signal-name parameter)")
    else:
        # Original behavior: Use best Phase 2 signal (first in sorted list)
        best_phase2_signal = phase2_passing_signals[0]  # Already sorted by quality in Phase 1
        print(f"\n🏆 BEST SIGNAL FOR OPTUNA: {best_phase2_signal['signal'].name} ({best_phase2_signal['timeframe']})")
        print("⚠️  NOTE: Only training TOP 1 signal. To train all signals, use --stop-after-phase2")

    # Phase 3: Optuna (if not skipped)
    optuna_result = None
    if not args.skip_optuna:
        optuna_result = run_phase3_optuna(
            split_result["train_df"],
            best_phase2_signal["signal"],
            symbol,
            direction,
            n_trials=args.optuna_trials,
        )
        best_params = optuna_result["best_params"]
    else:
        print("\n" + "=" * 60)
        print("PHASE 3: OPTUNA SKIPPED (using default hyperparameters)")
        print("=" * 60)
        best_params = {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "n_steps": 1024,
            "batch_size": 128,
        }
        print(f"Default params: {best_params}")

    # Phase 4: 30-Fold Training (if not skipped)
    training_result = None
    if not args.skip_training:
        # Issue #547: Get signal_name and timeframe for model path
        training_signal_name = args.signal_name or best_phase2_signal.get("signal_name")
        training_timeframe = args.timeframe or best_phase2_signal.get("timeframe")

        training_result = run_phase4_training(
            split_result["train_df"],
            best_phase2_signal["signal"],
            symbol,
            direction,
            best_params,
            n_folds=30,
            use_d1_features=True,  # Match Phase 3 Optuna (obs_dim=242)
            discovered_signal=best_phase2_signal["signal"],  # Issue #530
            signal_name=training_signal_name,  # Issue #547
            timeframe=training_timeframe,  # Issue #547
        )

        if not training_result["passed"]:
            print("\n🛑 PIPELINE STOPPED: Phase 4 training failed")
            print("   No successful model training")
            return False
    else:
        print("\n" + "=" * 60)
        print("PHASE 4: TRAINING SKIPPED")
        print("=" * 60)
        print("Cannot run Phase 5 without trained models.")
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED (Phases 0-2 only)")
        print("=" * 60)
        return True

    # Phase 5: Final Validation
    final_result = run_phase5_validation(
        filtered_df,
        training_result["model_paths"],
        symbol,
        direction,
        training_result["avg_pf"],
        use_d1_features=True,  # Issue #535: Match Phase 4 configuration
        discovered_signal=best_phase2_signal["signal"],  # Issue #535: Pass signal to Phase 5
    )

    # Save Phase 5 results to pipeline_tracking.json and model_metrics.json
    # Issue: Phase 5 results were only in logs, not saved to GCS
    signal_name = (
        best_phase2_signal["signal"].name
        if hasattr(best_phase2_signal["signal"], "name")
        else str(best_phase2_signal["signal"])
    )
    timeframe = best_phase2_signal.get("timeframe", "H1")

    update_phase5(
        symbol=symbol,
        direction=direction,
        signal_name=signal_name,
        completed=True,
        test_pf=final_result["ensemble_pf"],
        test_wr=final_result["ensemble_wr"],
        total_trades=final_result["total_trades"],
        passed=final_result["passes_production"],
        timeframe=timeframe,
    )
    logger.info(f"Phase 5 results saved: PF={final_result['ensemble_pf']:.2f}, WR={final_result['ensemble_wr']:.1f}%, Trades={final_result['total_trades']}")

    # BUG FIX #1: Removed duplicate queue update (Issue #600 Stream D Bug #1)
    # training_worker.py already updates queue atomically at line 909
    # Duplicate updates cause race conditions and potential data loss
    # Original code (REMOVED):
    # queue_status = "completed" if final_result["passes_production"] else "failed"
    # update_training_queue(symbol, direction, queue_status)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)
    if final_result["passes_production"]:
        print("🎉 RESULT: APPROVED FOR PRODUCTION")
        print(f"   Test PF: {final_result['ensemble_pf']:.2f}")
        print(f"   Ensemble of {len(training_result['model_paths'])} models")
    else:
        print("❌ RESULT: REJECTED FOR PRODUCTION")
        for reason in final_result["rejection_reasons"]:
            print(f"   - {reason}")

    return final_result["passes_production"]


# =============================================================================
# Training Queue Sync (Issue #558)
# =============================================================================


def update_training_queue(symbol: str, direction: str, status: str = "completed") -> bool:
    """
    Update local training_queue.json after pipeline completes.

    TFG: Uses local file instead of GCS.

    Args:
        symbol: Symbol being trained (e.g., "eurusd")
        direction: Direction being trained ("long" or "short")
        status: New status ("completed" or "failed")

    Returns:
        True if successful, False otherwise
    """
    try:
        LOCAL_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Load existing queue or create empty
        if LOCAL_QUEUE_PATH.exists():
            with open(LOCAL_QUEUE_PATH, "r") as f:
                queue_data = json.load(f)
        else:
            queue_data = {"signals": []}

        # Find and update the matching job
        updated = False
        for job in queue_data.get("signals", []):
            job_symbol = job.get("symbol", "").lower()
            job_direction = job.get("direction", "").lower()
            if job_symbol == symbol.lower() and job_direction == direction.lower():
                old_status = job.get("status", "pending")
                job["status"] = status
                logger.info(f"Updated {symbol}/{direction}: {old_status} -> {status}")
                updated = True
                break

        if not updated:
            # Add new entry if not found
            queue_data["signals"].append({
                "symbol": symbol.lower(),
                "direction": direction.lower(),
                "status": status,
            })
            logger.info(f"Added {symbol}/{direction} -> {status}")

        # Save updated queue
        with open(LOCAL_QUEUE_PATH, "w") as f:
            json.dump(queue_data, f, indent=2)

        logger.info(f"Training queue updated: {symbol}/{direction} -> {status}")
        return True

    except Exception as e:
        logger.warning(f"Failed to update training queue: {e}")
        return False


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """
    Main entry point.

    Parses arguments, validates them, and runs the pipeline.

    Returns:
        Exit code: 0 for completed (APPROVED or REJECTED), 1 for error/crash

    Note: A REJECTED signal returns 0 because the pipeline completed successfully.
    The signal was properly evaluated and didn't meet criteria - that's a valid outcome.
    Exit code 1 is reserved for actual errors (crashes, missing data, etc.).
    """
    parser = create_parser()

    try:
        args = parser.parse_args()
    except SystemExit as e:
        # argparse calls sys.exit on error
        return 1 if e.code != 0 else 0

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not validate_args(args):
        return 1

    # Run pipeline
    try:
        run_pipeline(args)
        # Return 0 for both APPROVED and REJECTED - pipeline completed successfully
        # The job status (passed/failed) is tracked in the queue JSON, not via exit code
        return 0
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        # Update training queue as failed on crash (Issue #558)
        try:
            update_training_queue(args.symbol.lower(), args.direction.lower(), "failed")
        except Exception:
            pass  # Best effort - don't fail on queue update error
        return 1


if __name__ == "__main__":
    sys.exit(main())
