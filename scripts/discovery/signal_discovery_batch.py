#!/usr/bin/env python3
"""
Signal Discovery Batch Runner

Runs Phase 1 signal discovery for all symbols × directions × timeframes.
Outputs: signal_discoveries.csv (one row per signal)

Schedule: Every 14 days via cron or Cloud Scheduler

Usage:
    python scripts/signal_discovery_batch.py
    python scripts/signal_discovery_batch.py --symbols eurusd,gbpusd --dry-run
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from multiprocessing import Pool, cpu_count

# Lifecycle management (Issue #611)
try:
    from scripts.lifecycle.signal_lifecycle_manager import (
        SignalLifecycleManager,
        LifecycleState,
    )
    LIFECYCLE_ENABLED = True
except ImportError:
    LIFECYCLE_ENABLED = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SYMBOLS = ['eurusd', 'gbpusd', 'usdjpy', 'eurjpy', 'usdcad', 'usdchf', 'eurcad', 'eurgbp']
DIRECTIONS = ['long', 'short']
TIMEFRAMES = ['M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D1']

# Paths - script is in scripts/discovery/, so go up 2 levels to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'
CONFIG_DIR = PROJECT_ROOT / 'config'
DATA_DIR = Path(os.environ.get('DATA_DIR', 'data/indicators'))

# Output files
SIGNAL_DISCOVERIES_CSV = RESULTS_DIR / 'signal_discoveries.csv'
DISCOVERY_BATCHES_JSON = RESULTS_DIR / 'discovery_batches.json'

# CSV columns for new simplified format
CSV_COLUMNS = [
    'signal_id',
    'discovery_batch',
    'discovery_date',
    'symbol',
    'direction',
    'signal_name',
    'timeframe',
    'win_rate',
    'trades',
    'p_value',
    'quality',
    'profit_factor_estimate',
    'status',
    'training_status',
    'previous_status'
]

# Filter criteria (can be overridden by config/training_filter.yaml)
DEFAULT_FILTER = {
    'min_win_rate': 0.58,
    'min_trades': 190,
    'max_p_value': 0.02,
    'min_quality': ['excellent', 'good'],
    'require_combo': True
}

# Default combo patterns (2+ indicator signals)
# ALL signals must combine 2+ indicators - single indicators are NOT allowed
DEFAULT_COMBO_PATTERNS = [
    '*RSI_Stoch*', '*Stoch_RSI*', '*SMA*RSI*', '*EMA_RSI*',
    '*MACD_Stoch*', '*BB_RSI*', '*RSI_BB*', '*Volume_EMA*',
    '*SMA*BB*', '*SMA*Stoch*', '*Triple*', '*MACD*RSI*',
    '*SMA*MACD*', '*SMA*cross*', '*EMA*MACD*'  # SMA/EMA + MACD, SMA crossovers
]

# Default excluded patterns (single-indicator signals)
DEFAULT_EXCLUDED_PATTERNS = [
    'Stoch_K_oversold*', 'Stoch_K_overbought*', 'RSI14_*',
    'EMA12_cross_EMA26*', 'SMA20_cross_SMA50*', 'MACD_cross_signal*',
    'BB_*_touch*', 'Volume_spike*'
]


def load_filter_config() -> Dict:
    """Load filter criteria from config file or use defaults."""
    config_path = CONFIG_DIR / 'training_filter.yaml'
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded filter config from {config_path}")

                # Get queue_filter (preferred) or filter section
                filter_config = config.get('queue_filter', config.get('filter', DEFAULT_FILTER))

                # Also load pattern lists from config
                filter_config['combo_patterns'] = config.get('combo_patterns', DEFAULT_COMBO_PATTERNS)
                filter_config['excluded_patterns'] = config.get('excluded_patterns', DEFAULT_EXCLUDED_PATTERNS)

                return filter_config
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")

    # Return defaults with patterns
    result = dict(DEFAULT_FILTER)
    result['combo_patterns'] = DEFAULT_COMBO_PATTERNS
    result['excluded_patterns'] = DEFAULT_EXCLUDED_PATTERNS
    return result


def generate_signal_id(symbol: str, direction: str, signal_name: str, timeframe: str) -> str:
    """Generate unique signal ID."""
    return f"{symbol}_{direction}_{signal_name}_{timeframe}"


def load_existing_signals(csv_path: Path) -> Dict[str, Dict]:
    """
    Load existing signals from CSV into a dict keyed by signal_id.
    This enables proper deduplication when merging new discoveries.
    """
    existing = {}
    if not csv_path.exists():
        return existing

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                signal_id = row.get('signal_id', '')
                if signal_id:
                    existing[signal_id] = row
        logger.info(f"Loaded {len(existing)} existing signals from {csv_path}")
    except Exception as e:
        logger.warning(f"Failed to load existing signals: {e}")

    return existing


def run_phase1_discovery(symbol: str, direction: str, data_dir: Path) -> List[Dict]:
    """
    Run Phase 1 signal discovery for a symbol/direction combination.
    Returns list of discovered signals.
    """
    logger.info(f"Running Phase 1 discovery: {symbol} {direction}")

    # Check if data file exists
    data_file = data_dir / f'technical_indicator_{symbol}.csv'
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return []

    # Run the hybrid pipeline in Phase 1 only mode
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'scripts' / 'pipelines' / 'run_hybrid_v4_pipeline.py'),
        '--symbol', symbol,
        '--direction', direction,
        '--data-dir', str(data_dir),
        '--phase1-only',  # Only run Phase 1
        '--output-json'   # Output results as JSON for parsing
    ]

    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{PROJECT_ROOT}:{PROJECT_ROOT / 'src'}"
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per symbol/direction
            cwd=str(PROJECT_ROOT),
            env=env
        )

        if result.returncode != 0:
            logger.warning(f"Phase 1 subprocess failed for {symbol} {direction}, trying fallback")
            logger.warning(f"  stderr: {result.stderr[:500] if result.stderr else 'none'}")
            # Try fallback to existing comprehensive_discovery files
            return parse_comprehensive_discovery(symbol, direction)

        # Parse JSON output from stdout - look for JSON_OUTPUT_START marker
        output = result.stdout
        json_start_marker = output.find('JSON_OUTPUT_START')
        json_end_marker = output.find('JSON_OUTPUT_END')

        if json_start_marker != -1 and json_end_marker != -1:
            # Extract JSON between markers
            json_str = output[json_start_marker + len('JSON_OUTPUT_START'):json_end_marker].strip()
            try:
                data = json.loads(json_str)
                signals = data.get('phase1_results', [])
                logger.info(f"Discovered {len(signals)} signals for {symbol} {direction}")
                return signals
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON output: {e}")
                logger.error(f"  JSON string: {json_str[:200]}...")
                return parse_comprehensive_discovery(symbol, direction)

        # Fallback: Try older JSON format
        json_start = output.find('{"phase1_results":')
        if json_start == -1:
            json_start = output.find('{"signals":')

        if json_start != -1:
            json_end = output.rfind('}') + 1
            json_str = output[json_start:json_end]
            try:
                data = json.loads(json_str)
                signals = data.get('phase1_results', data.get('signals', []))
                logger.info(f"Discovered {len(signals)} signals for {symbol} {direction}")
                return signals
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON output: {e}")
                return []

        # Final fallback: parse from comprehensive_discovery JSON files
        logger.warning(f"No JSON output found, falling back to comprehensive_discovery")
        return parse_comprehensive_discovery(symbol, direction)

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout running Phase 1 for {symbol} {direction}")
        return []
    except Exception as e:
        logger.error(f"Error running Phase 1 for {symbol} {direction}: {e}")
        return []


def parse_comprehensive_discovery(symbol: str, direction: str) -> List[Dict]:
    """
    Fallback: Parse signals from comprehensive_discovery JSON files.
    """
    json_file = RESULTS_DIR / 'comprehensive_discovery' / f'{symbol}_all_signals.json'
    if not json_file.exists():
        return []

    try:
        with open(json_file) as f:
            data = json.load(f)

        signals = data.get('top_signals', [])
        # Filter by direction
        filtered = [s for s in signals if s.get('direction', '').lower() == direction.lower()]
        return filtered
    except Exception as e:
        logger.error(f"Failed to parse {json_file}: {e}")
        return []


def matches_pattern(signal_name: str, pattern: str) -> bool:
    """
    Check if signal_name matches a glob-like pattern.
    Supports * as wildcard for any characters.
    """
    import fnmatch
    return fnmatch.fnmatch(signal_name, pattern)


def is_combo_signal(signal_name: str, combo_patterns: List[str], excluded_patterns: List[str]) -> bool:
    """
    Check if signal is a 2+ indicator combo signal.

    A signal passes the combo filter if:
    1. It matches at least one combo_pattern, AND
    2. It does NOT match any excluded_pattern

    Args:
        signal_name: The signal name to check
        combo_patterns: List of glob patterns for valid combo signals
        excluded_patterns: List of glob patterns for excluded single-indicator signals

    Returns:
        True if signal is a valid combo signal
    """
    # First check if signal is explicitly excluded
    for pattern in excluded_patterns:
        if matches_pattern(signal_name, pattern):
            return False

    # Then check if signal matches any combo pattern
    for pattern in combo_patterns:
        if matches_pattern(signal_name, pattern):
            return True

    return False


def filter_signals(signals: List[Dict], filter_config: Dict) -> List[Dict]:
    """
    Apply filter criteria to signals.

    Filters applied:
    1. min_win_rate: Minimum win rate threshold (decimal, e.g., 0.58 = 58%)
    2. min_trades: Minimum number of trades for statistical significance
    3. max_p_value: Maximum p-value for statistical significance
    4. min_quality: List of acceptable quality levels
    5. require_combo: If True, only allow 2+ indicator combo signals

    Returns:
        List of signals that pass all filters
    """
    filtered = []
    require_combo = filter_config.get('require_combo', False)
    combo_patterns = filter_config.get('combo_patterns', DEFAULT_COMBO_PATTERNS)
    excluded_patterns = filter_config.get('excluded_patterns', DEFAULT_EXCLUDED_PATTERNS)

    combo_rejected = 0
    stats_rejected = 0

    for s in signals:
        signal_name = s.get('signal_name', '')
        win_rate = s.get('win_rate', 0)
        if isinstance(win_rate, str):
            win_rate = float(win_rate)
        if win_rate > 1:  # Convert percentage to decimal
            win_rate = win_rate / 100

        trades = s.get('trades', 0)
        p_value = s.get('p_value', 1)
        quality = s.get('quality', s.get('status', 'unknown')).lower()

        # Apply combo filter FIRST (most restrictive for overlapping trade problem)
        if require_combo:
            if not is_combo_signal(signal_name, combo_patterns, excluded_patterns):
                combo_rejected += 1
                continue

        # Apply statistical filters
        if win_rate < filter_config.get('min_win_rate', 0):
            stats_rejected += 1
            continue
        if trades < filter_config.get('min_trades', 0):
            stats_rejected += 1
            continue
        if p_value > filter_config.get('max_p_value', 1):
            stats_rejected += 1
            continue

        # Check quality (supports both 'quality' and 'min_quality' keys)
        min_quality = filter_config.get('quality', filter_config.get('min_quality', []))
        if isinstance(min_quality, list) and len(min_quality) > 0 and quality not in [q.lower() for q in min_quality]:
            stats_rejected += 1
            continue

        filtered.append(s)

    if require_combo:
        logger.info(f"Combo filter: {combo_rejected} signals rejected (single-indicator)")
    logger.info(f"Stats filter: {stats_rejected} signals rejected (WR/trades/p-value/quality)")

    return filtered


def write_signal_discoveries_csv(
    signals: List[Dict],
    batch_id: str,
    discovery_date: str,
    append: bool = True
) -> int:
    """
    Write signals to signal_discoveries.csv in the new simplified format.
    Handles duplicates by:
    - Preserving training_status from existing signals if already progressed
    - Updating win_rate, trades, p_value with fresh discovery data
    Returns number of signals written.
    """
    # Load existing signals for deduplication
    existing_signals = load_existing_signals(SIGNAL_DISCOVERIES_CSV) if append else {}

    # Training statuses that should NOT be overwritten
    PRESERVE_STATUSES = ['in_progress', 'completed', 'production']

    # Merge new signals with existing
    merged_signals = dict(existing_signals)  # Start with existing
    new_count = 0
    updated_count = 0

    for s in signals:
        symbol = s.get('symbol', '').lower()
        direction = s.get('direction', '').lower()
        signal_name = s.get('signal_name', '')
        timeframe = s.get('timeframe', '')

        win_rate = s.get('win_rate', 0)
        if isinstance(win_rate, str):
            win_rate = float(win_rate)
        if win_rate > 1:
            win_rate = win_rate / 100

        signal_id = generate_signal_id(symbol, direction, signal_name, timeframe)

        row = {
            'signal_id': signal_id,
            'discovery_batch': batch_id,
            'discovery_date': discovery_date,
            'symbol': symbol,
            'direction': direction,
            'signal_name': signal_name,
            'timeframe': timeframe,
            'win_rate': round(win_rate, 4),
            'trades': s.get('trades', 0),
            'p_value': s.get('p_value', ''),
            'quality': s.get('quality', s.get('status', '')),
            'profit_factor_estimate': s.get('profit_factor', ''),
            'status': 'pending',
            'training_status': 'not_started'
        }

        # Check if signal already exists
        if signal_id in existing_signals:
            existing = existing_signals[signal_id]
            existing_training_status = existing.get('training_status', 'not_started')

            # Preserve training_status if it has progressed
            if existing_training_status in PRESERVE_STATUSES:
                row['training_status'] = existing_training_status
                row['status'] = existing.get('status', 'pending')
                logger.debug(f"Preserving training_status '{existing_training_status}' for {signal_id}")
            elif existing_training_status == 'failed':
                # Requeue failed signal that was rediscovered
                row['training_status'] = 'not_started'
                row['status'] = 'pending'
                row['previous_status'] = 'failed'
                logger.info(f"Requeuing failed signal {signal_id} (rediscovered in batch {batch_id})")

            updated_count += 1
        else:
            new_count += 1

        merged_signals[signal_id] = row

    # Write all merged signals
    written = 0
    with open(SIGNAL_DISCOVERIES_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in merged_signals.values():
            writer.writerow(row)
            written += 1

    logger.info(f"Wrote {written} signals to {SIGNAL_DISCOVERIES_CSV} ({new_count} new, {updated_count} updated)")
    return written


def update_discovery_batches(
    batch_id: str,
    started_at: str,
    completed_at: str,
    symbols_processed: List[str],
    signals_discovered: int,
    signals_passed_filter: int,
    filter_config: Dict
) -> None:
    """Update discovery_batches.json with this batch info."""
    batches_data = {'batches': []}

    if DISCOVERY_BATCHES_JSON.exists():
        try:
            with open(DISCOVERY_BATCHES_JSON) as f:
                batches_data = json.load(f)
        except:
            pass

    # Calculate next batch date (14 days from now)
    next_batch = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    from datetime import timedelta
    next_batch = next_batch + timedelta(days=14)

    batch_info = {
        'batch_id': batch_id,
        'started_at': started_at,
        'completed_at': completed_at,
        'symbols_processed': symbols_processed,
        'signals_discovered': signals_discovered,
        'signals_passed_filter': signals_passed_filter,
        'filter_criteria': filter_config,
        'next_batch_scheduled': next_batch.strftime('%Y-%m-%dT%H:%M:%SZ')
    }

    batches_data['batches'].append(batch_info)

    with open(DISCOVERY_BATCHES_JSON, 'w') as f:
        json.dump(batches_data, f, indent=2)

    logger.info(f"Updated {DISCOVERY_BATCHES_JSON}")


def update_signal_lifecycle(
    signals: List[Dict],
    batch_id: str,
    lifecycle_manager: Optional['SignalLifecycleManager'] = None
) -> Dict:
    """
    Update signal lifecycle tracking for discovered signals (Issue #611).

    Records new discoveries and checks for degradation in existing signals.
    This runs AFTER filter_signals to only track signals that passed filters.

    Args:
        signals: List of filtered signals from discovery
        batch_id: Discovery batch identifier
        lifecycle_manager: Optional manager instance (creates new if None)

    Returns:
        Dict with lifecycle statistics
    """
    if not LIFECYCLE_ENABLED:
        logger.warning("Lifecycle management not enabled (import failed)")
        return {"enabled": False}

    try:
        # Create or use provided manager
        if lifecycle_manager is None:
            lifecycle_manager = SignalLifecycleManager()

        stats = {
            "enabled": True,
            "new_signals": 0,
            "updated_signals": 0,
            "degraded_signals": 0,
            "quarantined_signals": 0,
        }

        for s in signals:
            symbol = s.get('symbol', '').lower()
            direction = s.get('direction', '').lower()
            signal_name = s.get('signal_name', '')
            timeframe = s.get('timeframe', '')

            signal_id = generate_signal_id(symbol, direction, signal_name, timeframe)

            # Prepare metrics
            win_rate = s.get('win_rate', 0)
            if isinstance(win_rate, str):
                win_rate = float(win_rate)
            if win_rate > 1:
                win_rate = win_rate / 100

            metrics = {
                "win_rate": win_rate,
                "trades": s.get('trades', 0),
                "p_value": s.get('p_value'),
                "quality": s.get('quality', s.get('status', '')),
            }

            # Check if existing signal
            is_new = signal_id not in lifecycle_manager.signals

            # Record discovery (handles both new and existing)
            result = lifecycle_manager.record_discovery(
                signal_id=signal_id,
                symbol=symbol,
                direction=direction,
                signal_name=signal_name,
                timeframe=timeframe,
                batch_id=batch_id,
                metrics=metrics,
            )

            if is_new:
                stats["new_signals"] += 1
            else:
                stats["updated_signals"] += 1

                # Check degradation result
                if result and result.is_degraded:
                    stats["degraded_signals"] += 1
                    if result.immediate_quarantine:
                        stats["quarantined_signals"] += 1

        # Save lifecycle state
        lifecycle_manager.save()

        logger.info(
            f"Lifecycle updated: {stats['new_signals']} new, "
            f"{stats['updated_signals']} updated, "
            f"{stats['degraded_signals']} degraded"
        )

        return stats

    except Exception as e:
        logger.error(f"Failed to update lifecycle: {e}")
        return {"enabled": True, "error": str(e)}


def _process_combo(args):
    """
    Process a single (symbol, direction, data_dir) combination.
    Must be at module level for multiprocessing pickle compatibility.
    """
    symbol, direction, data_dir = args
    signals = run_phase1_discovery(symbol, direction, data_dir)
    return (symbol, direction, signals)


def download_previous_csv_from_gcs(**kwargs) -> bool:
    """TFG: No GCS — uses local files only. Previous CSV is already local."""
    logger.info("TFG mode: using local signal_discoveries.csv (no GCS)")
    return SIGNAL_DISCOVERIES_CSV.exists()


def run_discovery_batch(
    symbols: List[str],
    directions: List[str],
    data_dir: Path,
    filter_config: Dict,
    dry_run: bool = False,
    parallel: bool = True
) -> Dict:
    """
    Run full discovery batch for all symbols/directions.
    Returns summary statistics.
    """
    batch_id = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    started_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    logger.info(f"Starting discovery batch: {batch_id}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Directions: {directions}")
    logger.info(f"Filter: {filter_config}")

    all_signals = []
    symbols_processed = []

    # Create list of (symbol, direction, data_dir) combinations
    # data_dir must be included in tuple for module-level function (pickle compatibility)
    combinations = [(s, d, data_dir) for s in symbols for d in directions]

    if parallel and not dry_run:
        # Parallel execution
        n_workers = min(cpu_count(), len(combinations))
        logger.info(f"Running {len(combinations)} combinations with {n_workers} parallel workers")

        with Pool(n_workers) as pool:
            results = pool.map(_process_combo, combinations)

        for symbol, direction, signals in results:
            if signals:
                # Add symbol/direction to each signal
                for s in signals:
                    s['symbol'] = symbol
                    s['direction'] = direction
                all_signals.extend(signals)
                if symbol not in symbols_processed:
                    symbols_processed.append(symbol)
    else:
        # Sequential execution
        for symbol, direction, data_dir_path in combinations:
            if dry_run:
                logger.info(f"[DRY RUN] Would run Phase 1 for {symbol} {direction}")
                # Use existing comprehensive_discovery data for dry run
                signals = parse_comprehensive_discovery(symbol, direction)
            else:
                signals = run_phase1_discovery(symbol, direction, data_dir_path)

            if signals:
                for s in signals:
                    s['symbol'] = symbol
                    s['direction'] = direction
                all_signals.extend(signals)
                if symbol not in symbols_processed:
                    symbols_processed.append(symbol)

    logger.info(f"Total signals discovered: {len(all_signals)}")

    # Apply filters
    filtered_signals = filter_signals(all_signals, filter_config)
    logger.info(f"Signals after filter: {len(filtered_signals)}")

    # Write to CSV
    discovery_date = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    if not dry_run:
        # Download previous CSV from GCS so merge logic can preserve statuses
        download_previous_csv_from_gcs()

        # Append to existing CSV to preserve signals from previous batches
        written = write_signal_discoveries_csv(
            filtered_signals,
            batch_id,
            discovery_date,
            append=True  # Merge with existing, don't replace
        )
    else:
        written = len(filtered_signals)
        logger.info(f"[DRY RUN] Would write {written} signals to CSV")

    completed_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    # Update batch tracking
    if not dry_run:
        update_discovery_batches(
            batch_id,
            started_at,
            completed_at,
            symbols_processed,
            len(all_signals),
            len(filtered_signals),
            filter_config
        )

        # Update signal lifecycle (Issue #611)
        lifecycle_stats = update_signal_lifecycle(filtered_signals, batch_id)
        if lifecycle_stats.get("enabled"):
            logger.info(f"Lifecycle: {lifecycle_stats}")

    return {
        'batch_id': batch_id,
        'symbols_processed': symbols_processed,
        'signals_discovered': len(all_signals),
        'signals_passed_filter': len(filtered_signals),
        'signals_written': written
    }


def main():
    parser = argparse.ArgumentParser(description='Signal Discovery Batch Runner')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated list of symbols (default: all)')
    parser.add_argument('--directions', type=str, default='long,short',
                        help='Comma-separated list of directions (default: long,short)')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR),
                        help='Directory containing technical_indicator_*.csv files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simulate discovery using existing comprehensive_discovery data')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Run sequentially instead of parallel')
    parser.add_argument('--upload-gcs', action='store_true',
                        help='Not used in TFG (no GCS)')
    parser.add_argument('--generate-queue', action='store_true',
                        help='Generate training_queue.json after discovery')
    parser.add_argument('--no-lifecycle', action='store_true',
                        help='Disable lifecycle tracking (Issue #611)')

    args = parser.parse_args()

    # Parse symbols and directions
    symbols = args.symbols.split(',') if args.symbols else SYMBOLS
    directions = args.directions.split(',')
    data_dir = Path(args.data_dir)

    # Load filter config
    filter_config = load_filter_config()

    # Run discovery batch
    result = run_discovery_batch(
        symbols=symbols,
        directions=directions,
        data_dir=data_dir,
        filter_config=filter_config,
        dry_run=args.dry_run,
        parallel=not args.no_parallel
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SIGNAL DISCOVERY BATCH COMPLETE")
    print("=" * 60)
    print(f"Batch ID:              {result['batch_id']}")
    print(f"Symbols Processed:     {', '.join(result['symbols_processed'])}")
    print(f"Signals Discovered:    {result['signals_discovered']}")
    print(f"Signals After Filter:  {result['signals_passed_filter']}")
    print(f"Signals Written:       {result['signals_written']}")
    print("=" * 60)

    # Generate training queue FIRST (so it can be uploaded)
    if args.generate_queue and not args.dry_run:
        print("\nGenerating training queue...")
        cmd = [sys.executable, str(PROJECT_ROOT / 'scripts' / 'training' / 'generate_training_queue.py')]
        subprocess.run(cmd)

    # TFG: No GCS upload, no fleet launch — results are local only
    if args.upload_gcs:
        print("\nTFG mode: --upload-gcs ignored (no GCS in local environment)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
