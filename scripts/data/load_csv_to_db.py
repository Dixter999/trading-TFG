#!/usr/bin/env python3
"""
Load CSV data into local PostgreSQL for Trading-TFG.

Requirements: pip install pandas psycopg2-binary

Usage:
    python scripts/data/load_csv_to_db.py                  # load all tables from data/
    python scripts/data/load_csv_to_db.py --sample          # load from data/sample/
    python scripts/data/load_csv_to_db.py --tables rates indicators
    python scripts/data/load_csv_to_db.py --append          # skip TRUNCATE
"""

import argparse
import io
import os
import re
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2 import sql

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_CONFIG = {
    "host": os.getenv("AI_MODEL_DB_HOST", "localhost"),
    "port": int(os.getenv("AI_MODEL_DB_PORT", "5432")),
    "dbname": os.getenv("AI_MODEL_DB_NAME", "ai_model"),
    "user": os.getenv("AI_MODEL_DB_USER", "tfg_user"),
    "password": os.getenv("AI_MODEL_DB_PASSWORD", "tfg_password"),
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # trading-TFG/

# Column order must match init_db.sql table definitions exactly.
RATES_COLS = [
    "rate_time", "symbol", "timeframe",
    "open", "high", "low", "close", "volume", "readable_date",
]

INDICATORS_COLS = [
    "id", "symbol", "timeframe", "timestamp",
    "open", "high", "low", "close", "volume",
    "sma_20", "sma_50", "sma_200",
    "ema_12", "ema_26", "ema_50",
    "rsi_14", "atr_14",
    "bb_upper_20", "bb_middle_20", "bb_lower_20",
    "macd_line", "macd_signal", "macd_histogram",
    "created_at", "updated_at",
    "stoch_k", "stoch_d",
    "ob_bullish_high", "ob_bullish_low",
    "ob_bearish_high", "ob_bearish_low",
]

PAPER_TRADES_COLS = [
    "id", "symbol", "direction",
    "entry_time", "entry_price", "exit_time", "exit_price",
    "sl_price", "tp_price", "size", "pnl_pips",
    "exit_reason", "entry_signal_data", "created_at",
    "entry_model", "signal_timeframe",
]

SIGNAL_DISCOVERIES_COLS = [
    "id", "discovery_date", "symbol", "direction",
    "lookback_years", "total_candles", "train_candles",
    "val_candles", "test_candles",
    "data_start_date", "data_end_date",
    "phase1_signals_tested", "phase1_signals_passed",
    "phase1_duration_seconds",
    "phase2_signals_validated", "phase2_duration_seconds",
    "top_signal_name", "top_signal_timeframe",
    "top_signal_wr", "top_signal_oos_wr",
    "top_signal_trades", "top_signal_pvalue",
    "phase1_results", "phase2_results",
    "pipeline_version", "hostname", "created_at",
    "phase3_trials_completed", "phase3_best_trial_number",
    "phase3_best_pf", "phase3_best_hyperparams",
    "phase3_duration_seconds",
    "phase4_folds_completed", "phase4_avg_pf",
    "phase4_std_pf", "phase4_avg_wr", "phase4_duration_seconds",
    "phase5_test_pf", "phase5_test_wr", "phase5_test_trades",
    "phase5_approved_for_production", "phase5_approval_date",
    "training_locked_at", "training_locked_by",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_connection():
    """Return a new psycopg2 connection."""
    return psycopg2.connect(**DB_CONFIG)


def copy_df_to_table(df: pd.DataFrame, table: str, columns: list[str]) -> int:
    """Bulk-insert a DataFrame via COPY FROM (fastest psycopg2 path)."""
    clean = df[columns].copy()
    clean.replace("", pd.NA, inplace=True)
    buf = io.StringIO()
    clean.to_csv(buf, index=False, header=False, sep="\t", na_rep="\\N")
    buf.seek(0)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.copy_from(buf, table, sep="\t", null="\\N", columns=columns)
        conn.commit()
        return len(df)
    finally:
        conn.close()


def truncate_table(table: str) -> None:
    """TRUNCATE a table and reset its SERIAL sequence."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(
                sql.Identifier(table)
            ))
        conn.commit()
    finally:
        conn.close()


def parse_rates_filename(filename: str) -> tuple[str, str] | None:
    """Extract (SYMBOL, TIMEFRAME) from e.g. 'eurusd_h1_rates.csv'."""
    m = re.match(r"^([a-z]{6})_([a-z0-9]+)_rates\.csv$", filename, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).upper(), m.group(2).upper()


# ---------------------------------------------------------------------------
# Per-table loaders
# ---------------------------------------------------------------------------

def _load_single_rates_file(args: tuple) -> tuple[str, int]:
    """Worker for parallel rates loading. Returns (filename, rows_loaded)."""
    filepath, symbol, timeframe = args
    df = pd.read_csv(filepath, dtype=str, keep_default_na=False)

    # Inject symbol and timeframe columns parsed from filename.
    df["symbol"] = symbol
    df["timeframe"] = timeframe

    # Ensure column order matches COPY target.
    for col in RATES_COLS:
        if col not in df.columns:
            df[col] = ""

    rows = copy_df_to_table(df, "rates", RATES_COLS)
    return (filepath.name, rows)


def load_rates(data_dir: Path, append: bool) -> int:
    """Load all *_rates.csv files from data_dir/rates/."""
    rates_dir = data_dir / "rates"
    if not rates_dir.is_dir():
        print(f"  [skip] {rates_dir} not found")
        return 0

    files = sorted(rates_dir.glob("*_rates.csv"))
    if not files:
        print(f"  [skip] no *_rates.csv files in {rates_dir}")
        return 0

    if not append:
        print("  TRUNCATE rates")
        truncate_table("rates")

    # Build work items: (Path, symbol, timeframe)
    work = []
    for f in files:
        parsed = parse_rates_filename(f.name)
        if not parsed:
            print(f"  [skip] cannot parse filename: {f.name}")
            continue
        work.append((f, parsed[0], parsed[1]))

    total = 0
    workers = min(cpu_count(), len(work))
    print(f"  Loading {len(work)} rate files with {workers} workers ...")

    with Pool(processes=workers) as pool:
        for filename, rows in pool.imap_unordered(_load_single_rates_file, work):
            total += rows
            print(f"    {filename}: {rows:,} rows")

    return total


def load_indicators(data_dir: Path, append: bool) -> int:
    """Load technical_indicator_*.csv files into per-symbol tables.

    Each file technical_indicator_{symbol}.csv is loaded into the
    corresponding technical_indicator_{symbol} table (used by the backend API).
    Also loads into the combined technical_indicators table for backward compat.
    """
    ind_dir = data_dir / "indicators"
    if not ind_dir.is_dir():
        print(f"  [skip] {ind_dir} not found")
        return 0

    files = sorted(ind_dir.glob("technical_indicator_*.csv"))
    if not files:
        print(f"  [skip] no indicator CSVs in {ind_dir}")
        return 0

    if not append:
        print("  TRUNCATE technical_indicators")
        truncate_table("technical_indicators")

    total = 0
    for f in files:
        print(f"  Loading {f.name} ...", end=" ", flush=True)
        df = pd.read_csv(f, dtype=str, keep_default_na=False)

        for col in INDICATORS_COLS:
            if col not in df.columns:
                df[col] = ""

        # Load into combined table
        rows = copy_df_to_table(df, "technical_indicators", INDICATORS_COLS)

        # Also load into per-symbol table (e.g. technical_indicator_eurusd)
        table_name = f.stem  # e.g. "technical_indicator_eurusd"
        try:
            if not append:
                truncate_table(table_name)
            copy_df_to_table(df, table_name, INDICATORS_COLS)
        except Exception as e:
            print(f"[warn] per-symbol table {table_name}: {e}")

        total += rows
        print(f"{rows:,} rows")

    return total


def load_paper_trades(data_dir: Path, append: bool) -> int:
    """Load paper_trades.csv from data_dir/trades/."""
    csv_path = data_dir / "trades" / "paper_trades.csv"
    if not csv_path.is_file():
        print(f"  [skip] {csv_path} not found")
        return 0

    if not append:
        print("  TRUNCATE paper_trades")
        truncate_table("paper_trades")

    print(f"  Loading {csv_path.name} ...", end=" ", flush=True)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    for col in PAPER_TRADES_COLS:
        if col not in df.columns:
            df[col] = ""

    rows = copy_df_to_table(df, "paper_trades", PAPER_TRADES_COLS)
    print(f"{rows:,} rows")
    return rows


def load_signal_discoveries(data_dir: Path, append: bool) -> int:
    """Load signal_discoveries.csv from data_dir/analysis/."""
    csv_path = data_dir / "analysis" / "signal_discoveries.csv"
    if not csv_path.is_file():
        print(f"  [skip] {csv_path} not found")
        return 0

    if not append:
        print("  TRUNCATE signal_discoveries")
        truncate_table("signal_discoveries")

    print(f"  Loading {csv_path.name} ...", end=" ", flush=True)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    for col in SIGNAL_DISCOVERIES_COLS:
        if col not in df.columns:
            df[col] = ""

    rows = copy_df_to_table(df, "signal_discoveries", SIGNAL_DISCOVERIES_COLS)
    print(f"{rows:,} rows")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

LOADERS = {
    "rates": load_rates,
    "indicators": load_indicators,
    "trades": load_paper_trades,
    "discoveries": load_signal_discoveries,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load CSV data into local PostgreSQL for Trading-TFG."
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Load from data/sample/ instead of full data directories.",
    )
    parser.add_argument(
        "--tables", nargs="+", choices=list(LOADERS.keys()),
        help="Load only specific tables (default: all).",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Skip TRUNCATE and append to existing data.",
    )
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / ("data/sample" if args.sample else "data")
    tables = args.tables or list(LOADERS.keys())

    print(f"Database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
    print(f"Data dir: {data_dir}")
    print(f"Tables:   {', '.join(tables)}")
    print(f"Mode:     {'append' if args.append else 'truncate + load'}")
    print()

    # Verify connection before starting.
    try:
        conn = get_connection()
        conn.close()
    except psycopg2.OperationalError as e:
        print(f"ERROR: Cannot connect to database: {e}", file=sys.stderr)
        print("Is PostgreSQL running? Check your .env / environment variables.", file=sys.stderr)
        sys.exit(1)

    grand_total = 0
    t0 = time.time()

    for table in tables:
        loader = LOADERS[table]
        print(f"--- {table} ---")
        try:
            rows = loader(data_dir, args.append)
            grand_total += rows
        except FileNotFoundError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
        except psycopg2.Error as e:
            print(f"  DB ERROR loading {table}: {e}", file=sys.stderr)
        print()

    elapsed = time.time() - t0
    print(f"Done. {grand_total:,} total rows loaded in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
