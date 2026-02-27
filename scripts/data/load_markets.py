#!/usr/bin/env python3
"""
Load rates CSVs into the markets database (per-symbol-per-timeframe tables).

Each CSV file like 'eurusd_h1_rates.csv' is loaded into the corresponding
'eurusd_h1_rates' table in the markets database.

Usage:
    python scripts/data/load_markets.py                    # load from data/rates/
    python scripts/data/load_markets.py --sample           # load from data/sample/rates/
    python scripts/data/load_markets.py --source /path     # load from external directory
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

DB_CONFIG = {
    "host": os.getenv("MARKETS_DB_HOST", "localhost"),
    "port": int(os.getenv("MARKETS_DB_PORT", "5432")),
    "dbname": os.getenv("MARKETS_DB_NAME", "markets"),
    "user": os.getenv("MARKETS_DB_USER", "tfg_user"),
    "password": os.getenv("MARKETS_DB_PASSWORD", "tfg_password"),
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FOREX_SYMBOLS = {"eurusd", "gbpusd", "usdjpy", "eurjpy", "usdcad", "eurcad", "usdchf", "eurgbp"}

# CSV columns (must match table schema order)
RATES_COLS = ["rate_time", "open", "high", "low", "close", "volume", "readable_date"]


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def parse_rates_filename(filename: str) -> tuple[str, str] | None:
    """Extract (symbol, timeframe) from e.g. 'eurusd_h1_rates.csv'."""
    m = re.match(r"^([a-z]{6})_([a-z0-9]+)_rates\.csv$", filename, re.IGNORECASE)
    if not m:
        return None
    symbol = m.group(1).lower()
    if symbol not in FOREX_SYMBOLS:
        return None
    return symbol, m.group(2).lower()


def _load_single_file(args: tuple) -> tuple[str, int]:
    """Worker: load one CSV into its matching table."""
    filepath, table_name = args
    df = pd.read_csv(filepath, dtype=str, keep_default_na=False)

    # Ensure expected columns
    for col in RATES_COLS:
        if col not in df.columns:
            df[col] = ""

    clean = df[RATES_COLS].copy()
    clean.replace("", pd.NA, inplace=True)
    buf = io.StringIO()
    clean.to_csv(buf, index=False, header=False, sep="\t", na_rep="\\N")
    buf.seek(0)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Create table if it doesn't exist (idempotent)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    rate_time     BIGINT PRIMARY KEY,
                    open          DECIMAL(18,8) NOT NULL,
                    high          DECIMAL(18,8) NOT NULL,
                    low           DECIMAL(18,8) NOT NULL,
                    close         DECIMAL(18,8) NOT NULL,
                    volume        DECIMAL(18,8) NOT NULL,
                    readable_date TIMESTAMP WITHOUT TIME ZONE
                )
            """)
            cur.execute(f"TRUNCATE TABLE {table_name}")
            cur.copy_from(buf, table_name, sep="\t", null="\\N", columns=RATES_COLS)
        conn.commit()
        return (filepath.name, len(df))
    finally:
        conn.close()


def load_markets(rates_dir: Path) -> int:
    """Load all rate CSVs from the given directory into the markets DB."""
    files = sorted(rates_dir.glob("*_rates.csv"))
    if not files:
        print(f"  [skip] no *_rates.csv files in {rates_dir}")
        return 0

    # Build work items: (Path, table_name)
    work = []
    for f in files:
        parsed = parse_rates_filename(f.name)
        if not parsed:
            continue
        symbol, tf = parsed
        table_name = f"{symbol}_{tf}_rates"
        work.append((f, table_name))

    if not work:
        print("  [skip] no matching forex rate files found")
        return 0

    workers = min(cpu_count(), len(work))
    print(f"  Loading {len(work)} rate files with {workers} workers ...")

    total = 0
    with Pool(processes=workers) as pool:
        for filename, rows in pool.imap_unordered(_load_single_file, work):
            total += rows
            print(f"    {filename}: {rows:,} rows")

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Load rates CSVs into markets database.")
    parser.add_argument("--sample", action="store_true", help="Load from data/sample/rates/")
    parser.add_argument("--source", type=str, help="External source directory with *_rates.csv files")
    args = parser.parse_args()

    if args.source:
        rates_dir = Path(args.source)
    elif args.sample:
        rates_dir = PROJECT_ROOT / "data" / "sample" / "rates"
    else:
        rates_dir = PROJECT_ROOT / "data" / "rates"

    print(f"Database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
    print(f"Source:   {rates_dir}")
    print()

    try:
        conn = get_connection()
        conn.close()
    except psycopg2.OperationalError as e:
        print(f"ERROR: Cannot connect to markets database: {e}", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    total = load_markets(rates_dir)
    elapsed = time.time() - t0
    print(f"\nDone. {total:,} total rows loaded in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
