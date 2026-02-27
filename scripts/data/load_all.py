#!/usr/bin/env python3
"""
Load all CSV data into both markets and ai_model databases.

This is the Docker db-loader entrypoint. It loads:
  1. Markets DB: per-symbol-per-timeframe rate tables from data/rates/
  2. AI Model DB: indicators, paper_trades, signal_discoveries from data/

Usage:
    python scripts/data/load_all.py              # load from data/
    python scripts/data/load_all.py --sample     # load from data/sample/
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_loader(script: str, extra_args: list[str] | None = None) -> bool:
    """Run a loader script as a subprocess."""
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "data" / script)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Load all CSV data into databases.")
    parser.add_argument("--sample", action="store_true", help="Load sample data only")
    args = parser.parse_args()

    extra = ["--sample"] if args.sample else []

    t0 = time.time()
    success = True

    # 1. Load markets database (rates)
    print("\n[1/2] Loading MARKETS database ...")
    if not run_loader("load_markets.py", extra):
        print("WARNING: Markets loading had errors", file=sys.stderr)
        success = False

    # 2. Load ai_model database (indicators, trades, discoveries)
    print("\n[2/2] Loading AI_MODEL database ...")
    if not run_loader("load_csv_to_db.py", extra):
        print("WARNING: AI Model loading had errors", file=sys.stderr)
        success = False

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"All loading complete in {elapsed:.1f}s.")
    if not success:
        print("Some loaders reported errors — check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
