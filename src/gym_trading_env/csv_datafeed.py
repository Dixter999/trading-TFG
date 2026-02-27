"""
CSV-based DataFeed for training without PostgreSQL dependency.

This module provides a CSV implementation of DataFeed that loads market data
and indicators directly from CSV files in data/csv/ directory.

Used during database maintenance periods or for offline training.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


class CSVDataFeed:
    """
    CSV-based DataFeed for loading market data from CSV files.

    Compatible interface with DataFeed but reads from CSV files
    in data/csv/ directory instead of PostgreSQL.

    CSV files expected format: {symbol}_{timeframe}_complete.csv
    Example: eurusd_h4_complete.csv, gold_pro_h1_complete.csv

    Attributes:
        symbol: Trading symbol (e.g., 'EURUSD', 'GOLD.PRO')
        timeframe: Trading timeframe (e.g., 'H1', 'H4', 'D1')
        csv_path: Path to the CSV file
    """

    def __init__(
        self,
        symbol: str = 'EURUSD',
        timeframe: str = 'H4',
        csv_dir: str = 'data/csv'
    ):
        """
        Initialize CSV DataFeed.

        Args:
            symbol: Trading symbol (default: 'EURUSD')
            timeframe: Trading timeframe (default: 'H4')
            csv_dir: Directory containing CSV files (default: 'data/csv')

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV file is empty or invalid
        """
        self.symbol = symbol
        self.timeframe = timeframe

        # Construct CSV filename
        symbol_lower = symbol.lower().replace('.', '_')
        timeframe_lower = timeframe.lower()
        csv_filename = f"{symbol_lower}_{timeframe_lower}_complete.csv"

        self.csv_path = Path(csv_dir) / csv_filename

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.csv_path}. "
                f"Expected format: {{symbol}}_{{timeframe}}_complete.csv"
            )

        # Load data immediately to validate
        self._data = None
        self._load_data()

    def _load_data(self) -> None:
        """Load and validate CSV data."""
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV {self.csv_path}: {e}")

        if df.empty:
            raise ValueError(f"CSV file is empty: {self.csv_path}")

        # Validate required columns
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. "
                f"Found columns: {list(df.columns)}"
            )

        # Filter by symbol and timeframe if columns exist
        if 'symbol' in df.columns and 'timeframe' in df.columns:
            df = df[
                (df['symbol'] == self.symbol) &
                (df['timeframe'] == self.timeframe)
            ].copy()

        if df.empty:
            raise ValueError(
                f"No data found for {self.symbol} {self.timeframe} in {self.csv_path}"
            )

        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)

        # Add rate_time column (expected by TradingEnv)
        # Use 'time' column as rate_time timestamp
        df['rate_time'] = pd.to_datetime(df['time'], unit='s')

        self._data = df

    def get_data(
        self,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get market data slice.

        Args:
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)

        Returns:
            DataFrame with market data and indicators
        """
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self._data)

        return self._data.iloc[start_idx:end_idx].copy()

    def __len__(self) -> int:
        """Return total number of data points."""
        return len(self._data)

    def load_data(self) -> pd.DataFrame:
        """
        Load and return market data.

        Compatible with DataFeed interface for TradingEnv.

        Returns:
            DataFrame with market data and indicators
        """
        return self._data.copy()

    @property
    def data(self) -> pd.DataFrame:
        """Get full dataset."""
        return self._data.copy()
