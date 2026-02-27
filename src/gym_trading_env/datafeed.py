"""
Custom DataFeed class for gym-trading-env with dual PostgreSQL database support.

This module provides a DataFeed implementation that loads EURUSD H1 candles from the
MARKETS database (eurusd_h1_rates table) and technical indicators from the AI_MODEL
database (technical_indicators table). Data is queried separately from each database
and merged in Python using pandas on rate_time, filtered by symbol and timeframe.

TDD Phase: REFACTOR - Separated database queries to avoid cross-database JOIN.
"""

from typing import Any, List, Optional, Tuple
import pandas as pd
from psycopg2.pool import ThreadedConnectionPool


# SQL query templates for separate database queries

# Query for OHLC data from MARKETS database
OHLC_QUERY = """
SELECT rate_time, open, high, low, close, volume
FROM eurusd_h1_rates
ORDER BY rate_time ASC
"""

# Query for technical indicators from AI_MODEL database
INDICATORS_QUERY = """
SELECT
    EXTRACT(EPOCH FROM timestamp)::bigint as rate_time,
    sma_20, sma_50, sma_200,
    ema_12, ema_26, ema_50,
    rsi_14, atr_14,
    bb_upper_20, bb_middle_20, bb_lower_20,
    macd_line, macd_signal, macd_histogram,
    stoch_k, stoch_d,
    ob_bullish_high, ob_bullish_low,
    ob_bearish_high, ob_bearish_low
FROM technical_indicators
WHERE timeframe = %s AND symbol = %s
ORDER BY timestamp ASC
"""


class DataFeed:
    """
    Custom DataFeed for loading market data and indicators from dual databases.

    This class loads EURUSD H1 candle data from the MARKETS database
    (eurusd_h1_rates table) and technical indicators from the AI_MODEL database
    (technical_indicators table), then merges them in Python using pandas
    on rate_time + symbol + timeframe.

    Attributes:
        markets_pool: Connection pool for markets database
        ai_model_pool: Connection pool for ai_model database
        symbol: Trading symbol (default: 'EURUSD')
        timeframe: Trading timeframe (default: 'H1')
    """

    def __init__(
        self,
        markets_pool: ThreadedConnectionPool,
        ai_model_pool: ThreadedConnectionPool,
        symbol: str = 'EURUSD',
        timeframe: str = 'H1'
    ):
        """
        Initialize DataFeed with dual database connection pools.

        Args:
            markets_pool: Connection pool for markets database
            ai_model_pool: Connection pool for ai_model database
            symbol: Trading symbol (default: 'EURUSD')
            timeframe: Trading timeframe (default: 'H1')

        Raises:
            TypeError: If required pools are not provided
        """
        if not isinstance(markets_pool, ThreadedConnectionPool):
            raise TypeError(
                "markets_pool must be a psycopg2.pool.ThreadedConnectionPool instance"
            )
        if not isinstance(ai_model_pool, ThreadedConnectionPool):
            raise TypeError(
                "ai_model_pool must be a psycopg2.pool.ThreadedConnectionPool instance"
            )

        self.markets_pool = markets_pool
        self.ai_model_pool = ai_model_pool
        self.symbol = symbol
        self.timeframe = timeframe

    def _load_ohlc_data(self) -> pd.DataFrame:
        """
        Load OHLC data from MARKETS database.

        Returns:
            DataFrame with columns: rate_time, open, high, low, close, volume

        Raises:
            Exception: If database query fails
        """
        conn = None
        try:
            conn = self.markets_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(OHLC_QUERY)
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                return pd.DataFrame(rows, columns=columns)
        finally:
            if conn is not None:
                self.markets_pool.putconn(conn)

    def _load_indicators(self) -> pd.DataFrame:
        """
        Load technical indicators from AI_MODEL database.

        Filters by symbol and timeframe. Converts all numeric values to float
        to ensure compatibility with MARKETS data (which uses double precision).

        Returns:
            DataFrame with rate_time and all indicator columns (as float)

        Raises:
            Exception: If database query fails
        """
        conn = None
        try:
            conn = self.ai_model_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(INDICATORS_QUERY, (self.timeframe, self.symbol))
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                df = pd.DataFrame(rows, columns=columns)

                # Convert all numeric columns to float (except rate_time which is already bigint)
                # This ensures compatibility with MARKETS data which uses double precision
                for col in df.columns:
                    if col != 'rate_time':  # Keep rate_time as bigint
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                return df
        finally:
            if conn is not None:
                self.ai_model_pool.putconn(conn)

    def _convert_to_dataframe(
        self, rows: List[Tuple], column_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convert database query results to pandas DataFrame.

        Args:
            rows: List of row tuples from database query
            column_names: Optional list of column names (if None, empty list used)

        Returns:
            DataFrame with query results
        """
        columns = column_names if column_names else []
        return pd.DataFrame(rows, columns=columns)

    def load_data(self) -> pd.DataFrame:
        """
        Load market data with indicators from dual databases.

        Queries MARKETS database for OHLC data and AI_MODEL database for indicators,
        then merges them on rate_time + symbol + timeframe.

        Returns:
            DataFrame with candle data and technical indicators merged on rate_time

        Raises:
            Exception: If database query or merge fails
        """
        # 1. Load OHLC data from MARKETS database
        ohlc_data = self._load_ohlc_data()

        # 2. Load indicators from AI_MODEL database (filtered by symbol + timeframe)
        indicators_data = self._load_indicators()

        # 3. Merge on rate_time (inner join to keep only matching timestamps)
        merged_data = pd.merge(
            ohlc_data,
            indicators_data,
            on='rate_time',
            how='inner',
            suffixes=('', '_indicator')  # Avoid column name conflicts
        )

        # 4. Sort by rate_time to ensure chronological order
        merged_data = merged_data.sort_values('rate_time').reset_index(drop=True)

        # 5. Replace NaN with None for proper NULL handling in RL environment
        # Pandas NaN values would propagate through calculations, but None is handled correctly
        merged_data = merged_data.where(pd.notna(merged_data), None)

        return merged_data

    def __enter__(self):
        """
        Context manager entry.

        Returns:
            self for use in with statement
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit.

        Handles cleanup when exiting context manager.

        Args:
            exc_type: Exception type if error occurred
            exc_val: Exception value if error occurred
            exc_tb: Exception traceback if error occurred

        Returns:
            False to propagate exceptions
        """
        # No special cleanup needed in current implementation
        return False
