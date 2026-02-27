"""
Minimal DataFeed - NO Order Block features

This is a simplified version of DataFeed that:
- Loads OHLC + essential technical indicators only
- Excludes ALL Order Block features from database query
- Optimized for Track 7C baseline training

For Track 7D (with OBs), use the original datafeed.py instead.
"""

from typing import Any, List, Optional, Tuple
import pandas as pd
from psycopg2.pool import ThreadedConnectionPool


# SQL query templates for separate database queries

# Query for OHLC data from MARKETS database (PARAMETERIZED by symbol)
# NOTE: Table name is constructed dynamically based on symbol + timeframe
OHLC_QUERY_TEMPLATE = """
SELECT rate_time, open, high, low, close, volume
FROM {table_name}
ORDER BY rate_time ASC
"""

# Query for technical indicators from AI_MODEL database
# SIMPLIFIED: NO Order Block columns (ob_bullish_*, ob_bearish_*)
INDICATORS_QUERY_MINIMAL = """
SELECT
    EXTRACT(EPOCH FROM timestamp)::bigint as rate_time,
    sma_20, sma_50, sma_200,
    ema_12, ema_26, ema_50,
    rsi_14, atr_14,
    bb_upper_20, bb_middle_20, bb_lower_20,
    macd_line, macd_signal, macd_histogram,
    stoch_k, stoch_d
FROM technical_indicators
WHERE timeframe = %s AND symbol = %s
ORDER BY timestamp ASC
"""


class MinimalDataFeed:
    """
    Minimal DataFeed for loading market data without Order Block features.

    This class loads EURUSD H1 candle data from the MARKETS database
    and essential technical indicators from the AI_MODEL database,
    explicitly EXCLUDING all Order Block features.

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
        Initialize MinimalDataFeed with dual database connection pools.

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
        Load OHLC data from MARKETS database using symbol-specific table.

        Constructs table name as: {symbol_lowercase}_{timeframe_lowercase}_rates
        Examples: eurusd_h1_rates, usdjpy_h1_rates, gbpusd_h1_rates

        Returns:
            DataFrame with columns: rate_time, open, high, low, close, volume

        Raises:
            Exception: If database query fails or table doesn't exist
        """
        # Construct table name: {symbol}_{timeframe}_rates (e.g., usdjpy_h1_rates)
        table_name = f"{self.symbol.lower()}_{self.timeframe.lower()}_rates"

        # Build query with table name
        query = OHLC_QUERY_TEMPLATE.format(table_name=table_name)

        conn = None
        try:
            conn = self.markets_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                return pd.DataFrame(rows, columns=columns)
        finally:
            if conn is not None:
                self.markets_pool.putconn(conn)

    def _load_indicators(self) -> pd.DataFrame:
        """
        Load technical indicators from AI_MODEL database (NO Order Blocks).

        Filters by symbol and timeframe. Converts all numeric values to float.
        EXCLUDES all Order Block features from the query.

        Returns:
            DataFrame with rate_time and essential indicator columns (NO OBs)

        Raises:
            Exception: If database query fails
        """
        conn = None
        try:
            conn = self.ai_model_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(INDICATORS_QUERY_MINIMAL, (self.timeframe, self.symbol))
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                df = pd.DataFrame(rows, columns=columns)

                # Convert all numeric columns to float (except rate_time)
                for col in df.columns:
                    if col != 'rate_time':
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                return df
        finally:
            if conn is not None:
                self.ai_model_pool.putconn(conn)

    def load_data(self) -> pd.DataFrame:
        """
        Load market data with essential indicators (NO Order Blocks).

        Queries MARKETS database for OHLC data and AI_MODEL database for indicators,
        then merges them on rate_time. Order Block features are excluded from the query.

        Returns:
            DataFrame with candle data and technical indicators (NO OBs) merged on rate_time

        Raises:
            Exception: If database query or merge fails
        """
        # 1. Load OHLC data from MARKETS database
        ohlc_data = self._load_ohlc_data()

        # 2. Load indicators from AI_MODEL database (NO OBs)
        indicators_data = self._load_indicators()

        # 3. Merge on rate_time (inner join)
        merged_data = pd.merge(
            ohlc_data,
            indicators_data,
            on='rate_time',
            how='inner',
            suffixes=('', '_indicator')
        )

        # 4. Sort by rate_time
        merged_data = merged_data.sort_values('rate_time').reset_index(drop=True)

        # 5. Replace NaN with None
        merged_data = merged_data.where(pd.notna(merged_data), None)

        return merged_data

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return False
