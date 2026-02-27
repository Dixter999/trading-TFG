"""
Database connection manager with connection pooling.

This module provides the DatabaseManager class for managing separate
connection pools for Markets Database (read-only) and AI Model Database (read-write).
"""

from typing import Optional, Dict, Any, List, Union
from datetime import timedelta
import time
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

from src.database.config import AppConfig
from src.database.exceptions import (
    DatabaseConnectionError,
    QueryExecutionError,
    DataValidationError,
)
from src.database.utils import (
    validate_timeframe,
    parse_timestamp,
    calculate_expected_candles,
    validate_table_name,
    validate_column_name,
    build_where_clause,
    validate_connection,
    retry_on_transient_error,
    TIMEFRAME_TABLE_MAP,
    VALID_TIMEFRAMES,
    TIMEFRAME_MINUTES,
)
from src.database.logger import StructuredLogger


class DatabaseManager:
    """Manages database connections with separate pools for Markets and AI Model databases.

    This class implements connection pooling using SQLAlchemy for both the Markets
    Database (read-only) and AI Model Database (read-write). It supports lazy
    initialization, health checks, and context manager protocol for automatic
    resource cleanup.

    Attributes:
        config: Application configuration containing database settings
        markets_engine: SQLAlchemy engine for Markets Database (read-only)
        ai_model_engine: SQLAlchemy engine for AI Model Database (read-write)

    Connection Pool Configuration:
        Markets DB: pool_size=15, max_overflow=25
        AI Model DB: pool_size=5, max_overflow=10

    Example:
        >>> with DatabaseManager() as db:
        ...     health = db.health_check()
        ...     print(health)
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize DatabaseManager with optional config path.

        Args:
            config_path: Optional path to configuration file. If None,
                        configuration is loaded from environment variables.

        Note:
            Connections are not established during initialization (lazy loading).
            Call connect() explicitly or use as context manager.
        """
        # Load configuration from environment
        self.config: AppConfig = AppConfig.from_env()

        # Initialize structured logger
        self.logger: StructuredLogger = StructuredLogger("database")

        # Initialize engines as None (lazy initialization)
        self.markets_engine: Optional[Engine] = None
        self.ai_model_engine: Optional[Engine] = None

    def connect(self) -> bool:
        """Create connection pools for both databases.

        Establishes connection pools using SQLAlchemy with pre-ping enabled
        to verify connections before use.

        Returns:
            bool: True if connections established successfully

        Raises:
            DatabaseConnectionError: If connection to either database fails

        Note:
            Connection pools are thread-safe by default with SQLAlchemy.
        """
        try:
            # Create Markets DB engine (read-only, larger pool)
            markets_url = (
                f"postgresql://{self.config.markets_db.user}:{self.config.markets_db.password}"
                f"@{self.config.markets_db.host}:{self.config.markets_db.port}"
                f"/{self.config.markets_db.database}"
            )
            self.markets_engine = create_engine(
                markets_url,
                poolclass=QueuePool,
                pool_size=self.config.markets_db.pool_size,
                max_overflow=self.config.markets_db.max_overflow,
                pool_pre_ping=True,  # Verify connections before use
                echo=False,
            )

            # Validate markets DB connection
            validate_connection(self.markets_engine)
            self.logger.info(
                "Markets DB engine created",
                pool_size=self.config.markets_db.pool_size,
                max_overflow=self.config.markets_db.max_overflow,
                pool_pre_ping=True,
            )

            # Create AI Model DB engine (read-write, smaller pool)
            ai_model_url = (
                f"postgresql://{self.config.ai_model_db.user}:{self.config.ai_model_db.password}"
                f"@{self.config.ai_model_db.host}:{self.config.ai_model_db.port}"
                f"/{self.config.ai_model_db.database}"
            )
            self.ai_model_engine = create_engine(
                ai_model_url,
                poolclass=QueuePool,
                pool_size=self.config.ai_model_db.pool_size,
                max_overflow=self.config.ai_model_db.max_overflow,
                pool_pre_ping=True,
                echo=False,
            )

            # Validate AI model DB connection
            validate_connection(self.ai_model_engine)
            self.logger.info(
                "AI Model DB engine created",
                pool_size=self.config.ai_model_db.pool_size,
                max_overflow=self.config.ai_model_db.max_overflow,
                pool_pre_ping=True,
            )

            return True

        except Exception as e:
            self.logger.error("Failed to create database connections", error=str(e))
            raise DatabaseConnectionError(f"Failed to create database connections: {e}")

    def disconnect(self) -> None:
        """Clean up and dispose of all database connections.

        Disposes both connection pools, releasing all resources.
        Safe to call multiple times or without prior connect().
        """
        if self.markets_engine is not None:
            self.markets_engine.dispose()
            self.markets_engine = None

        if self.ai_model_engine is not None:
            self.ai_model_engine.dispose()
            self.ai_model_engine = None

    def health_check(self) -> Dict[str, Any]:
        """Check health of both database connections.

        Executes a simple query (SELECT 1) against each database to verify
        connectivity and reports pool statistics.

        Returns:
            Dict containing health status for both databases:
            {
                "markets_db": {
                    "status": "healthy" | "unhealthy: <error>",
                    "pool_size": int,
                    "checked_in": int
                },
                "ai_model_db": {
                    "status": "healthy" | "unhealthy: <error>",
                    "pool_size": int,
                    "checked_in": int
                }
            }

        Note:
            Returns pool statistics even if connection test fails.
        """
        result: Dict[str, Any] = {
            "markets_db": {"status": "unknown", "pool_size": 0, "checked_in": 0},
            "ai_model_db": {"status": "unknown", "pool_size": 0, "checked_in": 0},
        }

        # Check Markets DB
        try:
            if self.markets_engine is not None:
                with self.markets_engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                result["markets_db"]["status"] = "healthy"
                result["markets_db"]["pool_size"] = self.markets_engine.pool.size()
                result["markets_db"][
                    "checked_in"
                ] = self.markets_engine.pool.checkedin()
        except Exception as e:
            result["markets_db"]["status"] = f"unhealthy: {str(e)}"

        # Check AI Model DB
        try:
            if self.ai_model_engine is not None:
                with self.ai_model_engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                result["ai_model_db"]["status"] = "healthy"
                result["ai_model_db"]["pool_size"] = self.ai_model_engine.pool.size()
                result["ai_model_db"][
                    "checked_in"
                ] = self.ai_model_engine.pool.checkedin()
        except Exception as e:
            result["ai_model_db"]["status"] = f"unhealthy: {str(e)}"

        return result

    def __enter__(self) -> "DatabaseManager":
        """Context manager entry point.

        Establishes database connections when entering context.

        Returns:
            DatabaseManager: Self reference for use in context

        Raises:
            DatabaseConnectionError: If connection fails
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point.

        Cleans up database connections when exiting context.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self.disconnect()

    @retry_on_transient_error(max_attempts=3)
    def get_latest_candles(self, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Get the most recent N candles for a given timeframe.

        Args:
            timeframe: Timeframe code (d1, h4, h1, m30, m15, m5, m1)
            limit: Number of candles to retrieve (default: 100)

        Returns:
            DataFrame with columns: rate_time, open, high, low, close, volume

        Raises:
            DataValidationError: If timeframe is invalid
            QueryExecutionError: If query fails

        Example:
            >>> db = DatabaseManager()
            >>> db.connect()
            >>> df = db.get_latest_candles("h1", limit=50)
            >>> print(df.head())
        """
        start_time = time.time()

        # Validate timeframe
        validated_tf = validate_timeframe(timeframe)

        self.logger.debug(
            "Fetching latest candles", timeframe=validated_tf, limit=limit
        )

        # Get table name for timeframe
        table_name = TIMEFRAME_TABLE_MAP[validated_tf]

        # Build query
        query = text(
            f"""
            SELECT rate_time, open, high, low, close, volume
            FROM {table_name}
            ORDER BY rate_time DESC
            LIMIT :limit
        """
        )

        # Execute query - let transient errors propagate to retry decorator
        with self.markets_engine.connect() as conn:
            result = conn.execute(query, {"limit": limit})
            rows = result.fetchall()

        # Convert to DataFrame
        df = pd.DataFrame(
            rows, columns=["rate_time", "open", "high", "low", "close", "volume"]
        )

        # Ensure correct data types
        if not df.empty:
            df["rate_time"] = pd.to_datetime(df["rate_time"])
            df["open"] = df["open"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(int)

        duration_ms = (time.time() - start_time) * 1000
        self.logger.info(
            "Fetched candles successfully",
            timeframe=validated_tf,
            rows=len(df),
            duration_ms=duration_ms,
        )

        return df

    @retry_on_transient_error(max_attempts=3)
    def get_candles_by_date_range(
        self, timeframe: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Get candles within a specific date range.

        Args:
            timeframe: Timeframe code (d1, h4, h1, m30, m15, m5, m1)
            start_date: Start datetime (ISO format: "2024-01-01T00:00:00Z")
            end_date: End datetime (ISO format: "2024-12-31T23:59:59Z")

        Returns:
            DataFrame with columns: rate_time, open, high, low, close, volume

        Raises:
            DataValidationError: If timeframe or dates are invalid
            QueryExecutionError: If query fails

        Example:
            >>> db = DatabaseManager()
            >>> db.connect()
            >>> df = db.get_candles_by_date_range(
            ...     "d1",
            ...     "2024-01-01T00:00:00Z",
            ...     "2024-01-31T23:59:59Z"
            ... )
        """
        start_time = time.time()

        # Validate timeframe
        validated_tf = validate_timeframe(timeframe)

        # Parse and validate timestamps
        start_dt = parse_timestamp(start_date)
        end_dt = parse_timestamp(end_date)

        self.logger.debug(
            "Fetching candles by date range",
            timeframe=validated_tf,
            start_date=start_date,
            end_date=end_date,
        )

        # Get table name for timeframe
        table_name = TIMEFRAME_TABLE_MAP[validated_tf]

        # Build query
        # Convert datetime to bigint nanoseconds since epoch for comparison
        # rate_time is stored as bigint (nanoseconds), not timestamp
        start_ns = int(start_dt.timestamp() * 1_000_000_000)
        end_ns = int(end_dt.timestamp() * 1_000_000_000)

        query = text(
            f"""
            SELECT rate_time, open, high, low, close, volume
            FROM {table_name}
            WHERE rate_time >= :start_ns
              AND rate_time <= :end_ns
            ORDER BY rate_time ASC
        """
        )

        # Execute query - let transient errors propagate to retry decorator
        with self.markets_engine.connect() as conn:
            result = conn.execute(query, {"start_ns": start_ns, "end_ns": end_ns})
            rows = result.fetchall()

        # Convert to DataFrame
        df = pd.DataFrame(
            rows, columns=["rate_time", "open", "high", "low", "close", "volume"]
        )

        # Ensure correct data types
        if not df.empty:
            df["rate_time"] = pd.to_datetime(df["rate_time"])
            df["open"] = df["open"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(int)

        duration_ms = (time.time() - start_time) * 1000
        self.logger.info(
            "Fetched candles by date range successfully",
            timeframe=validated_tf,
            rows=len(df),
            duration_ms=duration_ms,
        )

        return df

    @retry_on_transient_error(max_attempts=3)
    def get_candle_at_time(
        self, timeframe: str, timestamp: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific candle at a given timestamp.

        Args:
            timeframe: Timeframe code (d1, h4, h1, m30, m15, m5, m1)
            timestamp: Exact candle timestamp (ISO format)

        Returns:
            Dictionary with keys: rate_time, open, high, low, close, volume
            Returns None if candle not found

        Raises:
            DataValidationError: If timeframe or timestamp is invalid
            QueryExecutionError: If query fails

        Example:
            >>> db = DatabaseManager()
            >>> db.connect()
            >>> candle = db.get_candle_at_time("h1", "2024-01-15T14:00:00Z")
            >>> print(candle["close"])
        """
        start_time = time.time()

        # Validate timeframe
        validated_tf = validate_timeframe(timeframe)

        # Parse and validate timestamp
        target_dt = parse_timestamp(timestamp)

        self.logger.debug(
            "Fetching candle at specific time",
            timeframe=validated_tf,
            timestamp=timestamp,
        )

        # Get table name for timeframe
        table_name = TIMEFRAME_TABLE_MAP[validated_tf]

        # Build query
        query = text(
            f"""
            SELECT rate_time, open, high, low, close, volume
            FROM {table_name}
            WHERE rate_time = :timestamp
            LIMIT 1
        """
        )

        # Execute query - let transient errors propagate to retry decorator
        with self.markets_engine.connect() as conn:
            result = conn.execute(query, {"timestamp": target_dt})
            row = result.fetchone()

        # Return None if not found
        if row is None:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(
                "Candle not found at timestamp",
                timeframe=validated_tf,
                timestamp=timestamp,
                duration_ms=duration_ms,
            )
            return None

        # Convert to dictionary
        candle = {
            "rate_time": row[0],
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": int(row[5]),
        }

        duration_ms = (time.time() - start_time) * 1000
        self.logger.info(
            "Fetched candle at time successfully",
            timeframe=validated_tf,
            timestamp=timestamp,
            duration_ms=duration_ms,
        )

        return candle

    def get_all_timeframes(self) -> List[str]:
        """
        Get list of all available EURUSD timeframes.

        Returns:
            List of timeframe codes: ["d1", "h4", "h1", "m30", "m15", "m5", "m1"]

        Example:
            >>> db = DatabaseManager()
            >>> timeframes = db.get_all_timeframes()
            >>> print(timeframes)
            ['d1', 'h4', 'h1', 'm30', 'm15', 'm5', 'm1']
        """
        return VALID_TIMEFRAMES.copy()

    def validate_data_integrity(
        self, timeframe: str, start_date: str, end_date: str
    ) -> Dict[str, Any]:
        """
        Check for data gaps and integrity issues in a date range.

        Args:
            timeframe: Timeframe code
            start_date: Start datetime (ISO format)
            end_date: End datetime (ISO format)

        Returns:
            Dictionary with:
            - total_expected: Expected number of candles
            - total_actual: Actual number of candles found
            - gaps: List of missing timestamp ranges
            - data_issues: List of data quality issues (invalid OHLC, etc.)

        Example:
            >>> db = DatabaseManager()
            >>> db.connect()
            >>> report = db.validate_data_integrity(
            ...     "h1",
            ...     "2024-01-01T00:00:00Z",
            ...     "2024-01-31T23:59:59Z"
            ... )
            >>> print(f"Found {len(report['gaps'])} gaps in data")
        """
        # Validate timeframe
        validated_tf = validate_timeframe(timeframe)

        # Parse timestamps
        start_dt = parse_timestamp(start_date)
        end_dt = parse_timestamp(end_date)

        # Calculate expected number of candles
        total_expected = calculate_expected_candles(validated_tf, start_dt, end_dt)

        # Get actual candles in the range
        df = self.get_candles_by_date_range(validated_tf, start_date, end_date)
        total_actual = len(df)

        # Initialize result
        result = {
            "total_expected": total_expected,
            "total_actual": total_actual,
            "gaps": [],
            "data_issues": [],
        }

        # Detect gaps by comparing actual vs expected
        if total_actual < total_expected and not df.empty:
            # Build list of expected timestamps
            minutes_per_candle = TIMEFRAME_MINUTES[validated_tf]
            expected_timestamps = []
            current = start_dt
            while current <= end_dt:
                expected_timestamps.append(current)
                current += timedelta(minutes=minutes_per_candle)

            # Convert actual timestamps to set for faster lookup
            actual_timestamps = set(df["rate_time"].dt.to_pydatetime())

            # Find gaps
            gap_start = None
            for expected in expected_timestamps:
                if expected not in actual_timestamps:
                    if gap_start is None:
                        gap_start = expected
                else:
                    if gap_start is not None:
                        # End of gap found
                        result["gaps"].append(
                            {
                                "start": gap_start.isoformat(),
                                "end": (
                                    expected - timedelta(minutes=minutes_per_candle)
                                ).isoformat(),
                                "missing_candles": len(
                                    [
                                        t
                                        for t in expected_timestamps
                                        if gap_start <= t < expected
                                        and t not in actual_timestamps
                                    ]
                                ),
                            }
                        )
                        gap_start = None

            # Check if there's a gap at the end
            if gap_start is not None:
                result["gaps"].append(
                    {
                        "start": gap_start.isoformat(),
                        "end": end_dt.isoformat(),
                        "missing_candles": len(
                            [
                                t
                                for t in expected_timestamps
                                if gap_start <= t <= end_dt
                                and t not in actual_timestamps
                            ]
                        ),
                    }
                )

        # Check for data quality issues (e.g., invalid OHLC relationships)
        if not df.empty:
            for idx, row in df.iterrows():
                issues = []
                # High should be >= all other prices
                if (
                    row["high"] < row["open"]
                    or row["high"] < row["close"]
                    or row["high"] < row["low"]
                ):
                    issues.append("high price lower than other prices")

                # Low should be <= all other prices
                if (
                    row["low"] > row["open"]
                    or row["low"] > row["close"]
                    or row["low"] > row["high"]
                ):
                    issues.append("low price higher than other prices")

                # Volume should be non-negative
                if row["volume"] < 0:
                    issues.append("negative volume")

                if issues:
                    result["data_issues"].append(
                        {"timestamp": row["rate_time"].isoformat(), "issues": issues}
                    )

        return result

    # ========================================================================
    # AI Model DB CRUD Methods
    # ========================================================================

    @retry_on_transient_error(max_attempts=3)
    def execute_query(
        self, database: str, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute arbitrary SQL query on specified database.

        This method executes raw SQL queries with parameterized bindings
        for security. Use :param_name syntax in query for parameters.

        Supports both SELECT (query) and DML (INSERT/UPDATE/DELETE) statements.
        DML statements are automatically committed.

        Args:
            database: Database name ('markets' or 'ai_model')
            query: SQL query string with :param placeholders
            params: Optional dictionary of parameter bindings

        Returns:
            List of dictionaries, each representing a row with column names as keys
            For DML with RETURNING clause, returns the returned rows
            For DML without RETURNING, returns empty list

        Raises:
            QueryExecutionError: If query execution fails
            ValueError: If database name is invalid

        Example:
            >>> db = DatabaseManager()
            >>> db.connect()
            >>> # SELECT query on markets DB
            >>> results = db.execute_query(
            ...     "markets",
            ...     "SELECT * FROM eurusd_m1_rates LIMIT 1"
            ... )
            >>> # INSERT with RETURNING on AI model DB
            >>> results = db.execute_query(
            ...     "ai_model",
            ...     "INSERT INTO indicators (symbol) VALUES (:symbol) RETURNING id",
            ...     {"symbol": "EURUSD"}
            ... )

        Security:
            Always use parameterized queries. Never concatenate user input
            directly into query strings.
        """
        start_time = time.time()

        # Select appropriate engine
        if database == "markets":
            engine = self.markets_engine
            db_label = "Markets DB"
        elif database == "ai_model":
            engine = self.ai_model_engine
            db_label = "AI Model DB"
        else:
            raise ValueError(
                f"Invalid database '{database}'. Must be 'markets' or 'ai_model'"
            )

        self.logger.debug(
            f"Executing query on {db_label}",
            query=query[:100],  # Truncate long queries
            params=params,
        )

        # Execute query with explicit transaction management
        with engine.connect() as conn:
            result = conn.execute(text(query), params or {})

            # Check if query returns rows (SELECT, INSERT...RETURNING, etc.)
            # DML without RETURNING (DELETE, UPDATE without RETURNING) don't return rows
            try:
                rows = result.fetchall()
                keys = result.keys()
                data = [dict(zip(keys, row)) for row in rows]
            except Exception:
                # Query doesn't return rows (DELETE, UPDATE without RETURNING)
                data = []

            # Commit transaction for DML statements (INSERT, UPDATE, DELETE)
            # This ensures data is persisted to the database
            conn.commit()

        duration_ms = (time.time() - start_time) * 1000
        self.logger.info(
            "Query executed successfully", rows=len(data), duration_ms=duration_ms
        )

        return data

    @retry_on_transient_error(max_attempts=3)
    def insert_data(
        self, table: str, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> bool:
        """Insert one or more rows into a table in AI Model Database.

        Supports both single-row and batch inserts. Batch inserts are
        optimized for efficiency when inserting multiple rows.

        Args:
            table: Name of the table to insert into
            data: Single dictionary or list of dictionaries representing rows
                  Each dict key should be a column name

        Returns:
            True if insert was successful

        Raises:
            DataValidationError: If table name is invalid or data is empty
            QueryExecutionError: If insert operation fails

        Example:
            >>> db = DatabaseManager()
            >>> db.connect()
            >>> # Single row insert
            >>> db.insert_data("indicators", {
            ...     "symbol": "EURUSD",
            ...     "indicator_type": "RSI",
            ...     "value": 65.5
            ... })
            True
            >>> # Batch insert
            >>> db.insert_data("indicators", [
            ...     {"symbol": "EURUSD", "indicator_type": "RSI", "value": 65.5},
            ...     {"symbol": "EURUSD", "indicator_type": "MACD", "value": 0.0012}
            ... ])
            True

        Security:
            Table names are validated to prevent SQL injection.
            Values are bound using parameterized queries.
        """
        start_time = time.time()

        # Validate table name
        validated_table = validate_table_name(table)

        # Handle both single dict and list of dicts
        if isinstance(data, dict):
            if not data:
                raise DataValidationError("Cannot insert empty data")
            data_list = [data]
        elif isinstance(data, list):
            if not data:
                raise DataValidationError("Cannot insert empty data")
            data_list = data
        else:
            raise DataValidationError("Data must be dict or list of dicts")

        self.logger.debug(
            "Inserting data into AI Model DB",
            table=validated_table,
            rows=len(data_list),
        )

        try:
            with self.ai_model_engine.connect() as conn:
                for row in data_list:
                    columns = list(row.keys())
                    placeholders = [f":{col}" for col in columns]

                    query = text(
                        f"""
                        INSERT INTO {validated_table} ({', '.join(columns)})
                        VALUES ({', '.join(placeholders)})
                    """
                    )

                    conn.execute(query, row)
                conn.commit()

            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(
                "Data inserted successfully",
                table=validated_table,
                rows=len(data_list),
                duration_ms=duration_ms,
            )

            return True

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "Failed to insert data",
                table=validated_table,
                error=str(e),
                duration_ms=duration_ms,
            )
            raise QueryExecutionError(f"Failed to insert data: {e}")

    @retry_on_transient_error(max_attempts=3)
    def update_data(
        self, table: str, data: Dict[str, Any], condition: Dict[str, Any]
    ) -> int:
        """Update rows matching condition in AI Model Database.

        Uses parameterized WHERE clauses from build_where_clause utility
        for safe condition filtering.

        Args:
            table: Name of the table to update
            data: Dictionary of column_name: new_value pairs to update
            condition: Dictionary of column_name: value pairs for WHERE clause

        Returns:
            Number of rows updated

        Raises:
            DataValidationError: If table name is invalid or data is empty
            QueryExecutionError: If update operation fails

        Example:
            >>> db = DatabaseManager()
            >>> db.connect()
            >>> rows_updated = db.update_data(
            ...     "indicators",
            ...     {"value": 70.0},
            ...     {"symbol": "EURUSD", "indicator_type": "RSI"}
            ... )
            >>> print(f"Updated {rows_updated} rows")
            Updated 3 rows

        Security:
            Table names validated. Conditions use parameterized queries
            via build_where_clause() to prevent SQL injection.
        """
        start_time = time.time()

        # Validate table name
        validated_table = validate_table_name(table)

        # Validate data is not empty
        if not data:
            raise DataValidationError("Cannot update with empty data")

        self.logger.debug(
            "Updating data in AI Model DB",
            table=validated_table,
            data=data,
            condition=condition,
        )

        # Build SET clause
        set_parts = []
        params = {}
        for col, val in data.items():
            set_parts.append(f"{col} = :set_{col}")
            params[f"set_{col}"] = val

        # Build WHERE clause
        where_clause, where_params = build_where_clause(condition)
        params.update(where_params)

        # Build full query
        query_str = f"UPDATE {validated_table} SET {', '.join(set_parts)}"
        if where_clause:
            query_str += f" WHERE {where_clause}"

        try:
            with self.ai_model_engine.connect() as conn:
                result = conn.execute(text(query_str), params)
                conn.commit()
                rowcount = result.rowcount

            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(
                "Data updated successfully",
                table=validated_table,
                rows_updated=rowcount,
                duration_ms=duration_ms,
            )

            return rowcount

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "Failed to update data",
                table=validated_table,
                error=str(e),
                duration_ms=duration_ms,
            )
            raise QueryExecutionError(f"Failed to update data: {e}")

    @retry_on_transient_error(max_attempts=3)
    def delete_data(self, table: str, condition: Dict[str, Any]) -> int:
        """Delete rows matching condition from AI Model Database.

        Requires a condition for safety (prevents accidental full table deletes).
        Uses parameterized WHERE clauses via build_where_clause utility.

        Args:
            table: Name of the table to delete from
            condition: Dictionary of column_name: value pairs for WHERE clause
                      Cannot be empty (safety check)

        Returns:
            Number of rows deleted

        Raises:
            DataValidationError: If table name is invalid or condition is empty
            QueryExecutionError: If delete operation fails

        Example:
            >>> db = DatabaseManager()
            >>> db.connect()
            >>> rows_deleted = db.delete_data(
            ...     "indicators",
            ...     {"symbol": "EURUSD", "indicator_type": "RSI"}
            ... )
            >>> print(f"Deleted {rows_deleted} rows")
            Deleted 2 rows

        Security:
            Table names validated. Empty conditions rejected as safety check.
            Conditions use parameterized queries to prevent SQL injection.
        """
        start_time = time.time()

        # Validate table name
        validated_table = validate_table_name(table)

        # Safety check - require condition
        if not condition:
            raise DataValidationError("Delete requires a condition (safety check)")

        self.logger.debug(
            "Deleting data from AI Model DB", table=validated_table, condition=condition
        )

        # Build WHERE clause
        where_clause, params = build_where_clause(condition)

        # Build query
        query_str = f"DELETE FROM {validated_table}"
        if where_clause:
            query_str += f" WHERE {where_clause}"

        try:
            with self.ai_model_engine.connect() as conn:
                result = conn.execute(text(query_str), params)
                conn.commit()
                rowcount = result.rowcount

            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(
                "Data deleted successfully",
                table=validated_table,
                rows_deleted=rowcount,
                duration_ms=duration_ms,
            )

            return rowcount

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "Failed to delete data",
                table=validated_table,
                error=str(e),
                duration_ms=duration_ms,
            )
            raise QueryExecutionError(f"Failed to delete data: {e}")

    @retry_on_transient_error(max_attempts=3)
    def create_table(self, table_name: str, schema: Dict[str, str]) -> bool:
        """Create new table with specified schema in AI Model Database.

        Validates both table name and column names to prevent SQL injection.
        Schema dictionary maps column names to PostgreSQL data type definitions.

        Args:
            table_name: Name of the table to create
            schema: Dictionary mapping column_name to PostgreSQL type definition
                   Example: {"id": "SERIAL PRIMARY KEY", "name": "VARCHAR(100)"}

        Returns:
            True if table creation was successful

        Raises:
            DataValidationError: If table/column names invalid or schema empty
            QueryExecutionError: If table creation fails

        Example:
            >>> db = DatabaseManager()
            >>> db.connect()
            >>> db.create_table("test_indicators", {
            ...     "id": "SERIAL PRIMARY KEY",
            ...     "symbol": "VARCHAR(10) NOT NULL",
            ...     "indicator_type": "VARCHAR(50) NOT NULL",
            ...     "value": "NUMERIC(10, 4)",
            ...     "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ... })
            True

        Security:
            Both table and column names validated with validate_table_name()
            and validate_column_name() to prevent SQL injection.
        """
        start_time = time.time()

        # Validate table name
        validated_table = validate_table_name(table_name)

        # Validate schema is not empty
        if not schema:
            raise DataValidationError("Cannot create table with empty schema")

        self.logger.debug(
            "Creating table in AI Model DB", table=validated_table, columns=len(schema)
        )

        # Validate column names and build column definitions
        column_defs = []
        for col_name, col_type in schema.items():
            validated_col = validate_column_name(col_name)
            column_defs.append(f"{validated_col} {col_type}")

        # Build query
        query_str = f"CREATE TABLE {validated_table} ({', '.join(column_defs)})"

        try:
            with self.ai_model_engine.connect() as conn:
                conn.execute(text(query_str))
                conn.commit()

            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(
                "Table created successfully",
                table=validated_table,
                columns=len(schema),
                duration_ms=duration_ms,
            )

            return True

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "Failed to create table",
                table=validated_table,
                error=str(e),
                duration_ms=duration_ms,
            )
            raise QueryExecutionError(f"Failed to create table: {e}")

    @retry_on_transient_error(max_attempts=3)
    def drop_table(self, table_name: str, cascade: bool = False) -> bool:
        """Drop a table from AI Model Database.

        Supports CASCADE option to automatically drop dependent objects
        (foreign key constraints, views, etc.).

        Args:
            table_name: Name of the table to drop
            cascade: If True, automatically drop objects that depend on the table

        Returns:
            True if table was dropped successfully

        Raises:
            DataValidationError: If table name is invalid
            QueryExecutionError: If drop operation fails

        Example:
            >>> db = DatabaseManager()
            >>> db.connect()
            >>> db.drop_table("test_indicators")
            True
            >>> # Drop with cascade
            >>> db.drop_table("parent_table", cascade=True)
            True

        Warning:
            This operation is irreversible. All data in the table will be lost.

        Security:
            Table name validated with validate_table_name() to prevent
            SQL injection.
        """
        start_time = time.time()

        # Validate table name
        validated_table = validate_table_name(table_name)

        self.logger.debug(
            "Dropping table from AI Model DB", table=validated_table, cascade=cascade
        )

        # Build query
        query_str = f"DROP TABLE {validated_table}"
        if cascade:
            query_str += " CASCADE"

        try:
            with self.ai_model_engine.connect() as conn:
                conn.execute(text(query_str))
                conn.commit()

            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(
                "Table dropped successfully",
                table=validated_table,
                cascade=cascade,
                duration_ms=duration_ms,
            )

            return True

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "Failed to drop table",
                table=validated_table,
                error=str(e),
                duration_ms=duration_ms,
            )
            raise QueryExecutionError(f"Failed to drop table: {e}")

    @retry_on_transient_error(max_attempts=3)
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists in AI Model Database.

        Queries PostgreSQL's information_schema.tables to verify table existence.

        Args:
            table_name: Name of the table to check

        Returns:
            True if table exists, False otherwise

        Raises:
            DataValidationError: If table name is invalid
            QueryExecutionError: If query fails

        Example:
            >>> db = DatabaseManager()
            >>> db.connect()
            >>> if db.table_exists("indicators"):
            ...     print("Table exists")
            ... else:
            ...     print("Table does not exist")
            Table exists

        Security:
            Table name validated with validate_table_name() to prevent
            SQL injection.
        """
        start_time = time.time()

        # Validate table name
        validated_table = validate_table_name(table_name)

        self.logger.debug(
            "Checking table existence in AI Model DB", table=validated_table
        )

        query = text(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = :table_name
        """
        )

        try:
            with self.ai_model_engine.connect() as conn:
                result = conn.execute(query, {"table_name": validated_table})
                row = result.fetchone()
                exists = row is not None and row[0] > 0

            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(
                "Table existence check completed",
                table=validated_table,
                exists=exists,
                duration_ms=duration_ms,
            )

            return exists

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "Failed to check table existence",
                table=validated_table,
                error=str(e),
                duration_ms=duration_ms,
            )
            raise QueryExecutionError(f"Failed to check table existence: {e}")

    @retry_on_transient_error(max_attempts=3)
    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """Get schema for a table from AI Model Database.

        Queries PostgreSQL's information_schema.columns to retrieve
        column names and data types in ordinal position order.

        Args:
            table_name: Name of the table to inspect

        Returns:
            Dictionary mapping column names to PostgreSQL data types
            Returns empty dict if table doesn't exist

        Raises:
            DataValidationError: If table name is invalid
            QueryExecutionError: If query fails

        Example:
            >>> db = DatabaseManager()
            >>> db.connect()
            >>> schema = db.get_table_schema("indicators")
            >>> print(schema)
            {'id': 'integer', 'symbol': 'character varying', 'value': 'numeric'}
            >>> for col, dtype in schema.items():
            ...     print(f"{col}: {dtype}")
            id: integer
            symbol: character varying
            value: numeric

        Security:
            Table name validated with validate_table_name() to prevent
            SQL injection.
        """
        start_time = time.time()

        # Validate table name
        validated_table = validate_table_name(table_name)

        self.logger.debug(
            "Getting table schema from AI Model DB", table=validated_table
        )

        query = text(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = :table_name
            ORDER BY ordinal_position
        """
        )

        try:
            with self.ai_model_engine.connect() as conn:
                result = conn.execute(query, {"table_name": validated_table})
                rows = result.fetchall()

                schema = {row[0]: row[1] for row in rows}

            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(
                "Table schema retrieved successfully",
                table=validated_table,
                columns=len(schema),
                duration_ms=duration_ms,
            )

            return schema

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "Failed to get table schema",
                table=validated_table,
                error=str(e),
                duration_ms=duration_ms,
            )
            raise QueryExecutionError(f"Failed to get table schema: {e}")

    def transaction(self):
        """Context manager for atomic database transactions on AI Model Database.

        Provides automatic commit on success and rollback on exceptions.
        Use this for multi-operation atomic transactions to ensure data consistency.

        Yields:
            SQLAlchemy connection object for executing queries within transaction

        Raises:
            Any exception raised within the transaction block will cause rollback

        Example:
            >>> db = DatabaseManager()
            >>> db.connect()
            >>> # Single transaction with multiple operations
            >>> with db.transaction() as conn:
            ...     conn.execute(text("INSERT INTO table1 VALUES (:val)"), {"val": 1})
            ...     conn.execute(text("UPDATE table2 SET col = :val"), {"val": 2})
            ...     # Automatically commits if no exception

            >>> # Transaction with rollback on error
            >>> try:
            ...     with db.transaction() as conn:
            ...         conn.execute(text("INSERT INTO table1 VALUES (:val)"), {"val": 1})
            ...         raise ValueError("Oops!")  # This will rollback the insert
            ... except ValueError:
            ...     print("Transaction rolled back")
            Transaction rolled back

        Note:
            All operations within the transaction block must use the provided
            connection object, not db.execute_query() or other methods which
            create their own connections.
        """
        from contextlib import contextmanager

        @contextmanager
        def _transaction():
            with self.ai_model_engine.connect() as conn:
                with conn.begin():
                    yield conn

        return _transaction()
