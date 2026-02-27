"""
Database utility functions for validation, parsing, and calculations.

This module provides helper functions used throughout the database module
for timeframe validation, timestamp parsing, and data integrity calculations.
"""

import functools
import logging
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from src.database.exceptions import (
    DatabaseConnectionError,
    DataValidationError,
    QueryExecutionError,
)

# Configure logger
logger = logging.getLogger(__name__)


# Timeframe to table name mapping - ALL 16 timeframes from PostgreSQL
TIMEFRAME_TABLE_MAP: dict[str, str] = {
    "m1": "eurusd_m1_rates",
    "m5": "eurusd_m5_rates",
    "m10": "eurusd_m10_rates",
    "m15": "eurusd_m15_rates",
    "m20": "eurusd_m20_rates",
    "m30": "eurusd_m30_rates",
    "h1": "eurusd_h1_rates",
    "h2": "eurusd_h2_rates",
    "h3": "eurusd_h3_rates",
    "h4": "eurusd_h4_rates",
    "h6": "eurusd_h6_rates",
    "h8": "eurusd_h8_rates",
    "h12": "eurusd_h12_rates",
    "d1": "eurusd_d1_rates",
    "w1": "eurusd_w1_rates",
    "mn1": "eurusd_mn1_rates",
}

# Valid timeframe codes
VALID_TIMEFRAMES: list[str] = list(TIMEFRAME_TABLE_MAP.keys())

# Timeframe to minutes mapping - ALL 16 timeframes
TIMEFRAME_MINUTES: dict[str, int] = {
    "m1": 1,
    "m5": 5,
    "m10": 10,
    "m15": 15,
    "m20": 20,
    "m30": 30,
    "h1": 60,
    "h2": 120,
    "h3": 180,
    "h4": 240,
    "h6": 360,
    "h8": 480,
    "h12": 720,
    "d1": 1440,
    "w1": 10080,
    "mn1": 43200,
}

# Error message templates
ERROR_MESSAGES: dict[str, str] = {
    "connection_failed": (
        "Failed to connect to database '{db_name}' at {host}:{port}. "
        "Check network connectivity and database credentials. "
        "Error: {error}"
    ),
    "query_timeout": (
        "Query timed out after {timeout}s. "
        "Consider optimizing the query or increasing timeout. "
        "Query: {query}"
    ),
    "permission_denied": (
        "Permission denied for operation on table '{table}'. "
        "Verify database user '{user}' has required privileges. "
        "Operation: {operation}"
    ),
    "invalid_timeframe": (
        "Invalid timeframe '{timeframe}'. " "Valid options: {valid_timeframes}"
    ),
}


def retry_on_transient_error(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    transient_exceptions: tuple[type[Exception], ...] | None = None,
) -> Callable:
    """
    Decorator to retry function on transient errors with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
        backoff_factor: Multiplier for delay between attempts (default: 2.0)
        transient_exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function with retry logic

    Example:
        @retry_on_transient_error(max_attempts=3)
        def query_database():
            # Database query that might fail transiently
            pass
    """
    if transient_exceptions is None:
        transient_exceptions = (
            OperationalError,  # SQLAlchemy connection errors
            TimeoutError,
            ConnectionError,
        )

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except transient_exceptions as e:
                    if attempt == max_attempts:
                        logger.error(
                            f"Max retry attempts ({max_attempts}) exceeded",
                            extra={
                                "function": func.__name__,
                                "attempts": attempt,
                                "error": str(e),
                            },
                        )
                        raise QueryExecutionError(
                            f"Query failed after {max_attempts} attempts: {str(e)}"
                        ) from e

                    logger.warning(
                        f"Transient error on attempt {attempt}/{max_attempts}, retrying in {delay}s",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt,
                            "delay": delay,
                            "error": str(e),
                        },
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
                except Exception as e:
                    # Non-transient error - fail fast
                    logger.error(
                        "Non-transient error, failing immediately",
                        extra={
                            "function": func.__name__,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    raise

        return wrapper

    return decorator


def validate_connection(engine) -> bool:
    """
    Validate database connection before use.

    Args:
        engine: SQLAlchemy engine to validate

    Returns:
        True if connection is valid

    Raises:
        DatabaseConnectionError: If connection validation fails

    Example:
        >>> from sqlalchemy import create_engine
        >>> engine = create_engine("postgresql://...")
        >>> validate_connection(engine)
        True
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.debug("Connection validation successful")
        return True
    except Exception as e:
        logger.error("Connection validation failed", extra={"error": str(e)})
        raise DatabaseConnectionError(f"Connection validation failed: {str(e)}") from e


def validate_timeframe(timeframe: str) -> str:
    """
    Validate and normalize timeframe code.

    Args:
        timeframe: Timeframe code to validate (e.g., "D1", "h1", "M30")

    Returns:
        Normalized timeframe code in lowercase (e.g., "d1", "h1", "m30")

    Raises:
        DataValidationError: If timeframe is invalid or not in VALID_TIMEFRAMES

    Example:
        >>> validate_timeframe("D1")
        'd1'
        >>> validate_timeframe("h1")
        'h1'
        >>> validate_timeframe("invalid")
        Traceback (most recent call last):
            ...
        DataValidationError: Invalid timeframe: 'invalid'. Valid timeframes: d1, h4, h1, m30, m15, m5, m1
    """
    if timeframe is None:
        raise DataValidationError(
            f"Invalid timeframe: None. Valid timeframes: {', '.join(VALID_TIMEFRAMES)}"
        )

    if not isinstance(timeframe, str):
        raise DataValidationError(
            f"Invalid timeframe: {timeframe}. Valid timeframes: {', '.join(VALID_TIMEFRAMES)}"
        )

    normalized = timeframe.lower()

    if normalized not in VALID_TIMEFRAMES:
        raise DataValidationError(
            f"Invalid timeframe: '{timeframe}'. Valid timeframes: {', '.join(VALID_TIMEFRAMES)}"
        )

    return normalized


def parse_timestamp(timestamp: str) -> datetime:
    """
    Parse ISO format timestamp string to datetime with UTC timezone.

    Supports various ISO 8601 formats and ensures the result is in UTC timezone.
    If the timestamp has a non-UTC timezone offset, it will be converted to UTC.

    Args:
        timestamp: ISO format timestamp string (e.g., "2024-01-15T14:00:00Z")

    Returns:
        datetime object with UTC timezone

    Raises:
        DataValidationError: If timestamp is invalid or cannot be parsed

    Example:
        >>> parse_timestamp("2024-01-15T14:00:00Z")
        datetime.datetime(2024, 1, 15, 14, 0, 0, tzinfo=datetime.timezone.utc)
        >>> parse_timestamp("2024-01-15T14:00:00+05:00")
        datetime.datetime(2024, 1, 15, 9, 0, 0, tzinfo=datetime.timezone.utc)
    """
    if timestamp is None:
        raise DataValidationError(
            "Invalid timestamp: None. Expected ISO format timestamp string."
        )

    if not isinstance(timestamp, str):
        raise DataValidationError(
            f"Invalid timestamp: {timestamp}. Expected ISO format timestamp string."
        )

    try:
        # Try parsing with fromisoformat (handles most ISO 8601 formats)
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        # Ensure UTC timezone
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        elif dt.tzinfo != timezone.utc:
            dt = dt.astimezone(timezone.utc)

        return dt

    except (ValueError, AttributeError) as e:
        raise DataValidationError(
            f"Invalid timestamp: '{timestamp}'. Expected ISO format (e.g., '2024-01-15T14:00:00Z'). Error: {e}"
        )


def calculate_expected_candles(timeframe: str, start: datetime, end: datetime) -> int:
    """
    Calculate expected number of candles in a date range for given timeframe.

    This function calculates how many candles should exist between start and end
    datetimes based on the timeframe interval. Partial periods are not counted.

    Args:
        timeframe: Timeframe code (d1, h4, h1, m30, m15, m5, m1)
        start: Start datetime (should have timezone info)
        end: End datetime (should have timezone info)

    Returns:
        Integer count of expected candles

    Raises:
        DataValidationError: If timeframe is invalid

    Example:
        >>> from datetime import datetime, timezone
        >>> start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        >>> end = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        >>> calculate_expected_candles("m1", start, end)
        60
        >>> calculate_expected_candles("h1", start, end)
        1
    """
    # Validate timeframe
    normalized_tf = validate_timeframe(timeframe)

    # Get minutes per candle for this timeframe
    minutes_per_candle = TIMEFRAME_MINUTES[normalized_tf]

    # Calculate total minutes in the date range
    time_diff = end - start
    total_minutes = int(time_diff.total_seconds() / 60)

    # Calculate number of complete candles
    num_candles = total_minutes // minutes_per_candle

    return num_candles


def build_where_clause(conditions: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Build parameterized WHERE clause from dictionary of conditions.

    This function creates SQL WHERE clause strings with safe parameter binding
    to prevent SQL injection. Handles None values as IS NULL checks.

    Args:
        conditions: Dictionary mapping column names to values

    Returns:
        Tuple of (where_clause_string, parameters_dict)
        - where_clause_string: SQL WHERE clause without the "WHERE" keyword
        - parameters_dict: Dictionary of parameter bindings

    Example:
        >>> build_where_clause({"symbol": "EURUSD", "value": 65})
        ("symbol = :symbol AND value = :value", {"symbol": "EURUSD", "value": 65})
        >>> build_where_clause({"symbol": "EURUSD", "deleted_at": None})
        ("symbol = :symbol AND deleted_at IS NULL", {"symbol": "EURUSD"})
        >>> build_where_clause({})
        ("", {})
    """
    if not conditions:
        return ("", {})

    clauses = []
    params = {}

    for column, value in conditions.items():
        if value is None:
            # Use IS NULL for None values
            clauses.append(f"{column} IS NULL")
        else:
            # Use parameterized query for actual values
            clauses.append(f"{column} = :{column}")
            params[column] = value

    where_str = " AND ".join(clauses)
    return (where_str, params)


def validate_table_name(table_name: str) -> str:
    """
    Validate and sanitize table name to prevent SQL injection.

    Only alphanumeric characters and underscores are allowed. This prevents
    SQL injection attacks through malicious table names.

    Args:
        table_name: Table name to validate

    Returns:
        The validated table name (unchanged if valid)

    Raises:
        DataValidationError: If table name is invalid or contains forbidden characters

    Example:
        >>> validate_table_name("indicators")
        'indicators'
        >>> validate_table_name("eurusd_d1_rates")
        'eurusd_d1_rates'
        >>> validate_table_name("table'; DROP TABLE--")
        Traceback (most recent call last):
            ...
        DataValidationError: Invalid table name...
    """
    if table_name is None:
        raise DataValidationError(
            "Invalid table name: None. Table names must contain only alphanumeric characters and underscores."
        )

    if not isinstance(table_name, str):
        raise DataValidationError(
            f"Invalid table name: {table_name}. Table names must be strings containing only alphanumeric characters and underscores."
        )

    if not table_name:
        raise DataValidationError(
            "Invalid table name: empty string. Table names must contain only alphanumeric characters and underscores."
        )

    # Only allow alphanumeric and underscore characters
    if not re.match(r"^[a-zA-Z0-9_]+$", table_name):
        raise DataValidationError(
            f"Invalid table name: '{table_name}'. Table names must contain only alphanumeric characters and underscores."
        )

    return table_name


def validate_column_name(column_name: str) -> str:
    """
    Validate and sanitize column name to prevent SQL injection.

    Only alphanumeric characters and underscores are allowed. This prevents
    SQL injection attacks through malicious column names.

    Args:
        column_name: Column name to validate

    Returns:
        The validated column name (unchanged if valid)

    Raises:
        DataValidationError: If column name is invalid or contains forbidden characters

    Example:
        >>> validate_column_name("symbol")
        'symbol'
        >>> validate_column_name("indicator_type")
        'indicator_type'
        >>> validate_column_name("col'; DROP--")
        Traceback (most recent call last):
            ...
        DataValidationError: Invalid column name...
    """
    if column_name is None:
        raise DataValidationError(
            "Invalid column name: None. Column names must contain only alphanumeric characters and underscores."
        )

    if not isinstance(column_name, str):
        raise DataValidationError(
            f"Invalid column name: {column_name}. Column names must be strings containing only alphanumeric characters and underscores."
        )

    if not column_name:
        raise DataValidationError(
            "Invalid column name: empty string. Column names must contain only alphanumeric characters and underscores."
        )

    # Only allow alphanumeric and underscore characters
    if not re.match(r"^[a-zA-Z0-9_]+$", column_name):
        raise DataValidationError(
            f"Invalid column name: '{column_name}'. Column names must contain only alphanumeric characters and underscores."
        )

    return column_name
