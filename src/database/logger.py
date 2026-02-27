"""
Structured logging module for database operations.

This module provides JSON-formatted logging with query execution tracking
and connection pool statistics. All log output is in JSON format for easy
parsing and integration with log aggregation systems.

Example:
    >>> from database.logger import StructuredLogger
    >>> logger = StructuredLogger("database")
    >>> logger.info("Query executed", duration_ms=45.2, rows=10)
    {"timestamp": "2025-10-03T12:30:45.123456+00:00", "level": "INFO", ...}
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Set


# Standard LogRecord attributes that should not be included as extra fields
_STANDARD_RECORD_ATTRS: Set[str] = {
    "name",
    "msg",
    "args",
    "created",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "thread",
    "threadName",
    "exc_info",
    "exc_text",
    "stack_info",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """
    Custom formatter that outputs JSON-formatted log entries.

    This formatter converts Python logging.LogRecord objects into JSON strings,
    including standard fields (timestamp, level, message, etc.) and any extra
    fields passed via the `extra` parameter in logging calls.

    All timestamps are in ISO 8601 UTC format for consistency.

    Example:
        >>> formatter = JsonFormatter()
        >>> handler = logging.StreamHandler()
        >>> handler.setFormatter(formatter)
        >>> logger = logging.getLogger("test")
        >>> logger.addHandler(handler)
        >>> logger.info("Test message", extra={"user_id": 123})
        {"timestamp": "2025-10-03T12:30:45+00:00", "level": "INFO", ...}
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON string.

        Args:
            record: Log record to format

        Returns:
            JSON string with log data including:
            - timestamp: ISO 8601 UTC timestamp
            - level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            - message: Log message
            - module: Module name where log was called
            - function: Function name where log was called
            - line: Line number where log was called
            - Any extra fields from the `extra` parameter

        Example:
            >>> record = logging.LogRecord(
            ...     name="test", level=logging.INFO, pathname="test.py",
            ...     lineno=42, msg="Test", args=(), exc_info=None, func="test_func"
            ... )
            >>> formatter = JsonFormatter()
            >>> json_output = formatter.format(record)
            >>> import json
            >>> data = json.loads(json_output)
            >>> assert "timestamp" in data
            >>> assert data["level"] == "INFO"
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields from record if present
        # Any custom attributes added to the record will be included
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS:
                log_data[key] = value

        # Custom JSON encoder to handle datetime objects
        def json_serializer(obj):
            """Convert non-serializable objects to strings."""
            if hasattr(obj, 'isoformat'):  # datetime, date, time objects
                return obj.isoformat()
            return str(obj)

        return json.dumps(log_data, default=json_serializer)


class StructuredLogger:
    """
    JSON-formatted structured logging for database operations.

    This class provides a simple interface for logging database operations
    with automatic JSON formatting. All log methods accept keyword arguments
    that will be included as extra fields in the JSON output.

    Typical use cases:
    - Query execution time tracking
    - Connection pool statistics
    - Error tracking with context
    - Performance monitoring

    Attributes:
        logger: Underlying Python logger instance with JSON formatter

    Example:
        >>> logger = StructuredLogger("database")
        >>> logger.info("Connection established", host="localhost", port=5432)
        >>> logger.info("Query executed", duration_ms=45.2, rows=10, query="SELECT * FROM users")
        >>> logger.error("Connection failed", error="timeout", retry_count=3)
    """

    def __init__(self, name: str):
        """
        Initialize structured logger.

        Args:
            name: Logger name (typically module name like "database" or "database.queries")

        Example:
            >>> logger = StructuredLogger("database")
            >>> assert logger.logger.name == "database"
        """
        self.logger: logging.Logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Add JSON formatter to handler
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)

    def debug(self, message: str, **kwargs: Any) -> None:
        """
        Log DEBUG level message with optional extra fields.

        Args:
            message: Log message
            **kwargs: Extra fields to include in JSON output

        Example:
            >>> logger = StructuredLogger("database")
            >>> logger.debug("Preparing query", table="users", filters={"active": True})
        """
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """
        Log INFO level message with optional extra fields.

        Args:
            message: Log message
            **kwargs: Extra fields to include in JSON output

        Example:
            >>> logger = StructuredLogger("database")
            >>> logger.info("Query executed successfully", duration_ms=156.789, rows=42)
        """
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """
        Log WARNING level message with optional extra fields.

        Args:
            message: Log message
            **kwargs: Extra fields to include in JSON output

        Example:
            >>> logger = StructuredLogger("database")
            >>> logger.warning("Connection pool nearly exhausted", checked_out=9, pool_size=10)
        """
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """
        Log ERROR level message with optional extra fields.

        Args:
            message: Log message
            **kwargs: Extra fields to include in JSON output

        Example:
            >>> logger = StructuredLogger("database")
            >>> logger.error("Query failed", error="timeout", duration_ms=30000, query="SELECT ...")
        """
        self.logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """
        Log CRITICAL level message with optional extra fields.

        Args:
            message: Log message
            **kwargs: Extra fields to include in JSON output

        Example:
            >>> logger = StructuredLogger("database")
            >>> logger.critical("Database connection lost", error="network_failure", retry_exhausted=True)
        """
        self.logger.critical(message, extra=kwargs)
