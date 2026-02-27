"""
Custom exception classes for database operations.

This module defines a hierarchy of exceptions for handling various
database-related errors consistently throughout the application.
"""


class DatabaseError(Exception):
    """Base exception for all database-related errors.

    All database exceptions should inherit from this base class to allow
    for consistent error handling and logging.
    """

    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when a database connection fails.

    This exception is raised when:
    - Unable to establish connection to the database
    - Connection is lost unexpectedly
    - Authentication fails
    - Network issues prevent connection
    """

    pass


class QueryExecutionError(DatabaseError):
    """Raised when a database query execution fails.

    This exception is raised when:
    - SQL syntax errors occur
    - Query execution times out
    - Constraint violations occur
    - Database server returns an error
    """

    pass


class DataValidationError(DatabaseError):
    """Raised when data validation fails before or after database operations.

    This exception is raised when:
    - Input data doesn't match expected schema
    - Retrieved data doesn't pass validation
    - Type conversion fails
    - Data integrity constraints are violated
    """

    pass


class PermissionError(DatabaseError):
    """Raised when database permission is denied.

    This exception is raised when:
    - User lacks required privileges
    - Read-only access is attempted for write operations
    - Table or schema access is restricted
    - Row-level security policies prevent access
    """

    pass


class TimeoutError(DatabaseError):
    """Raised when a database operation times out.

    This exception is raised when:
    - Query execution exceeds timeout limit
    - Connection pool acquisition times out
    - Lock acquisition times out
    - Transaction times out
    """

    pass
