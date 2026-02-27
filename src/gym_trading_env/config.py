"""
Configuration module for gym_trading_env.

Handles database configuration loading from environment variables
with validation and connection parameter generation.
"""

import os
from typing import Any, Dict, Optional


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


# Default configuration constants
DEFAULT_DB_HOST = "localhost"
DEFAULT_DB_PORT = 5432
DEFAULT_POOL_MIN_CONN = 1
DEFAULT_POOL_MAX_CONN = 10


class DatabaseConfig:
    """
    Database configuration manager for dual PostgreSQL connections.

    Loads configuration from environment variables and provides
    connection strings and parameters for both markets and ai_model databases.

    Implements singleton pattern to ensure consistent configuration across the application.
    """

    _instance: Optional["DatabaseConfig"] = None

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize database configuration from environment variables."""
        # Reinitialize to pick up environment changes (needed for testing)
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from environment variables."""
        # Markets database configuration
        self.markets_host = os.getenv("MARKETS_DB_HOST", DEFAULT_DB_HOST)
        self.markets_database = os.getenv("MARKETS_DB_NAME", "markets")
        self.markets_user = os.getenv("MARKETS_DB_USER", "markets")
        self.markets_password = os.getenv("MARKETS_DB_PASSWORD", "")
        self.markets_port = int(os.getenv("MARKETS_DB_PORT", str(DEFAULT_DB_PORT)))

        # AI Model database configuration
        self.ai_model_host = os.getenv("AI_MODEL_DB_HOST", DEFAULT_DB_HOST)
        self.ai_model_database = os.getenv("AI_MODEL_DB_NAME", "ai_model")
        self.ai_model_user = os.getenv("AI_MODEL_DB_USER", "ai_model")
        self.ai_model_password = os.getenv("AI_MODEL_DB_PASSWORD", "")
        self.ai_model_port = int(os.getenv("AI_MODEL_DB_PORT", str(DEFAULT_DB_PORT)))

        # Connection pool configuration
        self.pool_min_conn = int(
            os.getenv("DB_POOL_MIN_CONN", str(DEFAULT_POOL_MIN_CONN))
        )
        self.pool_max_conn = int(
            os.getenv("DB_POOL_MAX_CONN", str(DEFAULT_POOL_MAX_CONN))
        )

    def validate(self) -> None:
        """
        Validate configuration.

        Raises:
            ConfigValidationError: If configuration is invalid.
        """
        # Check required fields are not empty
        if not self.markets_host:
            raise ConfigValidationError("Markets database host is required")
        if not self.ai_model_host:
            raise ConfigValidationError("AI Model database host is required")

        # Validate pool configuration
        if self.pool_min_conn >= self.pool_max_conn:
            raise ConfigValidationError(
                f"pool_min_conn ({self.pool_min_conn}) must be less than "
                f"pool_max_conn ({self.pool_max_conn})"
            )

        # Validate pool values are positive
        if self.pool_min_conn < 1:
            raise ConfigValidationError("pool_min_conn must be at least 1")
        if self.pool_max_conn < 1:
            raise ConfigValidationError("pool_max_conn must be at least 1")

    def _build_connection_string(
        self, user: str, password: str, host: str, port: int, database: str
    ) -> str:
        """
        Build PostgreSQL connection string.

        Args:
            user: Database user
            password: Database password
            host: Database host
            port: Database port
            database: Database name

        Returns:
            Connection string in format: postgresql://user:password@host:port/database
        """
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    def _build_connection_params(
        self, host: str, database: str, user: str, password: str, port: int
    ) -> Dict[str, Any]:
        """
        Build connection parameters dict (psycopg2 format).

        Args:
            host: Database host
            database: Database name
            user: Database user
            password: Database password
            port: Database port

        Returns:
            Dictionary with connection parameters for psycopg2.
        """
        return {
            "host": host,
            "database": database,
            "user": user,
            "password": password,
            "port": port,
        }

    def get_markets_connection_string(self) -> str:
        """
        Get PostgreSQL connection string for markets database.

        Returns:
            Connection string in format: postgresql://user:password@host:port/database
        """
        return self._build_connection_string(
            self.markets_user,
            self.markets_password,
            self.markets_host,
            self.markets_port,
            self.markets_database,
        )

    def get_ai_model_connection_string(self) -> str:
        """
        Get PostgreSQL connection string for ai_model database.

        Returns:
            Connection string in format: postgresql://user:password@host:port/database
        """
        return self._build_connection_string(
            self.ai_model_user,
            self.ai_model_password,
            self.ai_model_host,
            self.ai_model_port,
            self.ai_model_database,
        )

    def get_markets_connection_params(self) -> Dict[str, Any]:
        """
        Get connection parameters dict for markets database (psycopg2 format).

        Returns:
            Dictionary with connection parameters for psycopg2.
        """
        return self._build_connection_params(
            self.markets_host,
            self.markets_database,
            self.markets_user,
            self.markets_password,
            self.markets_port,
        )

    def get_ai_model_connection_params(self) -> Dict[str, Any]:
        """
        Get connection parameters dict for ai_model database (psycopg2 format).

        Returns:
            Dictionary with connection parameters for psycopg2.
        """
        return self._build_connection_params(
            self.ai_model_host,
            self.ai_model_database,
            self.ai_model_user,
            self.ai_model_password,
            self.ai_model_port,
        )

    @classmethod
    def _reset_instance(cls) -> None:
        """Reset singleton instance (for testing only)."""
        cls._instance = None
