"""
Database configuration classes using Pydantic v2.

This module provides configuration management for database connections,
loading settings from environment variables with validation.
"""

import os

from pydantic import BaseModel, Field, field_validator


class DatabaseConfig(BaseModel):
    """Configuration for a single database connection.

    Attributes:
        host: Database host address
        port: Database port (default: 5432)
        database: Database name
        user: Database user
        password: Database password
        pool_size: Connection pool size (default: 5)
        max_overflow: Maximum overflow connections (default: 10)
    """

    host: str = Field(..., description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(..., description="Database name")
    user: str = Field(..., description="Database user")
    password: str = Field(..., description="Database password")
    pool_size: int = Field(default=5, description="Connection pool size", gt=0)
    max_overflow: int = Field(default=10, description="Max overflow connections", ge=0)

    @field_validator("pool_size")
    @classmethod
    def validate_pool_size(cls, v: int) -> int:
        """Validate that pool_size is positive."""
        if v <= 0:
            raise ValueError("pool_size must be greater than 0")
        return v

    @field_validator("max_overflow")
    @classmethod
    def validate_max_overflow(cls, v: int) -> int:
        """Validate that max_overflow is non-negative."""
        if v < 0:
            raise ValueError("max_overflow must be non-negative")
        return v


class AppConfig(BaseModel):
    """Application configuration container for all database connections.

    Attributes:
        markets_db: Configuration for Markets Database (read-only)
        ai_model_db: Configuration for AI Model Database (read-write)
    """

    markets_db: DatabaseConfig
    ai_model_db: DatabaseConfig

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables.

        Environment variables expected:
            Markets DB:
                - MARKETS_DB_HOST (required)
                - MARKETS_DB_PORT (optional, default: 5432)
                - MARKETS_DB_NAME (required)
                - MARKETS_DB_USER (required)
                - MARKETS_DB_PASSWORD (required)
                - MARKETS_DB_POOL_SIZE (optional, default: 5)
                - MARKETS_DB_MAX_OVERFLOW (optional, default: 10)

            AI Model DB:
                - AI_MODEL_DB_HOST (required)
                - AI_MODEL_DB_PORT (optional, default: 5432)
                - AI_MODEL_DB_NAME (required)
                - AI_MODEL_DB_USER (required)
                - AI_MODEL_DB_PASSWORD (required)
                - AI_MODEL_DB_POOL_SIZE (optional, default: 5)
                - AI_MODEL_DB_MAX_OVERFLOW (optional, default: 10)

        Returns:
            AppConfig instance loaded from environment variables

        Raises:
            KeyError: If required environment variables are missing
            ValidationError: If environment variables contain invalid values
        """
        # Load Markets DB configuration
        markets_db = DatabaseConfig(
            host=os.environ["MARKETS_DB_HOST"],
            port=int(os.environ.get("MARKETS_DB_PORT", "5432")),
            database=os.environ["MARKETS_DB_NAME"],
            user=os.environ["MARKETS_DB_USER"],
            password=os.environ["MARKETS_DB_PASSWORD"],
            pool_size=int(os.environ.get("MARKETS_DB_POOL_SIZE", "5")),
            max_overflow=int(os.environ.get("MARKETS_DB_MAX_OVERFLOW", "10")),
        )

        # Load AI Model DB configuration
        ai_model_db = DatabaseConfig(
            host=os.environ["AI_MODEL_DB_HOST"],
            port=int(os.environ.get("AI_MODEL_DB_PORT", "5432")),
            database=os.environ["AI_MODEL_DB_NAME"],
            user=os.environ["AI_MODEL_DB_USER"],
            password=os.environ["AI_MODEL_DB_PASSWORD"],
            pool_size=int(os.environ.get("AI_MODEL_DB_POOL_SIZE", "5")),
            max_overflow=int(os.environ.get("AI_MODEL_DB_MAX_OVERFLOW", "10")),
        )

        return cls(markets_db=markets_db, ai_model_db=ai_model_db)
