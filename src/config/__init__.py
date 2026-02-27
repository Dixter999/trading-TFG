"""Configuration module for trading system.

This module provides centralized configuration for symbols, timeframes,
and related settings used across the trading platform.

Adding a New Symbol:
    1. Update SUPPORTED_SYMBOLS in src/config/symbols.py
    2. Ensure database tables exist (run migrations)
    3. All other configurations are auto-generated

Example:
    from src.config.symbols import SUPPORTED_SYMBOLS, VALID_TIMEFRAMES
    from src.config.symbols import generate_rate_table_name
"""

from src.config.symbols import (
    AI_MODEL_TABLES,
    MARKETS_TABLES,
    SUPPORTED_SYMBOLS,
    SYMBOL_TABLE_MAP,
    TIMEFRAME_CONFIG_BACKFILL,
    TIMEFRAME_MINUTES,
    TIMEFRAMES_FULL,
    TIMEFRAMES_MULTI_SYMBOL,
    VALID_TIMEFRAMES,
    generate_ai_model_tables,
    generate_indicator_table_name,
    generate_markets_tables,
    generate_rate_table_name,
    generate_symbol_table_map,
)

__all__ = [
    # Core symbol configuration
    "SUPPORTED_SYMBOLS",
    "VALID_TIMEFRAMES",
    # Timeframe configurations
    "TIMEFRAMES_FULL",
    "TIMEFRAMES_MULTI_SYMBOL",
    "TIMEFRAME_CONFIG_BACKFILL",
    "TIMEFRAME_MINUTES",
    # Pre-generated mappings
    "SYMBOL_TABLE_MAP",
    "MARKETS_TABLES",
    "AI_MODEL_TABLES",
    # Generator functions
    "generate_rate_table_name",
    "generate_indicator_table_name",
    "generate_symbol_table_map",
    "generate_markets_tables",
    "generate_ai_model_tables",
]
