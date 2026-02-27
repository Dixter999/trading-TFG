"""Centralized symbol configuration module.

This module provides a single source of truth for all trading symbols and related
configurations. To add a new symbol, simply add it to SUPPORTED_SYMBOLS - all other
configurations are generated dynamically.

Adding a New Symbol:
    1. Add the symbol to SUPPORTED_SYMBOLS list
    2. Ensure the database tables exist (run migrations)
    3. That's it! All other configurations are auto-generated.

Example:
    # To add AUDUSD, change:
    SUPPORTED_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY"]
    # To:
    SUPPORTED_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "AUDUSD"]
"""

# =============================================================================
# SYMBOL MAPPING - Internal to MT5 and Database
# =============================================================================

# Mapping of internal symbol names to MT5 Gateway symbol names
# Based on MT5_SYMBOL_DOCUMENTATION.md
_MT5_SYMBOL_MAP: dict[str, str] = {
    # GOLD/SILVER: Use .pro suffix for historical rates (not XAUUSD/XAGUSD)
    "GOLD": "GOLD.pro",      # GOLD.pro has historical data, XAUUSD only has ticks
    "SILVER": "SILVER.pro",  # SILVER.pro has historical data, XAGUSD only has ticks
    # FX pairs work both with and without .pro suffix
    # "EURUSD": "EURUSD.pro",  # Optional, works without .pro
    # "GBPUSD": "GBPUSD.pro",  # Optional, works without .pro
    # "USDJPY": "USDJPY.pro",  # Optional, works without .pro
    # "EURJPY": "EURJPY.pro",  # Optional, works without .pro
}

# Mapping of internal symbol names to database table prefixes
# Some symbols use different names in the database than their internal names
_DB_SYMBOL_MAP: dict[str, str] = {
    "SILVER": "XAGUSD",  # SILVER tables are named xagusd_*_rates in database
    # GOLD uses "gold" prefix (matches internal name)
    # FX pairs use lowercase of internal name (eurusd, gbpusd, etc.)
}


def get_mt5_symbol(symbol: str) -> str:
    """Convert internal symbol name to MT5 Gateway symbol name.

    Args:
        symbol: Internal symbol name (e.g., "GOLD", "EURUSD")

    Returns:
        MT5 symbol name (e.g., "GOLD.pro", "EURUSD")

    Example:
        >>> get_mt5_symbol("GOLD")
        'GOLD.pro'
        >>> get_mt5_symbol("EURUSD")
        'EURUSD'
        >>> get_mt5_symbol("SILVER")
        'SILVER.pro'
    """
    return _MT5_SYMBOL_MAP.get(symbol, symbol)


def get_db_symbol(symbol: str) -> str:
    """Convert internal symbol name to database table prefix.

    Args:
        symbol: Internal symbol name (e.g., "SILVER", "GOLD")

    Returns:
        Database table prefix (e.g., "XAGUSD", "gold")

    Example:
        >>> get_db_symbol("SILVER")
        'XAGUSD'
        >>> get_db_symbol("GOLD")
        'GOLD'
        >>> get_db_symbol("EURUSD")
        'EURUSD'
    """
    return _DB_SYMBOL_MAP.get(symbol, symbol)


# =============================================================================
# PRIMARY CONFIGURATION - SINGLE SOURCE OF TRUTH
# =============================================================================

# All supported trading symbols
# To add a new symbol: just add it to this list
# Note: AMZN removed - not available in MT5 broker (404 error)
# Issue #633: SILVER/XAGUSD removed - consistently fails with insufficient data
# Stale symbol cleanup (2026-02-24): GOLD, BTCUSD removed - FOREX ONLY system
SUPPORTED_SYMBOLS: list[str] = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "EURCAD", "USDCAD", "USDCHF", "EURGBP"]

# Valid timeframes for multi-symbol trading (M30 through D1)
# These are the timeframes used for indicator calculation and trading
# Issue #562: Added H3 to support eurusd_long_Stoch_K_oversold_long_25_H3 model
VALID_TIMEFRAMES: list[str] = ["M30", "H1", "H2", "H3", "H4", "H6", "H8", "H12", "D1"]

# =============================================================================
# TIMEFRAME CONFIGURATIONS
# =============================================================================

# Full timeframes for EURUSD in production (13 timeframes)
# NOTE: M1, M5, M15 are NOT in production database
TIMEFRAMES_FULL: list[str] = [
    "d1",
    "h1",
    "h2",
    "h3",
    "h4",
    "h6",
    "h8",
    "h12",
    "m10",
    "m20",
    "m30",
    "mn1",
    "w1",
]

# Timeframes for multi-symbol expansion (9 timeframes: M30 through D1)
# NOTE: Production database only has M30-D1 for multi-symbol trading
# M1, M5, M15 are NOT used in production
# Issue #562: Added h3 to support H3 model timeframes
TIMEFRAMES_MULTI_SYMBOL: list[str] = [
    "d1",
    "h1",
    "h2",
    "h3",
    "h4",
    "h6",
    "h8",
    "h12",
    "m30",
]

# Timeframe configuration for backfill - months of historical data per timeframe
# Issue #404: Updated higher timeframes to 96 months (8 years)
TIMEFRAME_CONFIG_BACKFILL: dict[str, int] = {
    "M1": 1,  # 1 month (M1 data is huge)
    "M5": 3,  # 3 months
    "M15": 6,  # 6 months
    "M30": 24,  # 2 years
    "H1": 48,  # 4 years
    "H2": 48,  # 4 years
    "H3": 48,  # 4 years (Issue #562)
    "H4": 96,  # 8 years
    "H6": 96,  # 8 years
    "H8": 96,  # 8 years
    "H12": 96,  # 8 years
    "D1": 96,  # 8 years
}

# Minutes per candle for each timeframe
TIMEFRAME_MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H2": 120,
    "H3": 180,  # Issue #562
    "H4": 240,
    "H6": 360,
    "H8": 480,
    "H12": 720,
    "D1": 1440,
    "W1": 10080,
    "MN1": 43200,  # Approximate: 30 days
}

# =============================================================================
# TABLE NAME GENERATION FUNCTIONS
# =============================================================================


def generate_rate_table_name(symbol: str, timeframe: str) -> str:
    """Generate the rate table name for a symbol/timeframe combination.

    Args:
        symbol: Trading symbol (e.g., "EURUSD", "GBPUSD", "SILVER")
        timeframe: Timeframe code (e.g., "H1", "M30")

    Returns:
        Table name in format: "{db_symbol}_{timeframe}_rates" (lowercase)

    Example:
        >>> generate_rate_table_name("EURUSD", "H1")
        'eurusd_h1_rates'
        >>> generate_rate_table_name("SILVER", "H1")
        'xagusd_h1_rates'
    """
    db_symbol = get_db_symbol(symbol)
    return f"{db_symbol.lower()}_{timeframe.lower()}_rates"


def generate_indicator_table_name(symbol: str) -> str:
    """Generate the indicator table name for a symbol.

    All symbols use per-symbol table pattern: 'technical_indicator_{symbol}'.
    Legacy 'technical_indicators' table is deprecated.

    Args:
        symbol: Trading symbol (e.g., "EURUSD", "GBPUSD", "SILVER")

    Returns:
        Table name for technical indicators

    Example:
        >>> generate_indicator_table_name("EURUSD")
        'technical_indicator_eurusd'
        >>> generate_indicator_table_name("GBPUSD")
        'technical_indicator_gbpusd'
        >>> generate_indicator_table_name("SILVER")
        'technical_indicator_xagusd'
    """
    db_symbol = get_db_symbol(symbol)
    return f"technical_indicator_{db_symbol.lower()}"


def generate_symbol_table_map() -> dict[str, dict[str, str]]:
    """Generate mapping of symbol -> timeframe -> table name.

    Dynamically generates the SYMBOL_TABLE_MAP for all supported symbols
    and valid timeframes.

    Returns:
        Dictionary mapping symbol -> timeframe -> table name

    Example:
        >>> result = generate_symbol_table_map()
        >>> result["EURUSD"]["H1"]
        'eurusd_h1_rates'
    """
    result: dict[str, dict[str, str]] = {}

    for symbol in SUPPORTED_SYMBOLS:
        result[symbol] = {}
        for timeframe in VALID_TIMEFRAMES:
            result[symbol][timeframe] = generate_rate_table_name(symbol, timeframe)

    return result


# =============================================================================
# BACKUP CONFIGURATION GENERATION
# =============================================================================


def generate_markets_tables() -> list[str]:
    """Generate list of all markets database tables for backup.

    Returns:
        List of rate table names for all symbols and their timeframes.
        Based on database schema:
        - EURUSD: TIMEFRAMES_FULL (13 timeframes: M10, M20, M30, H1-H12, D1, H3, W1, MN1)
        - All other symbols: VALID_TIMEFRAMES (9 timeframes: M30, H1-H12, D1)
    """
    tables: list[str] = []

    for symbol in SUPPORTED_SYMBOLS:
        # Determine which timeframes this symbol uses
        if symbol == "EURUSD":
            # Full historical timeframes (13 total)
            timeframes = TIMEFRAMES_FULL
        else:
            # Trading timeframes only (8 total: M30-D1)
            timeframes = [tf.lower() for tf in VALID_TIMEFRAMES]

        # Generate table names using database symbol mapping
        db_symbol = get_db_symbol(symbol)
        for tf in timeframes:
            tables.append(f"{db_symbol.lower()}_{tf}_rates")

    return tables


def generate_ai_model_tables() -> list[str]:
    """Generate list of all AI model database tables for backup.

    Returns:
        List of AI model table names including technical indicator tables
        for all supported symbols.
    """
    # Base tables (schema, analysis, trading, etc.)
    base_tables: list[str] = [
        # Schema migrations
        "alembic_version",
        # Cluster analysis tables
        "cluster_assignments",
        "cluster_performance",
        "cluster_templates",
        "kmeans_optimization_results",
        "pattern_clusters",
        "scaler_params",
        # Market structure tables
        "bos_events",
        "order_blocks",
        "order_block_touches",
        "swing_points",
        # Economic and sentiment data
        "economic_events",
        "reddit_posts",
        "seasons",
        # Trading tables
        "leaderboard_snapshots",
        "trades",
        # Paper trading tables
        "paper_decision_log",
        "paper_performance",
        "paper_positions",
        "paper_trades",
        # Analysis tables
        "baseline_results",
        "gate_decisions",
        "monte_carlo_results",
        # LLM/AI tables
        "llm_models",
        "model_chats",
    ]

    # Add technical indicator tables for all symbols
    indicator_tables: list[str] = []
    for symbol in SUPPORTED_SYMBOLS:
        indicator_tables.append(generate_indicator_table_name(symbol))

    # Issue #529: Add symbol-specific EURUSD table for training performance
    # Dual-table strategy:
    # - technical_indicators (legacy, ALL symbols) - used by frontend
    # - technical_indicator_eurusd (NEW, EURUSD only) - used by training scripts
    # This enables 9x faster D1 data loading (80MB vs 722MB)
    if "technical_indicator_eurusd" not in indicator_tables:
        # Insert after technical_indicators for logical ordering
        try:
            legacy_idx = indicator_tables.index("technical_indicators")
            indicator_tables.insert(legacy_idx + 1, "technical_indicator_eurusd")
        except ValueError:
            # If technical_indicators not found (shouldn't happen), append
            indicator_tables.append("technical_indicator_eurusd")

    return base_tables + indicator_tables


# =============================================================================
# PRE-GENERATED CONSTANTS (for backward compatibility)
# =============================================================================

# Pre-generated symbol table map for direct import
SYMBOL_TABLE_MAP: dict[str, dict[str, str]] = generate_symbol_table_map()

# Pre-generated markets tables list
MARKETS_TABLES: list[str] = generate_markets_tables()

# Pre-generated AI model tables list
AI_MODEL_TABLES: list[str] = generate_ai_model_tables()
