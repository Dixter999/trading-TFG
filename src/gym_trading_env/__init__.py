"""
gym_trading_env - Custom Gymnasium Trading Environment with Dual Database Integration

This package provides a custom trading environment for reinforcement learning agents
that integrates with dual PostgreSQL databases (markets and ai_model) to provide
real-time market data and pre-calculated technical indicators.
"""

__version__ = "0.1.0"

# Import main classes (use relative imports to avoid circular dependencies)
from .config import DatabaseConfig
from .data_splitter import DataSplitter
from .datafeed import DataFeed
from .db_pool import PoolManager, PoolType
from .normalizer import Normalizer
from .simplified_env import SimplifiedTradingEnv
from .minimal_datafeed import MinimalDataFeed
from .minimal_normalizer import MinimalNormalizer

# Optional imports that may have extra dependencies
try:
    from .eurusd_h1_env import TradingEnv
except ImportError:
    TradingEnv = None

__all__ = [
    "DatabaseConfig",
    "DataSplitter",
    "DataFeed",
    "PoolManager",
    "PoolType",
    "Normalizer",
    "TradingEnv",
    "SimplifiedTradingEnv",
    "MinimalDataFeed",
    "MinimalNormalizer",
]
