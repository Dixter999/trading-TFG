"""
Paper Trading Engine for Position Management (Issue #328).

This module provides a paper trading engine that manages virtual positions
and executes trades based on model signals from the inference service.

The main components are:
- PaperTradingEngine: Core trading logic for processing candles
- PaperTradingRunner: Docker service entry point (Issue #434)
- PositionManager: Position lifecycle management
- TradeLogger: Trade persistence
- PaperTradingConfig: Configuration management (Issue #430)
"""

from src.paper_trading.config import (
    AlertsConfig,
    ExitScaffoldConfig,
    PaperTradingConfig,
    RiskConfig,
    SymbolConfig,
)
from src.paper_trading.engine import PaperTradingEngine
from src.paper_trading.main import PaperTradingRunner, configure_logging
from src.paper_trading.models import (
    ExitReason,
    Position,
    PositionDirection,
    PositionState,
    Trade,
)
from src.paper_trading.position_manager import PositionManager
from src.paper_trading.trade_logger import TradeLogger

__all__ = [
    # Configuration (Issue #430)
    "PaperTradingConfig",
    "SymbolConfig",
    "RiskConfig",
    "ExitScaffoldConfig",
    "AlertsConfig",
    # Models
    "Position",
    "PositionDirection",
    "PositionState",
    "Trade",
    "ExitReason",
    # Core components
    "PositionManager",
    "TradeLogger",
    "PaperTradingEngine",
    "PaperTradingRunner",
    "configure_logging",
]
