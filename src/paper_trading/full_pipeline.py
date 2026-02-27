"""
Full Trading Pipeline for orchestrating the complete trading workflow.

Issue #332: Comprehensive Decision Logging & Full Trading Pipeline Integration
Stream E: Full Pipeline Integration

This module provides the FullTradingPipeline class that wires all components:
- PatternIntegration (detects patterns)
- EntrySignalGenerator (converts to entry signals)
- LeverageCalculator (calculates position size)
- TradingEngine (opens positions)
- TradeDecisionLogger (logs full decision context)
- PerformanceTracker (updates metrics on close)

Pipeline Flow:
    PatternIntegration.get_recent_patterns()
            |
            v
    EntrySignalGenerator.check_entry_signal()
            |
            v
    LeverageCalculator.calculate_position_size()
            |
            v
    TradingEngine.open_position()
            |
            v
    TradeDecisionLogger.log_decision()
            |
            v
    PerformanceTracker.record_trade() (on close)
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from src.paper_trading.decision_logger import DecisionContext, TradeDecisionLogger
from src.paper_trading.entry_signal_generator import (
    EntrySignalGenerator,
    SignalGeneratorConfig,
    TradingSystemProtocol,
)
from src.paper_trading.leverage_calculator import LeverageCalculator
from src.paper_trading.performance_tracker import (
    PerformanceMetrics,
    PerformanceTracker,
    TradeResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class PatternIntegrationProtocol(Protocol):
    """Protocol for pattern integration."""

    async def get_recent_patterns(self, symbol: str) -> list[dict[str, Any]]:
        """
        Get recently detected patterns for a symbol.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")

        Returns:
            List of pattern dictionaries with type, direction, confidence, etc.
        """
        ...

    async def get_indicators(self, symbol: str) -> dict[str, dict[str, float]]:
        """
        Get current indicator values by timeframe.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")

        Returns:
            Dict mapping timeframe to indicator values
        """
        ...

    async def get_current_price(self, symbol: str) -> float:
        """
        Get current market price for a symbol.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")

        Returns:
            Current market price
        """
        ...


class TradingEngineProtocol(Protocol):
    """Protocol for trading engine."""

    async def open_position(
        self,
        symbol: str,
        direction: str,
        size: float,
        tp: float,
        sl: float,
    ) -> str:
        """
        Open a new position.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            direction: Trade direction ("LONG" or "SHORT")
            size: Position size in lots
            tp: Take profit price
            sl: Stop loss price

        Returns:
            Position ID string
        """
        ...

    async def get_open_positions(self) -> list[dict[str, Any]]:
        """
        Get list of open positions.

        Returns:
            List of position dictionaries
        """
        ...

    async def get_account_status(self) -> dict[str, Any]:
        """
        Get current account status.

        Returns:
            Dict with balance, equity, open_positions, daily_pnl
        """
        ...


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PipelineConfig:
    """
    Configuration for the full trading pipeline.

    Attributes:
        symbol: Trading symbol (default: "EURUSD")
        initial_balance: Starting balance for performance tracking (default: 10000.0)
        risk_percent: Risk per trade as percentage (default: 1.0)
        log_dir: Directory for decision logs (default: "logs/decisions")
        min_confidence: Minimum signal confidence to trade (default: 0.6)
        max_positions: Maximum concurrent positions (default: 3)
        check_interval_seconds: How often to check for signals (default: 60.0)
    """

    symbol: str = "EURUSD"
    initial_balance: float = 10000.0
    risk_percent: float = 1.0
    log_dir: str = "logs/decisions"
    min_confidence: float = 0.6
    max_positions: int = 3
    check_interval_seconds: float = 60.0


# =============================================================================
# FullTradingPipeline
# =============================================================================


class FullTradingPipeline:
    """
    Orchestrates the complete trading pipeline.

    Integrates:
    - Pattern detection -> Entry signals
    - Position sizing with leverage
    - Position management
    - Decision logging
    - Performance tracking

    The pipeline flow is:
    1. Get patterns and indicators from PatternIntegration
    2. Generate entry signal via EntrySignalGenerator
    3. Calculate position size via LeverageCalculator
    4. Open position via TradingEngine
    5. Log decision via TradeDecisionLogger
    6. Track performance via PerformanceTracker

    Usage:
        >>> config = PipelineConfig(symbol="EURUSD", risk_percent=1.0)
        >>> pipeline = FullTradingPipeline(
        ...     config=config,
        ...     pattern_integration=pattern_integration,
        ...     trading_engine=trading_engine,
        ...     trading_system=trading_system,
        ... )
        >>> await pipeline.start()
    """

    def __init__(
        self,
        config: PipelineConfig,
        pattern_integration: PatternIntegrationProtocol | None = None,
        trading_engine: TradingEngineProtocol | None = None,
        trading_system: TradingSystemProtocol | None = None,
    ) -> None:
        """
        Initialize FullTradingPipeline.

        Args:
            config: Pipeline configuration
            pattern_integration: Optional pattern integration for market data
            trading_engine: Optional trading engine for position management
            trading_system: Optional trading system for model-based decisions
        """
        self.config = config
        self.pattern_integration = pattern_integration
        self.trading_engine = trading_engine

        # Initialize components
        self.signal_generator = EntrySignalGenerator(
            trading_system=trading_system,
            config=SignalGeneratorConfig(
                min_confidence=config.min_confidence,
                max_positions=config.max_positions,
            ),
        )
        self.leverage_calculator = LeverageCalculator()
        self.decision_logger = TradeDecisionLogger(log_dir=config.log_dir)
        self.performance_tracker = PerformanceTracker(
            initial_balance=config.initial_balance
        )

        self._running = False

        logger.info(
            f"FullTradingPipeline initialized: "
            f"symbol={config.symbol}, "
            f"risk_percent={config.risk_percent}, "
            f"min_confidence={config.min_confidence}"
        )

    async def start(self) -> None:
        """
        Start the trading pipeline loop.

        Runs the main loop that periodically checks for signals
        and executes trades. Use stop() to terminate.
        """
        self._running = True
        logger.info("Starting Full Trading Pipeline")

        while self._running:
            try:
                await self.check_and_execute()
                await asyncio.sleep(self.config.check_interval_seconds)
            except Exception as e:
                logger.error(f"Pipeline error: {e}")

    async def stop(self) -> None:
        """
        Stop the trading pipeline.

        Sets the running flag to False, causing the main loop to exit.
        """
        self._running = False
        logger.info("Stopping Full Trading Pipeline")

    async def check_and_execute(self) -> str | None:
        """
        Single pipeline iteration.

        Performs the complete pipeline flow:
        1. Get patterns and indicators
        2. Check for entry signal
        3. Calculate position size
        4. Open position
        5. Log decision

        Returns:
            Position ID if position opened, None otherwise.
        """
        if not self.pattern_integration or not self.trading_engine:
            logger.warning("Pattern integration or trading engine not configured")
            return None

        try:
            # 1. Get current market state
            patterns = await self.pattern_integration.get_recent_patterns(
                self.config.symbol
            )
            indicators = await self.pattern_integration.get_indicators(
                self.config.symbol
            )
            current_price = await self.pattern_integration.get_current_price(
                self.config.symbol
            )
            account_status = await self.trading_engine.get_account_status()

            # 2. Check for entry signal
            signal = self.signal_generator.check_entry_signal(
                symbol=self.config.symbol,
                patterns=patterns,
                indicators=indicators,
                current_price=current_price,
                account_status=account_status,
            )

            if signal is None:
                logger.debug("No entry signal")
                return None

            # 3. Calculate position size
            position_size = self.leverage_calculator.calculate_position_size(
                symbol=self.config.symbol,
                balance=account_status.get("balance", self.config.initial_balance),
                risk_percent=self.config.risk_percent,
                stop_loss_pips=signal.sl_pips,
                current_price=current_price,
            )

            # 4. Open position
            direction = "LONG" if signal.direction == 1 else "SHORT"
            position_id = await self.trading_engine.open_position(
                symbol=self.config.symbol,
                direction=direction,
                size=position_size,
                tp=signal.take_profit_price,
                sl=signal.stop_loss_price,
            )

            # 5. Log decision
            context = DecisionContext(
                timestamp=signal.timestamp,
                symbol=self.config.symbol,
                direction=direction,
                entry_price=current_price,
                take_profit=signal.take_profit_price,
                stop_loss=signal.stop_loss_price,
                position_size=position_size,
                leverage=self.leverage_calculator.get_leverage(self.config.symbol),
                risk_amount=account_status.get("balance", self.config.initial_balance)
                * (self.config.risk_percent / 100),
                risk_percent=self.config.risk_percent,
                observation=signal.observation,
                indicators=signal.indicators,
                patterns=signal.patterns,
                model_outputs=signal.model_outputs,
                account_status=account_status,
            )
            self.decision_logger.log_decision(context)

            logger.info(f"Opened {direction} position: {position_id}")
            return position_id

        except Exception as e:
            logger.error(f"Error in check_and_execute: {e}")
            return None

    def record_trade_result(self, trade: TradeResult) -> None:
        """
        Record a completed trade and update metrics.

        Should be called when a position is closed to update
        performance tracking.

        Args:
            trade: TradeResult with trade outcome details
        """
        self.performance_tracker.record_trade(trade)

    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Get current performance metrics.

        Returns:
            PerformanceMetrics with all calculated metrics
        """
        return self.performance_tracker.get_metrics()
