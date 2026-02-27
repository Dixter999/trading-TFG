"""
Paper Trading Docker Service Entry Point (Issue #434).

This module provides the main entry point for running the paper trading
system as a Docker service. It handles:
- Environment variable configuration
- Signal handling (SIGTERM, SIGINT) for graceful shutdown
- Logging configuration for Docker stdout
- Main polling loop for trade evaluation

Usage:
    # As a module
    python -m src.paper_trading.main

    # Or directly
    python src/paper_trading/main.py

Environment Variables:
    PAPER_TRADING_SYMBOLS: Comma-separated symbols (default: EURUSD,GBPUSD,USDJPY,EURJPY)
    PAPER_TRADING_POLL_INTERVAL: Poll interval in seconds (default: 60)
    PAPER_TRADING_ENABLED: Enable/disable trading (default: true)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

# Runtime imports for trading components
from src.paper_trading.config import PaperTradingConfig
from src.paper_trading.db_decision_logger import TradeDecisionDBLogger
from src.paper_trading.position_manager import PositionManager
from src.paper_trading.ppo_entry_evaluator import INDICATOR_TABLES, PPOEntryEvaluator
from src.paper_trading.hybrid_entry_evaluator import HybridEntryEvaluator
from src.paper_trading.hybrid_exit_evaluator import (
    HybridExitEvaluator,
    DEFAULT_MODEL_DIR_V2,
    DEFAULT_MODEL_DIR_V4,
    MODEL_VERSION_V2,
    MODEL_VERSION_V4,
)
from src.paper_trading.risk_manager import RiskConfig, RiskManager
from src.paper_trading.signal_preview_evaluator import SignalPreviewEvaluator
from src.paper_trading.live_performance_tracker import LivePerformanceTracker
from src.database.connection_manager import DatabaseManager

# TFG: asyncpg for async DB operations
try:
    import asyncpg
except ImportError:
    asyncpg = None

# TFG: Infrastructure stubs (no MT5 gateway, no Telegram, no WebSocket)
async def send_telegram_alert(message: str) -> None:
    """Stub: Telegram alerts disabled in TFG local environment."""
    pass

def format_position_opened(**kwargs) -> str:
    """Stub: returns formatted string for logging only."""
    return f"[TFG] Position opened: {kwargs.get('symbol', '?')} {kwargs.get('direction', '?')}"

def format_position_closed(**kwargs) -> str:
    """Stub: returns formatted string for logging only."""
    return f"[TFG] Position closed: {kwargs.get('symbol', '?')} pnl={kwargs.get('pnl_pips', 0)} pips"

WebSocketPriceClient = None  # Not available in TFG
RealtimeSLChecker = None  # Not available in TFG

# TFG: Live trading imports removed (no MT5 gateway in local environment)

if TYPE_CHECKING:
    pass  # All imports moved to runtime

# Default configuration values
DEFAULT_SYMBOLS: list[str] = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY"]
DEFAULT_POLL_INTERVAL: int = 60
DEFAULT_ENABLED: bool = True

# Hot reload configuration (Issue #612)
CONFIG_PATH = "config/paper_trading.yaml"
HOT_RELOAD_INTERVAL_POLLS = 5  # Check every 5 polls (5 minutes at 60s interval)
HOT_RELOAD_ENABLED = True


def configure_logging() -> None:
    """Configure logging for Docker compatibility.

    Sets up logging to stdout with timestamps for proper Docker log collection.
    Uses ISO 8601 format for timestamps.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Set log level from environment
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))


class PaperTradingRunner:
    """Runs the paper trading engine as a continuous service.

    This class manages the lifecycle of the paper trading system:
    - Configuration from environment or dict
    - Main polling loop
    - Signal handling for graceful shutdown
    - Logging and monitoring

    Attributes:
        config: Configuration dictionary containing symbols, poll_interval, etc.
        logger: Logger instance for this runner.
        _running: Internal flag indicating if the runner is active.
        entry_evaluator: PPO entry evaluator for generating trade signals.
        position_manager: Manager for tracking open positions.
        risk_manager: Risk management validation.
        decision_logger: Logger for trade decisions to database.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        entry_evaluator: PPOEntryEvaluator | HybridEntryEvaluator | None = None,
        exit_evaluator: HybridExitEvaluator | None = None,
        position_manager: PositionManager | None = None,
        risk_manager: RiskManager | None = None,
        decision_logger: TradeDecisionDBLogger | None = None,
        db_manager: DatabaseManager | None = None,
        ws_client=None,
        sl_checker=None,
    ) -> None:
        """Initialize with configuration from environment or dict.

        Args:
            config: Optional configuration dictionary. If not provided,
                    uses default values.
            entry_evaluator: Optional entry evaluator (PPO or Hybrid).
            exit_evaluator: Optional hybrid exit evaluator for RL-managed exits.
            position_manager: Optional position manager instance.
            risk_manager: Optional risk manager instance.
            decision_logger: Optional decision logger instance.
            db_manager: Optional database manager for fetching real prices.
            ws_client: Not used in TFG (no WebSocket).
            sl_checker: Not used in TFG (no real-time SL).
        """
        if config is None:
            config = {
                "symbols": DEFAULT_SYMBOLS.copy(),
                "poll_interval": DEFAULT_POLL_INTERVAL,
                "enabled": DEFAULT_ENABLED,
            }

        self.config: dict[str, Any] = config
        self.logger = logging.getLogger(__name__)
        self._running: bool = False

        # Paper trading account balance (starting capital) - Issue #631
        # Read from config, with fallback to default $10,000 USD
        account_config = config.get("account", {})
        initial_balance = Decimal(str(account_config.get("initial_balance", "10000.00")))
        self.account_currency: str = account_config.get("currency", "USD")
        self.exchange_rate_to_usd: Decimal = Decimal(str(account_config.get("exchange_rate_to_usd", "1.0")))

        # Convert to USD for margin calculations if not already USD
        if self.account_currency != "USD":
            self.account_balance: Decimal = initial_balance * self.exchange_rate_to_usd
            self.logger.info(
                f"Account balance: {initial_balance} {self.account_currency} = "
                f"${self.account_balance:.2f} USD (rate: {self.exchange_rate_to_usd})"
            )
        else:
            self.account_balance: Decimal = initial_balance
            self.logger.info(f"Account balance: ${self.account_balance:.2f} USD")

        # Live balance update tracking (dynamic position sizing)
        self._last_balance_update: float = 0.0
        self._balance_update_interval: int = 300  # 5 minutes
        self._balance_fetch_failures: int = 0
        self._peak_balance: Decimal = self.account_balance

        # Trading components (injected or created later)
        self.entry_evaluator = entry_evaluator
        self.exit_evaluator = exit_evaluator  # Issue #495: RL-managed exits
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.decision_logger = decision_logger
        self.db_manager = db_manager  # For fetching real prices (Fix #4)

        # Real-time SL protection components
        self.ws_client = ws_client
        self.sl_checker = sl_checker
        self._sl_checker_task: asyncio.Task | None = None

        # Signal Preview Evaluator (Issue #629)
        self.signal_preview_evaluator: SignalPreviewEvaluator | None = None

        # Background task references to prevent GC (same pattern as LiveTradingAdapter)
        self._background_tasks: set = set()

        # Dedup tracking: prevent processing same candle twice (Fix #2)
        self.last_processed_candle_time: dict[str, datetime] = {}

        # Post-close cooldown: prevent immediate re-entry after SL/TP hit
        # Issue #631: Reduced from 120 to 30 minutes to allow M30 signals
        # to re-enter on the next candle if conditions are met
        self.last_close_time: dict[str, datetime] = {}
        # Track last RL evaluation M30 candle timestamp per position_id
        # Matches training cadence: model was trained on M30 bars (hybrid_env primary_timeframe="m30")
        self._last_rl_eval_m30_ts: dict[int, int] = {}
        self.post_close_cooldown_minutes: int = self.config.get(
            "post_close_cooldown_minutes", 30
        )

        # Minimum bars before RL exit model evaluates (avoid unreliable bars_held=0 predictions)
        # Issue #628: Reduced from 3 to 1 - with 3 bars, RL never gets to evaluate
        # before SL hits on higher timeframes (H4=12h, D1=3d before first eval!)
        self.min_bars_for_rl_exit: int = self.config.get("min_bars_for_rl_exit", 1)

        # Periodic summary tracking (Enhanced logging feature)
        self._poll_counter: int = 0
        self._rejected_signals: dict[str, dict] = {}  # Track rejected signals for summary

        # Risk violation dedup: only log same symbol+direction+reason once per 30 min
        self._last_risk_violation_log: dict[str, datetime] = {}
        self._risk_violation_dedup_minutes: int = 30
        self._risk_violation_suppressed: dict[str, int] = {}  # count suppressed per key

        # Hot reload tracking (Issue #612)
        self._config_mtime: float = 0.0  # Track config file modification time
        self._hot_reload_enabled: bool = HOT_RELOAD_ENABLED

        # Lifecycle client for hot reload filtering (Issue #618)
        self._lifecycle_client = None

        self.logger.info(
            f"PaperTradingRunner initialized: "
            f"symbols={self.config.get('symbols')}, "
            f"poll_interval={self.config.get('poll_interval')}s, "
            f"enabled={self.config.get('enabled')}, "
            f"realtime_sl={'enabled' if self.ws_client else 'disabled'}"
        )

    def _init_lifecycle_client(self) -> None:
        """Initialize lifecycle client for hot reload filtering (Issue #618).

        Creates a LifecycleClient instance for querying signal lifecycle states
        from PostgreSQL. This enables filtering signals by lifecycle state
        during hot reload.

        Gracefully handles initialization failures - hot reload will work
        without lifecycle filtering if client fails to initialize.
        """
        if not self._hot_reload_enabled:
            return

        try:
            from src.paper_trading.lifecycle_client import LifecycleClient

            self._lifecycle_client = LifecycleClient()

            # Validate connection
            if self._lifecycle_client.validate_connection():
                self.logger.info("LifecycleClient initialized for hot reload")
            else:
                self.logger.warning(
                    "LifecycleClient connection failed - hot reload will work "
                    "without lifecycle filtering"
                )
                self._lifecycle_client = None

        except Exception as e:
            self.logger.warning(
                f"Could not initialize LifecycleClient: {e}. "
                f"Hot reload will work without lifecycle filtering."
            )
            self._lifecycle_client = None

    async def _validate_signal_count(self) -> None:
        """Validate configured vs tradeable signal counts, alert on mismatch.

        Compares total configured signals against lifecycle-tradeable signals.
        Sends Telegram CRITICAL alert if any configured signal is blocked by
        lifecycle state (quarantined/retired).

        Called at startup and periodically (~60 min) to detect drift.
        """
        if self._lifecycle_client is None:
            return

        # Build full list of configured signal IDs from _signal_config
        # _signal_config: Dict[str, List[Dict[str, str]]] = {symbol: [{signal, direction, timeframe}]}
        signal_config = getattr(self, "_signal_config", None)
        if not signal_config:
            return

        configured_ids = set()
        for symbol, sig_list in signal_config.items():
            for sig in sig_list:
                sig_name = sig.get("signal", "")
                direction = sig.get("direction", "")
                timeframe = sig.get("timeframe", "")
                if sig_name and direction and timeframe:
                    signal_id = f"{symbol}_{direction}_{sig_name}_{timeframe}"
                    configured_ids.add(signal_id)

        if not configured_ids:
            self.logger.warning("Signal count validation: 0 configured signals")
            return

        # Check each configured signal against lifecycle
        blocked_signals = []
        for signal_id in sorted(configured_ids):
            if not self._lifecycle_client.is_signal_tradeable(signal_id):
                state = self._lifecycle_client.get_signal_state(signal_id)
                blocked_signals.append((signal_id, state or "unknown"))

        tradeable_count = len(configured_ids) - len(blocked_signals)

        if blocked_signals:
            blocked_list = "\n".join(
                f"  - {sid} ({state})" for sid, state in blocked_signals
            )
            msg = (
                f"<b>SIGNAL COUNT MISMATCH</b>\n\n"
                f"Configured: {len(configured_ids)}\n"
                f"Tradeable: {tradeable_count}\n"
                f"Blocked: {len(blocked_signals)}\n\n"
                f"<b>Blocked signals:</b>\n{blocked_list}\n\n"
                f"These signals passed Phase 5 but are blocked by lifecycle state. "
                f"Check signal_lifecycle table."
            )
            self.logger.error(
                f"Signal count mismatch: {len(configured_ids)} configured, "
                f"{tradeable_count} tradeable, {len(blocked_signals)} blocked"
            )
            asyncio.create_task(send_telegram_alert(msg))
        else:
            self.logger.info(
                f"Signal count validation OK: {len(configured_ids)} configured, "
                f"all tradeable"
            )

    @classmethod
    def from_env(cls) -> PaperTradingRunner:
        """Create runner from environment variables.

        Environment Variables:
            PAPER_TRADING_SYMBOLS: Comma-separated list of symbols
            PAPER_TRADING_POLL_INTERVAL: Poll interval in seconds
            PAPER_TRADING_ENABLED: Enable flag (true/false)

        Returns:
            PaperTradingRunner instance configured from environment.
        """
        # Parse symbols
        symbols_str = os.getenv("PAPER_TRADING_SYMBOLS", "").strip()
        if symbols_str:
            symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        else:
            symbols = DEFAULT_SYMBOLS.copy()

        # Use defaults if empty list
        if not symbols:
            symbols = DEFAULT_SYMBOLS.copy()

        # Parse poll interval
        poll_interval_str = os.getenv("PAPER_TRADING_POLL_INTERVAL", "").strip()
        try:
            poll_interval = (
                int(poll_interval_str) if poll_interval_str else DEFAULT_POLL_INTERVAL
            )
        except ValueError:
            poll_interval = DEFAULT_POLL_INTERVAL

        # Parse enabled flag
        enabled_str = os.getenv("PAPER_TRADING_ENABLED", "true").strip().lower()
        enabled = enabled_str in ("true", "1", "yes", "on")

        config = {
            "symbols": symbols,
            "poll_interval": poll_interval,
            "enabled": enabled,
        }

        return cls(config=config)

    def setup_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown.

        Registers handlers for:
        - SIGTERM: Graceful shutdown (Docker stop)
        - SIGINT: Keyboard interrupt (Ctrl+C)
        """

        def signal_handler(signum: int, frame: Any) -> None:
            sig_name = signal.Signals(signum).name
            self.logger.info(f"Received {sig_name}, initiating graceful shutdown...")
            self.stop()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        self.logger.debug("Signal handlers registered (SIGTERM, SIGINT)")

    def stop(self) -> None:
        """Signal the runner to stop gracefully.

        Sets the running flag to False, which causes the main loop
        to exit after completing the current iteration.
        """
        self.logger.info("Stop requested, will exit after current iteration...")
        self._running = False

    async def run(self) -> None:
        """Run the main trading loop.

        This is the main entry point for the service. It:
        1. Checks if trading is enabled
        2. Sets up signal handlers
        3. Starts real-time SL protection (WebSocket + SL checker)
        4. Runs the polling loop until stopped
        5. Cleans up WebSocket connections on exit

        The loop polls all configured symbols at the configured interval,
        checking for entry/exit signals.
        """
        if not self.config.get("enabled", True):
            self.logger.info("Paper trading is disabled, exiting...")
            return

        self._running = True
        self.setup_signal_handlers()

        poll_interval = self.config.get("poll_interval", DEFAULT_POLL_INTERVAL)
        symbols = self.config.get("symbols", DEFAULT_SYMBOLS)

        # Start real-time SL protection as background tasks
        await self._start_realtime_sl_protection()

        # Validate signal count at startup
        await self._validate_signal_count()

        self.logger.info(
            f"Starting paper trading service: "
            f"{len(symbols)} symbols, {poll_interval}s interval"
        )

        try:
            while self._running:
                try:
                    await self._poll_once()
                except Exception as e:
                    self.logger.error(f"Error during poll: {e}", exc_info=True)

                # Increment poll counter and log periodic summary
                self._poll_counter += 1

                # Log summary every 15 polls (15 minutes if poll_interval=60s)
                if self._poll_counter % 15 == 0 and self._rejected_signals:
                    self._log_active_signals_summary()

                # Hot reload check every N polls (Issue #612)
                if (
                    self._hot_reload_enabled
                    and self._poll_counter % HOT_RELOAD_INTERVAL_POLLS == 0
                ):
                    if self._check_config_changed():
                        await self._perform_hot_reload()

                # Periodic signal count validation (every 60 polls ~60 min)
                if self._poll_counter % 60 == 0 and self._poll_counter > 0:
                    await self._validate_signal_count()

                # Wait for next poll interval (or exit if stopped)
                if self._running:
                    await asyncio.sleep(poll_interval)
        finally:
            # Clean up WebSocket connections
            await self._stop_realtime_sl_protection()

        self.logger.info("Paper trading service stopped")

    async def _start_realtime_sl_protection(self) -> None:
        """Start WebSocket client and SL checker as background tasks."""
        if self.ws_client is None or self.sl_checker is None:
            self.logger.info("Real-time SL protection not configured, skipping")
            return

        try:
            # Connect WebSocket client to all symbols
            await self.ws_client.connect_all()
            self.logger.info(
                f"WebSocket connected for real-time SL protection: "
                f"{self.ws_client.connection_status}"
            )

            # Start SL checker as background task
            self._sl_checker_task = asyncio.create_task(self.sl_checker.run())
            self.logger.info("Real-time SL checker started as background task")

        except Exception as e:
            self.logger.error(f"Failed to start real-time SL protection: {e}")
            # Continue without real-time SL - fallback to 60s polling

    async def _stop_realtime_sl_protection(self) -> None:
        """Stop WebSocket client and SL checker gracefully."""
        if self.sl_checker is not None:
            self.sl_checker.stop()

        if self._sl_checker_task is not None:
            self._sl_checker_task.cancel()
            try:
                await self._sl_checker_task
            except asyncio.CancelledError:
                pass
            self._sl_checker_task = None

        if self.ws_client is not None:
            await self.ws_client.disconnect_all()

        self.logger.info("Real-time SL protection stopped")

    async def _update_account_balance(self) -> None:
        """Fetch live account balance from MT5 and update self.account_balance.

        Uses equity (balance + floating P&L) for more accurate risk sizing.
        Only runs when position_manager is a LiveTradingAdapter with get_account_info.
        Fetches at most once every _balance_update_interval seconds.
        On failure, keeps last known balance (never zeroes out).
        """
        import time as _time

        # Only update if live trading adapter is available
        if not hasattr(self.position_manager, "get_account_info"):
            return

        now = _time.time()
        if now - self._last_balance_update < self._balance_update_interval:
            return

        self._last_balance_update = now

        try:
            account_info = await self.position_manager.get_account_info()
            if account_info is None:
                self._balance_fetch_failures += 1
                self.logger.warning(
                    f"Account balance fetch failed ({self._balance_fetch_failures} consecutive). "
                    f"Keeping last known balance: ${self.account_balance:.2f} USD"
                )
                return

            # Use equity (includes floating P&L) for risk sizing
            equity = Decimal(str(account_info.get("equity", 0)))
            currency = account_info.get("currency", "USD")

            # Convert to USD if needed
            if currency != "USD" and self.exchange_rate_to_usd != Decimal("1.0"):
                new_balance = equity * self.exchange_rate_to_usd
            else:
                new_balance = equity

            # Log significant changes (>1%)
            old_balance = self.account_balance
            if old_balance > 0:
                change_pct = abs(float(new_balance - old_balance) / float(old_balance)) * 100
                if change_pct > 1.0:
                    self.logger.info(
                        f"Account balance updated: ${old_balance:.2f} -> ${new_balance:.2f} USD "
                        f"({'+' if new_balance > old_balance else ''}{float(new_balance - old_balance):.2f}, "
                        f"{change_pct:.1f}%)"
                    )

            self.account_balance = new_balance
            if new_balance > self._peak_balance:
                self._peak_balance = new_balance
                self.logger.info(f"New peak balance: ${self._peak_balance:.2f} USD")
            self._balance_fetch_failures = 0

            self.logger.debug(
                f"Account balance: ${self.account_balance:.2f} USD "
                f"(equity={equity} {currency})"
            )

        except Exception as e:
            self._balance_fetch_failures += 1
            self.logger.warning(
                f"Account balance update error ({self._balance_fetch_failures} consecutive): {e}. "
                f"Keeping ${self.account_balance:.2f} USD"
            )

    async def _poll_once(self) -> None:
        """Execute a single poll iteration.

        Checks all configured symbols for trading signals.
        Override this method to implement actual trading logic.

        Issue #569 Stream E: Pre-fetches ALL timeframes for ALL symbols proactively.
        """
        # Update account balance from live MT5 (every 5 min)
        await self._update_account_balance()

        symbols = self.config.get("symbols", DEFAULT_SYMBOLS)

        # Skip entry evaluation when market is closed (reduces DB noise)
        if self.risk_manager is not None and not self.risk_manager.is_market_open():
            if self._poll_counter % 60 == 0:  # Log once per hour
                self.logger.info("Market is closed (weekend), skipping signal evaluation")
            # Still check exits (sync positions, update unrealized P&L)
            if self.position_manager is not None:
                if hasattr(self.position_manager, "sync_positions"):
                    try:
                        sync_result = await self.position_manager.sync_positions()
                        for trade in sync_result.get("closed_trades", []):
                            self.logger.info(f"Position closed (broker sync): {trade.symbol} - {trade}")
                            self._send_telegram(format_position_closed(
                                symbol=trade.symbol,
                                direction=trade.direction.value,
                                pnl_pips=float(trade.pnl_pips),
                                exit_reason=trade.exit_reason.value,
                                entry_price=float(trade.entry_price),
                                exit_price=float(trade.exit_price),
                            ))
                    except Exception as e:
                        self.logger.error(f"Failed to sync positions: {e}")
                current_prices = self._get_current_prices()
                self.position_manager.update_unrealized_pnl(current_prices)
            return

        self.logger.debug(f"Polling {len(symbols)} symbols...")

        # Issue #569 Stream E: Pre-fetch ALL timeframes for ALL symbols
        # This ensures data is cached and ready for signal evaluation
        if hasattr(self.entry_evaluator, "_prefetch_all_timeframes"):
            self.entry_evaluator._prefetch_all_timeframes()

        # Phase 1: Check exits for all symbols (SL/TP + RL exits)
        for symbol in symbols:
            try:
                await self._check_exits_for_symbol(symbol)
            except Exception as e:
                self.logger.error(f"Error checking exits for {symbol}: {e}", exc_info=True)

        # Phase 2: Collect all valid entry signals across all symbols
        all_ready: list[dict] = []
        for symbol in symbols:
            try:
                entries = self._collect_valid_entries(symbol)
                all_ready.extend(entries)
            except Exception as e:
                self.logger.error(f"Error collecting entries for {symbol}: {e}", exc_info=True)

        # Phase 3: Budget and open positions for all ready signals
        if all_ready:
            try:
                await self._budget_and_open(all_ready)
            except Exception as e:
                self.logger.error(f"Error in budget_and_open: {e}", exc_info=True)

        # Issue #629: Evaluate signal preview for confidence tracking
        await self._evaluate_signal_preview()

    async def _evaluate_signal_preview(self) -> None:
        """Evaluate signal preview confidence for all signals (Issue #629).

        This runs every poll cycle and stores preview snapshots in the database.
        """
        if not self.signal_preview_evaluator:
            return

        try:
            # Evaluate all signals
            previews = await self.signal_preview_evaluator.evaluate_all_signals()

            if previews:
                # Group by candle close time
                groups = self.signal_preview_evaluator.group_by_candle_close(previews)

                # Log summary
                if groups:
                    next_group = groups[0]
                    high_conf = sum(1 for s in previews if s.confidence >= 80)
                    self.logger.debug(
                        f"Signal Preview: {len(previews)} signals evaluated, "
                        f"{high_conf} high confidence, "
                        f"next close at {next_group.close_time.strftime('%H:%M UTC')}"
                    )

                # Save snapshot to database (for accuracy analysis later)
                await self.signal_preview_evaluator.save_snapshot(previews)

        except Exception as e:
            self.logger.error(f"Signal preview evaluation failed: {e}")

    def _log_active_signals_summary(self) -> None:
        """Log a periodic summary of active signals waiting for fresh candles.

        This provides visibility into what signals are pending and when they
        will be re-evaluated. Called every 15 poll cycles (typically 15 minutes).
        """
        from datetime import datetime, timezone

        if not self._rejected_signals:
            return

        current_time = datetime.now(timezone.utc)
        self.logger.info("=" * 60)
        self.logger.info("📊 ACTIVE SIGNALS SUMMARY (Waiting for Fresh Candles)")
        self.logger.info("=" * 60)

        # Sort by next candle close time
        sorted_signals = sorted(
            self._rejected_signals.items(),
            key=lambda x: x[1]["next_candle_close"]
        )

        for signal_key, info in sorted_signals:
            time_until = (info["next_candle_close"] - current_time).total_seconds()
            minutes_left = int(time_until // 60)
            hours_left = minutes_left // 60
            mins_remaining = minutes_left % 60

            if hours_left > 0:
                time_str = f"{hours_left}h {mins_remaining}m"
            else:
                time_str = f"{minutes_left}m"

            self.logger.info(
                f"  • {info['symbol']:8} {info['direction']:5} [{info['timeframe']:3}] "
                f"(conf={info['confidence']:.2f}) → "
                f"Next candle: {info['next_candle_close'].strftime('%H:%M UTC')} "
                f"(in {time_str})"
            )

        self.logger.info("=" * 60)
        self.logger.info(f"Total signals pending: {len(self._rejected_signals)}")
        self.logger.info("=" * 60)

    def _check_config_changed(self) -> bool:
        """Check if config file has been modified (Issue #612).

        Compares file modification time to detect ConfigMap updates.

        Returns:
            True if config has changed since last check
        """
        import os
        try:
            if not os.path.exists(CONFIG_PATH):
                return False

            current_mtime = os.path.getmtime(CONFIG_PATH)

            if self._config_mtime == 0.0:
                # First check - store current mtime
                self._config_mtime = current_mtime
                return False

            if current_mtime > self._config_mtime:
                self.logger.info(
                    f"Config change detected: mtime {self._config_mtime} -> {current_mtime}"
                )
                self._config_mtime = current_mtime
                return True

            return False

        except Exception as e:
            self.logger.warning(f"Failed to check config mtime: {e}")
            return False

    async def _perform_hot_reload(self) -> None:
        """Hot reload models from updated config (Issue #612, #618).

        Reads updated paper_trading.yaml and incrementally syncs models:
        - Loads NEW models added to config
        - Unloads REMOVED models (quarantined/retired)
        - Keeps unchanged models in memory

        Issue #618: Enhanced to filter by lifecycle state from PostgreSQL.
        Only ACTIVE and DEGRADED signals are loaded.
        QUARANTINED and RETIRED signals are filtered out.

        This avoids the 4-hour full reload on daily reconcile.
        """
        import yaml

        if self.exit_evaluator is None:
            self.logger.debug("No exit evaluator - skipping hot reload")
            return

        try:
            self.logger.info("=" * 60)
            self.logger.info("HOT RELOAD - Starting incremental model sync")
            self.logger.info("=" * 60)

            # Read updated config
            with open(CONFIG_PATH, "r") as f:
                config = yaml.safe_load(f)

            # Extract approved models from config
            approved_models = []
            symbols_config = config.get("symbols", {})

            for symbol, symbol_config in symbols_config.items():
                if not symbol_config.get("enabled", False):
                    continue

                for sig in symbol_config.get("signals", []):
                    if not sig.get("enabled", False):
                        continue

                    # Construct model_dir (model key)
                    signal_name = sig.get("signal", "")
                    direction = sig.get("direction", "")
                    timeframe = sig.get("timeframe", "")

                    model_dir = f"{symbol.lower()}_{direction}_{signal_name}_{timeframe}"

                    approved_models.append({
                        "model_dir": model_dir,
                        "symbol": symbol.lower(),
                        "direction": direction,
                        "signal_name": signal_name,
                        "timeframe": timeframe,
                    })

            self.logger.info(f"Config has {len(approved_models)} approved models")

            # Issue #618: Filter by lifecycle state from PostgreSQL
            if self._lifecycle_client is not None:
                try:
                    tradeable_signals = self._lifecycle_client.get_tradeable_signal_ids()

                    if tradeable_signals:
                        # Filter approved_models to only include tradeable signals
                        original_count = len(approved_models)
                        approved_models = [
                            m for m in approved_models
                            if m.get("model_dir") in tradeable_signals
                        ]
                        filtered_count = original_count - len(approved_models)

                        self.logger.info(
                            f"Lifecycle filter: {len(tradeable_signals)} tradeable signals, "
                            f"filtered out {filtered_count} non-tradeable from config"
                        )
                    else:
                        self.logger.info(
                            "No tradeable signals in lifecycle DB - loading all config models"
                        )

                except Exception as e:
                    self.logger.warning(
                        f"Lifecycle filter failed: {e}. Loading all config models."
                    )

            self.logger.info(f"Syncing {len(approved_models)} filtered models")

            # Perform incremental sync
            result = self.exit_evaluator.sync_models_incremental(approved_models)

            self.logger.info(
                f"Hot reload complete: "
                f"loaded={len(result.get('loaded', []))}, "
                f"unloaded={len(result.get('unloaded', []))}, "
                f"unchanged={len(result.get('unchanged', []))}"
            )
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"Hot reload failed: {e}", exc_info=True)

    async def _check_exits_for_symbol(self, symbol: str) -> None:
        """Check exits (SL/TP + RL) for a single symbol.

        Handles position sync, SL/TP exits, RL-managed exits, and P&L updates.
        Entry evaluation is handled separately by _collect_valid_entries.

        Args:
            symbol: Trading symbol to check (e.g., "EURUSD")
        """
        self.logger.debug(f"Checking exits for {symbol}...")

        # Step 1: Check SL/TP exits first (always)
        if self.position_manager is not None:
            # CRITICAL: Sync live positions before RL evaluation
            if hasattr(self.position_manager, "sync_positions"):
                try:
                    sync_result = await self.position_manager.sync_positions()
                    # Notify for positions closed by broker (native SL/TP hit)
                    # NOTE: log_position_close is NOT called here — _on_broker_close callback
                    # already logs the decision inside sync_positions() to avoid duplicates.
                    for trade in sync_result.get("closed_trades", []):
                        self._send_telegram(format_position_closed(
                            symbol=trade.symbol,
                            direction=trade.direction.value,
                            pnl_pips=float(trade.pnl_pips),
                            exit_reason=trade.exit_reason.value,
                            entry_price=float(trade.entry_price),
                            exit_price=float(trade.exit_price),
                        ))
                        if self.exit_evaluator is not None:
                            self.exit_evaluator.record_trade(trade)
                        trade_signal, trade_tf = self._extract_signal_info_from_entry_model(
                            trade.entry_model, trade.symbol
                        )
                        cooldown_key = f"{trade.symbol}:{trade_signal}:{trade_tf}"
                        self.last_close_time[cooldown_key] = trade.exit_time
                        self.logger.info(
                            f"Post-close cooldown started for {cooldown_key}: "
                            f"{self.post_close_cooldown_minutes} minutes"
                        )
                except Exception as e:
                    self.logger.error(f"Failed to sync positions: {e}")

            # Fetch current prices for all symbols
            current_prices = self._get_current_prices()

            # Use async check_exits to route SL/TP closes through MT5 (Issue #637)
            if hasattr(self.position_manager, 'check_exits_async'):
                closed_trades = await self.position_manager.check_exits_async(current_prices)
            else:
                closed_trades = self.position_manager.check_exits(current_prices)

            # Log any closed trades and record cooldown
            for trade in closed_trades:
                self.logger.info(f"Position closed (SL/TP): {trade.symbol} - {trade}")
                if self.decision_logger is not None:
                    self.decision_logger.log_position_close(trade)
                self._send_telegram(format_position_closed(
                    symbol=trade.symbol,
                    direction=trade.direction.value,
                    pnl_pips=float(trade.pnl_pips),
                    exit_reason=trade.exit_reason.value,
                    entry_price=float(trade.entry_price),
                    exit_price=float(trade.exit_price),
                ))
                # Issue #588: Record trade for exit model self-awareness
                if self.exit_evaluator is not None:
                    self.exit_evaluator.record_trade(trade)
                # Record close time for post-close cooldown (signal-specific key)
                trade_signal, trade_tf = self._extract_signal_info_from_entry_model(
                    trade.entry_model, trade.symbol
                )
                cooldown_key = f"{trade.symbol}:{trade_signal}:{trade_tf}"
                self.last_close_time[cooldown_key] = trade.exit_time
                self.logger.info(
                    f"Post-close cooldown started for {cooldown_key}: "
                    f"{self.post_close_cooldown_minutes} minutes"
                )

            # Clean up M30 eval tracking for SL/TP-closed positions
            if closed_trades and self._last_rl_eval_m30_ts:
                open_ids = {p.id for p in self.position_manager.get_all_positions().values()}
                stale_ids = [pid for pid in self._last_rl_eval_m30_ts if pid not in open_ids]
                for pid in stale_ids:
                    self._last_rl_eval_m30_ts.pop(pid, None)

            # Update unrealized P&L in database for frontend display
            self.position_manager.update_unrealized_pnl(current_prices)

        # Step 2: Check RL-managed exit for ALL open positions (multi-position support)
        if self.position_manager is not None:
            open_positions = self.position_manager.get_positions(symbol)
            if open_positions:
                # Check if RL exit is enabled for this symbol
                symbol_config = self._get_symbol_config(symbol)
                use_rl_exit = symbol_config.get("use_rl_exit")
                if use_rl_exit is None:
                    use_rl_exit = True  # Default: RL exit enabled
                exit_strategy = symbol_config.get("exit_strategy", "hybrid_v4")

                if use_rl_exit and self.exit_evaluator is not None:
                    current_price = self._get_current_price(symbol)

                    from datetime import timezone

                    now = datetime.now(timezone.utc)

                    for position in list(open_positions):  # Copy list, may modify during iteration
                        entry_time = position.entry_time
                        if entry_time.tzinfo is None:
                            entry_time = entry_time.replace(tzinfo=timezone.utc)

                        # Issue #560: Extract signal_name and timeframe for signal-specific model
                        signal_name, timeframe = self._extract_signal_info(position, symbol)

                        # Issue #608 v4: Evaluate RL exit at M30 candle close only
                        # Training uses M30 as primary_timeframe (hybrid_env.py:1043).
                        # Model makes HOLD/CLOSE decisions per M30 bar during training.
                        M30_SECONDS = 30 * 60  # 1800 seconds
                        INDICATOR_ARRIVAL_BUFFER = 300  # 5 min for indicator-calculator delay

                        current_ts = int(now.timestamp())
                        latest_m30_close_ts = (current_ts // M30_SECONDS) * M30_SECONDS
                        last_eval_ts = self._last_rl_eval_m30_ts.get(position.id, 0)

                        if latest_m30_close_ts <= last_eval_ts:
                            self.logger.debug(
                                f"RL exit skipped for {symbol} pos={position.id}: "
                                f"already evaluated M30 candle {latest_m30_close_ts}"
                            )
                            continue

                        time_since_m30_close = current_ts - latest_m30_close_ts
                        if time_since_m30_close < INDICATOR_ARRIVAL_BUFFER:
                            self.logger.debug(
                                f"RL exit deferred for {symbol} pos={position.id}: "
                                f"waiting for M30 indicators ({time_since_m30_close}s < {INDICATOR_ARRIVAL_BUFFER}s)"
                            )
                            continue

                        # Mark as evaluated and proceed
                        self._last_rl_eval_m30_ts[position.id] = latest_m30_close_ts

                        # Calculate bars_held using signal timeframe (for min_bars check and model input)
                        TIMEFRAME_MINUTES = {
                            "M30": 30, "H1": 60, "H2": 120, "H3": 180,
                            "H4": 240, "H6": 360, "H8": 480, "H12": 720, "D1": 1440
                        }
                        tf_minutes = TIMEFRAME_MINUTES.get(timeframe, 30)
                        hours_held = (now - entry_time).total_seconds() / 3600
                        bars_held = int(hours_held / (tf_minutes / 60))

                        self.logger.info(
                            f"RL exit evaluation for {symbol} pos={position.id} [{timeframe}]: "
                            f"M30 candle {latest_m30_close_ts}, bars_held={bars_held}"
                        )

                        if bars_held < self.min_bars_for_rl_exit:
                            self.logger.debug(
                                f"RL exit skipped for {symbol} pos={position.id}: bars_held={bars_held} "
                                f"< min={self.min_bars_for_rl_exit}"
                            )
                            continue  # Skip this position, check next

                        exit_signal = self.exit_evaluator.check_position(
                            symbol=symbol,
                            direction=1 if position.direction.value == "long" else -1,
                            entry_price=float(position.entry_price),
                            current_price=float(current_price),
                            sl_price=float(position.sl_price),
                            tp_price=float(position.tp_price),
                            bars_held=bars_held,
                            signal_name=signal_name,
                            timeframe=timeframe,
                        )

                        # Log every RL exit decision (HOLD and CLOSE) for analysis
                        if self.decision_logger is not None and hasattr(exit_signal, 'ensemble_meta'):
                            try:
                                self.decision_logger.log_exit_decision(exit_signal, position)
                            except Exception as log_err:
                                self.logger.warning(f"Failed to log exit decision: {log_err}")

                        if exit_signal.should_close:
                            self.logger.info(
                                f"RL exit signal: {symbol} pos={position.id} CLOSE "
                                f"(conf={exit_signal.confidence:.2f}, "
                                f"pnl={exit_signal.unrealized_pnl_pips:.1f} pips, reason={exit_signal.reason})"
                            )
                            from src.paper_trading.models import ExitReason

                            try:
                                # Use awaited close for RL exits (Issue #637)
                                if hasattr(self.position_manager, 'close_position_async'):
                                    closed_trade = await self.position_manager.close_position_async(
                                        symbol=symbol,
                                        exit_price=current_price,
                                        exit_time=now,
                                        exit_reason=ExitReason.MODEL_EXIT,
                                        model_version=exit_signal.model_version,
                                        position_id=position.id,
                                    )
                                else:
                                    closed_trade = self.position_manager.close_position(
                                        symbol=symbol,
                                        exit_price=current_price,
                                        exit_time=now,
                                        exit_reason=ExitReason.MODEL_EXIT,
                                        model_version=exit_signal.model_version,
                                        position_id=position.id,
                                    )
                                if closed_trade and self.decision_logger is not None:
                                    self.decision_logger.log_position_close(closed_trade)
                                if closed_trade:
                                    self._send_telegram(format_position_closed(
                                        symbol=closed_trade.symbol,
                                        direction=closed_trade.direction.value,
                                        pnl_pips=float(closed_trade.pnl_pips),
                                        exit_reason=closed_trade.exit_reason.value,
                                        entry_price=float(closed_trade.entry_price),
                                        exit_price=float(closed_trade.exit_price),
                                    ))
                                    # Signal-specific cooldown key
                                    cooldown_key = f"{symbol}:{signal_name}:{timeframe}"
                                    self.last_close_time[cooldown_key] = now
                                    self.logger.info(
                                        f"Post-close cooldown started for {cooldown_key}: "
                                        f"{self.post_close_cooldown_minutes} minutes (RL exit)"
                                    )
                                    # Clean up M30 eval tracking for closed position
                                    self._last_rl_eval_m30_ts.pop(position.id, None)
                            except (ValueError, KeyError) as close_err:
                                self.logger.warning(
                                    f"Ghost position detected for {symbol} pos={position.id}: "
                                    f"{close_err}. Force-removing from memory."
                                )
                                self.position_manager.force_remove_positions(symbol)
                                # Clean up M30 eval tracking for force-removed position
                                self._last_rl_eval_m30_ts.pop(position.id, None)
                else:
                    self.logger.debug(
                        f"RL exit disabled for {symbol} (exit_strategy={exit_strategy}), "
                        f"using SL/TP only"
                    )

    def _collect_valid_entries(self, symbol: str) -> list[dict]:
        """Collect all valid entry signals for a symbol.

        Evaluates entry signals and runs them through all validation gates.
        Returns signal dicts ready for batch budget allocation.

        Args:
            symbol: Trading symbol to evaluate

        Returns:
            List of dicts with keys: key, symbol, direction, timeframe,
            profit_factor, signal_obj
        """
        if self.entry_evaluator is None:
            return []

        # Use batch evaluate_entries if available, else fall back to single evaluate_entry
        if hasattr(self.entry_evaluator, 'evaluate_entries'):
            all_signals = self.entry_evaluator.evaluate_entries(symbol)
        else:
            single = self.entry_evaluator.evaluate_entry(symbol)
            all_signals = [single] if single is not None else []

        if not all_signals:
            return []

        # Filter signals through validation gates
        valid_entries = []
        for signal in all_signals:
            if not self._validate_signal_for_entry(symbol, signal):
                continue

            signal_key = f"{symbol}:{signal.direction}:{getattr(signal, 'signal_source', 'UNKNOWN')}:{signal.timeframe}"

            valid_entries.append({
                "key": signal_key,
                "symbol": symbol,
                "direction": signal.direction,
                "timeframe": signal.timeframe,
                "profit_factor": 1.0,  # PF no longer drives sizing; LivePerformanceTracker does
                "signal_obj": signal,
            })

        if valid_entries:
            self.logger.info(
                f"Collected {len(valid_entries)} valid entries for {symbol}"
            )

        return valid_entries

    def _get_signal_pf(self, symbol: str, signal) -> float:
        """Look up Phase 5 Profit Factor for a signal from config.

        Args:
            symbol: Trading symbol
            signal: Entry signal with signal_source and timeframe attributes

        Returns:
            Profit factor (float), defaults to 1.0 if not found
        """
        symbol_configs = self.config.get("symbol_configs", {})
        symbol_cfg = symbol_configs.get(symbol, {}) if isinstance(symbol_configs, dict) else {}
        signals_list = symbol_cfg.get("signals", [])

        signal_source = getattr(signal, "signal_source", None)
        timeframe = getattr(signal, "timeframe", None)

        for sig_cfg in signals_list:
            if (sig_cfg.get("signal") == signal_source and
                    sig_cfg.get("timeframe") == timeframe):
                pf = sig_cfg.get("profit_factor")
                if pf is not None:
                    return float(pf)

        # Fallback: try risk manager's approved models
        if self.risk_manager is not None and signal_source and timeframe:
            return self.risk_manager._get_phase5_pf(symbol, signal_source, timeframe)

        return 1.0

    async def _budget_and_open(self, all_ready: list[dict]) -> None:
        """Budget and open positions for all ready signals using SignalBudgetAllocator.

        Calculates PF-weighted lot sizes across all symbols, then opens each position.

        Args:
            all_ready: List of signal dicts from _collect_valid_entries
        """
        import inspect
        from src.paper_trading.models import PositionDirection

        if self.risk_manager is None or self.position_manager is None:
            self.logger.warning("No risk/position manager, cannot open positions")
            return

        self.logger.info(
            f"Budget allocation: {len(all_ready)} signals across "
            f"{len(set(s['symbol'] for s in all_ready))} symbols"
        )

        # Build signals_by_timeframe from config (all configured signals)
        signals_by_tf: dict[str, int] = {}
        symbol_configs = self.config.get("symbol_configs", {})
        if isinstance(symbol_configs, dict):
            for sym_cfg in symbol_configs.values():
                for sig in sym_cfg.get("signals", []):
                    tf = sig.get("timeframe", "H4")
                    signals_by_tf[tf] = signals_by_tf.get(tf, 0) + 1

        self.logger.info(f"signals_by_tf={dict(signals_by_tf)}, total_configured={sum(signals_by_tf.values())}")

        # Calculate batch sizes using SignalBudgetAllocator
        batch_sizes = self.risk_manager.calculate_batch_position_sizes(
            ready_signals=all_ready,
            account_balance=self.account_balance,
            peak_balance=self._peak_balance,
            signals_by_tf=signals_by_tf,
        )

        if not batch_sizes:
            self.logger.info("No budget allocated (all slots used or insufficient balance)")
            return

        self.logger.info(f"Batch allocation: {dict(batch_sizes)}")

        # Open each position with allocated lot size
        for entry in all_ready:
            signal = entry["signal_obj"]
            symbol = entry["symbol"]
            signal_key = entry["key"]

            allocated_size = batch_sizes.get(signal_key, Decimal("0"))
            if allocated_size <= 0:
                self.logger.info(f"Skipping {signal_key}: allocated 0 lots")
                continue

            direction = (
                PositionDirection.LONG
                if signal.direction.upper() == "LONG"
                else PositionDirection.SHORT
            )

            # Re-check risk limits before each open
            can_open, reason = self.risk_manager.can_open_position(symbol, direction)
            if not can_open:
                self.logger.warning(
                    f"Risk manager rejected: {symbol} {signal.direction} - {reason}"
                )
                if self.decision_logger is not None and "Market is closed" not in reason:
                    dedup_key = f"{symbol}:{signal.direction}:{reason}"
                    now = datetime.now(timezone.utc)
                    last_logged = self._last_risk_violation_log.get(dedup_key)
                    if last_logged is None or (now - last_logged).total_seconds() > self._risk_violation_dedup_minutes * 60:
                        suppressed = self._risk_violation_suppressed.pop(dedup_key, 0)
                        log_reason = f"{reason} (suppressed {suppressed}x)" if suppressed else reason
                        self.decision_logger.log_risk_violation(signal, log_reason)
                        self._last_risk_violation_log[dedup_key] = now
                    else:
                        self._risk_violation_suppressed[dedup_key] = self._risk_violation_suppressed.get(dedup_key, 0) + 1
                continue

            entry_price = self._get_current_price(symbol)
            sl_price, tp_price = self.position_manager.calculate_exit_prices(
                entry_price, direction, symbol
            )

            entry_model = f"Hybrid_V4 + {getattr(signal, 'signal_source', 'UNKNOWN')}"

            result = self.position_manager.open_position(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                entry_time=signal.timestamp,
                size=allocated_size,
                tp_price=tp_price,
                sl_price=sl_price,
                entry_model=entry_model,
                signal_timeframe=getattr(signal, 'timeframe', None),
            )
            if inspect.iscoroutine(result):
                position = await result
            else:
                position = result

            if position is None:
                self.logger.error(
                    f"Failed to open position: {symbol} {direction.value} "
                    f"entry={entry_price} - MT5 Gateway error or circuit breaker blocked"
                )
                continue

            self.logger.info(
                f"Position opened: {symbol} {direction.value} "
                f"entry={entry_price} size={allocated_size} SL={sl_price} TP={tp_price} "
                f"model={entry_model}"
            )

            if self.decision_logger is not None:
                self.decision_logger.log_position_open(position, signal)

            self._send_telegram(format_position_opened(
                symbol=symbol,
                direction=direction.value,
                signal_source=getattr(signal, 'signal_source', 'UNKNOWN'),
                size=float(allocated_size),
                entry_price=float(entry_price),
                sl=float(sl_price),
                tp=float(tp_price),
            ))

            # Lock signal to prevent re-entry on same crossover
            self.entry_evaluator.lock_signal(
                symbol=symbol,
                crossover_origin=signal.crossover_origin,
                signal_source=signal.signal_source,
            )
            self.logger.debug(
                f"Locked signal for {symbol}:{signal.signal_source} "
                f"crossover={signal.crossover_origin}"
            )

            # Update dedup tracking
            dedup_key = f"{symbol}:{signal.signal_source}:{signal.timeframe}"
            self.last_processed_candle_time[dedup_key] = signal.candle_time

    def _send_telegram(self, message: str) -> None:
        """Fire-and-forget Telegram alert. Never blocks or raises."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                task = asyncio.create_task(send_telegram_alert(message))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
        except Exception as e:
            self.logger.debug(f"Telegram send skipped: {e}")

    def _validate_signal_for_entry(self, symbol: str, signal) -> bool:
        """Validate a single signal through all entry gates.

        Checks cooldown, dedup, candle freshness, candle age, and candle close.
        Returns True if signal passes all gates and is valid for entry.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            signal: HybridEntrySignal to validate

        Returns:
            True if signal is valid for opening a position
        """
        from datetime import timedelta, timezone

        # Gate 1: Post-close cooldown
        signal_cooldown_key = f"{symbol}:{signal.signal_source}:{signal.timeframe}"
        if signal_cooldown_key in self.last_close_time:
            now = datetime.now(timezone.utc)
            last_close = self.last_close_time[signal_cooldown_key]
            if last_close.tzinfo is None:
                last_close = last_close.replace(tzinfo=timezone.utc)
            elapsed = now - last_close
            cooldown = timedelta(minutes=self.post_close_cooldown_minutes)
            if elapsed < cooldown:
                remaining = cooldown - elapsed
                self.logger.debug(
                    f"Post-close cooldown active for {signal_cooldown_key}: "
                    f"{remaining.total_seconds() / 60:.0f} min remaining"
                )
                return False

        # Gate 2: Existing position dedup (same signal+timeframe)
        existing_positions = self.position_manager.get_positions(symbol) if self.position_manager else []
        for pos in existing_positions:
            pos_signal, pos_tf = self._extract_signal_info(pos, symbol)
            if pos_signal and pos_tf:
                existing_key = f"{pos_signal}_{pos_tf}"
                signal_key = f"{signal.signal_source}_{signal.timeframe}"
                if existing_key == signal_key:
                    self.logger.debug(
                        f"Position already exists for {symbol} signal={signal.signal_source} "
                        f"tf={signal.timeframe}, skipping duplicate"
                    )
                    return False

        # Gate 3: Candle dedup (already processed this candle)
        dedup_key = f"{symbol}:{signal.signal_source}:{signal.timeframe}"
        last_candle_time = self.last_processed_candle_time.get(dedup_key)
        if last_candle_time is not None and signal.candle_time == last_candle_time:
            self.logger.debug(
                f"Skipping duplicate signal for {dedup_key} - "
                f"candle_time {signal.candle_time} already processed"
            )
            return False

        self.logger.info(
            f"Entry signal generated: {symbol} {signal.direction} "
            f"confidence={signal.confidence:.2f} timeframe={signal.timeframe}"
        )

        # Gate 4: Signal must be from the most recent closed candle
        TIMEFRAME_MINUTES = {
            "M30": 30, "H1": 60, "H2": 120, "H3": 180, "H4": 240,
            "H6": 360, "H8": 480, "H12": 720, "D1": 1440,
        }

        current_time = datetime.now(timezone.utc)
        period_minutes = TIMEFRAME_MINUTES.get(signal.timeframe, 30)
        period_seconds = period_minutes * 60
        current_ts = int(current_time.timestamp())
        latest_close_ts = (current_ts // period_seconds) * period_seconds
        latest_candle_close = datetime.fromtimestamp(latest_close_ts, tz=timezone.utc)

        expected_close = signal.crossover_origin + timedelta(minutes=period_minutes)
        time_diff_from_expected = abs((expected_close - latest_candle_close).total_seconds())
        time_diff_exact = abs((signal.crossover_origin - latest_candle_close).total_seconds())
        is_valid_signal = time_diff_from_expected <= 60 or time_diff_exact <= 60

        if not is_valid_signal:
            next_candle_close = latest_candle_close + timedelta(minutes=period_minutes)
            time_until_next = (next_candle_close - current_time).total_seconds()
            minutes_remaining = int(time_until_next // 60)
            seconds_remaining = int(time_until_next % 60)

            self.logger.warning(
                f"Signal from OLD CANDLE for {symbol} {signal.direction} [{signal.timeframe}]: "
                f"crossover at {signal.crossover_origin}, expected close at {expected_close}, "
                f"latest candle close at {latest_candle_close} - SKIPPING"
            )
            self.logger.info(
                f"Next {signal.timeframe} candle closes at {next_candle_close.strftime('%Y-%m-%d %H:%M:%S UTC')} "
                f"(in {minutes_remaining}m {seconds_remaining}s). Signal will be re-evaluated then."
            )

            signal_key = f"{symbol}_{signal.direction}_{signal.timeframe}"
            self._rejected_signals[signal_key] = {
                "symbol": symbol,
                "direction": signal.direction,
                "timeframe": signal.timeframe,
                "confidence": getattr(signal, "confidence", 0.0),
                "next_candle_close": next_candle_close,
                "crossover_time": signal.crossover_origin,
            }
            return False

        # Gate 5: Stale signal check (candle closed too long ago)
        MAX_SIGNAL_AGE_SECONDS = {
            "M30": 2700, "H1": 3600, "H2": 5400, "H3": 5400, "H4": 7200,
            "H6": 7200, "H8": 7200, "H12": 7200, "D1": 10800,
        }

        max_age = MAX_SIGNAL_AGE_SECONDS.get(signal.timeframe, 600)
        time_since_close = (current_time - latest_candle_close).total_seconds()

        if time_since_close > max_age:
            next_candle_close = latest_candle_close + timedelta(minutes=period_minutes)
            time_until_next = (next_candle_close - current_time).total_seconds()
            minutes_remaining = int(time_until_next // 60)
            seconds_remaining = int(time_until_next % 60)

            self.logger.warning(
                f"Signal STALE (candle too old) for {symbol} {signal.direction} [{signal.timeframe}]: "
                f"candle closed at {latest_candle_close.strftime('%Y-%m-%d %H:%M:%S UTC')}, "
                f"{time_since_close:.0f}s ago (max allowed: {max_age}s) - SKIPPING"
            )
            self.logger.info(
                f"This signal was detected {time_since_close/60:.1f} minutes after candle close. "
                f"Training expects positions to open immediately at candle close. "
                f"Next {signal.timeframe} candle closes in {minutes_remaining}m {seconds_remaining}s."
            )

            signal_key = f"{symbol}_{signal.direction}_{signal.timeframe}"
            self._rejected_signals[signal_key] = {
                "symbol": symbol,
                "direction": signal.direction,
                "timeframe": signal.timeframe,
                "confidence": getattr(signal, "confidence", 0.0),
                "next_candle_close": next_candle_close,
                "crossover_time": signal.crossover_origin,
                "rejection_reason": "stale_candle",
                "time_since_close": time_since_close,
            }
            return False

        # Log acceptance
        self.logger.info(
            f"Signal ACCEPTED (fresh candle): {symbol} {signal.direction} [{signal.timeframe}] "
            f"crossover={signal.crossover_origin}, latest_close={latest_candle_close}, "
            f"age={time_since_close:.0f}s (max={max_age}s)"
        )

        # Gate 6: Candle must be actually closed (not pre-close)
        is_closed, seconds_until = self._is_candle_closed(
            signal.timeframe, current_time, signal.crossover_origin
        )
        if not is_closed:
            candle_close = signal.crossover_origin + timedelta(minutes=period_minutes)
            self.logger.info(
                f"Waiting for {signal.timeframe} candle to close before opening position: "
                f"{symbol} {signal.direction} - candle closes in {seconds_until}s at "
                f"{candle_close.strftime('%H:%M:%S UTC')}"
            )

            signal_key = f"{symbol}_{signal.direction}_{signal.timeframe}"
            self._rejected_signals[signal_key] = {
                "symbol": symbol,
                "direction": signal.direction,
                "timeframe": signal.timeframe,
                "confidence": getattr(signal, "confidence", 0.0),
                "next_candle_close": candle_close,
                "crossover_time": signal.crossover_origin,
                "rejection_reason": "candle_not_closed",
            }
            return False

        # Gate 7: Weekend market close protection
        # Block entries where the next RL evaluation (next candle close) falls
        # at or after Friday market close (22:00 UTC). Without at least one RL
        # evaluation opportunity, positions sit through the weekend gap unmanaged.
        MARKET_CLOSE_HOUR_FRIDAY = 22  # Friday 22:00 UTC
        if current_time.weekday() == 4:  # Friday
            next_candle_close = latest_candle_close + timedelta(minutes=period_minutes)
            friday_market_close = current_time.replace(
                hour=MARKET_CLOSE_HOUR_FRIDAY, minute=0, second=0, microsecond=0
            )
            if next_candle_close >= friday_market_close:
                self.logger.info(
                    f"WEEKEND GATE: Blocking {symbol} {signal.direction} [{signal.timeframe}] - "
                    f"next candle close at {next_candle_close.strftime('%Y-%m-%d %H:%M UTC')} "
                    f"is at/after market close ({MARKET_CLOSE_HOUR_FRIDAY}:00 UTC Friday). "
                    f"Position would have no RL evaluation before weekend gap."
                )
                if self.decision_logger is not None:
                    self.decision_logger.log_risk_violation(
                        signal,
                        f"weekend_gate: next candle close {next_candle_close.strftime('%H:%M')} "
                        f">= market close {MARKET_CLOSE_HOUR_FRIDAY}:00 on Friday"
                    )
                return False

        # Remove from rejected signals tracking (it's now being processed)
        signal_key = f"{symbol}_{signal.direction}_{signal.timeframe}"
        self._rejected_signals.pop(signal_key, None)

        return True

    def _get_current_prices(self) -> dict[str, Decimal]:
        """Get current prices for all configured symbols from database.

        Queries each symbol's indicator table for the latest close price.

        Returns:
            Dictionary mapping symbol to current price.
        """
        symbols = self.config.get("symbols", DEFAULT_SYMBOLS)
        prices = {}
        for symbol in symbols:
            try:
                prices[symbol] = self._get_current_price(symbol)
            except ValueError as e:
                self.logger.error(f"Skipping price for {symbol}: {e}")
        return prices

    def _get_current_price(self, symbol: str) -> Decimal:
        """Get current price for a single symbol from database.

        Queries the rates table (Markets DB) for the latest close price.
        This provides ~60-second price resolution from the data-syncer,
        enabling intra-candle SL/TP detection instead of waiting for
        M30 candle closes.

        Falls back to indicator table (AI Model DB) if rates table
        query fails.

        Args:
            symbol: Trading symbol to get price for.

        Returns:
            Current market price for the symbol.

        Raises:
            ValueError: If symbol not found or no data available.
        """
        if self.db_manager is None:
            self.logger.warning(
                f"No db_manager configured, using fallback price for {symbol}"
            )
            return Decimal("1.0950")

        # Primary: query rates table (Markets DB, updated every ~60s by data-syncer)
        rates_table = f"{symbol.lower()}_m30_rates"
        rates_query = f"""
            SELECT close
            FROM {rates_table}
            ORDER BY rate_time DESC
            LIMIT 1
        """

        try:
            results = self.db_manager.execute_query("markets", rates_query, {})
            if results and len(results) > 0:
                close_price = results[0].get("close")
                if close_price is not None:
                    price = Decimal(str(close_price))
                    self._check_price_sanity(symbol, price)
                    return price
        except ValueError:
            raise  # re-raise price sanity failures
        except Exception as e:
            self.logger.warning(
                f"Rates table query failed for {symbol}: {e}, falling back to indicator table"
            )

        # Fallback: query indicator table (AI Model DB, updates on M30 candle close)
        table_name = INDICATOR_TABLES.get(symbol)
        if table_name is None:
            self.logger.warning(
                f"No indicator table for {symbol}, using fallback price"
            )
            return Decimal("1.0950")

        indicator_query = f"""
            SELECT close
            FROM {table_name}
            WHERE timeframe = 'M30'
            ORDER BY timestamp DESC
            LIMIT 1
        """

        try:
            results = self.db_manager.execute_query("ai_model", indicator_query, {})
            if results and len(results) > 0:
                close_price = results[0].get("close")
                if close_price is not None:
                    return Decimal(str(close_price))

            self.logger.warning(f"No price data found for {symbol}, using fallback")
            return Decimal("1.0950")

        except Exception as e:
            self.logger.error(f"Error fetching price for {symbol}: {e}")
            return Decimal("1.0950")

    # Expected close-price ranges per symbol (wide bounds to avoid false positives)
    _PRICE_RANGES = {
        "EURUSD": (0.80, 1.60),
        "GBPUSD": (1.00, 2.00),
        "USDJPY": (75.0, 200.0),
        "EURJPY": (90.0, 250.0),
        "USDCAD": (1.00, 1.80),
        "EURCAD": (1.10, 2.00),
        "USDCHF": (0.60, 1.20),
        "EURGBP": (0.65, 1.10),
    }

    def _check_price_sanity(self, symbol: str, price: Decimal) -> None:
        """Raise ValueError if *price* is outside the expected range for *symbol*."""
        bounds = self._PRICE_RANGES.get(symbol.upper())
        if bounds is None:
            return
        lo, hi = bounds
        if not (Decimal(str(lo)) <= price <= Decimal(str(hi))):
            self.logger.error(
                f"PRICE SANITY FAILED: {symbol} price={price} "
                f"outside expected range [{lo}, {hi}]"
            )
            raise ValueError(
                f"Price {price} for {symbol} outside expected range [{lo}, {hi}]"
            )

    def _get_symbol_config(self, symbol: str) -> dict[str, Any]:
        """Get symbol-specific configuration.

        Args:
            symbol: Trading symbol to get config for.

        Returns:
            Dictionary with symbol configuration or empty dict if not found.
        """
        symbol_configs = self.config.get("symbol_configs", {})
        return symbol_configs.get(symbol, {})

    def _is_candle_closed(self, timeframe: str, current_time: datetime, signal_candle_time: datetime) -> tuple[bool, int]:
        """Check if signal's candle has closed and we're in valid entry window (Issue #635).

        This prevents opening positions on signals from candles that haven't closed yet.
        Training behavior: positions open exactly at candle close.
        Production must match: only allow entry when the signal's candle has fully closed.

        The key insight: we need to verify that the signal's candle_time + candle_duration
        is in the past, meaning that candle has fully closed.

        Example for M30 signal from 20:00 candle at current time 20:26 UTC:
        - Signal candle (20:00) closes at 20:30 (20:00 + 30 minutes)
        - Current time 20:26 is BEFORE 20:30 → candle NOT closed → return (False, 240)
        - Current time 20:30+ is AT/AFTER close → candle closed → return (True, 0)

        Args:
            timeframe: Signal timeframe (e.g., 'M30', 'H1', 'H4')
            current_time: Current UTC datetime
            signal_candle_time: The candle timestamp from the signal

        Returns:
            Tuple of (is_closed: bool, seconds_until_close: int)
            - is_closed: True if signal's candle has closed
            - seconds_until_close: Seconds until candle closes (0 if already closed)
        """
        # Timeframe periods in minutes
        TIMEFRAME_MINUTES = {
            "M30": 30,
            "H1": 60,
            "H2": 120,
            "H3": 180,
            "H4": 240,
            "H6": 360,
            "H8": 480,
            "H12": 720,
            "D1": 1440,
        }

        minutes = TIMEFRAME_MINUTES.get(timeframe, 30)

        # Calculate when this signal's candle closes
        # Signal candle time is the START of the candle
        # Candle closes at candle_time + duration
        candle_close_time = signal_candle_time + timedelta(minutes=minutes)

        # Check if we're past the candle close
        if current_time >= candle_close_time:
            return (True, 0)
        else:
            seconds_until = int((candle_close_time - current_time).total_seconds())
            return (False, seconds_until)

    def _extract_signal_info(
        self, position: Any, symbol: str
    ) -> tuple[str | None, str | None]:
        """Extract signal_name and timeframe for signal-specific model loading.

        Issue #560: The exit evaluator needs to know which signal-specific model
        to use for exit decisions. This method extracts the signal name from the
        position's entry_model field and looks up the timeframe from config.

        Issue #631 fix: Look up signal-specific timeframe from the signals list,
        not the symbol-level default. This ensures RL exit evaluates at the
        correct candle close for the signal's timeframe.

        Args:
            position: The open position with entry_model field.
            symbol: Trading symbol to look up timeframe for.

        Returns:
            Tuple of (signal_name, timeframe) where either may be None if not found.

        Example:
            position.entry_model = "Hybrid_V4 + Stoch_RSI_long_15_25"
            Returns: ("Stoch_RSI_long_15_25", "M30")  # Signal-specific timeframe
        """
        signal_name: str | None = None
        timeframe: str | None = None

        # Extract signal_name from entry_model (format: "Hybrid_V4 + signal_name")
        if hasattr(position, "entry_model") and position.entry_model:
            entry_model = position.entry_model
            if " + " in entry_model:
                # Split "Hybrid_V4 + Stoch_RSI_long_15_25" -> "Stoch_RSI_long_15_25"
                signal_name = entry_model.split(" + ", 1)[1].strip()
            elif entry_model.startswith("Hybrid_V4"):
                # Fallback: just "Hybrid_V4" without signal name
                signal_name = None
            else:
                # Non-hybrid model, use as-is
                signal_name = entry_model

        # Prefer signal_timeframe stored on position at open time (avoids config
        # ambiguity when the same signal exists on multiple timeframes)
        if hasattr(position, "signal_timeframe") and position.signal_timeframe:
            timeframe = position.signal_timeframe
            self.logger.debug(
                f"Using stored signal_timeframe for {signal_name}: {timeframe}"
            )
        else:
            # Fallback: look up timeframe from symbol config
            symbol_config = self._get_symbol_config(symbol)

            # Issue #631 fix: Look up signal-specific timeframe from signals list
            if signal_name and "signals" in symbol_config:
                signals_list = symbol_config.get("signals", [])
                for signal_cfg in signals_list:
                    if signal_cfg.get("signal") == signal_name:
                        timeframe = signal_cfg.get("timeframe")
                        self.logger.debug(
                            f"Found signal-specific timeframe for {signal_name}: {timeframe}"
                        )
                        break

            # Fallback to symbol-level timeframe if signal not found
            if timeframe is None:
                symbol_config = self._get_symbol_config(symbol)
                timeframe = symbol_config.get("timeframe")

            # Final fallback: try to get timeframe from runner config
            if timeframe is None:
                timeframe = self.config.get("entry_timeframe", "H4")

        self.logger.debug(
            f"Extracted signal info for {symbol}: signal_name={signal_name}, "
            f"timeframe={timeframe}"
        )

        return signal_name, timeframe

    def _extract_signal_info_from_entry_model(
        self, entry_model: str | None, symbol: str
    ) -> tuple[str | None, str | None]:
        """Extract signal_name and timeframe from an entry_model string.

        Lightweight wrapper for use with Trade objects (which have entry_model
        but aren't Position objects). Reuses the same parsing and config lookup
        logic as _extract_signal_info.
        """
        # Create a minimal object with entry_model attribute
        class _EntryModelHolder:
            pass

        holder = _EntryModelHolder()
        holder.entry_model = entry_model
        return self._extract_signal_info(holder, symbol)


async def main() -> None:
    """Main entry point for the paper trading service.

    Initializes all trading components and starts the service:
    1. Loads configuration from YAML + environment overrides
    2. Establishes database connection
    3. Creates trading components (PositionManager, RiskManager, etc.)
    4. Wires components into PaperTradingRunner
    5. Runs the main polling loop
    """
    configure_logging()

    logger = logging.getLogger(__name__)
    logger.info("Starting Paper Trading Service...")

    # Load configuration from YAML + environment overrides
    config = PaperTradingConfig.from_env()
    enabled_symbols = config.get_enabled_symbols()
    logger.info(f"Configuration loaded: {len(enabled_symbols)} symbols enabled")
    logger.info(f"Enabled symbols: {enabled_symbols}")

    # Initialize database connection
    db_manager = DatabaseManager()
    db_manager.connect()
    logger.info("Database connection established")

    # Create asyncpg pool for async operations (Issue #629: SignalPreviewEvaluator)
    if asyncpg is not None:
        ai_model_pool = await asyncpg.create_pool(
            host=os.environ.get('AI_MODEL_DB_HOST', 'localhost'),
            port=int(os.environ.get('AI_MODEL_DB_PORT', '5432')),
            database=os.environ.get('AI_MODEL_DB_NAME', 'ai_model'),
            user=os.environ.get('AI_MODEL_DB_USER', 'tfg_user'),
            password=os.environ.get('AI_MODEL_DB_PASSWORD', 'tfg_password'),
            min_size=1,
            max_size=5,
        )
        logger.info("AsyncPG pool created for signal preview operations")
    else:
        ai_model_pool = None
        logger.warning("asyncpg not available - signal preview will be limited")

    # Bootstrap live performance tracker from historical trades
    live_tracker = LivePerformanceTracker()
    await live_tracker.bootstrap_from_db(ai_model_pool)
    logger.info("LivePerformanceTracker bootstrapped from paper_trades history")

    # Persist config params (margin/leverage/PF tiers) to DB for TS/frontend
    from src.paper_trading.balance_allocator import persist_config_params
    await persist_config_params(ai_model_pool)

    # Initialize trading components in dependency order
    position_manager = PositionManager(db_session=db_manager)
    logger.info("PositionManager initialized with database persistence")

    # Wire live_tracker to PositionManager BEFORE wrapping in LiveTradingAdapter
    # (LiveTradingAdapter reassigns position_manager variable, so wiring after
    # would set live_tracker on the adapter, not the inner PM that calls record_trade)
    position_manager.live_tracker = live_tracker
    logger.info("LivePerformanceTracker wired to PositionManager")

    # TFG: Always paper trading mode (no MT5 gateway)
    logger.info("TFG mode - using paper trading only (no broker integration)")

    # Dynamic max_concurrent_positions from signal count
    total_enabled_signals = sum(
        len(sym_config.get_active_signals())
        for sym_config in config.symbols.values()
        if sym_config.enabled
    )
    dynamic_max_concurrent = min(total_enabled_signals, 25)  # cap at 25 to avoid budget dilution

    # Create RiskConfig from Pydantic config
    risk_config = RiskConfig(
        max_concurrent_positions=dynamic_max_concurrent,
        max_positions_per_symbol=config.risk.max_positions_per_symbol,
        max_daily_trades=config.risk.max_daily_trades,
        max_daily_loss_pips=config.risk.max_daily_loss_pips,
        risk_per_trade_pct=config.risk.risk_per_trade_pct,
        lookahead_candles=config.risk.lookahead_candles,
        signal_hit_rate=config.risk.signal_hit_rate,
    )
    risk_manager = RiskManager(config=risk_config, position_manager=position_manager)
    risk_manager.live_tracker = live_tracker
    logger.info(
        f"RiskManager initialized: max_positions={risk_config.max_concurrent_positions} "
        f"(dynamic from {total_enabled_signals} signals), "
        f"max_daily_trades={risk_config.max_daily_trades}, "
        f"max_per_symbol={risk_config.max_positions_per_symbol}"
    )

    # Initialize decision logger for trade audit trail
    decision_logger = TradeDecisionDBLogger(db_session=db_manager)
    logger.info("TradeDecisionDBLogger initialized")

    # Wire decision logger to LiveTradingAdapter for broker-closed positions (SL/TP hit)
    if hasattr(position_manager, '_on_broker_close'):
        position_manager._on_broker_close = decision_logger.log_position_close
        logger.info("Decision logger wired to LiveTradingAdapter for broker close events")

    # Initialize lifecycle client for signal filtering
    lifecycle_client = None
    try:
        from src.paper_trading.lifecycle_client import LifecycleClient
        lifecycle_client = LifecycleClient.from_env()
        if lifecycle_client.validate_connection():
            logger.info("LifecycleClient initialized for entry + hot reload filtering")
        else:
            logger.warning("LifecycleClient connection failed - signals unfiltered")
            lifecycle_client = None
    except Exception as e:
        logger.warning(f"Could not initialize LifecycleClient: {e}")

    # Wire lifecycle_client to the inner PositionManager for performance reporting
    # (position_manager may be a LiveTradingAdapter wrapping the real PM)
    if lifecycle_client is not None:
        inner_pm = getattr(position_manager, '_position_manager', position_manager)
        inner_pm.lifecycle_client = lifecycle_client
        logger.info("LifecycleClient wired to PositionManager for performance reporting")

    # Initialize hybrid evaluators (Issue #495, #510: Hybrid V4 architecture)
    # Entry: Rule-based signals from config (H4 timeframe)
    # Exit: RL-managed using trained hybrid_v4 PPO models (30-fold ensemble)

    # Build signal_config from YAML configuration for HybridEntryEvaluator
    # Issue #541: Supports MULTIPLE signals per symbol
    # Issue #569: Includes per-signal timeframe for proper evaluation
    signal_config: Dict[str, List[Dict[str, str]]] = {}
    entry_timeframe = "D1"  # Default fallback

    for symbol in enabled_symbols:
        sym_cfg = config.symbols.get(symbol)
        if sym_cfg and sym_cfg.signals:
            # Get ALL active signals for this symbol
            active_signals = [s for s in sym_cfg.signals if s.enabled]
            if active_signals:
                # Store list of all signal configs for this symbol
                # Issue #569: Include timeframe for per-signal evaluation
                signal_config[symbol] = [
                    {
                        "signal": sig.signal,
                        "direction": sig.direction,
                        "timeframe": sig.timeframe,  # Issue #569: Per-signal timeframe
                    }
                    for sig in active_signals
                ]
                # Use symbol's default timeframe as fallback
                entry_timeframe = sym_cfg.timeframe if sym_cfg.timeframe else "H4"
                # Log all active signals with their timeframes
                for sig in active_signals:
                    logger.info(
                        f"Signal config for {symbol}: signal={sig.signal}, "
                        f"direction={sig.direction}, timeframe={sig.timeframe}"
                    )
                logger.info(
                    f"{symbol}: {len(active_signals)} signals configured for evaluation"
                )

    # Log total signal count at startup for monitoring
    total_configured_signals = sum(len(sigs) for sigs in signal_config.values())
    logger.info(
        f"Total configured signals: {total_configured_signals} across "
        f"{len(signal_config)} symbols"
    )

    # Startup signal count alert if dangerously low
    if total_configured_signals == 0:
        alert_msg = (
            "<b>CRITICAL: 0 SIGNALS CONFIGURED</b>\n\n"
            "Paper trading pod started with ZERO signals.\n"
            "Check paper_trading.yaml and approved_models."
        )
        logger.critical("ZERO signals configured at startup!")
        asyncio.create_task(send_telegram_alert(alert_msg))

    entry_evaluator = HybridEntryEvaluator(
        symbols=enabled_symbols,
        db_manager=db_manager,
        timeframe=entry_timeframe,
        signal_config=signal_config,
        lifecycle_client=lifecycle_client,
    )
    logger.info(
        f"HybridEntryEvaluator initialized: timeframe={entry_timeframe}, "
        f"signals={list(signal_config.keys())}"
    )

    # Auto-detect model version: prefer v4 if directory exists
    v4_model_dir = Path(DEFAULT_MODEL_DIR_V4)
    v2_model_dir = Path(DEFAULT_MODEL_DIR_V2)

    if v4_model_dir.exists() and any(v4_model_dir.iterdir()):
        # Use v4 models with direction-specific ensemble
        exit_evaluator = HybridExitEvaluator(
            symbols=enabled_symbols,
            directions=["long", "short"],  # Support both directions
            model_dir=str(v4_model_dir),
            model_version=MODEL_VERSION_V4,
            ensemble_method="mean",  # Average probabilities across folds
            db_manager=db_manager,  # Issue #588: Enable indicator queries
            confidence_threshold=config.exit_scaffold.confidence_threshold,
        )
        logger.info(
            f"HybridExitEvaluator initialized with hybrid_v4 models (ensemble), "
            f"confidence_threshold={config.exit_scaffold.confidence_threshold}"
        )
    else:
        # Fall back to v2 models
        exit_evaluator = HybridExitEvaluator(
            symbols=enabled_symbols,
            model_dir=str(v2_model_dir),
            model_version=MODEL_VERSION_V2,
        )
        logger.info("HybridExitEvaluator initialized with hybrid_v2 models")

    # Build runner config dict
    runner_config = {
        "symbols": enabled_symbols,
        "poll_interval": config.poll_interval,
        "enabled": config.enabled,
        "symbol_configs": config.get_symbol_configs(),  # Per-symbol exit strategy
        # Account configuration for dynamic position sizing (Issue #631)
        "account": {
            "initial_balance": config.account.initial_balance,
            "currency": config.account.currency,
            "exchange_rate_to_usd": config.account.exchange_rate_to_usd,
        },
    }

    # Log per-symbol exit strategies
    for symbol in enabled_symbols:
        sym_cfg = config.symbols.get(symbol)
        if sym_cfg:
            logger.info(
                f"Symbol {symbol}: use_rl_exit={sym_cfg.use_rl_exit}, "
                f"exit_strategy={sym_cfg.exit_strategy}"
            )

    # TFG: No real-time SL protection (no WebSocket/MT5 gateway)
    ws_client = None
    sl_checker = None
    logger.info("TFG mode - real-time SL protection disabled (no WebSocket)")

    # Initialize Signal Preview Evaluator (Issue #629)
    # Build config with full symbol configs (convert SignalConfig objects to dicts)
    preview_config = {
        "symbols": {
            symbol: {
                "enabled": True,
                "signals": [
                    {
                        "signal": sig.signal,
                        "direction": sig.direction,
                        "timeframe": sig.timeframe,
                        "enabled": sig.enabled,
                    }
                    for sig in config.symbols[symbol].signals
                    if sig.enabled
                ]
            }
            for symbol in enabled_symbols
            if symbol in config.symbols
        }
    }
    logger.info(f"preview_config structure: symbols type={type(preview_config['symbols'])}, sample keys={list(preview_config['symbols'].keys())[:2] if isinstance(preview_config['symbols'], dict) else 'not a dict'}")
    signal_preview_evaluator = SignalPreviewEvaluator(
        db_pool=ai_model_pool,  # AsyncPG pool for indicator lookups and snapshot saves
        config=preview_config,
        exit_evaluator=exit_evaluator,  # For 30-model consensus
        entry_evaluator=entry_evaluator,      # For crossover/lock status
        position_manager=position_manager,    # For open position checks
    )
    logger.info(f"SignalPreviewEvaluator initialized: {sum(len(s['signals']) for s in preview_config['symbols'].values())} signals")

    # Create runner with all components wired (Issue #495: Hybrid architecture)
    runner = PaperTradingRunner(
        config=runner_config,
        entry_evaluator=entry_evaluator,
        exit_evaluator=exit_evaluator,  # Issue #495: RL-managed exits
        position_manager=position_manager,
        risk_manager=risk_manager,
        decision_logger=decision_logger,
        db_manager=db_manager,  # Fix #4: Pass db_manager for real price fetching
        ws_client=ws_client,
        sl_checker=sl_checker,
    )
    # Share lifecycle_client with runner for hot reload filtering
    runner._lifecycle_client = lifecycle_client
    # Share signal_config with runner for signal count validation alerts
    runner._signal_config = signal_config
    # Attach signal preview evaluator to runner (Issue #629)
    runner.signal_preview_evaluator = signal_preview_evaluator

    # Log account configuration (Issue #631)
    logger.info(
        f"Account configured: {runner.account_currency} balance={runner_config.get('account', {}).get('initial_balance', 10000)}, "
        f"USD equivalent=${runner.account_balance:.2f} (rate: {runner.exchange_rate_to_usd})"
    )

    logger.info("All trading components wired - starting main loop")

    try:
        await runner.run()
    finally:
        # Cleanup database connections on shutdown
        if ai_model_pool is not None:
            await ai_model_pool.close()
            logger.info("AsyncPG pool closed")
        db_manager.disconnect()
        logger.info("Database connection closed")


if __name__ == "__main__":
    asyncio.run(main())
