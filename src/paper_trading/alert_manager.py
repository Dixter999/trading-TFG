"""
Alert Manager for Paper Trading Engine.

Issue #426: Alert System for Anomalies
Track 6: Paper Trading

This module provides:
- AlertType enum for categorizing alert types
- AlertSeverity enum for alert severity levels
- Alert dataclass for alert data
- AlertConfig dataclass for configuring alert thresholds
- AlertManager class for monitoring and alerting on anomalies

Alert Types:
- DRAWDOWN_WARNING: Drawdown exceeds warning threshold (>30 pips)
- DRAWDOWN_CRITICAL: Drawdown exceeds critical threshold (>50 pips)
- DAILY_LOSS_LIMIT: Daily loss limit exceeded
- SYSTEM_ERROR: System exceptions occurred
- INACTIVITY: No signals generated within threshold
- LOW_WIN_RATE: Win rate below threshold (<30%)
- DATA_STALE: Candle data older than threshold (>10 min)

Tables used:
- paper_alerts: Storage for generated alerts
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class DatabaseManager(Protocol):
    """Protocol for database manager dependency."""

    def execute_query(
        self, db_name: str, query: str, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Execute a query and return results."""
        ...

    def insert_data(self, table: str, data: dict[str, Any]) -> None:
        """Insert data into a table."""
        ...


class PerformanceTrackerProtocol(Protocol):
    """Protocol for performance tracker dependency."""

    def get_current_drawdown(self, symbol: str) -> Decimal:
        """Get current drawdown in pips."""
        ...

    def get_daily_pnl(self, symbol: str) -> Decimal:
        """Get daily P&L in pips."""
        ...

    def get_last_signal_time(self, symbol: str) -> Optional[datetime]:
        """Get timestamp of last signal generated."""
        ...

    def get_last_candle_time(self, symbol: str) -> Optional[datetime]:
        """Get timestamp of last candle received."""
        ...

    def get_recent_metrics(self, symbol: str) -> Any:
        """Get recent performance metrics."""
        ...


class AlertType(Enum):
    """Types of alerts that can be generated."""

    DRAWDOWN_WARNING = "drawdown_warning"
    DRAWDOWN_CRITICAL = "drawdown_critical"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    SYSTEM_ERROR = "system_error"
    INACTIVITY = "inactivity"
    LOW_WIN_RATE = "low_win_rate"
    DATA_STALE = "data_stale"


class AlertSeverity(Enum):
    """Severity levels for alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert data structure.

    Attributes:
        alert_type: Type of alert from AlertType enum
        severity: Severity level from AlertSeverity enum
        timestamp: When the alert was generated
        message: Human-readable alert message
        symbol: Trading symbol associated with alert (can be None for system alerts)
        value: Optional numeric value that triggered the alert
        threshold: Optional threshold value that was exceeded
    """

    alert_type: AlertType
    severity: AlertSeverity
    timestamp: datetime
    message: str
    symbol: Optional[str]
    value: Optional[Decimal] = None
    threshold: Optional[Decimal] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary for database storage.

        Returns:
            Dictionary with all fields suitable for database insertion.
        """
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "message": self.message,
            "symbol": self.symbol,
            "value": float(self.value) if self.value is not None else None,
            "threshold": float(self.threshold) if self.threshold is not None else None,
        }


@dataclass
class AlertConfig:
    """Configuration for alert thresholds.

    Attributes:
        drawdown_warning_pips: Drawdown level that triggers warning (default: 30)
        drawdown_critical_pips: Drawdown level that triggers critical alert (default: 50)
        daily_loss_limit_pips: Daily loss limit in pips (default: 100)
        inactivity_hours: Hours without signals before alerting (default: 4)
        low_win_rate_threshold: Win rate decimal below which to alert (default: 0.30)
        data_stale_minutes: Minutes after which candle data is stale (default: 10)
        alert_cooldown_minutes: Minutes between same alert type (default: 30)
        min_trades_for_alert: Minimum trades required for win rate alert (default: 10)
    """

    drawdown_warning_pips: int = 30
    drawdown_critical_pips: int = 50
    daily_loss_limit_pips: int = 100
    inactivity_hours: int = 4
    low_win_rate_threshold: float = 0.30
    data_stale_minutes: int = 10
    alert_cooldown_minutes: int = 30
    min_trades_for_alert: int = 10


class AlertManager:
    """Alert manager for paper trading anomaly detection.

    Monitors paper trading performance and generates alerts for:
    - Excessive drawdown (warning at 30 pips, critical at 50 pips)
    - Daily loss limit exceeded
    - Inactivity (no signals for extended periods)
    - Low win rate
    - Stale data

    Implements cooldown mechanism to prevent alert spam.

    Attributes:
        _config: AlertConfig with thresholds
        _performance_tracker: Performance tracker for metrics
        _db_manager: Database manager for alert storage
        _last_alert_times: Dictionary tracking last alert time per type/symbol
    """

    def __init__(
        self,
        config: AlertConfig,
        performance_tracker: PerformanceTrackerProtocol,
        db_manager: DatabaseManager,
    ) -> None:
        """Initialize the AlertManager.

        Args:
            config: AlertConfig with threshold settings
            performance_tracker: Performance tracker for metrics access
            db_manager: Database manager for alert storage
        """
        self._config = config
        self._performance_tracker = performance_tracker
        self._db_manager = db_manager
        self._last_alert_times: dict[str, datetime] = {}
        logger.debug("AlertManager initialized with config: %s", config)

    def check_all(self, symbol: str) -> list[Alert]:
        """Check all alert conditions for a symbol.

        Runs all individual check methods and returns aggregated alerts.
        Applies cooldown filtering to prevent alert spam.

        Args:
            symbol: Trading symbol to check

        Returns:
            List of Alert objects for conditions that triggered
        """
        alerts: list[Alert] = []

        # Run all checks
        alerts.extend(self._check_drawdown(symbol))
        alerts.extend(self._check_daily_loss(symbol))
        alerts.extend(self._check_inactivity(symbol))
        alerts.extend(self._check_win_rate(symbol))
        alerts.extend(self._check_data_freshness(symbol))

        # Filter by cooldown and record new alerts
        filtered_alerts = []
        for alert in alerts:
            if self._should_alert(alert.alert_type, symbol):
                self._record_alert_sent(alert.alert_type, symbol)
                filtered_alerts.append(alert)

        return filtered_alerts

    def _check_drawdown(self, symbol: Optional[str]) -> list[Alert]:
        """Check for excessive drawdown.

        Args:
            symbol: Trading symbol to check

        Returns:
            List containing drawdown alert if threshold exceeded
        """
        try:
            if symbol is None:
                return []

            drawdown = self._performance_tracker.get_current_drawdown(symbol)

            # Check critical first (higher priority)
            if drawdown > self._config.drawdown_critical_pips:
                return [
                    Alert(
                        alert_type=AlertType.DRAWDOWN_CRITICAL,
                        severity=AlertSeverity.CRITICAL,
                        timestamp=datetime.now(timezone.utc),
                        message=f"Drawdown exceeded critical threshold: {drawdown} pips",
                        symbol=symbol,
                        value=drawdown,
                        threshold=Decimal(str(self._config.drawdown_critical_pips)),
                    )
                ]

            # Check warning threshold
            if drawdown > self._config.drawdown_warning_pips:
                return [
                    Alert(
                        alert_type=AlertType.DRAWDOWN_WARNING,
                        severity=AlertSeverity.WARNING,
                        timestamp=datetime.now(timezone.utc),
                        message=f"Drawdown exceeded warning threshold: {drawdown} pips",
                        symbol=symbol,
                        value=drawdown,
                        threshold=Decimal(str(self._config.drawdown_warning_pips)),
                    )
                ]

            return []

        except Exception as e:
            logger.error("Error checking drawdown for %s: %s", symbol, e)
            return []

    def _check_daily_loss(self, symbol: str) -> list[Alert]:
        """Check for daily loss limit exceeded.

        Args:
            symbol: Trading symbol to check

        Returns:
            List containing daily loss alert if limit exceeded
        """
        try:
            daily_pnl = self._performance_tracker.get_daily_pnl(symbol)

            # Daily loss is negative, check if absolute value exceeds limit
            if daily_pnl < 0 and abs(daily_pnl) > self._config.daily_loss_limit_pips:
                return [
                    Alert(
                        alert_type=AlertType.DAILY_LOSS_LIMIT,
                        severity=AlertSeverity.CRITICAL,
                        timestamp=datetime.now(timezone.utc),
                        message=f"Daily loss limit exceeded: {abs(daily_pnl)} pips",
                        symbol=symbol,
                        value=abs(daily_pnl),
                        threshold=Decimal(str(self._config.daily_loss_limit_pips)),
                    )
                ]

            return []

        except Exception as e:
            logger.error("Error checking daily loss for %s: %s", symbol, e)
            return []

    def _check_inactivity(self, symbol: str) -> list[Alert]:
        """Check for signal inactivity.

        Args:
            symbol: Trading symbol to check

        Returns:
            List containing inactivity alert if threshold exceeded
        """
        try:
            last_signal_time = self._performance_tracker.get_last_signal_time(symbol)
            now = datetime.now(timezone.utc)

            # If no signals ever, alert
            if last_signal_time is None:
                return [
                    Alert(
                        alert_type=AlertType.INACTIVITY,
                        severity=AlertSeverity.WARNING,
                        timestamp=now,
                        message=f"No signals ever generated for {symbol}",
                        symbol=symbol,
                    )
                ]

            # Check if last signal is older than threshold
            inactivity_threshold = timedelta(hours=self._config.inactivity_hours)
            if now - last_signal_time > inactivity_threshold:
                hours_inactive = (now - last_signal_time).total_seconds() / 3600
                return [
                    Alert(
                        alert_type=AlertType.INACTIVITY,
                        severity=AlertSeverity.WARNING,
                        timestamp=now,
                        message=f"No signals for {hours_inactive:.1f} hours",
                        symbol=symbol,
                        value=Decimal(str(round(hours_inactive, 1))),
                        threshold=Decimal(str(self._config.inactivity_hours)),
                    )
                ]

            return []

        except Exception as e:
            logger.error("Error checking inactivity for %s: %s", symbol, e)
            return []

    def _check_win_rate(self, symbol: str) -> list[Alert]:
        """Check for low win rate.

        Args:
            symbol: Trading symbol to check

        Returns:
            List containing low win rate alert if below threshold
        """
        try:
            metrics = self._performance_tracker.get_recent_metrics(symbol)

            if metrics is None:
                return []

            # Require minimum trades for statistical significance
            if metrics.total_trades < self._config.min_trades_for_alert:
                return []

            # Win rate from metrics is percentage, convert to decimal for comparison
            win_rate_decimal = float(metrics.win_rate) / 100

            if win_rate_decimal < self._config.low_win_rate_threshold:
                return [
                    Alert(
                        alert_type=AlertType.LOW_WIN_RATE,
                        severity=AlertSeverity.WARNING,
                        timestamp=datetime.now(timezone.utc),
                        message=f"Win rate below threshold: {float(metrics.win_rate):.1f}%",
                        symbol=symbol,
                        value=metrics.win_rate,
                        threshold=Decimal(str(self._config.low_win_rate_threshold * 100)),
                    )
                ]

            return []

        except Exception as e:
            logger.error("Error checking win rate for %s: %s", symbol, e)
            return []

    def _check_data_freshness(self, symbol: str) -> list[Alert]:
        """Check for stale candle data.

        Args:
            symbol: Trading symbol to check

        Returns:
            List containing data stale alert if threshold exceeded
        """
        try:
            last_candle_time = self._performance_tracker.get_last_candle_time(symbol)
            now = datetime.now(timezone.utc)

            if last_candle_time is None:
                return [
                    Alert(
                        alert_type=AlertType.DATA_STALE,
                        severity=AlertSeverity.WARNING,
                        timestamp=now,
                        message=f"No candle data received for {symbol}",
                        symbol=symbol,
                    )
                ]

            # Check if data is stale
            stale_threshold = timedelta(minutes=self._config.data_stale_minutes)
            if now - last_candle_time > stale_threshold:
                minutes_stale = (now - last_candle_time).total_seconds() / 60
                return [
                    Alert(
                        alert_type=AlertType.DATA_STALE,
                        severity=AlertSeverity.WARNING,
                        timestamp=now,
                        message=f"Candle data is {minutes_stale:.1f} minutes old",
                        symbol=symbol,
                        value=Decimal(str(round(minutes_stale, 1))),
                        threshold=Decimal(str(self._config.data_stale_minutes)),
                    )
                ]

            return []

        except Exception as e:
            logger.error("Error checking data freshness for %s: %s", symbol, e)
            return []

    @staticmethod
    def _make_cooldown_key(alert_type: AlertType, symbol: Optional[str]) -> str:
        """Create consistent cooldown key for alert type and symbol.

        Args:
            alert_type: Type of alert
            symbol: Trading symbol

        Returns:
            Cooldown key string
        """
        return f"{alert_type.value}:{symbol}"

    def _should_alert(self, alert_type: AlertType, symbol: Optional[str]) -> bool:
        """Check if an alert should be sent based on cooldown.

        Args:
            alert_type: Type of alert
            symbol: Trading symbol

        Returns:
            True if alert should be sent, False if in cooldown
        """
        key = self._make_cooldown_key(alert_type, symbol)

        if key not in self._last_alert_times:
            return True

        last_alert_time = self._last_alert_times[key]
        cooldown = timedelta(minutes=self._config.alert_cooldown_minutes)
        now = datetime.now(timezone.utc)

        return now - last_alert_time > cooldown

    def _record_alert_sent(self, alert_type: AlertType, symbol: Optional[str]) -> None:
        """Record that an alert was sent for cooldown tracking.

        Args:
            alert_type: Type of alert
            symbol: Trading symbol
        """
        key = self._make_cooldown_key(alert_type, symbol)
        self._last_alert_times[key] = datetime.now(timezone.utc)

    def send_alert(self, alert: Alert) -> None:
        """Send an alert and record it.

        Logs the alert and stores it in the database.

        Args:
            alert: Alert to send
        """
        # Log based on severity
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(
                "ALERT [%s] %s: %s",
                alert.symbol,
                alert.alert_type.value,
                alert.message,
            )
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(
                "ALERT [%s] %s: %s",
                alert.symbol,
                alert.alert_type.value,
                alert.message,
            )
        else:
            logger.info(
                "ALERT [%s] %s: %s",
                alert.symbol,
                alert.alert_type.value,
                alert.message,
            )

        # Store in database
        self._db_manager.insert_data("paper_alerts", alert.to_dict())

        # Record for cooldown
        self._record_alert_sent(alert.alert_type, alert.symbol)

    def create_system_error_alert(
        self, error: Exception, symbol: Optional[str] = None
    ) -> list[Alert]:
        """Create a system error alert from an exception.

        Args:
            error: Exception that occurred
            symbol: Optional trading symbol associated with error

        Returns:
            List containing system error alert
        """
        return [
            Alert(
                alert_type=AlertType.SYSTEM_ERROR,
                severity=AlertSeverity.CRITICAL,
                timestamp=datetime.now(timezone.utc),
                message=f"System error: {str(error)}",
                symbol=symbol,
            )
        ]

    def get_active_alerts(
        self, symbol: Optional[str] = None, hours: int = 24
    ) -> list[dict[str, Any]]:
        """Get active alerts from the database.

        Args:
            symbol: Optional symbol to filter by
            hours: Number of hours to look back (default: 24)

        Returns:
            List of alert dictionaries from database
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        if symbol:
            query = """
                SELECT *
                FROM paper_alerts
                WHERE symbol = :symbol
                  AND timestamp >= :cutoff
                ORDER BY timestamp DESC
            """
            params = {"symbol": symbol, "cutoff": cutoff}
        else:
            query = """
                SELECT *
                FROM paper_alerts
                WHERE timestamp >= :cutoff
                ORDER BY timestamp DESC
            """
            params = {"cutoff": cutoff}

        return self._db_manager.execute_query("ai_model", query, params)

    def check_multiple_symbols(self, symbols: list[str]) -> dict[str, list[Alert]]:
        """Check all alerts for multiple symbols.

        Args:
            symbols: List of trading symbols to check

        Returns:
            Dictionary mapping symbol to list of alerts
        """
        results: dict[str, list[Alert]] = {}
        for symbol in symbols:
            results[symbol] = self.check_all(symbol)
        return results

    def clear_cooldown(self, alert_type: AlertType, symbol: Optional[str] = None) -> None:
        """Clear cooldown for a specific alert type/symbol.

        Useful for testing or manual intervention.

        Args:
            alert_type: Type of alert to clear cooldown for
            symbol: Trading symbol (None for all symbols of this type)
        """
        if symbol is not None:
            key = self._make_cooldown_key(alert_type, symbol)
            if key in self._last_alert_times:
                del self._last_alert_times[key]
        else:
            # Clear all cooldowns for this alert type
            keys_to_remove = [
                k for k in self._last_alert_times if k.startswith(f"{alert_type.value}:")
            ]
            for key in keys_to_remove:
                del self._last_alert_times[key]
