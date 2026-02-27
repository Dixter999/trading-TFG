"""
Alerting configuration for MT5 Live Trading System.

This module provides alerting rules and notifications for critical
events in the live trading system.

Alert types:
- System health alerts
- Trading operation alerts
- API error alerts
- Position discrepancy alerts
- Circuit breaker alerts
"""

import asyncio
import logging
import smtplib
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class Alert(ABC):
    """Base class for alerts."""

    def __init__(self, message: str, severity: str = "warning"):
        self.message = message
        self.severity = severity
        self.timestamp = datetime.utcnow()
        self.resolved = False

    @abstractmethod
    async def send(self) -> bool:
        """Send the alert."""
        pass


class EmailAlert(Alert):
    """Email alert implementation."""

    def __init__(
        self,
        message: str,
        severity: str = "warning",
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        smtp_user: str = "",
        smtp_password: str = "",
        recipients: List[str] = None,
    ):
        super().__init__(message, severity)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.recipients = recipients or []

    async def send(self) -> bool:
        """Send email alert."""
        try:
            # Create message
            msg = MimeMultipart()
            msg["From"] = self.smtp_user
            msg["To"] = ", ".join(self.recipients)
            msg["Subject"] = f"[{self.severity.upper()}] MT5 Live Trading Alert"

            # Email body
            body = f"""
            MT5 Live Trading System Alert
            
            Severity: {self.severity.upper()}
            Time: {self.timestamp.isoformat()}
            Message: {self.message}
            
            This is an automated alert from the MT5 Live Trading System.
            """

            msg.attach(MimeText(body, "plain"))

            # Send email
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            if self.smtp_user:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)

            server.send_message(msg)
            server.quit()

            logger.info(f"Email alert sent: {self.message}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class SlackAlert(Alert):
    """Slack alert implementation."""

    def __init__(
        self,
        message: str,
        severity: str = "warning",
        webhook_url: str = "",
        channel: str = "#alerts",
    ):
        super().__init__(message, severity)
        self.webhook_url = webhook_url
        self.channel = channel

    async def send(self) -> bool:
        """Send Slack alert."""
        try:
            # Color based on severity
            color_map = {
                "info": "#36a64f",  # green
                "warning": "#ff9500",  # orange
                "error": "#ff0000",  # red
                "critical": "#8b0000",  # dark red
            }
            color = color_map.get(self.severity, "#ff9500")

            # Create payload
            payload = {
                "channel": self.channel,
                "username": "MT5 Trading Bot",
                "icon_emoji": ":chart_with_upwards_trend:",
                "attachments": [
                    {
                        "color": color,
                        "title": f"MT5 Live Trading Alert - {self.severity.upper()}",
                        "text": self.message,
                        "ts": int(self.timestamp.timestamp()),
                    }
                ],
            }

            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Slack alert sent: {self.message}")
                        return True
                    else:
                        logger.error(f"Failed to send Slack alert: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class AlertManager:
    """Manages alerting for the live trading system."""

    def __init__(self, config: Dict):
        self.config = config
        self.alert_history = []
        self.alert_cooldowns = {}

        # Alert configurations
        self.email_config = config.get("email", {})
        self.slack_config = config.get("slack", {})

        # Initialize alert channels
        self.alert_channels = []
        if self.email_config.get("enabled", False):
            self.alert_channels.append("email")
        if self.slack_config.get("enabled", False):
            self.alert_channels.append("slack")

    async def send_alert(
        self,
        message: str,
        severity: str = "warning",
        alert_type: str = "general",
        cooldown_minutes: int = 5,
    ) -> bool:
        """Send alert through configured channels."""

        # Check cooldown
        if self._is_on_cooldown(alert_type, cooldown_minutes):
            logger.debug(f"Alert {alert_type} is on cooldown")
            return False

        # Record alert
        alert_record = {
            "message": message,
            "severity": severity,
            "type": alert_type,
            "timestamp": datetime.utcnow(),
        }
        self.alert_history.append(alert_record)

        # Send through each channel
        success = True
        for channel in self.alert_channels:
            try:
                if channel == "email":
                    alert = EmailAlert(
                        message=message,
                        severity=severity,
                        smtp_host=self.email_config.get("smtp_host"),
                        smtp_port=self.email_config.get("smtp_port", 587),
                        smtp_user=self.email_config.get("smtp_user"),
                        smtp_password=self.email_config.get("smtp_password"),
                        recipients=self.email_config.get("recipients", []),
                    )
                elif channel == "slack":
                    alert = SlackAlert(
                        message=message,
                        severity=severity,
                        webhook_url=self.slack_config.get("webhook_url"),
                        channel=self.slack_config.get("channel", "#alerts"),
                    )
                else:
                    continue

                channel_success = await alert.send()
                success = success and channel_success

            except Exception as e:
                logger.error(f"Failed to send {channel} alert: {e}")
                success = False

        # Update cooldown
        self.alert_cooldowns[alert_type] = datetime.utcnow()

        return success

    def _is_on_cooldown(self, alert_type: str, cooldown_minutes: int) -> bool:
        """Check if alert type is on cooldown."""
        if alert_type not in self.alert_cooldowns:
            return False

        last_alert = self.alert_cooldowns[alert_type]
        cooldown_until = last_alert + timedelta(minutes=cooldown_minutes)

        return datetime.utcnow() < cooldown_until

    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [a for a in self.alert_history if a["timestamp"] > cutoff]


class TradingAlertRules:
    """Alert rules for live trading system."""

    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.last_health_check = datetime.utcnow()
        self.last_position_check = datetime.utcnow()

    async def check_system_health(self, metrics: Dict):
        """Check system health and send alerts if needed."""
        try:
            # Circuit breaker open
            if metrics.get("circuit_breaker_state") == 1:  # OPEN
                await self.alert_manager.send_alert(
                    "Circuit breaker is OPEN - trading is suspended",
                    severity="error",
                    alert_type="circuit_breaker_open",
                    cooldown_minutes=30,
                )

            # High API error rate
            error_rate = metrics.get("api_error_rate", 0)
            if error_rate > 0.1:  # 10% error rate
                await self.alert_manager.send_alert(
                    f"High API error rate: {error_rate:.1%}",
                    severity="error",
                    alert_type="high_error_rate",
                    cooldown_minutes=15,
                )

            # System unhealthy
            if metrics.get("system_health") == 0:
                await self.alert_manager.send_alert(
                    "System health check failed - multiple issues detected",
                    severity="critical",
                    alert_type="system_unhealthy",
                    cooldown_minutes=60,
                )

            # Too many position discrepancies
            discrepancies = metrics.get("position_discrepancies", {})
            total_discrepancies = sum(discrepancies.values())
            if total_discrepancies > 5:
                await self.alert_manager.send_alert(
                    f"High number of position discrepancies: {total_discrepancies}",
                    severity="warning",
                    alert_type="position_discrepancies",
                    cooldown_minutes=30,
                )

            # Long time since last trade (if active)
            last_trade = metrics.get("last_trade_timestamp", 0)
            if last_trade > 0:
                time_since = datetime.utcnow().timestamp() - last_trade
                if time_since > 7200:  # 2 hours
                    await self.alert_manager.send_alert(
                        f"No trades executed for {time_since / 3600:.1f} hours",
                        severity="info",
                        alert_type="no_trades",
                        cooldown_minutes=240,  # 4 hours
                    )

            self.last_health_check = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error checking system health: {e}")

    async def check_position_alerts(self, positions: List[Dict]):
        """Check position-related alerts."""
        try:
            # Too many open positions
            if len(positions) > 10:
                await self.alert_manager.send_alert(
                    f"High number of open positions: {len(positions)}",
                    severity="warning",
                    alert_type="too_many_positions",
                    cooldown_minutes=60,
                )

            # Large losses
            for pos in positions:
                if pos.get("profit", 0) < -100:  # $100 loss
                    await self.alert_manager.send_alert(
                        f"Large loss detected on {pos.get('symbol')} {pos.get('ticket')}: ${pos.get('profit')}",
                        severity="error",
                        alert_type="large_loss",
                        cooldown_minutes=30,
                    )

            # Large profits
            for pos in positions:
                if pos.get("profit", 0) > 500:  # $500 profit
                    await self.alert_manager.send_alert(
                        f"Large profit on {pos.get('symbol')} {pos.get('ticket')}: ${pos.get('profit')}",
                        severity="info",
                        alert_type="large_profit",
                        cooldown_minutes=60,
                    )

            self.last_position_check = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error checking position alerts: {e}")

    async def check_trade_execution_alerts(self, trade_result: Dict):
        """Check trade execution for alerts."""
        try:
            result = trade_result.get("result", "unknown")
            symbol = trade_result.get("symbol", "unknown")
            direction = trade_result.get("direction", "unknown")

            if result == "failed":
                error_msg = trade_result.get("error", "Unknown error")
                await self.alert_manager.send_alert(
                    f"Trade failed for {symbol} {direction}: {error_msg}",
                    severity="error",
                    alert_type="trade_failed",
                    cooldown_minutes=5,
                )

            elif result == "success":
                # Log successful trades (info level)
                await self.alert_manager.send_alert(
                    f"Trade executed: {symbol} {direction}",
                    severity="info",
                    alert_type="trade_success",
                    cooldown_minutes=1,
                )

        except Exception as e:
            logger.error(f"Error checking trade alerts: {e}")


def create_alert_config_template() -> Dict:
    """Create template for alert configuration."""

    config = {
        "email": {
            "enabled": False,
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_user": "your-email@gmail.com",
            "smtp_password": "your-app-password",
            "recipients": ["trader@example.com", "ops@example.com"],
        },
        "slack": {
            "enabled": False,
            "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
            "channel": "#trading-alerts",
        },
        "alert_rules": {
            "circuit_breaker_open": {
                "enabled": True,
                "severity": "error",
                "cooldown_minutes": 30,
            },
            "high_error_rate": {
                "enabled": True,
                "threshold": 0.1,  # 10%
                "severity": "error",
                "cooldown_minutes": 15,
            },
            "system_unhealthy": {
                "enabled": True,
                "severity": "critical",
                "cooldown_minutes": 60,
            },
            "position_discrepancies": {
                "enabled": True,
                "threshold": 5,
                "severity": "warning",
                "cooldown_minutes": 30,
            },
            "no_trades": {
                "enabled": True,
                "threshold_hours": 2,
                "severity": "info",
                "cooldown_minutes": 240,
            },
            "too_many_positions": {
                "enabled": True,
                "threshold": 10,
                "severity": "warning",
                "cooldown_minutes": 60,
            },
            "large_loss": {
                "enabled": True,
                "threshold": -100,  # $100
                "severity": "error",
                "cooldown_minutes": 30,
            },
            "large_profit": {
                "enabled": True,
                "threshold": 500,  # $500
                "severity": "info",
                "cooldown_minutes": 60,
            },
        },
    }

    return config


# Export alert configuration template for documentation
ALERT_CONFIG_TEMPLATE = create_alert_config_template()
