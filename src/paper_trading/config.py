"""
Configuration Management for Paper Trading Engine (Issue #430).

This module provides a centralized configuration system using Pydantic that allows:
- Enabling/disabling symbols
- Adjusting risk parameters
- Managing alert thresholds
- Environment variable overrides
- Bidirectional trading support (Issue #510)

All configuration changes can be made without code modifications.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class SignalConfig(BaseModel):
    """Configuration for a single trading signal (LONG or SHORT).

    Supports bidirectional trading by allowing multiple signals per symbol.

    Attributes:
        signal: Signal type name (e.g., "EMA12_cross_EMA26_down")
        direction: Signal direction ("long" or "short")
        timeframe: Training timeframe for this signal (M30, H1, H2, H3, H4, H6, H8, H12, D1)
                   Optional for backward compatibility - falls back to symbol timeframe
        enabled: Whether this signal is actively monitored
        use_rl_exit: Whether to use RL model for exit decisions (default True)
        best_fold: Best fold number from training (optional)
        oos_wr: Out-of-sample win rate from signal discovery
        profit_factor: Profit factor from Phase 5 testing (optional)
        model_version: Model version identifier (e.g., "hybrid_v4")
    """

    signal: str
    direction: str
    timeframe: Optional[str] = None
    enabled: bool = True
    use_rl_exit: bool = True
    best_fold: Optional[int] = None
    oos_wr: Optional[float] = None
    profit_factor: Optional[float] = None
    model_version: Optional[str] = None

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        """Validate direction is long or short."""
        v = v.lower()
        if v not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got '{v}'")
        return v

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v: Optional[str]) -> Optional[str]:
        """Validate timeframe is a valid trading timeframe.

        Valid timeframes: M30, H1, H2, H3, H4, H6, H8, H12, D1
        None is allowed for backward compatibility (falls back to symbol default).
        """
        if v is None:
            return None
        v = v.upper()
        valid_timeframes = ("M30", "H1", "H2", "H3", "H4", "H6", "H8", "H12", "D1")
        if v not in valid_timeframes:
            raise ValueError(
                f"Timeframe must be one of {valid_timeframes}, got '{v}'"
            )
        return v


class SymbolConfig(BaseModel):
    """Configuration for a single trading symbol.

    Updated to support bidirectional trading (Issue #510).

    Attributes:
        enabled: Whether the symbol is enabled for trading
        max_position_size: Maximum position size in lots (0.0-1.0)
        timeframe: Entry signal timeframe (H4 or D1, default D1)
        signals: List of signal configurations (supports bidirectional)
        exit_strategy: Exit strategy name for logging/audit (default "hybrid_v4")

    Legacy attributes (backward compatibility):
        signal: Old single signal format (deprecated)
        signal_direction: Old single direction format (deprecated)
        use_rl_exit: Old single RL exit flag (deprecated)
    """

    enabled: bool = True
    max_position_size: float = Field(default=0.1, gt=0.0, le=1.0)
    timeframe: str = Field(default="D1")
    signals: List[SignalConfig] = Field(default_factory=list)
    exit_strategy: str = "hybrid_v4"

    # Legacy fields for backward compatibility
    signal: Optional[str] = None
    signal_direction: Optional[str] = None
    use_rl_exit: bool = True

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        """Validate timeframe is H4 or D1."""
        v = v.upper()
        if v not in ("H4", "D1"):
            raise ValueError(f"Timeframe must be 'H4' or 'D1', got '{v}'")
        return v

    @model_validator(mode="after")
    def convert_legacy_format(self) -> "SymbolConfig":
        """Convert legacy single-signal format to new signals list format.

        If signals list is empty but signal/signal_direction are provided,
        convert to new format for backward compatibility.
        """
        if not self.signals and self.signal and self.signal_direction:
            # Convert legacy format to new format
            self.signals = [
                SignalConfig(
                    signal=self.signal,
                    direction=self.signal_direction,
                    enabled=True,
                    use_rl_exit=self.use_rl_exit,
                )
            ]
        return self

    def get_active_signals(self) -> List[SignalConfig]:
        """Get list of enabled signals for this symbol.

        Returns:
            List of SignalConfig objects where enabled=True
        """
        return [sig for sig in self.signals if sig.enabled]


class RiskConfig(BaseModel):
    """Risk management configuration parameters.

    Attributes:
        max_concurrent_positions: Maximum number of open positions (dynamic, derived from signal count)
        max_positions_per_symbol: Maximum positions per symbol (default: 8)
        max_daily_trades: Maximum trades allowed per day (default: 30)
        max_daily_loss_pips: Maximum allowed daily loss in pips (default: 150)
        risk_per_trade_pct: Risk percentage per trade (default: 0.01 = 1%)
    """

    max_concurrent_positions: int = Field(default=25, gt=0)
    max_positions_per_symbol: int = Field(default=8, ge=1, le=16)
    max_daily_trades: int = Field(default=30, ge=0)
    max_daily_loss_pips: int = Field(default=150, ge=0)
    risk_per_trade_pct: float = Field(default=0.01, ge=0.0, le=1.0)
    lookahead_candles: int = Field(default=6, ge=1, le=24)
    signal_hit_rate: float = Field(default=0.04, ge=0.001, le=0.5)


class ExitScaffoldConfig(BaseModel):
    """Exit scaffold configuration - FROZEN at 30/30.

    The 30/30 scaffold is intentionally frozen and cannot be modified.
    This ensures consistent backtesting and forward testing conditions.

    Attributes:
        sl_pips: Stop loss distance in pips (frozen at 30)
        tp_pips: Take profit distance in pips (frozen at 30)
        confidence_threshold: Minimum RL ensemble consensus to trigger CLOSE (0.50-0.90).
            Default 0.55 filters out coin-flip exits (50-55% bucket).
    """

    sl_pips: int = 30
    tp_pips: int = 30
    confidence_threshold: float = Field(default=0.55, ge=0.50, le=0.90)

    @field_validator("sl_pips", "tp_pips")
    @classmethod
    def validate_frozen_value(cls, v: int, info) -> int:
        """Validate that exit scaffold values are frozen at 30 pips."""
        if v != 30:
            raise ValueError(f"Exit scaffold is frozen at 30 pips, got {v}")
        return v


class AlertsConfig(BaseModel):
    """Alert threshold configuration.

    Attributes:
        drawdown_warning_pips: Pips of drawdown before warning (default: 50)
        drawdown_critical_pips: Pips of drawdown before critical alert (default: 80)
        inactivity_hours: Hours of inactivity before alert (default: 4)
        low_win_rate_threshold: Win rate threshold for alert (default: 0.30)
    """

    drawdown_warning_pips: int = Field(default=50, ge=0)
    drawdown_critical_pips: int = Field(default=80, ge=0)
    inactivity_hours: int = Field(default=4, ge=0)
    low_win_rate_threshold: float = Field(default=0.30, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_warning_less_than_critical(self) -> "AlertsConfig":
        """Ensure warning threshold is less than critical threshold."""
        if self.drawdown_warning_pips >= self.drawdown_critical_pips:
            raise ValueError(
                f"drawdown_warning_pips ({self.drawdown_warning_pips}) must be "
                f"less than drawdown_critical_pips ({self.drawdown_critical_pips})"
            )
        return self


class DatabaseConfig(BaseModel):
    """Database connection configuration.

    Attributes:
        host: Database hostname or IP address
        port: Database port (default: 5432)
        name: Database name
        user: Database username
        password: Database password
    """

    host: str
    port: int = Field(default=5432, ge=1, le=65535)
    name: str
    user: str
    password: str

    @property
    def connection_string(self) -> str:
        """Generate a PostgreSQL connection string.

        Returns:
            PostgreSQL connection string in format:
            postgresql://user:password@host:port/name
        """
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class DatabasesConfig(BaseModel):
    """Container for multiple database configurations.

    Holds configuration for both the markets database (OHLCV data, indicators)
    and the AI model database (order blocks, pattern data).

    Attributes:
        markets: Configuration for the markets database
        ai_model: Configuration for the AI model database
    """

    markets: DatabaseConfig
    ai_model: DatabaseConfig


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker failure protection (Issue #596).

    Attributes:
        failure_threshold: Number of failures before opening circuit (default: 5)
        timeout_seconds: Timeout in seconds before attempting half-open (default: 300)
        half_open_attempts: Number of half-open test requests (default: 2)
    """

    failure_threshold: int = Field(default=5, ge=1)
    timeout_seconds: int = Field(default=300, ge=1)
    half_open_attempts: int = Field(default=2, ge=1)


class LiveTradingConfig(BaseModel):
    """Configuration for live trading (disabled in TFG).

    Attributes:
        enabled: Whether live trading is enabled (default: False, always False in TFG)
        gateway_url: Not used in TFG
        gateway_timeout: HTTP timeout for gateway requests in seconds (default: 10)
        max_position_size: Maximum position size in lots (default: 0.01)
        sync_interval: Position sync interval in seconds (default: 60)
        pre_trade_sync: Sync with broker before opening position (default: True)
        symbol_mapping: Mapping of internal symbols to broker symbols (default: {})
    """

    enabled: bool = Field(default=False)
    gateway_url: str = Field(default="http://localhost:8001")  # TFG: not used
    gateway_timeout: int = Field(default=10, ge=1, le=60)
    max_position_size: float = Field(default=0.01, gt=0, le=100)
    sync_interval: int = Field(default=60, ge=10, le=3600)
    pre_trade_sync: bool = Field(default=True)
    symbol_mapping: dict[str, str] = Field(default_factory=dict)


class AccountConfig(BaseModel):
    """Account configuration for dynamic position sizing (Issue #631).

    Attributes:
        initial_balance: Initial account balance in account currency
        currency: Account currency code (USD, PLN, EUR, etc.)
        exchange_rate_to_usd: Exchange rate to convert to USD for margin calculations
    """

    initial_balance: float = Field(default=10000.0, gt=0)
    currency: str = Field(default="USD")
    exchange_rate_to_usd: float = Field(default=1.0, gt=0)


class PaperTradingConfig(BaseModel):
    """Main paper trading configuration.

    This is the root configuration class that contains all sub-configurations
    for the paper trading system.

    Attributes:
        enabled: Whether paper trading is enabled (default: True)
        poll_interval: Polling interval in seconds (default: 60)
        symbols: Dictionary of symbol configurations
        risk: Risk management configuration
        exit_scaffold: Exit scaffold configuration (frozen 30/30)
        alerts: Alert threshold configuration
        database: Optional database configurations (markets + ai_model)
        live_trading: Live trading configuration (Issue #596)
        circuit_breaker: Circuit breaker configuration (Issue #596)
        account: Account configuration for dynamic position sizing (Issue #631)
    """

    enabled: bool = True
    poll_interval: int = Field(default=60, gt=0)
    symbols: Dict[str, SymbolConfig]
    risk: RiskConfig
    exit_scaffold: ExitScaffoldConfig
    alerts: AlertsConfig
    database: Optional[DatabasesConfig] = None
    live_trading: LiveTradingConfig = Field(default_factory=LiveTradingConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    account: AccountConfig = Field(default_factory=AccountConfig)

    @field_validator("symbols")
    @classmethod
    def validate_at_least_one_symbol(
        cls, v: Dict[str, SymbolConfig]
    ) -> Dict[str, SymbolConfig]:
        """Ensure at least one symbol is configured."""
        if not v:
            raise ValueError("Configuration must have at least one symbol")
        return v

    @classmethod
    def from_yaml(cls, path: str) -> "PaperTradingConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file

        Returns:
            PaperTradingConfig instance

        Raises:
            FileNotFoundError: If the file does not exist
            yaml.YAMLError: If the YAML is invalid
            ValidationError: If the configuration is invalid
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        # Extract configuration from YAML structure
        paper_trading_section = data.get("paper_trading", {})
        symbols_section = data.get("symbols", {})
        risk_section = data.get("risk", {})
        exit_scaffold_section = data.get("exit_scaffold", {})
        alerts_section = data.get("alerts", {})
        database_section = data.get("database", {})
        live_trading_section = data.get("live_trading", {})
        circuit_breaker_section = data.get("circuit_breaker", {})
        account_section = data.get("account", {})

        # Build symbols dict with SymbolConfig instances
        symbols = {}
        for symbol_name, symbol_data in symbols_section.items():
            if symbol_data is None:
                symbols[symbol_name] = SymbolConfig()
            else:
                symbols[symbol_name] = SymbolConfig(**symbol_data)

        # Build database config if present
        database_config = None
        if database_section:
            markets_data = database_section.get("markets", {})
            ai_model_data = database_section.get("ai_model", {})
            if markets_data and ai_model_data:
                database_config = DatabasesConfig(
                    markets=DatabaseConfig(**markets_data),
                    ai_model=DatabaseConfig(**ai_model_data),
                )

        return cls(
            enabled=paper_trading_section.get("enabled", True),
            poll_interval=paper_trading_section.get("poll_interval", 60),
            symbols=symbols,
            risk=RiskConfig(**risk_section) if risk_section else RiskConfig(),
            exit_scaffold=(
                ExitScaffoldConfig(**exit_scaffold_section)
                if exit_scaffold_section
                else ExitScaffoldConfig()
            ),
            alerts=AlertsConfig(**alerts_section) if alerts_section else AlertsConfig(),
            database=database_config,
            live_trading=(
                LiveTradingConfig(**live_trading_section)
                if live_trading_section
                else LiveTradingConfig()
            ),
            circuit_breaker=(
                CircuitBreakerConfig(**circuit_breaker_section)
                if circuit_breaker_section
                else CircuitBreakerConfig()
            ),
            account=(
                AccountConfig(**account_section)
                if account_section
                else AccountConfig()
            ),
        )

    @classmethod
    def from_env(cls) -> "PaperTradingConfig":
        """Load configuration from environment variables.

        First loads from YAML file (specified by PAPER_TRADING_CONFIG_PATH or default),
        then applies environment variable overrides.

        Environment variables:
            PAPER_TRADING_CONFIG_PATH: Path to YAML config (default: config/paper_trading.yaml)
            PAPER_TRADING_ENABLED: Override enabled flag
            PAPER_TRADING_POLL_INTERVAL: Override poll interval
            PAPER_TRADING_MAX_CONCURRENT_POSITIONS: Override max positions
            PAPER_TRADING_MAX_DAILY_TRADES: Override max daily trades
            PAPER_TRADING_SYMBOL_{SYMBOL}_ENABLED: Override symbol enabled

        Returns:
            PaperTradingConfig instance

        Raises:
            FileNotFoundError: If the config file does not exist
        """
        config_path = os.getenv(
            "PAPER_TRADING_CONFIG_PATH", "config/paper_trading.yaml"
        )

        # Load base config from YAML
        config = cls.from_yaml(config_path)

        # Apply environment variable overrides
        if os.getenv("PAPER_TRADING_ENABLED"):
            enabled_str = os.getenv("PAPER_TRADING_ENABLED", "").lower()
            config = config.model_copy(
                update={"enabled": enabled_str in ("true", "1", "yes")}
            )

        if os.getenv("PAPER_TRADING_POLL_INTERVAL"):
            poll_interval = int(os.getenv("PAPER_TRADING_POLL_INTERVAL", "60"))
            config = config.model_copy(update={"poll_interval": poll_interval})

        # Apply risk parameter overrides
        risk_updates = {}
        if os.getenv("PAPER_TRADING_MAX_CONCURRENT_POSITIONS"):
            risk_updates["max_concurrent_positions"] = int(
                os.getenv("PAPER_TRADING_MAX_CONCURRENT_POSITIONS", "8")
            )
        if os.getenv("PAPER_TRADING_MAX_DAILY_TRADES"):
            risk_updates["max_daily_trades"] = int(
                os.getenv("PAPER_TRADING_MAX_DAILY_TRADES", "40")
            )

        if risk_updates:
            new_risk = config.risk.model_copy(update=risk_updates)
            config = config.model_copy(update={"risk": new_risk})

        # Apply symbol-specific overrides
        symbols_copy = dict(config.symbols)
        for symbol_name in symbols_copy:
            symbol_updates = {}

            # Check for enabled override
            env_key = f"PAPER_TRADING_SYMBOL_{symbol_name}_ENABLED"
            if os.getenv(env_key):
                enabled_str = os.getenv(env_key, "").lower()
                symbol_updates["enabled"] = enabled_str in ("true", "1", "yes")

            # Check for timeframe override
            timeframe_key = f"PAPER_TRADING_SYMBOL_{symbol_name}_TIMEFRAME"
            if os.getenv(timeframe_key):
                timeframe = os.getenv(timeframe_key, "").upper()
                symbol_updates["timeframe"] = timeframe

            # Apply all updates for this symbol
            if symbol_updates:
                symbols_copy[symbol_name] = symbols_copy[symbol_name].model_copy(
                    update=symbol_updates
                )

        config = config.model_copy(update={"symbols": symbols_copy})

        return config

    def get_enabled_symbols(self) -> List[str]:
        """Get list of enabled symbol names.

        Returns:
            List of symbol names that are enabled
        """
        return [name for name, cfg in self.symbols.items() if cfg.enabled]

    def is_symbol_enabled(self, symbol: str) -> bool:
        """Check if a specific symbol is enabled.

        Args:
            symbol: Symbol name to check

        Returns:
            True if symbol is configured and enabled, False otherwise
        """
        if symbol not in self.symbols:
            return False
        return self.symbols[symbol].enabled

    def get_symbol_configs(self) -> Dict[str, Dict]:
        """Get symbol configurations as dict for runner.

        Returns:
            Dictionary mapping symbol names to their config dicts.
        """
        return {name: cfg.model_dump() for name, cfg in self.symbols.items()}

    def to_yaml(self, path: str) -> None:
        """Export configuration to a YAML file.

        Args:
            path: Path to write the YAML file
        """
        # Convert to dict with proper YAML structure
        data = {
            "paper_trading": {
                "enabled": self.enabled,
                "poll_interval": self.poll_interval,
            },
            "symbols": {name: cfg.model_dump() for name, cfg in self.symbols.items()},
            "risk": self.risk.model_dump(),
            "exit_scaffold": self.exit_scaffold.model_dump(),
            "alerts": self.alerts.model_dump(),
        }

        # Include database config if present
        if self.database is not None:
            data["database"] = {
                "markets": self.database.markets.model_dump(),
                "ai_model": self.database.ai_model.model_dump(),
            }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
