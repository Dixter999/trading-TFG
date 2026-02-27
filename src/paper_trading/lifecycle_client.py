"""
Lifecycle client for paper trading hot reload (Issue #618).

This module provides a lightweight client to query signal lifecycle states
from PostgreSQL, enabling the hot reload functionality to filter signals
by their lifecycle state.

Database: ai_model
Table: signal_lifecycle (created by migration 028)

Usage:
    client = LifecycleClient()
    active_signals = client.get_active_signal_ids()
    is_tradeable = client.is_signal_tradeable("gbpusd_long_Stoch_RSI_H1")
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional, Set

from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

# Tradeable states - signals in these states can be loaded for trading
TRADEABLE_STATES = {"active", "degraded"}


class LifecycleClient:
    """Client to query signal lifecycle states from PostgreSQL.

    Provides read operations for lifecycle state management used by
    the paper trading hot reload functionality.

    Attributes:
        _host: Database host
        _port: Database port
        _database: Database name
        _user: Database user
        _password: Database password
        _connection_string: Full connection string
        _engine: SQLAlchemy engine (lazy-loaded)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "ai_model",
        user: str = "tfg_user",
        password: str = "tfg_password",
    ) -> None:
        """Initialize database connection parameters.

        Args:
            host: Database host (default: localhost)
            port: Database port (default: 5432)
            database: Database name (default: ai_model)
            user: Database user (default: tfg_user)
            password: Database password (default: tfg_password)
        """
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password

        self._connection_string = (
            f"postgresql://{user}:{password}@{host}:{port}/{database}"
        )

        # Lazy-loaded engine
        self._engine = None

        logger.debug(
            f"LifecycleClient initialized for {host}:{port}/{database}"
        )

    @classmethod
    def from_env(cls) -> "LifecycleClient":
        """Create LifecycleClient from environment variables.

        Environment Variables:
            LIFECYCLE_DB_HOST: Database host (default: localhost)
            LIFECYCLE_DB_PORT: Database port (default: 5432)
            LIFECYCLE_DB_NAME: Database name (default: ai_model)
            LIFECYCLE_DB_USER: Database user (default: tfg_user)
            LIFECYCLE_DB_PASSWORD: Database password (default: tfg_password)

        Returns:
            LifecycleClient instance configured from environment
        """
        return cls(
            host=os.environ.get("AI_MODEL_DB_HOST", "localhost"),
            port=int(os.environ.get("AI_MODEL_DB_PORT", "5432")),
            database=os.environ.get("AI_MODEL_DB_NAME", "ai_model"),
            user=os.environ.get("AI_MODEL_DB_USER", "tfg_user"),
            password=os.environ.get("AI_MODEL_DB_PASSWORD", "tfg_password"),
        )

    def _get_engine(self):
        """Get or create SQLAlchemy engine (lazy initialization).

        Returns:
            SQLAlchemy engine instance
        """
        if self._engine is None:
            self._engine = create_engine(
                self._connection_string,
                poolclass=QueuePool,
                pool_size=2,
                max_overflow=5,
                pool_pre_ping=True,
            )
        return self._engine

    def validate_connection(self) -> bool:
        """Validate database connection.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.debug("LifecycleClient connection validated")
            return True
        except Exception as e:
            logger.warning(f"LifecycleClient connection validation failed: {e}")
            return False

    def get_active_signal_ids(self) -> Set[str]:
        """Get set of signal_ids that are ACTIVE.

        Queries the signal_lifecycle table for signals with
        lifecycle_state = 'active'.

        Returns:
            Set of signal IDs like {'gbpusd_long_Stoch_RSI_H1', ...}
        """
        query = text("""
            SELECT signal_id
            FROM signal_lifecycle
            WHERE lifecycle_state = 'active'
        """)

        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                result = conn.execute(query)
                return {row[0] for row in result}
        except Exception as e:
            logger.warning(f"Failed to get active signal IDs: {e}")
            return set()

    def get_tradeable_signal_ids(self) -> Set[str]:
        """Get set of signal_ids that are tradeable (ACTIVE or DEGRADED).

        Signals in ACTIVE or DEGRADED states can still be loaded for trading.
        Only QUARANTINED/RETIRED signals should be excluded.

        Returns:
            Set of signal IDs like {'gbpusd_long_Stoch_RSI_H1', ...}
        """
        query = text("""
            SELECT signal_id
            FROM signal_lifecycle
            WHERE lifecycle_state IN ('active', 'degraded')
        """)

        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                result = conn.execute(query)
                return {row[0] for row in result}
        except Exception as e:
            logger.warning(f"Failed to get tradeable signal IDs: {e}")
            return set()

    def get_signal_state(self, signal_id: str) -> Optional[str]:
        """Get lifecycle state for a single signal.

        Args:
            signal_id: Unique signal identifier

        Returns:
            State string ('active', 'degraded', 'quarantined', etc.)
            or None if not found
        """
        query = text("""
            SELECT lifecycle_state
            FROM signal_lifecycle
            WHERE signal_id = :signal_id
        """)

        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                result = conn.execute(query, {"signal_id": signal_id})
                row = result.fetchone()
                if row:
                    return row[0]
                return None
        except Exception as e:
            logger.warning(f"Failed to get signal state for {signal_id}: {e}")
            return None

    def get_model_hash(self, signal_id: str) -> Optional[str]:
        """Get stored model hash for a signal.

        Used to detect retrained models by comparing hashes.

        Args:
            signal_id: Unique signal identifier

        Returns:
            Model hash string or None if not found
        """
        query = text("""
            SELECT model_hash
            FROM signal_lifecycle
            WHERE signal_id = :signal_id
        """)

        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                result = conn.execute(query, {"signal_id": signal_id})
                row = result.fetchone()
                if row:
                    return row[0]
                return None
        except Exception as e:
            logger.warning(f"Failed to get model hash for {signal_id}: {e}")
            return None

    def update_model_hash(self, signal_id: str, model_hash: str) -> bool:
        """Update model hash after loading new version.

        Args:
            signal_id: Unique signal identifier
            model_hash: New model hash value

        Returns:
            True if update successful, False otherwise
        """
        query = text("""
            UPDATE signal_lifecycle
            SET model_hash = :model_hash
            WHERE signal_id = :signal_id
        """)

        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                result = conn.execute(
                    query,
                    {"signal_id": signal_id, "model_hash": model_hash}
                )
                conn.commit()

                if result.rowcount == 0:
                    logger.debug(f"Signal not found for hash update: {signal_id}")
                    return False

                logger.debug(f"Updated model hash for {signal_id}")
                return True
        except Exception as e:
            logger.warning(f"Failed to update model hash for {signal_id}: {e}")
            return False

    def is_signal_tradeable(self, signal_id: str) -> bool:
        """Check if signal is tradeable.

        Fail-open policy (Issue #661): Signals NOT FOUND in the lifecycle
        table are treated as tradeable. All configured signals passed Phase 5
        approval, which is permanent (Issue #645). Only signals explicitly
        marked as quarantined/retired are blocked.

        Args:
            signal_id: Unique signal identifier

        Returns:
            True if signal is tradeable, False only if explicitly non-tradeable
        """
        state = self.get_signal_state(signal_id)
        if state is None:
            # Fail-open: signal not in lifecycle table but in config = tradeable
            logger.debug(
                f"Signal {signal_id} not in lifecycle table, allowing (fail-open)"
            )
            return True
        return state in TRADEABLE_STATES

    def report_signal_performance(
        self, signal_id: str, current_wr: float, total_trades: int
    ) -> bool:
        """Report live performance metrics for a signal.

        Updates the performance_metrics JSONB column in signal_lifecycle
        with current win rate and trade count from paper trading.

        Args:
            signal_id: Unique signal identifier
            current_wr: Current rolling win rate (0.0-1.0)
            total_trades: Total number of trades for this signal

        Returns:
            True if update successful, False otherwise
        """
        query = text("""
            UPDATE signal_lifecycle
            SET performance_metrics = jsonb_set(
                COALESCE(performance_metrics, '{}'),
                '{live}',
                :metrics::jsonb
            )
            WHERE signal_id = :signal_id
        """)

        import json
        metrics_json = json.dumps({
            "current_wr": round(current_wr, 4),
            "total_trades": total_trades,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        })

        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                result = conn.execute(
                    query,
                    {"signal_id": signal_id, "metrics": metrics_json}
                )
                conn.commit()

                if result.rowcount == 0:
                    logger.debug(f"Signal not found for performance update: {signal_id}")
                    return False

                logger.debug(
                    f"Reported performance for {signal_id}: WR={current_wr:.1%}, trades={total_trades}"
                )
                return True
        except Exception as e:
            logger.warning(f"Failed to report performance for {signal_id}: {e}")
            return False

    @staticmethod
    def calculate_rolling_wr(wins: int, total: int) -> float:
        """Calculate rolling win rate from win/total counts.

        Args:
            wins: Number of winning trades
            total: Total number of trades

        Returns:
            Win rate as float (0.0-1.0), or 0.0 if no trades
        """
        if total == 0:
            return 0.0
        return wins / total

    @staticmethod
    def should_report_performance(trade_count: int) -> bool:
        """Check if performance should be reported (every 10th trade).

        Args:
            trade_count: Current trade count for the signal

        Returns:
            True if should report, False otherwise
        """
        if trade_count == 0:
            return False
        return trade_count % 10 == 0
