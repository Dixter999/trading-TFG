"""
Database connection pool managers for dual database access.

This module provides connection pooling for both markets and ai_model databases
using psycopg2's ThreadedConnectionPool.

TDD Phase: REFACTOR - Improved implementation with better structure.
"""

from contextlib import contextmanager
from enum import Enum
from typing import Dict, Any, List
import psycopg2.pool
from psycopg2.extensions import connection as Connection


class PoolType(Enum):
    """Enumeration of available database pools."""
    MARKETS = "markets"
    AI_MODEL = "ai_model"


class PoolManager:
    """
    Manages dual database connection pools for markets and ai_model databases.

    Provides thread-safe connection pooling with context managers for automatic
    connection lifecycle management.
    """

    def __init__(
        self,
        markets_config: Dict[str, Any],
        ai_model_config: Dict[str, Any],
        minconn: int = 1,
        maxconn: int = 10
    ):
        """
        Initialize dual connection pools.

        Args:
            markets_config: Dictionary with keys: host, database, user, password
            ai_model_config: Dictionary with keys: host, database, user, password
            minconn: Minimum number of connections in pool (default: 1)
            maxconn: Maximum number of connections in pool (default: 10)
        """
        self._minconn = minconn
        self._maxconn = maxconn

        # Track active connections for leak detection
        self._active_connections: Dict[PoolType, List[Connection]] = {
            PoolType.MARKETS: [],
            PoolType.AI_MODEL: []
        }

        # Create connection pools
        self._pools: Dict[PoolType, psycopg2.pool.ThreadedConnectionPool] = {
            PoolType.MARKETS: self._create_pool(markets_config),
            PoolType.AI_MODEL: self._create_pool(ai_model_config)
        }

    def _create_pool(self, config: Dict[str, Any]) -> psycopg2.pool.ThreadedConnectionPool:
        """
        Create a ThreadedConnectionPool with the given configuration.

        Args:
            config: Dictionary with keys: host, database, user, password, port

        Returns:
            Configured ThreadedConnectionPool
        """
        return psycopg2.pool.ThreadedConnectionPool(
            minconn=self._minconn,
            maxconn=self._maxconn,
            host=config['host'],
            port=config['port'],  # Include port for PostgreSQL connection
            database=config['database'],
            user=config['user'],
            password=config['password']
        )

    def _get_connection(self, pool_type: PoolType) -> Connection:
        """
        Generic method to get a connection from specified pool.

        Args:
            pool_type: The pool to get connection from

        Returns:
            psycopg2 connection object

        Raises:
            psycopg2.pool.PoolError: If pool is exhausted
        """
        pool = self._pools[pool_type]
        conn = pool.getconn()
        self._active_connections[pool_type].append(conn)
        return conn

    def _return_connection(self, pool_type: PoolType, conn: Connection) -> None:
        """
        Generic method to return a connection to specified pool.

        Args:
            pool_type: The pool to return connection to
            conn: Connection to return
        """
        pool = self._pools[pool_type]
        pool.putconn(conn)
        if conn in self._active_connections[pool_type]:
            self._active_connections[pool_type].remove(conn)

    def get_markets_connection(self) -> Connection:
        """
        Get a connection from the markets database pool.

        Returns:
            psycopg2 connection object

        Raises:
            psycopg2.pool.PoolError: If pool is exhausted
        """
        return self._get_connection(PoolType.MARKETS)

    def get_ai_model_connection(self) -> Connection:
        """
        Get a connection from the ai_model database pool.

        Returns:
            psycopg2 connection object

        Raises:
            psycopg2.pool.PoolError: If pool is exhausted
        """
        return self._get_connection(PoolType.AI_MODEL)

    def return_markets_connection(self, conn: Connection) -> None:
        """
        Return a connection to the markets database pool.

        Args:
            conn: Connection to return
        """
        self._return_connection(PoolType.MARKETS, conn)

    def return_ai_model_connection(self, conn: Connection) -> None:
        """
        Return a connection to the ai_model database pool.

        Args:
            conn: Connection to return
        """
        self._return_connection(PoolType.AI_MODEL, conn)

    @contextmanager
    def get_markets_connection_ctx(self):
        """
        Context manager for markets database connection.

        Automatically returns connection when exiting context, even on exception.

        Example:
            with pool_manager.get_markets_connection_ctx() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM eurusd_h1_rates")
        """
        conn = self.get_markets_connection()
        try:
            yield conn
        finally:
            self.return_markets_connection(conn)

    @contextmanager
    def get_ai_model_connection_ctx(self):
        """
        Context manager for ai_model database connection.

        Automatically returns connection when exiting context, even on exception.

        Example:
            with pool_manager.get_ai_model_connection_ctx() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM technical_indicators")
        """
        conn = self.get_ai_model_connection()
        try:
            yield conn
        finally:
            self.return_ai_model_connection(conn)

    def get_active_connections_count(self, pool_name: str) -> int:
        """
        Get count of active (not returned) connections for leak detection.

        Args:
            pool_name: Either 'markets' or 'ai_model'

        Returns:
            Number of active connections

        Raises:
            ValueError: If pool_name is not valid
        """
        try:
            pool_type = PoolType(pool_name)
            return len(self._active_connections[pool_type])
        except ValueError:
            raise ValueError(
                f"Unknown pool name: {pool_name}. "
                f"Valid options: {', '.join([pt.value for pt in PoolType])}"
            )

    def close_all(self) -> None:
        """
        Close both connection pools.

        This should be called when shutting down the application.
        """
        for pool in self._pools.values():
            pool.closeall()
