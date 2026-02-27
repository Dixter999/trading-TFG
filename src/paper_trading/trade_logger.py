"""
Trade Logger for Paper Trading Engine (Issue #328).

This module provides PostgreSQL logging for closed trades.

Table Schema (paper_trades):
    id SERIAL PRIMARY KEY
    symbol VARCHAR(20) NOT NULL
    direction VARCHAR(10) NOT NULL
    entry_time TIMESTAMP NOT NULL
    exit_time TIMESTAMP
    entry_price DECIMAL(10,5) NOT NULL
    exit_price DECIMAL(10,5)
    size DECIMAL(10,2) NOT NULL
    tp_price DECIMAL(10,5)
    sl_price DECIMAL(10,5)
    pnl_pips DECIMAL(10,2)
    exit_reason VARCHAR(20)
    model_version VARCHAR(100)
    created_at TIMESTAMP DEFAULT NOW()
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from src.paper_trading.models import Trade

# Table schema definition
PAPER_TRADES_SCHEMA = {
    "id": "SERIAL PRIMARY KEY",
    "symbol": "VARCHAR(20) NOT NULL",
    "direction": "VARCHAR(10) NOT NULL",
    "entry_time": "TIMESTAMP NOT NULL",
    "exit_time": "TIMESTAMP",
    "entry_price": "DECIMAL(10,5) NOT NULL",
    "exit_price": "DECIMAL(10,5)",
    "size": "DECIMAL(10,2) NOT NULL",
    "tp_price": "DECIMAL(10,5)",
    "sl_price": "DECIMAL(10,5)",
    "pnl_pips": "DECIMAL(10,2)",
    "exit_reason": "VARCHAR(20)",
    "model_version": "VARCHAR(100)",
    "created_at": "TIMESTAMP DEFAULT NOW()",
}


class TradeLogger:
    """Logs closed trades to PostgreSQL database.

    This class provides:
    - Trade logging to paper_trades table
    - Trade retrieval with filtering
    - Trade statistics calculation

    Attributes:
        _db_manager: Reference to DatabaseManager for database operations
    """

    def __init__(self, db_manager: Any) -> None:
        """Initialize TradeLogger with database manager.

        Args:
            db_manager: DatabaseManager instance for database operations
        """
        self._db_manager = db_manager

    def ensure_table(self) -> None:
        """Ensure paper_trades table exists in database.

        Creates the table if it doesn't exist.
        """
        if not self._db_manager.table_exists("paper_trades"):
            self._db_manager.create_table("paper_trades", PAPER_TRADES_SCHEMA)

    def log_trade(self, trade: Trade) -> None:
        """Log a closed trade to the database.

        Args:
            trade: Trade record to log
        """
        data = trade.to_dict()
        self._db_manager.insert_data("paper_trades", data)

    def get_trades(
        self,
        symbol: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get trades from database with optional filtering.

        Args:
            symbol: Filter by trading symbol
            start_date: Filter by entry time >= start_date
            end_date: Filter by entry time <= end_date
            limit: Maximum number of trades to return

        Returns:
            List of trade dictionaries
        """
        conditions = []
        params = {}

        if symbol:
            conditions.append("symbol = :symbol")
            params["symbol"] = symbol

        if start_date:
            conditions.append("entry_time >= :start_date")
            params["start_date"] = start_date

        if end_date:
            conditions.append("entry_time <= :end_date")
            params["end_date"] = end_date

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT * FROM paper_trades
            {where_clause}
            ORDER BY entry_time DESC
            LIMIT :limit
        """
        params["limit"] = limit

        return self._db_manager.execute_query("ai_model", query, params)

    def get_stats(
        self,
        symbol: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Calculate trade statistics.

        Args:
            symbol: Filter by trading symbol
            start_date: Filter by entry time >= start_date
            end_date: Filter by entry time <= end_date

        Returns:
            Dictionary with statistics:
            - total_trades: Total number of trades
            - winning_trades: Number of profitable trades
            - losing_trades: Number of losing trades
            - total_pnl_pips: Total P&L in pips
            - avg_pnl_pips: Average P&L per trade
        """
        conditions = []
        params = {}

        if symbol:
            conditions.append("symbol = :symbol")
            params["symbol"] = symbol

        if start_date:
            conditions.append("entry_time >= :start_date")
            params["start_date"] = start_date

        if end_date:
            conditions.append("entry_time <= :end_date")
            params["end_date"] = end_date

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT
                COUNT(*) as total_trades,
                COUNT(CASE WHEN pnl_pips > 0 THEN 1 END) as winning_trades,
                COUNT(CASE WHEN pnl_pips < 0 THEN 1 END) as losing_trades,
                COALESCE(SUM(pnl_pips), 0) as total_pnl_pips,
                COALESCE(AVG(pnl_pips), 0) as avg_pnl_pips
            FROM paper_trades
            {where_clause}
        """

        results = self._db_manager.execute_query("ai_model", query, params)

        if results:
            return results[0]

        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl_pips": Decimal("0"),
            "avg_pnl_pips": Decimal("0"),
        }
