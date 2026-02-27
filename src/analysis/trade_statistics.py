"""
Trade statistics calculation module.

Author: python-backend-engineer (Stream B)
Issue: #249
Created: 2025-11-11 (RED phase - stub)

Calculates:
- Win rate and profit factor
- Average win/loss and win/loss ratio
- Trade distribution by exit type (stop loss, take profit, timeout)
"""

from dataclasses import dataclass
from typing import List, Dict


def calculate_win_rate(trades: List[Dict]) -> float:
    """
    Calculate win rate percentage.

    Args:
        trades: List of trade dictionaries with 'pnl' field

    Returns:
        Win rate as percentage (0-100)
    """
    if not trades:
        return 0.0

    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    return (len(winning_trades) / len(trades)) * 100.0


def calculate_profit_factor(trades: List[Dict]) -> float:
    """
    Calculate profit factor (total wins / total losses).

    Args:
        trades: List of trade dictionaries with 'pnl' field

    Returns:
        Profit factor (> 1 is profitable)
    """
    if not trades:
        return 0.0

    total_wins = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
    total_losses = abs(sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0))

    if total_losses == 0:
        return 0.0 if total_wins == 0 else 0.0  # Return 0 for consistency

    return total_wins / total_losses


def calculate_average_win(trades: List[Dict]) -> float:
    """
    Calculate average winning trade P&L.

    Args:
        trades: List of trade dictionaries with 'pnl' field

    Returns:
        Average win amount
    """
    winning_trades = [t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0]

    if not winning_trades:
        return 0.0

    return sum(winning_trades) / len(winning_trades)


def calculate_average_loss(trades: List[Dict]) -> float:
    """
    Calculate average losing trade P&L.

    Args:
        trades: List of trade dictionaries with 'pnl' field

    Returns:
        Average loss amount (negative)
    """
    losing_trades = [t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0]

    if not losing_trades:
        return 0.0

    return sum(losing_trades) / len(losing_trades)


def calculate_win_loss_ratio(trades: List[Dict]) -> float:
    """
    Calculate win/loss ratio (avg win / abs(avg loss)).

    Args:
        trades: List of trade dictionaries with 'pnl' field

    Returns:
        Win/loss ratio
    """
    avg_win = calculate_average_win(trades)
    avg_loss = calculate_average_loss(trades)

    if avg_loss == 0:
        return 0.0

    return avg_win / abs(avg_loss)


def analyze_exit_distribution(trades: List[Dict]) -> Dict:
    """
    Analyze trade distribution by exit type.

    Args:
        trades: List of trade dictionaries with 'exit_reason' field

    Returns:
        Dictionary with exit type counts and percentages
    """
    if not trades:
        return {
            "total": 0,
            "stop_loss": 0,
            "take_profit": 0,
            "timeout": 0,
            "unknown": 0,
            "stop_loss_pct": 0.0,
            "take_profit_pct": 0.0,
            "timeout_pct": 0.0,
            "unknown_pct": 0.0,
        }

    # Count exit types
    exit_counts = {
        "stop_loss": 0,
        "take_profit": 0,
        "timeout": 0,
        "unknown": 0,
    }

    for trade in trades:
        exit_reason = trade.get("exit_reason", "unknown")
        if exit_reason in exit_counts:
            exit_counts[exit_reason] += 1
        else:
            exit_counts["unknown"] += 1

    total = len(trades)

    # Calculate percentages
    return {
        "total": total,
        "stop_loss": exit_counts["stop_loss"],
        "take_profit": exit_counts["take_profit"],
        "timeout": exit_counts["timeout"],
        "unknown": exit_counts["unknown"],
        "stop_loss_pct": round((exit_counts["stop_loss"] / total) * 100, 2),
        "take_profit_pct": round((exit_counts["take_profit"] / total) * 100, 2),
        "timeout_pct": round((exit_counts["timeout"] / total) * 100, 2),
        "unknown_pct": round((exit_counts["unknown"] / total) * 100, 2),
    }


@dataclass
class TradeStatistics:
    """
    Container for trade statistics.

    Attributes:
        total_trades: Total number of trades
        winning_trades: Number of winning trades
        losing_trades: Number of losing trades
        win_rate: Win rate percentage
        profit_factor: Profit factor
        average_win: Average winning trade
        average_loss: Average losing trade
        win_loss_ratio: Win/loss ratio
        exit_distribution: Distribution of exits by type
    """

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    win_loss_ratio: float
    exit_distribution: Dict

    @classmethod
    def from_trades(cls, trades: List[Dict]) -> "TradeStatistics":
        """
        Create TradeStatistics from trade list.

        Args:
            trades: List of trade dictionaries

        Returns:
            TradeStatistics instance
        """
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.get("pnl", 0) > 0])
        losing_trades = len([t for t in trades if t.get("pnl", 0) <= 0])

        return cls(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=calculate_win_rate(trades),
            profit_factor=calculate_profit_factor(trades),
            average_win=calculate_average_win(trades),
            average_loss=calculate_average_loss(trades),
            win_loss_ratio=calculate_win_loss_ratio(trades),
            exit_distribution=analyze_exit_distribution(trades),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "win_loss_ratio": self.win_loss_ratio,
            "exit_distribution": self.exit_distribution,
        }
