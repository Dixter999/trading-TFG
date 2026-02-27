"""
Feature extraction modules for gym_trading_env.

This module provides feature extraction utilities for:
- Order Blocks (entry signals)
- Market Phase detection (trend analysis)

Author: python-backend-engineer
Issue: #254, #258
Created: 2025-11-12
"""

from .order_block_features import (
    OrderBlockSignal,
    extract_order_block_features
)
from .market_phase import (
    MarketPhase,
    detect_market_phase
)

__all__ = [
    "OrderBlockSignal",
    "extract_order_block_features",
    "MarketPhase",
    "detect_market_phase"
]
