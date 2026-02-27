"""
Hybrid action space for RL-refined Order Block trading.

This module provides discrete actions that allow the RL agent to:
- Skip weak Order Blocks (SKIP)
- Accept baseline entry (ENTER)
- Increase position size (ENTER_2X)
- Tighten stop-loss (ADJUST_SL)
- Extend take-profit (ADJUST_TP)
"""

from gym_trading_env.actions.hybrid_actions import (
    HybridAction,
    apply_hybrid_action,
)

__all__ = [
    "HybridAction",
    "apply_hybrid_action",
]
