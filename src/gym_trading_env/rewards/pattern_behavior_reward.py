"""
Pattern-Based Behavior Reward System.

This module implements a reward function that rewards correct trading decisions
based on market phase, pattern presence, risk management quality, and trade
execution timing - NOT based on PnL outcome.

The goal is to teach the RL agent to make CORRECT TRADING DECISIONS, regardless
of outcome randomness from market noise.

TDD Phase: RED → GREEN → REFACTOR
Author: python-backend-engineer (Stream B)
Issue: #258
Created: 2025-11-13
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .base import BaseReward
from ..features.order_block_features import OrderBlockSignal
from ..actions.hybrid_actions import HybridAction


class MarketPhase(Enum):
    """
    Market phase classification based on price structure and volatility.

    Phases:
        ACCUMULATION: Consolidation period, low volatility, awaiting breakout
        DISTRIBUTION: Consolidation after move, potential reversal setup
        MARKUP: Bullish trend with higher highs and higher lows
        MARKDOWN: Bearish trend with lower highs and lower lows
    """
    ACCUMULATION = 0
    DISTRIBUTION = 1
    MARKUP = 2
    MARKDOWN = 3


@dataclass
class PositionInfo:
    """
    Position and risk management information.

    Attributes:
        direction (str): Position direction - "LONG" or "SHORT"
        entry_price (float): Entry price level
        sl_distance_pips (float): Stop-loss distance from entry in pips
        tp_distance_pips (float): Take-profit distance from entry in pips
        rr_ratio (float): Risk-reward ratio (TP/SL)
    """
    direction: str
    entry_price: float
    sl_distance_pips: float
    tp_distance_pips: float
    rr_ratio: float


class PatternBehaviorReward(BaseReward):
    """
    Pattern-based behavior reward function.

    Rewards correct trading decisions based on:
    1. Market phase alignment (trade with trend, skip consolidation)
    2. Pattern recognition quality (OB strength, validity)
    3. Risk management quality (SL/TP placement)
    4. Trade outcome (10% weight, not primary)

    This reward system focuses on DECISION QUALITY, not outcome luck.

    Example:
        >>> reward_fn = PatternBehaviorReward()
        >>> signal = OrderBlockSignal(...)
        >>> position = PositionInfo(...)
        >>> reward = reward_fn.calculate(
        ...     action=HybridAction.ENTER,
        ...     market_phase=MarketPhase.MARKUP,
        ...     ob_signal=signal,
        ...     position=position,
        ...     trade_outcome="TP2"
        ... )
    """

    def calculate(
        self,
        action: HybridAction,
        market_phase: MarketPhase,
        ob_signal: OrderBlockSignal,
        position: Optional[PositionInfo],
        trade_outcome: Optional[str] = None,
        **kwargs
    ) -> float:
        """
        Calculate behavior-based reward.

        Args:
            action: Action taken by agent (SKIP, ENTER, etc.)
            market_phase: Current market phase
            ob_signal: Order Block signal information
            position: Position and risk management info (None if SKIP)
            trade_outcome: Trade outcome - "SL", "TP1", "TP2", "TP3", or None
            **kwargs: Additional parameters for BaseReward compatibility

        Returns:
            Total reward combining all components

        Example:
            >>> signal = OrderBlockSignal(direction="LONG", strength=0.85, ...)
            >>> position = PositionInfo(direction="LONG", sl_distance_pips=50, ...)
            >>> reward = reward_fn.calculate(
            ...     action=HybridAction.ENTER,
            ...     market_phase=MarketPhase.MARKUP,
            ...     ob_signal=signal,
            ...     position=position,
            ...     trade_outcome="TP2"
            ... )
        """
        # TDD GREEN PHASE: Combine all evaluation components

        total_reward = 0.0

        # 1. Market phase reward - full weight
        direction = position.direction if position else None
        total_reward += self._evaluate_phase_action(action, market_phase, direction)

        # 2. Pattern recognition reward - full weight
        total_reward += self._evaluate_pattern_action(action, ob_signal)

        # 3. Risk management reward - full weight (only if position exists)
        if position is not None:
            total_reward += self._evaluate_risk_management(
                position.sl_distance_pips,
                position.tp_distance_pips,
                position.rr_ratio
            )

        # 4. Trade outcome bonus - 10% weight to avoid rewarding luck
        if trade_outcome is not None:
            outcome_reward = self._evaluate_outcome(trade_outcome)
            total_reward += outcome_reward * 0.1

        return total_reward

    def _evaluate_phase_action(
        self,
        action: HybridAction,
        market_phase: MarketPhase,
        direction: Optional[str] = None
    ) -> float:
        """
        Evaluate reward based on market phase and action alignment.

        Rewards:
        - ACCUMULATION/DISTRIBUTION + SKIP = +2 (patience)
        - ACCUMULATION/DISTRIBUTION + ENTER = -3 (impatience)
        - MARKUP + LONG = +5 (with-trend)
        - MARKUP + SHORT = -5 (counter-trend)
        - MARKDOWN + SHORT = +5 (with-trend)
        - MARKDOWN + LONG = -5 (counter-trend)
        - MARKUP/MARKDOWN + SKIP = 0 (neutral)

        Args:
            action: Action taken by agent
            market_phase: Current market phase
            direction: Position direction ("LONG" or "SHORT"), required for trending phases

        Returns:
            Reward value based on phase-action alignment
        """
        # TDD GREEN PHASE: Minimal implementation to pass tests

        # Handle SKIP action
        if action == HybridAction.SKIP:
            # Reward patience during consolidation phases
            if market_phase in (MarketPhase.ACCUMULATION, MarketPhase.DISTRIBUTION):
                return 2.0
            # Neutral during trending phases
            return 0.0

        # Handle ENTER action (and variants like ENTER_2X, ADJUST_SL, ADJUST_TP)
        # Consolidation phases - penalize impatience
        if market_phase in (MarketPhase.ACCUMULATION, MarketPhase.DISTRIBUTION):
            return -3.0

        # Trending phases - reward/penalize based on direction alignment
        if market_phase == MarketPhase.MARKUP:
            # Bullish trend - reward longs, penalize shorts
            if direction == "LONG":
                return 5.0
            elif direction == "SHORT":
                return -5.0

        if market_phase == MarketPhase.MARKDOWN:
            # Bearish trend - reward shorts, penalize longs
            if direction == "SHORT":
                return 5.0
            elif direction == "LONG":
                return -5.0

        # Default: neutral (shouldn't reach here if all cases handled)
        return 0.0

    def _evaluate_pattern_action(
        self,
        action: HybridAction,
        ob_signal: OrderBlockSignal
    ) -> float:
        """
        Evaluate reward based on pattern recognition and action.

        Rewards:
        - Strong OB (>0.8) + ENTER = +5 (correct aggression)
        - Moderate OB (0.3-0.8) + ENTER = +3 (reasonable entry)
        - Weak OB (<0.3) + ENTER = -2 (poor signal)
        - No OB + ENTER = -3 (gambling)
        - No OB + SKIP = +1 (correct patience)
        - Valid OB + SKIP = -1 (missed opportunity)

        Args:
            action: Action taken by agent
            ob_signal: Order Block signal information

        Returns:
            Reward value based on pattern-action alignment
        """
        # TDD GREEN PHASE: Minimal implementation to pass tests

        # Check if signal is valid (OB detected)
        is_valid_ob = ob_signal.is_valid and ob_signal.direction != "NONE"

        # Handle SKIP action
        if action == HybridAction.SKIP:
            if is_valid_ob:
                # Valid OB but skipped - missed opportunity
                return -1.0
            else:
                # No OB and skipped - correct patience
                return 1.0

        # Handle ENTER action (and variants)
        if not is_valid_ob:
            # No OB but entered - gambling
            return -3.0

        # Valid OB + ENTER - reward based on signal strength
        strength = ob_signal.strength

        if strength > 0.8:
            # Strong signal - correct aggression
            return 5.0
        elif strength >= 0.3:
            # Moderate signal - reasonable entry
            return 3.0
        else:
            # Weak signal - poor entry
            return -2.0

    def _evaluate_risk_management(
        self,
        sl_distance_pips: float,
        tp_distance_pips: float,
        rr_ratio: float
    ) -> float:
        """
        Evaluate reward based on risk management quality.

        Rewards:
        - SL at optimal distance (30-100 pips) = +2
        - SL too tight (<30 pips) = -1
        - SL too wide (>100 pips) = -1
        - TP at 2:1 RR = +2
        - TP at 3:1 RR = +3
        - TP at <1:1 RR = -2 (poor risk-reward)

        Args:
            sl_distance_pips: Stop-loss distance from entry in pips
            tp_distance_pips: Take-profit distance from entry in pips
            rr_ratio: Risk-reward ratio (TP/SL)

        Returns:
            Reward value based on risk management quality
        """
        # TDD GREEN PHASE: Minimal implementation to pass tests

        reward = 0.0

        # Evaluate SL distance - SL violations should dominate
        # Bad SL placement means the trade setup is fundamentally flawed
        if sl_distance_pips < 30:
            # SL too tight - heavy penalty that outweighs TP bonus
            reward -= 3.0
        elif sl_distance_pips > 100:
            # SL too wide - heavy penalty that outweighs TP bonus
            reward -= 3.0
        # Note: Good SL (30-100 pips) gets no separate reward, it's expected baseline

        # Evaluate RR ratio / TP placement
        # RR is the primary reward signal for risk management
        # Poor RR should dominate even with good SL
        if rr_ratio < 1.0:
            # Poor risk-reward ratio - heavy penalty
            reward -= 4.0
        elif rr_ratio >= 3.0:
            # Excellent 3:1 RR
            reward += 3.0
        elif rr_ratio >= 2.0:
            # Good 2:1 RR
            reward += 2.0
        # Note: 1.0 <= RR < 2.0 gets no bonus/penalty for TP

        return reward

    def _evaluate_outcome(
        self,
        outcome: Optional[str]
    ) -> float:
        """
        Evaluate bonus reward based on trade outcome.

        This component has ONLY 10% weight in final reward to avoid
        rewarding luck over decision quality.

        Rewards:
        - SL reached = -2 (pattern failed)
        - TP1 reached = +3 (good trade)
        - TP2 reached = +5 (excellent trade)
        - TP3 reached = +10 (exceptional trade)
        - None = 0 (pending trade)

        Args:
            outcome: Trade outcome - "SL", "TP1", "TP2", "TP3", or None

        Returns:
            Reward value based on trade outcome
        """
        # TDD GREEN PHASE: Minimal implementation to pass tests

        if outcome is None:
            return 0.0

        outcome_rewards = {
            "SL": -2.0,
            "TP1": 3.0,
            "TP2": 5.0,
            "TP3": 10.0
        }

        return outcome_rewards.get(outcome, 0.0)
