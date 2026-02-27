"""
Trade Decision Logger for comprehensive trade decision logging.

Issue #332: Comprehensive Decision Logging & Full Trading Pipeline Integration
Stream A: Decision Logger

This module provides detailed logging of all trade decisions with complete
context including observation vectors, indicators, patterns, and model outputs.

The log format provides full traceability for every trade decision:
- Decision details (entry, TP, SL, position size)
- Model inputs (observation vector decomposition)
- Indicator analysis by timeframe
- Pattern context
- Model outputs (PPO, SAC, ensemble)
- Account status
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# Constants
# ==============================================================================

# Observation dimension constants (from observation_builder.py)
BASE_OBS_DIM = 96
CONTEXT_DIM = 5
EXTENDED_OBS_DIM = BASE_OBS_DIM + CONTEXT_DIM  # 101

# Context feature indices within observation vector
IDX_DIRECTION = 96  # Direction indicator (+1 LONG, -1 SHORT)
IDX_ENTRY_PRICE = 97  # Entry price (normalized)
IDX_ATR = 98  # Current ATR
IDX_CONFLUENCE = 99  # Confluence strength
IDX_SESSION = 100  # Session indicator

# Pip value by symbol suffix
PIP_VALUES = {
    "JPY": 0.01,  # JPY pairs use 0.01 pip
    "DEFAULT": 0.0001,  # Standard pip value for most pairs
}


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class DecisionContext:
    """
    Complete context for a trade decision.

    Contains all information needed to understand and reproduce
    a trade decision, including market state, model outputs,
    and account status.

    Attributes:
        timestamp: When the decision was made (UTC)
        symbol: Trading symbol (e.g., "EURUSD")
        direction: Trade direction ("LONG" or "SHORT")
        entry_price: Price at which to enter the trade
        take_profit: Take profit price level
        stop_loss: Stop loss price level
        position_size: Size of the position (in lots)
        leverage: Leverage multiplier
        risk_amount: Dollar amount at risk
        risk_percent: Percentage of account at risk
        observation: 101-dimensional observation vector
        indicators: Dict of indicator values by timeframe
        patterns: List of detected patterns
        model_outputs: Dict of model output values
        account_status: Dict of account status values
    """

    timestamp: datetime
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    take_profit: float
    stop_loss: float
    position_size: float
    leverage: int
    risk_amount: float
    risk_percent: float
    observation: np.ndarray  # 101-dim observation vector
    indicators: dict[str, dict[str, float]]  # By timeframe
    patterns: list[dict[str, Any]]  # Detected patterns
    model_outputs: dict[str, Any]  # PPO/SAC outputs
    account_status: dict[str, float]  # Balance, equity, etc.


# ==============================================================================
# TradeDecisionLogger Class
# ==============================================================================


class TradeDecisionLogger:
    """
    Logs comprehensive trade decision context for paper trading.

    Formats and logs all decision context including observation vectors,
    technical indicators, detected patterns, and model outputs.

    The log format provides full traceability:
    - Decision summary (entry, TP, SL, position sizing)
    - Observation vector decomposition
    - Indicator values by timeframe
    - Pattern context
    - Model outputs and confidence
    - Account status

    Attributes:
        log_dir: Directory for log files
        console_output: Whether to also print to console

    Example:
        >>> logger = TradeDecisionLogger(log_dir="logs/decisions")
        >>> context = DecisionContext(...)
        >>> logger.log_decision(context)
    """

    def __init__(
        self,
        log_dir: str = "logs/decisions",
        console_output: bool = False,
    ) -> None:
        """
        Initialize TradeDecisionLogger.

        Args:
            log_dir: Directory path for log files (default: "logs/decisions")
            console_output: Whether to print decisions to console (default: False)
        """
        self.log_dir = Path(log_dir)
        self.console_output = console_output

    def log_decision(self, context: DecisionContext) -> None:
        """
        Log a complete trade decision with all context.

        Creates/appends to a log file named by date and writes
        the formatted decision. Optionally prints to console.

        Args:
            context: DecisionContext with all trade decision details
        """
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Format the decision
        formatted = self.format_decision(context)

        # Write to file (date-based filename)
        log_file = (
            self.log_dir / f"decisions_{context.timestamp.strftime('%Y%m%d')}.log"
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(formatted)
            f.write("\n\n")

        # Log to Python logger
        logger.info(
            f"Trade decision logged: {context.direction} {context.symbol} @ {context.entry_price}"
        )

        # Console output if enabled
        if self.console_output:
            print(formatted)

    def format_decision(self, context: DecisionContext) -> str:
        """
        Format decision context as a comprehensive readable string.

        Creates a detailed multi-section log entry with all context
        needed to understand and reproduce the trade decision.

        Args:
            context: DecisionContext with all trade decision details

        Returns:
            Formatted string with all decision details
        """
        sections = []

        # Header
        sections.append(self._format_header(context))

        # Decision section
        sections.append(self._format_decision_section(context))

        # Model inputs (observation)
        sections.append(self._format_observation_section(context.observation))

        # Indicator analysis
        sections.append(self._format_indicators_section(context.indicators))

        # Pattern context
        sections.append(self._format_patterns_section(context.patterns))

        # Model outputs
        sections.append(self._format_model_outputs_section(context.model_outputs))

        # Expected outcome
        sections.append(self._format_expected_outcome_section(context))

        # Account status
        sections.append(self._format_account_status_section(context.account_status))

        # Footer
        sections.append(self._format_footer())

        return "\n".join(sections)

    def _format_header(self, context: DecisionContext) -> str:
        """Format the log header with timestamp."""
        separator = "=" * 79
        timestamp_str = context.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")

        return f"""{separator}
TRADE DECISION LOG - {timestamp_str}
{separator}"""

    def _format_footer(self) -> str:
        """Format the log footer."""
        return "=" * 79

    def _format_decision_section(self, context: DecisionContext) -> str:
        """Format the main decision details section."""
        tp_pips = self._calculate_pips_with_direction(
            context.symbol,
            context.entry_price,
            context.take_profit,
            context.direction,
        )
        sl_pips = self._calculate_pips_with_direction(
            context.symbol,
            context.entry_price,
            context.stop_loss,
            context.direction,
        )

        # Format pips with sign
        tp_sign = "+" if tp_pips >= 0 else ""
        sl_sign = "+" if sl_pips >= 0 else ""

        return f"""
DECISION: {context.direction} {context.symbol} @ {context.entry_price:.5f}
   |- Take Profit: {context.take_profit:.5f} ({tp_sign}{tp_pips} pips)
   |- Stop Loss: {context.stop_loss:.5f} ({sl_sign}{sl_pips} pips)
   |- Position Size: {context.position_size} lots
   |- Leverage: x{context.leverage}
   |- Risk: ${context.risk_amount:.2f} ({context.risk_percent:.1f}% of balance)"""

    def _format_observation_section(self, obs: np.ndarray) -> str:
        """
        Extract and format observation vector components.

        Decomposes the 101-dimensional observation vector into
        readable components.

        Args:
            obs: 101-dimensional observation vector

        Returns:
            Formatted string with observation components
        """
        # Extract context features (indices 96-100)
        direction_val = obs[IDX_DIRECTION] if len(obs) > IDX_DIRECTION else 0.0
        entry_price_norm = obs[IDX_ENTRY_PRICE] if len(obs) > IDX_ENTRY_PRICE else 0.0
        atr = obs[IDX_ATR] if len(obs) > IDX_ATR else 0.0
        confluence = obs[IDX_CONFLUENCE] if len(obs) > IDX_CONFLUENCE else 0.0
        session = obs[IDX_SESSION] if len(obs) > IDX_SESSION else 0.0

        # Determine direction string
        direction_str = (
            "LONG" if direction_val > 0 else "SHORT" if direction_val < 0 else "HOLD"
        )

        # Format session description
        session_desc = self._get_session_description(session)

        return f"""
MODEL INPUTS (Observation Vector):
   |- Direction Indicator: {direction_val:+.1f} ({direction_str})
   |- Entry Price (norm): {entry_price_norm:.2f}
   |- ATR: {atr:.4f}
   |- Confluence: {confluence:.2f}
   |- Session: {session:.2f} ({session_desc})"""

    def _format_indicators_section(
        self, indicators: dict[str, dict[str, float]]
    ) -> str:
        """
        Format indicator values grouped by timeframe.

        Args:
            indicators: Dict of indicator values by timeframe

        Returns:
            Formatted string with indicators by timeframe
        """
        if not indicators:
            return """
INDICATOR ANALYSIS (by timeframe):
   |- No indicator data available"""

        lines = ["\nINDICATOR ANALYSIS (by timeframe):"]

        for tf, values in sorted(indicators.items()):
            # Format indicator values
            parts = []
            if "rsi_14" in values:
                parts.append(f"RSI={values['rsi_14']:.1f}")
            if "macd_line" in values:
                parts.append(f"MACD={values['macd_line']:+.4f}")
            if "bb_position" in values:
                parts.append(f"BB_pos={values['bb_position']:.2f}")
            if "sma_20" in values:
                parts.append(f"SMA20={values['sma_20']:.5f}")

            # If no known indicators, show all
            if not parts:
                parts = [f"{k}={v:.4f}" for k, v in values.items()]

            indicator_str = ", ".join(parts) if parts else "No data"
            lines.append(f"   |- {tf}:  {indicator_str}")

        return "\n".join(lines)

    def _format_patterns_section(self, patterns: list[dict[str, Any]]) -> str:
        """
        Format detected pattern context.

        Args:
            patterns: List of detected patterns with type, direction, etc.

        Returns:
            Formatted string with pattern details
        """
        if not patterns:
            return """
PATTERN CONTEXT:
   |- No patterns detected"""

        lines = ["\nPATTERN CONTEXT:"]

        for pattern in patterns:
            pattern_type = pattern.get("type", "unknown")
            direction = pattern.get("direction", "neutral")
            price = pattern.get("price", 0.0)
            confidence = pattern.get("confidence", 0.0)

            # Format pattern type for display
            display_type = pattern_type.replace("_", " ").title()

            lines.append(
                f"   |- {display_type}: {direction.title()} @ {price:.5f} (conf: {confidence:.2f})"
            )

        return "\n".join(lines)

    def _format_model_outputs_section(self, outputs: dict[str, Any]) -> str:
        """
        Format model output values.

        Args:
            outputs: Dict of model outputs (PPO, SAC, confidence, etc.)

        Returns:
            Formatted string with model outputs
        """
        lines = ["\nMODEL OUTPUT:"]

        # PPO direction
        ppo_action = outputs.get("ppo_action", 0)
        ppo_logprob = outputs.get("ppo_logprob", 0.0)
        action_name = {0: "HOLD", 1: "LONG", 2: "SHORT"}.get(ppo_action, "UNKNOWN")
        lines.append(
            f"   |- PPO Direction: {action_name} (action={ppo_action}, logprob={ppo_logprob:.2f})"
        )

        # SAC exit
        sac_tp = outputs.get("sac_tp", 0.0)
        sac_sl = outputs.get("sac_sl", 0.0)
        lines.append(f"   |- SAC Exit: TP={sac_tp:.4f}, SL={sac_sl:.4f} (raw actions)")

        # Confidence
        confidence = outputs.get("confidence", 0.0)
        lines.append(f"   |- Confidence: {confidence:.2f}")

        # Ensemble agreement
        ensemble = outputs.get("ensemble_agreement", 0.0)
        lines.append(f"   |- Ensemble Agreement: {ensemble:.2f}")

        return "\n".join(lines)

    def _format_expected_outcome_section(self, context: DecisionContext) -> str:
        """
        Format expected outcome based on risk/reward.

        Args:
            context: DecisionContext with trade details

        Returns:
            Formatted string with expected outcome
        """
        rr_ratio = self._calculate_risk_reward(context)

        # Estimate win probability based on R:R (placeholder logic)
        # In production, this would come from model backtest results
        win_prob = min(0.65, 0.45 + rr_ratio * 0.1)

        # Calculate expected PnL
        expected_pnl = (
            (
                win_prob
                * abs(
                    self._calculate_pips_with_direction(
                        context.symbol,
                        context.entry_price,
                        context.take_profit,
                        context.direction,
                    )
                )
                - (1 - win_prob)
                * abs(
                    self._calculate_pips_with_direction(
                        context.symbol,
                        context.entry_price,
                        context.stop_loss,
                        context.direction,
                    )
                )
            )
            * context.position_size
            * 10
        )  # Rough pip value

        return f"""
EXPECTED OUTCOME:
   |- Win Probability: {win_prob * 100:.1f}% (estimate)
   |- Expected PnL: ${expected_pnl:.2f}
   |- Risk/Reward: {rr_ratio:.2f}"""

    def _format_account_status_section(self, status: dict[str, float]) -> str:
        """
        Format account status section.

        Args:
            status: Dict with balance, equity, etc.

        Returns:
            Formatted string with account status
        """
        balance = status.get("balance", 0.0)
        equity = status.get("equity", balance)
        open_positions = int(status.get("open_positions", 0))
        daily_pnl = status.get("daily_pnl", 0.0)
        drawdown = status.get("drawdown", 0.0)

        return f"""
ACCOUNT STATUS:
   |- Balance: ${balance:,.2f}
   |- Equity: ${equity:,.2f}
   |- Open Positions: {open_positions}
   |- Daily P&L: ${daily_pnl:,.2f}
   |- Drawdown: {drawdown:.1f}%"""

    def _calculate_pips(self, symbol: str, price1: float, price2: float) -> int:
        """
        Calculate pip difference between two prices.

        Args:
            symbol: Trading symbol (e.g., "EURUSD", "USDJPY")
            price1: First price (usually entry)
            price2: Second price (TP or SL)

        Returns:
            Pip difference as integer (positive or negative)
        """
        # Determine pip value based on symbol
        pip_value = PIP_VALUES["DEFAULT"]
        if "JPY" in symbol.upper():
            pip_value = PIP_VALUES["JPY"]

        # Calculate pip difference
        pip_diff = (price2 - price1) / pip_value

        return int(round(pip_diff))

    def _calculate_pips_with_direction(
        self,
        symbol: str,
        entry_price: float,
        target_price: float,
        direction: str,
    ) -> int:
        """
        Calculate pip difference considering trade direction.

        For LONG: profit when target > entry
        For SHORT: profit when target < entry

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            target_price: Target price (TP or SL)
            direction: Trade direction ("LONG" or "SHORT")

        Returns:
            Pip difference (positive for profit direction)
        """
        raw_pips = self._calculate_pips(symbol, entry_price, target_price)

        # For SHORT positions, invert the sign
        if direction.upper() == "SHORT":
            return -raw_pips

        return raw_pips

    def _calculate_risk_reward(self, context: DecisionContext) -> float:
        """
        Calculate risk/reward ratio for a trade.

        Args:
            context: DecisionContext with trade details

        Returns:
            Risk/reward ratio (reward / risk)
        """
        # Calculate reward and risk in pips
        reward_pips = abs(
            self._calculate_pips(
                context.symbol, context.entry_price, context.take_profit
            )
        )
        risk_pips = abs(
            self._calculate_pips(context.symbol, context.entry_price, context.stop_loss)
        )

        if risk_pips == 0:
            return 0.0

        return reward_pips / risk_pips

    def _get_session_description(self, session_value: float) -> str:
        """
        Get human-readable session description from session indicator.

        The session indicator is sin(hour/24 * 2*pi):
        - ~0.5 to 1.0: Tokyo session
        - ~0.0 to -0.5: London session
        - ~-0.5 to -1.0: NY session

        Args:
            session_value: Session indicator value [-1, 1]

        Returns:
            Session name string
        """
        if session_value > 0.5:
            return "Tokyo"
        elif session_value > 0:
            return "Tokyo/London overlap"
        elif session_value > -0.5:
            return "London"
        else:
            return "NY"
