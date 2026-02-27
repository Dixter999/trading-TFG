"""
Signal Parser for dynamic signal detection.

Issue #561: Eliminates manual handlers by parsing signal definitions from YAML.

This module provides a declarative approach to signal detection by:
1. Loading signal definitions from YAML configuration
2. Evaluating conditions against market data DataFrames
3. Supporting multiple condition types (threshold, comparison, crossover)
4. Resolving indicator column name aliases automatically

Usage:
    parser = SignalParser()
    signal, timestamp = parser.evaluate_signal("RSI_oversold_long", df)
    if signal:
        print(f"Signal: {signal} at {timestamp}")
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Direction constants
SIGNAL_LONG = "LONG"
SIGNAL_SHORT = "SHORT"

# Default config path
DEFAULT_CONFIG_PATH = os.getenv("SIGNAL_DEFINITIONS_PATH", "config/signal_definitions.yaml")

# Indicator column name mappings (support multiple variants)
INDICATOR_ALIASES: dict[str, list[str]] = {
    "rsi_14": ["rsi_14", "rsi"],
    "rsi": ["rsi_14", "rsi"],
    "sma_20": ["sma_20", "sma20"],
    "sma_50": ["sma_50", "sma50"],
    "sma_200": ["sma_200", "sma200"],
    "stoch_k": ["stoch_k", "stochk", "slowk"],
    "stoch_d": ["stoch_d", "stochd", "slowd"],
    "macd": ["macd", "macd_line"],
    "macd_signal": ["macd_signal", "signal_line", "macdsignal", "macd_signal_line"],
    "bb_upper": ["bb_upper", "bollinger_upper", "upper_band", "bb_upper_20"],
    "bb_lower": ["bb_lower", "bollinger_lower", "lower_band", "bb_lower_20"],
    "ema_12": ["ema_12", "ema12"],
    "ema_26": ["ema_26", "ema26"],
    "close": ["close", "Close"],
    "open": ["open", "Open"],
    "high": ["high", "High"],
    "low": ["low", "Low"],
}

# Timestamp column aliases
TIMESTAMP_ALIASES: list[str] = [
    "timestamp",
    "time",
    "datetime",
    "date",
    "Timestamp",
    "Time",
]


class SignalParser:
    """Parse and evaluate signal definitions from YAML configuration.

    This class provides a dynamic approach to signal detection by loading
    signal definitions from a YAML file and evaluating them against market
    data DataFrames.

    Attributes:
        definitions: Dictionary of signal definitions loaded from YAML
        config_path: Path to the YAML configuration file

    Example:
        >>> parser = SignalParser()
        >>> df = pd.DataFrame({'rsi_14': [28], 'timestamp': [datetime.now()]})
        >>> signal, ts = parser.evaluate_signal('RSI_oversold_long', df)
        >>> print(signal)  # 'LONG'
    """

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize SignalParser.

        Args:
            config_path: Path to YAML config file. Defaults to config/signal_definitions.yaml
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.definitions: dict[str, Any] = {}
        self._load_definitions()

    def _load_definitions(self) -> None:
        """Load signal definitions from YAML file."""
        try:
            path = Path(self.config_path)
            if not path.exists():
                logger.warning(f"Signal definitions file not found: {self.config_path}")
                return

            with open(path) as f:
                data = yaml.safe_load(f)

            if data and "signals" in data:
                self.definitions = data["signals"]
                logger.info(
                    f"Loaded {len(self.definitions)} signal definitions from {self.config_path}"
                )
            else:
                logger.warning(f"No 'signals' section found in {self.config_path}")

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
        except Exception as e:
            logger.error(f"Failed to load signal definitions: {e}")

    def evaluate_signal(
        self,
        signal_name: str,
        df: pd.DataFrame,
    ) -> tuple[str | None, datetime | None]:
        """Evaluate a signal against current market data.

        Args:
            signal_name: Name of signal (e.g., "RSI_oversold_long")
            df: DataFrame with indicator data (must have timestamp column)

        Returns:
            (direction, timestamp) if signal triggered, (None, None) otherwise
        """
        if df is None or df.empty:
            return None, None

        if signal_name not in self.definitions:
            logger.warning(f"Signal '{signal_name}' not found in definitions")
            return None, None

        signal_def = self.definitions[signal_name]
        conditions = signal_def.get("conditions", [])
        direction = signal_def.get("direction", "").upper()

        if not conditions:
            logger.warning(f"Signal '{signal_name}' has no conditions defined")
            return None, None

        if direction not in (SIGNAL_LONG, SIGNAL_SHORT):
            logger.warning(
                f"Invalid direction '{direction}' for signal '{signal_name}'"
            )
            return None, None

        # Get the last row for evaluation (current candle)
        latest = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) >= 2 else None

        # Evaluate all conditions (AND logic)
        all_conditions_met = True
        for condition in conditions:
            if not self._evaluate_condition(latest, prev_row, condition, df):
                all_conditions_met = False
                break

        if all_conditions_met:
            timestamp = self._get_timestamp(latest)
            return direction, timestamp

        return None, None

    def _evaluate_condition(
        self,
        row: pd.Series,
        prev_row: pd.Series | None,
        condition: dict[str, Any],
        df: pd.DataFrame,
    ) -> bool:
        """Evaluate a single condition against a data row.

        Args:
            row: Current row of data
            prev_row: Previous row of data (for crossover detection)
            condition: Condition definition from YAML
            df: Full DataFrame (for cross detection lookback)

        Returns:
            True if condition is met, False otherwise
        """
        condition_type = condition.get("type")

        # Handle crossover conditions
        if condition_type in ("cross_above", "cross_below"):
            return self._evaluate_cross_condition(condition, df)

        # Handle standard comparison conditions
        indicator_name = condition.get("indicator")
        operator = condition.get("operator")
        value = condition.get("value")
        compare_to = condition.get("compare_to")
        tolerance = condition.get("tolerance")

        if not indicator_name or not operator:
            logger.warning("Invalid condition: missing indicator or operator")
            return False

        # Get indicator value
        indicator_value = self._get_indicator_value(row, indicator_name)
        if indicator_value is None:
            return False

        # Determine comparison value
        if compare_to:
            # Indicator vs indicator comparison
            compare_value = self._get_indicator_value(row, compare_to)
            if compare_value is None:
                return False
            # Apply tolerance if specified
            if tolerance is not None:
                compare_value = compare_value * tolerance
        elif value is not None:
            # Indicator vs fixed value comparison
            compare_value = value
        else:
            logger.warning("Condition missing both 'value' and 'compare_to'")
            return False

        # Perform comparison
        return self._compare(indicator_value, operator, compare_value)

    def _evaluate_cross_condition(
        self,
        condition: dict[str, Any],
        df: pd.DataFrame,
    ) -> bool:
        """Evaluate a crossover condition.

        Args:
            condition: Cross condition with 'fast' and 'slow' indicators
            df: Full DataFrame for lookback

        Returns:
            True if crossover detected, False otherwise
        """
        if len(df) < 2:
            return False

        fast_name = condition.get("fast")
        slow_name = condition.get("slow")
        cross_type = condition.get("type")

        if not fast_name or not slow_name:
            logger.warning("Cross condition missing 'fast' or 'slow' indicator")
            return False

        # Get current and previous values
        current = df.iloc[-1]
        previous = df.iloc[-2]

        fast_current = self._get_indicator_value(current, fast_name)
        fast_previous = self._get_indicator_value(previous, fast_name)
        slow_current = self._get_indicator_value(current, slow_name)
        slow_previous = self._get_indicator_value(previous, slow_name)

        if None in (fast_current, fast_previous, slow_current, slow_previous):
            return False

        if cross_type == "cross_above":
            # Fast was below or equal slow, now fast is above slow
            crossed = fast_previous <= slow_previous and fast_current > slow_current
        elif cross_type == "cross_below":
            # Fast was above or equal slow, now fast is below slow
            crossed = fast_previous >= slow_previous and fast_current < slow_current
        else:
            logger.warning(f"Unknown cross type: {cross_type}")
            return False

        return crossed

    def _get_indicator_value(
        self,
        row: pd.Series,
        indicator_name: str,
    ) -> float | None:
        """Get indicator value from row, resolving aliases.

        Args:
            row: Data row (Series)
            indicator_name: Indicator name to look up

        Returns:
            Float value if found, None otherwise
        """
        # Get list of column name aliases
        aliases = INDICATOR_ALIASES.get(indicator_name, [indicator_name])

        for alias in aliases:
            if alias in row.index:
                value = row[alias]
                # Handle NaN values
                if pd.isna(value):
                    continue
                try:
                    return float(value)
                except (ValueError, TypeError):
                    continue

        # Column not found with any alias
        return None

    def _compare(self, value: float, operator: str, compare_value: float) -> bool:
        """Perform comparison operation.

        Args:
            value: Left operand
            operator: Comparison operator string
            compare_value: Right operand

        Returns:
            Result of comparison
        """
        operators = {
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "==": lambda a, b: np.isclose(a, b),
            "!=": lambda a, b: not np.isclose(a, b),
        }

        op_func = operators.get(operator)
        if op_func is None:
            logger.warning(f"Unknown operator: {operator}")
            return False

        return op_func(value, compare_value)

    def _get_timestamp(self, row: pd.Series) -> datetime | None:
        """Extract timestamp from row.

        Args:
            row: Data row

        Returns:
            Timestamp if found, None otherwise
        """
        for alias in TIMESTAMP_ALIASES:
            if alias in row.index:
                ts = row[alias]
                if pd.notna(ts):
                    if isinstance(ts, datetime):
                        return ts
                    try:
                        return pd.Timestamp(ts).to_pydatetime()
                    except Exception:
                        continue
        return None

    def get_all_signals(self) -> list[str]:
        """Return list of all defined signal names.

        Returns:
            List of signal names
        """
        return list(self.definitions.keys())

    def is_dynamic_signal(self, signal_name: str) -> bool:
        """Check if signal is defined in YAML (vs hardcoded).

        Args:
            signal_name: Signal name to check

        Returns:
            True if signal is defined in YAML
        """
        return signal_name in self.definitions

    def get_signal_definition(self, signal_name: str) -> dict[str, Any] | None:
        """Get the full definition of a signal.

        Args:
            signal_name: Signal name

        Returns:
            Signal definition dict or None
        """
        return self.definitions.get(signal_name)

    def get_signals_by_direction(self, direction: str) -> list[str]:
        """Get all signals for a given direction.

        Args:
            direction: 'long' or 'short'

        Returns:
            List of signal names matching direction
        """
        direction = direction.lower()
        return [
            name
            for name, defn in self.definitions.items()
            if defn.get("direction", "").lower() == direction
        ]

    def get_signals_by_category(self, category: str) -> list[str]:
        """Get all signals for a given category.

        Args:
            category: Category name

        Returns:
            List of signal names matching category
        """
        return [
            name
            for name, defn in self.definitions.items()
            if defn.get("category", "") == category
        ]

    def validate_signal_definition(self, signal_name: str) -> tuple[bool, list[str]]:
        """Validate a signal definition for completeness.

        Args:
            signal_name: Signal to validate

        Returns:
            (is_valid, list_of_errors)
        """
        errors: list[str] = []

        if signal_name not in self.definitions:
            return False, [f"Signal '{signal_name}' not found"]

        defn = self.definitions[signal_name]

        # Check direction
        direction = defn.get("direction", "")
        if direction.upper() not in (SIGNAL_LONG, SIGNAL_SHORT):
            errors.append(f"Invalid direction: '{direction}'")

        # Check conditions
        conditions = defn.get("conditions", [])
        if not conditions:
            errors.append("No conditions defined")

        for i, cond in enumerate(conditions):
            cond_type = cond.get("type")

            if cond_type in ("cross_above", "cross_below"):
                if not cond.get("fast"):
                    errors.append(f"Condition {i}: missing 'fast' indicator")
                if not cond.get("slow"):
                    errors.append(f"Condition {i}: missing 'slow' indicator")
            else:
                if not cond.get("indicator"):
                    errors.append(f"Condition {i}: missing 'indicator'")
                if not cond.get("operator"):
                    errors.append(f"Condition {i}: missing 'operator'")
                if cond.get("value") is None and not cond.get("compare_to"):
                    errors.append(f"Condition {i}: missing 'value' or 'compare_to'")

        return len(errors) == 0, errors
