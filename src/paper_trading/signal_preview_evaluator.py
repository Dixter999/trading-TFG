"""
Signal Preview Evaluator for Paper Trading (Issue #629).

Evaluates all configured signals and calculates confidence levels
WITHOUT triggering actual trades. This is a READ-ONLY preview system
that shows which signals are close to triggering at the next candle close.

Key Features:
- Evaluates all signals every 60 seconds (synced with data-syncer)
- Groups signals by next candle close time
- Calculates confidence % for each signal condition
- Stores snapshots for accuracy analysis
- Does NOT affect actual trading decisions
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
import json

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

logger = logging.getLogger(__name__)

# Global cache for signal definitions from YAML
_signal_definitions_cache: Optional[Dict[str, Dict]] = None


# Timeframe definitions in minutes
TIMEFRAME_MINUTES = {
    "M30": 30,
    "H1": 60,
    "H2": 120,
    "H3": 180,
    "H4": 240,
    "H6": 360,
    "H8": 480,
    "H12": 720,
    "D1": 1440,
}


def load_signal_definitions() -> Dict[str, Dict]:
    """
    Load signal definitions from config/signal_definitions.yaml.

    Returns:
        Dict mapping signal_name to {"conditions": [...], "direction": "long"/"short"}
    """
    global _signal_definitions_cache
    if _signal_definitions_cache is not None:
        return _signal_definitions_cache

    if yaml is None:
        logger.warning("PyYAML not available, using inferred conditions")
        return {}

    # Try multiple paths (gcs-config is the shared volume from init container)
    config_paths = [
        "/app/gcs-config/signal_definitions.yaml",
        os.path.join(os.path.dirname(__file__), "..", "..", "config", "signal_definitions.yaml"),
        "config/signal_definitions.yaml",
        "/home/dixter/Projects/trading/config/signal_definitions.yaml",
    ]

    for path in config_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    config = yaml.safe_load(f)
                _signal_definitions_cache = {}
                for name, sig in config.get("signals", {}).items():
                    _signal_definitions_cache[name] = {
                        "conditions": sig.get("conditions", []),
                        "direction": sig.get("direction", "long"),
                        "required_indicators": sig.get("required_indicators", []),
                    }
                logger.info(f"Loaded {len(_signal_definitions_cache)} signal definitions from {path}")
                return _signal_definitions_cache
            except Exception as e:
                logger.warning(f"Failed to load signal definitions from {path}: {e}")

    logger.warning("No signal_definitions.yaml found, using inferred conditions")
    _signal_definitions_cache = {}
    return _signal_definitions_cache


# Indicator name aliases (same as CLI monitor)
INDICATOR_ALIASES = {
    "rsi": "rsi_14",
    "rsi_14": "rsi_14",
    "stoch_k": "stoch_k",
    "stochk": "stoch_k",
    "slowk": "stoch_k",
    "sma_20": "sma_20",
    "sma20": "sma_20",
    "sma_50": "sma_50",
    "sma50": "sma_50",
    "sma_200": "sma_200",
    "sma200": "sma_200",
    "macd": "macd_line",
    "macd_line": "macd_line",
    "macd_signal": "macd_signal",
    "macd_signal_line": "macd_signal",
    "macd_histogram": "macd_histogram",
    "bb_upper": "bb_upper",
    "bb_upper_20": "bb_upper",
    "bb_lower": "bb_lower",
    "bb_lower_20": "bb_lower",
    "bb_middle": "bb_middle",
    "bb_middle_20": "bb_middle",
    "close": "close",
}


def normalize_indicator_name(name: str) -> str:
    """Normalize indicator name using aliases."""
    return INDICATOR_ALIASES.get(name.lower(), name)


@dataclass
class ConditionResult:
    """Result of evaluating a single condition."""
    name: str
    met: bool
    current_value: float
    required_value: str  # e.g., "< 30", "> 50"
    confidence: float  # 0-100


@dataclass
class ModelConsensus:
    """Result of 30-fold ensemble model prediction."""
    total_models: int = 30
    models_agree: int = 0  # How many agree with action
    action: str = "HOLD"  # "HOLD" or "EXIT"
    confidence: float = 0.0  # Average probability
    available: bool = False  # Whether model exists for this signal

    @property
    def agreement_ratio(self) -> str:
        """Return agreement as X/30 format."""
        return f"{self.models_agree}/{self.total_models}"


@dataclass
class SignalPreview:
    """Preview of a single signal's current state."""
    symbol: str
    direction: str
    signal_name: str
    timeframe: str
    confidence: float  # 0-100
    conditions: List[ConditionResult]
    next_candle_close: datetime
    indicator_values: Dict[str, float]
    model_consensus: Optional[ModelConsensus] = None  # 30-fold model agreement
    status: str = "APPROACHING"      # READY/LOCKED/BLOCKED/CONDITIONS_MET/APPROACHING
    blocked_reason: str = ""         # e.g., "position_exists", "signal_locked"


@dataclass
class CandleGroup:
    """Group of signals that close at the same time."""
    close_time: datetime
    seconds_until: int
    timeframes: List[str]
    signals: List[SignalPreview]

    @property
    def signal_count(self) -> int:
        return len(self.signals)

    @property
    def high_confidence_count(self) -> int:
        return sum(1 for s in self.signals if s.confidence >= 80)


class SignalPreviewEvaluator:
    """
    Evaluates signals for PREVIEW purposes only.
    Does NOT trigger trades - read-only evaluation.
    """

    def __init__(self, db_pool=None, config: Dict = None, exit_evaluator=None,
                 entry_evaluator=None, position_manager=None):
        """
        Initialize the evaluator.

        Args:
            db_pool: Database connection pool for storing snapshots
            config: Paper trading config with signal definitions
            exit_evaluator: HybridExitEvaluator for 30-model consensus (optional)
            entry_evaluator: HybridEntryEvaluator for crossover/lock status (optional)
            position_manager: PositionManager for open position checks (optional)
        """
        self.exit_evaluator = exit_evaluator
        self.entry_evaluator = entry_evaluator
        self.position_manager = position_manager
        self.db_pool = db_pool
        self.config = config or {}
        self._indicator_cache: Dict[str, Dict] = {}
        self._cache_timestamp: Optional[datetime] = None

    def get_next_candle_closes(self) -> Dict[str, datetime]:
        """
        Calculate next candle close time for each timeframe.

        Returns:
            Dict mapping timeframe to next close datetime (UTC)
        """
        now = datetime.now(timezone.utc)
        current_ts = int(now.timestamp())

        closes = {}
        for tf, minutes in TIMEFRAME_MINUTES.items():
            period_seconds = minutes * 60
            next_close_ts = ((current_ts // period_seconds) + 1) * period_seconds
            closes[tf] = datetime.fromtimestamp(next_close_ts, tz=timezone.utc)

        return closes

    def get_seconds_until_close(self, timeframe: str) -> int:
        """Get seconds until next candle close for a timeframe."""
        closes = self.get_next_candle_closes()
        if timeframe not in closes:
            return 0
        now = datetime.now(timezone.utc)
        delta = closes[timeframe] - now
        return max(0, int(delta.total_seconds()))

    async def get_indicators(self, symbol: str, timeframe: str) -> Dict[str, float]:
        """
        Get current indicator values for a symbol/timeframe.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Timeframe (e.g., "H1")

        Returns:
            Dict of indicator name to current value
        """
        cache_key = f"{symbol}_{timeframe}"

        # Return cached if fresh (within 60 seconds)
        if self._cache_timestamp and cache_key in self._indicator_cache:
            age = (datetime.now(timezone.utc) - self._cache_timestamp).total_seconds()
            if age < 60:
                return self._indicator_cache[cache_key]

        if not self.db_pool:
            logger.warning("No database pool available for indicator lookup")
            return {}

        try:
            # Query latest indicators from database
            table_name = f"technical_indicator_{symbol.lower()}"
            query = f"""
                SELECT
                    rsi_14, stoch_k, stoch_d,
                    sma_20, sma_50, sma_200,
                    ema_12, ema_26, ema_50,
                    macd_line, macd_signal, macd_histogram,
                    bb_upper_20, bb_middle_20, bb_lower_20,
                    atr_14, close
                FROM {table_name}
                WHERE timeframe = $1
                ORDER BY timestamp DESC
                LIMIT 1
            """

            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(query, timeframe)

            if not row:
                return {}

            indicators = {
                "rsi_14": float(row["rsi_14"]) if row["rsi_14"] else None,
                "stoch_k": float(row["stoch_k"]) if row["stoch_k"] else None,
                "stoch_d": float(row["stoch_d"]) if row["stoch_d"] else None,
                "sma_20": float(row["sma_20"]) if row["sma_20"] else None,
                "sma_50": float(row["sma_50"]) if row["sma_50"] else None,
                "sma_200": float(row["sma_200"]) if row["sma_200"] else None,
                "ema_12": float(row["ema_12"]) if row["ema_12"] else None,
                "ema_26": float(row["ema_26"]) if row["ema_26"] else None,
                "ema_50": float(row["ema_50"]) if row["ema_50"] else None,
                "macd_line": float(row["macd_line"]) if row["macd_line"] else None,
                "macd_signal": float(row["macd_signal"]) if row["macd_signal"] else None,
                "macd_histogram": float(row["macd_histogram"]) if row["macd_histogram"] else None,
                "bb_upper": float(row["bb_upper_20"]) if row["bb_upper_20"] else None,
                "bb_middle": float(row["bb_middle_20"]) if row["bb_middle_20"] else None,
                "bb_lower": float(row["bb_lower_20"]) if row["bb_lower_20"] else None,
                "atr_14": float(row["atr_14"]) if row["atr_14"] else None,
                "close": float(row["close"]) if row["close"] else None,
            }

            # Update cache
            self._indicator_cache[cache_key] = indicators
            self._cache_timestamp = datetime.now(timezone.utc)

            return indicators

        except Exception as e:
            logger.error(f"Failed to get indicators for {symbol} {timeframe}: {e}")
            return {}

    def calculate_threshold_confidence(
        self,
        current: float,
        threshold: float,
        condition: str
    ) -> Tuple[bool, float]:
        """
        Calculate confidence for threshold conditions (RSI < 30, Stoch > 80, etc.)

        Args:
            current: Current indicator value
            threshold: Threshold value
            condition: "less_than" or "greater_than"

        Returns:
            Tuple of (condition_met, confidence_percentage)
        """
        if current is None:
            return (False, 0.0)

        if condition == "less_than":
            if current <= threshold:
                return (True, 100.0)
            # How close? Within 20% of threshold = high confidence
            distance = current - threshold
            max_distance = threshold * 0.5  # 50% away = 0% confidence
            if max_distance <= 0:
                return (False, 0.0)
            confidence = max(0, (1 - distance / max_distance)) * 100
            return (False, min(confidence, 99.9))

        elif condition == "greater_than":
            if current >= threshold:
                return (True, 100.0)
            distance = threshold - current
            max_distance = threshold * 0.5
            if max_distance <= 0:
                return (False, 0.0)
            confidence = max(0, (1 - distance / max_distance)) * 100
            return (False, min(confidence, 99.9))

        return (False, 0.0)

    def calculate_crossover_confidence(
        self,
        fast: float,
        slow: float,
        direction: str = "above"
    ) -> Tuple[bool, float]:
        """
        Calculate confidence for crossover conditions.

        Args:
            fast: Fast indicator value (e.g., SMA20)
            slow: Slow indicator value (e.g., SMA50)
            direction: "above" (fast > slow) or "below" (fast < slow)

        Returns:
            Tuple of (condition_met, confidence_percentage)
        """
        if fast is None or slow is None:
            return (False, 0.0)

        if direction == "above":
            if fast > slow:
                return (True, 100.0)
            # How close to crossing?
            gap = slow - fast
            relative_gap = gap / slow if slow != 0 else 1
            confidence = max(0, (1 - relative_gap * 10)) * 100  # 10% gap = 0%
            return (False, min(confidence, 99.9))

        elif direction == "below":
            if fast < slow:
                return (True, 100.0)
            gap = fast - slow
            relative_gap = gap / slow if slow != 0 else 1
            confidence = max(0, (1 - relative_gap * 10)) * 100
            return (False, min(confidence, 99.9))

        return (False, 0.0)

    def evaluate_signal_conditions(
        self,
        signal_name: str,
        direction: str,
        indicators: Dict[str, float]
    ) -> List[ConditionResult]:
        """
        Evaluate all conditions for a signal using config/signal_definitions.yaml.

        This now uses the YAML definitions as the single source of truth,
        matching the CLI monitor behavior.

        Args:
            signal_name: Name of the signal (e.g., "MACD_Stoch_long")
            direction: "long" or "short"
            indicators: Current indicator values

        Returns:
            List of ConditionResult for each condition
        """
        # Load signal definitions from YAML (cached)
        signal_defs = load_signal_definitions()
        signal_def = signal_defs.get(signal_name, {})
        yaml_conditions = signal_def.get("conditions", [])

        results = []

        # If we have YAML definitions, use them as source of truth
        if yaml_conditions:
            for cond in yaml_conditions:
                indicator_name = normalize_indicator_name(cond.get("indicator", ""))
                operator = cond.get("operator", ">")
                compare_to = cond.get("compare_to")
                value = cond.get("value")

                current_value = indicators.get(indicator_name)
                if current_value is None:
                    # Try common aliases
                    for alias, canonical in INDICATOR_ALIASES.items():
                        if canonical == indicator_name:
                            current_value = indicators.get(alias)
                            if current_value is not None:
                                break

                # Determine threshold and evaluate
                if compare_to:
                    # Indicator-to-indicator comparison (e.g., SMA20 > SMA50)
                    compare_indicator = normalize_indicator_name(compare_to)
                    threshold = indicators.get(compare_indicator)
                    if threshold is None:
                        results.append(ConditionResult(
                            name=f"{indicator_name} {operator} {compare_to}",
                            met=False,
                            current_value=current_value or 0,
                            required_value=f"{operator} {compare_to} (no data)",
                            confidence=0
                        ))
                        continue

                    cross_dir = "above" if operator in (">", ">=") else "below"
                    met, conf = self.calculate_crossover_confidence(current_value, threshold, cross_dir)

                    # Format nice name
                    ind_name = indicator_name.upper().replace("_", "")
                    cmp_name = compare_to.upper().replace("_", "")
                    name = f"{ind_name} {operator} {cmp_name}"
                    req_val = f"{operator} {threshold:.5f}" if threshold and threshold < 10 else f"{operator} {threshold:.2f}" if threshold else f"{operator} ?"

                    results.append(ConditionResult(
                        name=name,
                        met=met,
                        current_value=current_value or 0,
                        required_value=req_val,
                        confidence=conf
                    ))

                elif value is not None:
                    # Threshold comparison (e.g., Stoch_K < 30)
                    threshold = float(value)
                    condition_type = "less_than" if operator in ("<", "<=") else "greater_than"
                    met, conf = self.calculate_threshold_confidence(current_value, threshold, condition_type)

                    # Format nice name
                    ind_friendly = indicator_name.replace("_", " ").title().replace(" ", "_")
                    name = f"{ind_friendly} {operator} {int(threshold) if threshold == int(threshold) else threshold}"

                    results.append(ConditionResult(
                        name=name,
                        met=met,
                        current_value=current_value or 0,
                        required_value=f"{operator} {int(threshold) if threshold == int(threshold) else threshold}",
                        confidence=conf
                    ))

            if results:
                return results

        # Fallback: Use inferred conditions if YAML not found (legacy behavior)
        return self._evaluate_inferred_conditions(signal_name, direction, indicators)

    def _evaluate_inferred_conditions(
        self,
        signal_name: str,
        direction: str,
        indicators: Dict[str, float]
    ) -> List[ConditionResult]:
        """
        Legacy fallback: Infer conditions from signal name.
        Used only when signal_definitions.yaml doesn't have the signal.
        """
        results = []
        signal_lower = signal_name.lower()

        # RSI conditions
        if "rsi" in signal_lower:
            rsi = indicators.get("rsi_14")
            if "oversold" in signal_lower or (direction == "long" and "rsi" in signal_lower):
                threshold = 30
                met, conf = self.calculate_threshold_confidence(rsi, threshold, "less_than")
                results.append(ConditionResult(
                    name="RSI < 30",
                    met=met,
                    current_value=rsi or 0,
                    required_value="< 30",
                    confidence=conf
                ))
            elif "overbought" in signal_lower or (direction == "short" and "rsi" in signal_lower):
                threshold = 70
                met, conf = self.calculate_threshold_confidence(rsi, threshold, "greater_than")
                results.append(ConditionResult(
                    name="RSI > 70",
                    met=met,
                    current_value=rsi or 0,
                    required_value="> 70",
                    confidence=conf
                ))
            else:
                # Generic RSI condition based on direction
                if direction == "long":
                    threshold = 50
                    met, conf = self.calculate_threshold_confidence(rsi, threshold, "less_than")
                    results.append(ConditionResult(
                        name="RSI < 50",
                        met=met,
                        current_value=rsi or 0,
                        required_value="< 50",
                        confidence=conf
                    ))
                else:
                    threshold = 50
                    met, conf = self.calculate_threshold_confidence(rsi, threshold, "greater_than")
                    results.append(ConditionResult(
                        name="RSI > 50",
                        met=met,
                        current_value=rsi or 0,
                        required_value="> 50",
                        confidence=conf
                    ))

        # Stochastic conditions - now try to get threshold from YAML first
        if "stoch" in signal_lower:
            stoch_k = indicators.get("stoch_k")
            # Extract threshold from signal name if present (e.g., Stoch_K_oversold_long_25)
            threshold = 30 if direction == "long" else 70  # Better defaults matching YAML
            for part in signal_name.split("_"):
                if part.isdigit():
                    threshold = int(part)
                    break

            if direction == "long":
                met, conf = self.calculate_threshold_confidence(stoch_k, threshold, "less_than")
                results.append(ConditionResult(
                    name=f"Stoch_K < {threshold}",
                    met=met,
                    current_value=stoch_k or 0,
                    required_value=f"< {threshold}",
                    confidence=conf
                ))
            else:
                met, conf = self.calculate_threshold_confidence(stoch_k, threshold, "greater_than")
                results.append(ConditionResult(
                    name=f"Stoch_K > {threshold}",
                    met=met,
                    current_value=stoch_k or 0,
                    required_value=f"> {threshold}",
                    confidence=conf
                ))

        # SMA crossover conditions
        if "sma" in signal_lower:
            sma_20 = indicators.get("sma_20")
            sma_50 = indicators.get("sma_50")
            sma_200 = indicators.get("sma_200")

            if "20" in signal_lower and "50" in signal_lower:
                cross_dir = "above" if direction == "long" else "below"
                met, conf = self.calculate_crossover_confidence(sma_20, sma_50, cross_dir)
                results.append(ConditionResult(
                    name=f"SMA20 {'>' if direction == 'long' else '<'} SMA50",
                    met=met,
                    current_value=sma_20 or 0,
                    required_value=f"{'>' if direction == 'long' else '<'} {sma_50 or 0:.5f}",
                    confidence=conf
                ))
            if "50" in signal_lower and "200" in signal_lower:
                cross_dir = "above" if direction == "long" else "below"
                met, conf = self.calculate_crossover_confidence(sma_50, sma_200, cross_dir)
                results.append(ConditionResult(
                    name=f"SMA50 {'>' if direction == 'long' else '<'} SMA200",
                    met=met,
                    current_value=sma_50 or 0,
                    required_value=f"{'>' if direction == 'long' else '<'} {sma_200 or 0:.5f}",
                    confidence=conf
                ))
            if "20" in signal_lower and "200" in signal_lower and "50" not in signal_lower:
                cross_dir = "above" if direction == "long" else "below"
                met, conf = self.calculate_crossover_confidence(sma_20, sma_200, cross_dir)
                results.append(ConditionResult(
                    name=f"SMA20 {'>' if direction == 'long' else '<'} SMA200",
                    met=met,
                    current_value=sma_20 or 0,
                    required_value=f"{'>' if direction == 'long' else '<'} {sma_200 or 0:.5f}",
                    confidence=conf
                ))

        # MACD conditions
        if "macd" in signal_lower:
            macd_line = indicators.get("macd_line")
            macd_signal = indicators.get("macd_signal")
            macd_hist = indicators.get("macd_histogram")

            if direction == "long":
                # MACD > 0 or MACD > Signal
                if macd_hist is not None:
                    met, conf = self.calculate_threshold_confidence(macd_hist, 0, "greater_than")
                    results.append(ConditionResult(
                        name="MACD Histogram > 0",
                        met=met,
                        current_value=macd_hist,
                        required_value="> 0",
                        confidence=conf
                    ))
            else:
                if macd_hist is not None:
                    met, conf = self.calculate_threshold_confidence(macd_hist, 0, "less_than")
                    results.append(ConditionResult(
                        name="MACD Histogram < 0",
                        met=met,
                        current_value=macd_hist,
                        required_value="< 0",
                        confidence=conf
                    ))

        # Bollinger Bands conditions
        if "bb" in signal_lower:
            close = indicators.get("close")
            bb_lower = indicators.get("bb_lower")
            bb_upper = indicators.get("bb_upper")

            if direction == "long" and bb_lower is not None and close is not None:
                # Price near lower band
                distance = close - bb_lower
                band_width = (bb_upper - bb_lower) if bb_upper and bb_lower else 1
                relative_pos = distance / band_width if band_width > 0 else 0.5
                # Closer to lower band = higher confidence for long
                conf = max(0, (1 - relative_pos * 2)) * 100
                met = relative_pos < 0.2  # Within 20% of lower band
                results.append(ConditionResult(
                    name="Price near BB Lower",
                    met=met,
                    current_value=close,
                    required_value=f"< {bb_lower:.5f}" if bb_lower else "< BB Lower",
                    confidence=min(conf, 100)
                ))
            elif direction == "short" and bb_upper is not None and close is not None:
                distance = bb_upper - close
                band_width = (bb_upper - bb_lower) if bb_upper and bb_lower else 1
                relative_pos = distance / band_width if band_width > 0 else 0.5
                conf = max(0, (1 - relative_pos * 2)) * 100
                met = relative_pos < 0.2
                results.append(ConditionResult(
                    name="Price near BB Upper",
                    met=met,
                    current_value=close,
                    required_value=f"> {bb_upper:.5f}" if bb_upper else "> BB Upper",
                    confidence=min(conf, 100)
                ))

        # Triple Momentum (MACD + RSI + Stoch)
        if "triple" in signal_lower and "momentum" in signal_lower:
            # Already handled by individual conditions above
            pass

        # If no specific conditions matched, add a generic one
        if not results:
            results.append(ConditionResult(
                name="Signal Active",
                met=True,
                current_value=0,
                required_value="Active",
                confidence=50.0
            ))

        return results

    def calculate_overall_confidence(self, conditions: List[ConditionResult]) -> float:
        """
        Calculate overall signal confidence from individual conditions.
        Uses geometric mean to penalize low individual scores.

        Args:
            conditions: List of condition results

        Returns:
            Overall confidence 0-100
        """
        if not conditions:
            return 0.0

        # Geometric mean
        product = 1.0
        for c in conditions:
            product *= (c.confidence / 100.0)

        geometric_mean = (product ** (1 / len(conditions))) * 100
        return round(geometric_mean, 1)

    def get_model_consensus(
        self,
        symbol: str,
        direction: str,
        signal_name: str,
        timeframe: str
    ) -> ModelConsensus:
        """
        Get 30-fold model consensus for a signal.

        This queries the HybridExitEvaluator to see how many of the 30 models
        would agree on the exit decision if a position were open.

        Args:
            symbol: Trading symbol
            direction: "long" or "short"
            signal_name: Signal name
            timeframe: Timeframe

        Returns:
            ModelConsensus with agreement count
        """
        consensus = ModelConsensus()

        if not self.exit_evaluator:
            return consensus

        try:
            # Build model key (matches hybrid_v4 naming)
            model_key = f"{symbol.upper()}_{direction}"

            # Check if model exists
            if model_key not in self.exit_evaluator.models:
                return consensus

            ensemble = self.exit_evaluator.models.get(model_key, [])
            if not ensemble:
                return consensus

            consensus.total_models = len(ensemble)
            consensus.available = True

            # We need an observation to get predictions
            # For preview, we can use dummy obs or try to get real indicators
            # Here we just report availability and count of loaded models
            # Full prediction would require building the observation vector

            # For now, mark as available with loaded model count
            # The actual prediction would happen at candle close
            consensus.models_agree = consensus.total_models
            consensus.action = "AVAILABLE"
            consensus.confidence = 100.0

            return consensus

        except Exception as e:
            logger.error(f"Failed to get model consensus for {symbol}: {e}")
            return consensus

    # Status sort priority (lower = higher priority)
    STATUS_PRIORITY = {
        "READY": 1,
        "LOCKED": 2,
        "BLOCKED": 3,
        "CONDITIONS_MET": 4,
        "APPROACHING": 5,
    }

    def _compute_status(
        self,
        symbol: str,
        direction: str,
        signal_name: str,
        timeframe: str,
        conditions: list,
    ) -> Tuple[str, str]:
        """Compute signal status based on position/crossover/lock state.

        Args:
            symbol: Trading symbol
            direction: "long" or "short"
            signal_name: Signal name
            timeframe: Signal timeframe
            conditions: List of ConditionResult with .met bool

        Returns:
            Tuple of (status, blocked_reason)
        """
        # 1. Check for existing position from THIS signal → BLOCKED
        # Per user requirement: each signal can open independently, even for same symbol+direction.
        # Only BLOCK if the same signal_name already has an open position.
        if self.position_manager is not None:
            try:
                positions = self.position_manager.get_positions(symbol, direction)
                for pos in positions:
                    # entry_model format: "Hybrid_V4 + SIGNAL_NAME"
                    if pos.entry_model and signal_name in pos.entry_model:
                        return "BLOCKED", "position_exists"
            except Exception as e:
                logger.warning(f"Position check failed for {symbol} {direction}: {e}")

        # 2. If not all conditions met → APPROACHING
        all_met = conditions and all(c.met for c in conditions)
        if not all_met:
            return "APPROACHING", ""

        # All conditions met (95%+), check crossover status
        if self.entry_evaluator is not None:
            try:
                has_crossover, is_locked = self.entry_evaluator.check_crossover_status(
                    symbol, signal_name, direction, timeframe
                )

                if is_locked:
                    return "LOCKED", "signal_locked"

                if has_crossover:
                    return "READY", ""

                # All conditions met but no fresh crossover
                return "CONDITIONS_MET", ""

            except Exception as e:
                logger.warning(f"Crossover check failed for {symbol}:{signal_name}: {e}")

        # Fallback: no entry evaluator, can't determine crossover status
        return "CONDITIONS_MET", ""

    async def evaluate_signal(
        self,
        symbol: str,
        signal_config: Dict
    ) -> Optional[SignalPreview]:
        """
        Evaluate a single signal and return preview.

        Args:
            symbol: Trading symbol
            signal_config: Signal configuration from paper_trading.yaml

        Returns:
            SignalPreview or None if evaluation fails
        """
        try:
            signal_name = signal_config.get("signal", "")
            direction = signal_config.get("direction", "long")
            timeframe = signal_config.get("timeframe", "H1")

            if not signal_config.get("enabled", True):
                return None

            # Get indicators
            indicators = await self.get_indicators(symbol, timeframe)
            if not indicators:
                return None

            # Evaluate conditions
            conditions = self.evaluate_signal_conditions(signal_name, direction, indicators)

            # Calculate overall confidence
            confidence = self.calculate_overall_confidence(conditions)

            # Get next candle close
            closes = self.get_next_candle_closes()
            next_close = closes.get(timeframe, datetime.now(timezone.utc))

            # Get 30-model consensus (if exit evaluator available)
            model_consensus = self.get_model_consensus(
                symbol, direction, signal_name, timeframe
            )

            # Compute actionable status
            status, blocked_reason = self._compute_status(
                symbol, direction, signal_name, timeframe, conditions
            )

            return SignalPreview(
                symbol=symbol,
                direction=direction,
                signal_name=signal_name,
                timeframe=timeframe,
                confidence=confidence,
                conditions=conditions,
                next_candle_close=next_close,
                indicator_values=indicators,
                model_consensus=model_consensus,
                status=status,
                blocked_reason=blocked_reason,
            )

        except Exception as e:
            logger.error(f"Failed to evaluate signal {signal_config}: {e}")
            return None

    async def evaluate_all_signals(self) -> List[SignalPreview]:
        """
        Evaluate all configured signals.

        Returns:
            List of SignalPreview for all signals
        """
        previews = []

        symbols_config = self.config.get("symbols", {})
        logger.info(f"SignalPreview config type: {type(self.config)}, symbols type: {type(symbols_config)}, config keys: {list(self.config.keys()) if isinstance(self.config, dict) else 'not a dict'}")

        for symbol, symbol_config in symbols_config.items():
            if not symbol_config.get("enabled", True):
                continue

            signals = symbol_config.get("signals", [])
            for signal_config in signals:
                preview = await self.evaluate_signal(symbol, signal_config)
                if preview:
                    previews.append(preview)

        return previews

    def group_by_candle_close(self, previews: List[SignalPreview]) -> List[CandleGroup]:
        """
        Group signals by their next candle close time.

        Args:
            previews: List of signal previews

        Returns:
            List of CandleGroup sorted by close time
        """
        # Group by close time
        groups: Dict[datetime, List[SignalPreview]] = {}

        for preview in previews:
            close_time = preview.next_candle_close
            if close_time not in groups:
                groups[close_time] = []
            groups[close_time].append(preview)

        # Convert to CandleGroup objects
        now = datetime.now(timezone.utc)
        result = []

        for close_time, signals in sorted(groups.items()):
            # Get unique timeframes in this group
            timeframes = sorted(set(s.timeframe for s in signals),
                              key=lambda tf: TIMEFRAME_MINUTES.get(tf, 0))

            # Sort signals by status priority (READY first), then confidence descending
            signals.sort(key=lambda s: (self.STATUS_PRIORITY.get(s.status, 5), -s.confidence))

            seconds_until = max(0, int((close_time - now).total_seconds()))

            result.append(CandleGroup(
                close_time=close_time,
                seconds_until=seconds_until,
                timeframes=timeframes,
                signals=signals
            ))

        return result

    async def save_snapshot(self, previews: List[SignalPreview]) -> int:
        """
        Save preview snapshot to database.

        Args:
            previews: List of signal previews

        Returns:
            Number of rows inserted
        """
        if not self.db_pool or not previews:
            return 0

        try:
            now = datetime.now(timezone.utc)

            # Batch insert
            values = []
            for p in previews:
                # Store full condition details for frontend display
                conditions_data = [
                    {
                        "name": c.name,
                        "met": c.met,
                        "current": f"{c.current_value:.4f}" if isinstance(c.current_value, (int, float)) else str(c.current_value),
                        "required": c.required_value,
                        "confidence": round(c.confidence, 1)
                    }
                    for c in p.conditions
                ]
                values.append((
                    now,
                    p.symbol,
                    p.direction,
                    p.signal_name,
                    p.timeframe,
                    p.confidence,
                    json.dumps(conditions_data),  # Full details, not just {name: met}
                    json.dumps(p.indicator_values),
                    p.next_candle_close,
                    p.status,
                    p.blocked_reason,
                ))

            query = """
                INSERT INTO signal_preview_snapshots
                (timestamp, symbol, direction, signal_name, timeframe,
                 confidence, conditions_met, indicator_values, next_candle_close,
                 status, blocked_reason)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """

            async with self.db_pool.acquire() as conn:
                await conn.executemany(query, values)

            logger.info(f"Saved {len(previews)} signal preview snapshots")
            return len(previews)

        except Exception as e:
            logger.error(f"Failed to save preview snapshot: {e}")
            return 0

    def to_api_response(self, groups: List[CandleGroup]) -> Dict:
        """
        Convert candle groups to API response format.

        Args:
            groups: List of CandleGroup

        Returns:
            Dict suitable for JSON API response
        """
        now = datetime.now(timezone.utc)

        candle_groups = []
        total_signals = 0
        total_high_conf = 0

        for group in groups:
            signals_data = []
            for s in group.signals:
                conditions_data = [
                    {
                        "name": c.name,
                        "met": c.met,
                        "current": f"{c.current_value:.4f}" if isinstance(c.current_value, float) else str(c.current_value),
                        "required": c.required_value
                    }
                    for c in s.conditions
                ]

                signal_entry = {
                    "symbol": s.symbol,
                    "direction": s.direction,
                    "signal_name": s.signal_name,
                    "timeframe": s.timeframe,
                    "confidence": s.confidence,
                    "conditions": conditions_data,
                    "status": s.status,
                    "blocked_reason": s.blocked_reason,
                }

                # Add model consensus (30-fold agreement) if available
                if s.model_consensus and s.model_consensus.available:
                    signal_entry["model_consensus"] = {
                        "agreement": s.model_consensus.agreement_ratio,
                        "models_agree": s.model_consensus.models_agree,
                        "total_models": s.model_consensus.total_models,
                        "action": s.model_consensus.action,
                        "confidence": s.model_consensus.confidence
                    }

                signals_data.append(signal_entry)

            candle_groups.append({
                "close_time": group.close_time.isoformat(),
                "seconds_until": group.seconds_until,
                "timeframes": group.timeframes,
                "signal_count": group.signal_count,
                "high_confidence_count": group.high_confidence_count,
                "signals": signals_data
            })

            total_signals += group.signal_count
            total_high_conf += group.high_confidence_count

        return {
            "data": {
                "candle_groups": candle_groups,
                "last_update": now.isoformat()
            },
            "metadata": {
                "total_signals": total_signals,
                "total_high_confidence": total_high_conf,
                "next_close": groups[0].close_time.isoformat() if groups else None,
                "timestamp": now.isoformat()
            }
        }
