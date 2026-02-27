"""
Hybrid Exit Evaluator for Paper Trading (Issue #494, Updated for #495, #538, #560).

Uses trained hybrid PPO models to decide when to exit positions.
This implements the exit side of the hybrid architecture where
entries are rule-based (signal patterns) and exits are RL-managed.

Key Features:
- Supports both hybrid_v2 (legacy) and hybrid_v4 (current) models
- hybrid_v4: Signal-specific models with 30-fold ensemble (Issue #560)
- hybrid_v2: Single model per symbol (backward compatible)
- Builds observation from position state and market data
- Predicts HOLD (0) vs CLOSE (1) actions

Model Versions:
- hybrid_v2: models/hybrid_v2/{symbol}/fold_001_seed_{seed}.zip (single model)
- hybrid_v4: models/hybrid_v4/{symbol}_{direction}_{signal}_{timeframe}/fold_XX.zip (30-fold ensemble)

Architecture:
    Position State + Market Data -> Observation -> PPO Model(s) -> HOLD/CLOSE
"""

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Model directories
DEFAULT_MODEL_DIR_V2 = "models/hybrid_v2"
DEFAULT_MODEL_DIR_V4 = "models/hybrid_v4"
DEFAULT_MODEL_DIR = DEFAULT_MODEL_DIR_V4  # Default to v4

# Model versions
MODEL_VERSION_V2 = "hybrid_v2"
MODEL_VERSION_V4 = "hybrid_v4"

# Action mappings
ACTION_HOLD = 0
ACTION_CLOSE = 1

# Symbol seeds used during training (for v2 backward compatibility)
SYMBOL_SEEDS: Dict[str, int] = {
    "EURUSD": 1234,
    "GBPUSD": 42,
    "USDJPY": 5678,
    "EURJPY": 456,
    "XAGUSD": 2024,
}

# Number of folds for v4 ensemble
V4_NUM_FOLDS = 30

# Parallel model loading workers (Issue #623: Speed up model loading)
# Set PARALLEL_MODEL_WORKERS=1 to disable parallel loading
PARALLEL_MODEL_WORKERS = int(os.environ.get("PARALLEL_MODEL_WORKERS", "4"))

# Pip values per symbol (Issue #571: Added all deployed symbols)
PIP_VALUES: Dict[str, float] = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "EURJPY": 0.01,
    "XAGUSD": 0.001,
    "USDCAD": 0.0001,  # Issue #571
    "EURCAD": 0.0001,  # Issue #571
    "EURGBP": 0.0001,  # Issue #571: For future deployment
    "USDCHF": 0.0001,  # Issue #571: For future deployment
}

# SL/TP configuration (30/30 pips - symmetric 1:1 R:R per Issue #526, #588)
# Matches mandatory 30/30 SL/TP rule (see .claude/rules/sl-tp-configuration.md)
SL_PIPS = 30
TP_PIPS = 30

# Max bars in position before forcing close
MAX_POSITION_BARS = 50

# Indicator table names per symbol (Issue #588)
INDICATOR_TABLES: Dict[str, str] = {
    "EURUSD": "technical_indicators",
    "GBPUSD": "technical_indicator_gbpusd",
    "USDJPY": "technical_indicator_usdjpy",
    "EURJPY": "technical_indicator_eurjpy",
    "XAGUSD": "technical_indicator_xagusd",
    "USDCAD": "technical_indicator_usdcad",
    "EURCAD": "technical_indicator_eurcad",
    "EURGBP": "technical_indicator_eurgbp",
    "USDCHF": "technical_indicator_usdchf",
}

# All indicator columns in DB (16 indicators + OHLCV)
INDICATOR_COLUMNS = (
    "timestamp, open, high, low, close, volume, "
    "sma_20, sma_50, sma_200, ema_12, ema_26, ema_50, "
    "rsi_14, atr_14, bb_upper_20, bb_middle_20, bb_lower_20, "
    "macd_line, macd_signal, macd_histogram, stoch_k, stoch_d"
)

# Primary indicator columns for normalization (16 features, indices 0-15)
PRIMARY_INDICATOR_COLS = [
    "sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "ema_50",
    "rsi_14", "atr_14",
    "bb_upper_20", "bb_middle_20", "bb_lower_20",
    "macd_line", "macd_signal", "macd_histogram",
    "stoch_k", "stoch_d",
]

# Context timeframes for base features (indices 32-159, 8 TFs x 16 features each)
BASE_CONTEXT_TIMEFRAMES = ["H1", "H2", "H3", "H4", "H6", "H8", "H12", "D1"]

# V2 timeframes (indices 222-321, 8 TFs x 12 features each)
V2_TIMEFRAMES = ["W1", "D1", "H12", "H8", "H6", "H4", "H2", "H1"]
V2_FEATURES_PER_TF = 12

# Cache TTL in seconds
INDICATOR_CACHE_TTL = 300  # 5 minutes

# Volatility constants
ATR_ROLLING_WINDOW = 252
VOLATILITY_REGIME_LOW = 0.33
VOLATILITY_REGIME_HIGH = 0.67

# Observation dimensions
BASE_OBS_DIM = 222  # Base multi-TF features
D1_FEATURES_DIM = 15  # D1 daily timeframe context features
POSITION_DIM = 5  # Position features
TOTAL_OBS_DIM = BASE_OBS_DIM + D1_FEATURES_DIM + POSITION_DIM  # 242

# Legacy alias for backward compatibility with tests
MULTI_TF_V2_DIM = D1_FEATURES_DIM


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class HybridExitSignal:
    """Exit signal generated by hybrid RL model.

    Attributes:
        symbol: Trading symbol
        action: Predicted action (HOLD=0, CLOSE=1)
        confidence: Model confidence (0.0 to 1.0)
        timestamp: UTC timestamp of evaluation
        position_bars: Bars in position
        unrealized_pnl_pips: Current unrealized P&L in pips
        reason: Human-readable reason for signal
        model_version: Model identifier
        direction: Position direction (for v4 models)
    """

    symbol: str
    action: int
    confidence: float
    timestamp: datetime
    position_bars: int
    unrealized_pnl_pips: float
    reason: str
    model_version: str = MODEL_VERSION_V4
    direction: Optional[str] = None  # "long" or "short" for v4
    ensemble_meta: Optional[Dict[str, Any]] = None  # Vote breakdown from v4 ensemble

    @property
    def should_close(self) -> bool:
        """Returns True if action is CLOSE."""
        return self.action == ACTION_CLOSE


# =============================================================================
# Position State Tracker
# =============================================================================


@dataclass
class PositionState:
    """Track position state for observation building.

    Attributes:
        symbol: Trading symbol
        direction: 1 for LONG, -1 for SHORT
        entry_price: Entry price
        entry_time: Entry timestamp
        current_price: Current market price
        sl_price: Stop loss price
        tp_price: Take profit price
        bars_held: Number of bars since entry
        signal_name: Signal that opened the position (Issue #560)
        timeframe: Timeframe of the signal model (Issue #560)
    """

    symbol: str
    direction: int  # 1=LONG, -1=SHORT
    entry_price: float
    entry_time: datetime
    current_price: float
    sl_price: float
    tp_price: float
    bars_held: int = 0
    signal_name: Optional[str] = None  # e.g., "Stoch_RSI_long_15_25"
    timeframe: Optional[str] = None  # e.g., "H1", "H4", "M30"

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L in price terms."""
        if self.direction == 1:  # LONG
            return self.current_price - self.entry_price
        else:  # SHORT
            return self.entry_price - self.current_price

    @property
    def unrealized_pnl_pips(self) -> float:
        """Calculate unrealized P&L in pips."""
        pip_value = PIP_VALUES.get(self.symbol.upper(), 0.0001)
        return self.unrealized_pnl / pip_value

    @property
    def distance_to_sl(self) -> float:
        """Calculate distance to SL in price terms."""
        if self.direction == 1:  # LONG
            return self.current_price - self.sl_price
        else:  # SHORT
            return self.sl_price - self.current_price

    @property
    def distance_to_tp(self) -> float:
        """Calculate distance to TP in price terms."""
        if self.direction == 1:  # LONG
            return self.tp_price - self.current_price
        else:  # SHORT
            return self.current_price - self.tp_price


# =============================================================================
# Hybrid Exit Evaluator
# =============================================================================


class HybridExitEvaluator:
    """Hybrid exit evaluator using trained RL models.

    Supports both hybrid_v2 (single model per symbol) and hybrid_v4
    (direction-specific, 30-fold ensemble) models.

    Usage (v4 - recommended):
        evaluator = HybridExitEvaluator(
            symbols=["GBPUSD"],
            directions=["long"],  # Specify direction for v4
            model_dir="models/hybrid_v4",
            model_version="hybrid_v4",
        )

    Usage (v2 - backward compatible):
        evaluator = HybridExitEvaluator(
            symbols=["EURUSD", "GBPUSD"],
            model_dir="models/hybrid_v2",
            model_version="hybrid_v2",
        )

        # Build position state
        state = PositionState(
            symbol="GBPUSD",
            direction=1,  # LONG
            entry_price=1.2650,
            entry_time=datetime.now(),
            current_price=1.2680,
            sl_price=1.2620,
            tp_price=1.2680,
            bars_held=10,
        )

        signal = evaluator.evaluate_exit(state)
        if signal.should_close:
            print("RL recommends closing position")
    """

    # Issue #598: Fail-closed validation gate toggle.
    # When True, unknown/missing models are REJECTED (safe default).
    # When False, revert to legacy fail-open behavior for gradual rollout.
    # CHANGED: Set to False to trust init container's filtering via approved_models.yaml
    # The init container already syncs ONLY Phase 5 passed models (PF >= 1.2)
    # Double validation against pipeline_tracking.json was causing false rejections
    _FAIL_CLOSED_ENABLED: bool = False

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        directions: Optional[List[str]] = None,
        model_dir: str = DEFAULT_MODEL_DIR,
        model_version: str = MODEL_VERSION_V4,
        confidence_threshold: float = 0.55,
        ensemble_method: str = "mean",  # "mean" or "majority"
        db_manager: Any = None,
    ) -> None:
        """Initialize HybridExitEvaluator.

        Args:
            symbols: List of symbols to load models for
            directions: List of directions for v4 models ("long", "short")
            model_dir: Directory containing hybrid models
            model_version: Model version ("hybrid_v2" or "hybrid_v4")
            confidence_threshold: Minimum confidence for exit signal
            ensemble_method: For v4 ensemble - "mean" (average probs) or "majority" (voting)
            db_manager: Database manager for indicator queries (Issue #588)
        """
        self.symbols = symbols or list(SYMBOL_SEEDS.keys())
        self.directions = directions or ["long", "short"]
        self.model_dir = Path(model_dir)
        self.model_version = model_version
        self.confidence_threshold = confidence_threshold
        self.ensemble_method = ensemble_method

        # Model storage
        # For v2: Dict[symbol, model]
        # For v4: Dict[f"{symbol}_{direction}", List[model]] (ensemble)
        self.models: Dict[str, Any] = {}
        self.model_paths: Dict[str, List[str]] = {}
        self.model_hashes: Dict[str, str] = {}

        # Issue #588: Database access for indicator queries
        self.db_manager = db_manager

        # Issue #588: Indicator cache with TTL
        self._indicator_cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamps: Dict[str, float] = {}

        # Issue #588: Trade history for self-awareness features
        self._trade_history: List[Dict[str, Any]] = []
        self._consecutive_wins: int = 0
        self._consecutive_losses: int = 0
        self._last_exit_bar: Optional[int] = None
        self._current_bar_counter: int = 0
        self._profit_factor: float = 0.0

        # Load models
        if self.model_version == MODEL_VERSION_V4:
            self._load_models_v4()
        else:
            self._load_models_v2()

        logger.info(
            f"HybridExitEvaluator initialized: {len(self.models)} model(s) loaded "
            f"(version={self.model_version})"
        )

    def record_trade(
        self,
        trade: Any = None,
        *,
        pnl_pips: Optional[float] = None,
        exit_reason: str = "unknown",
        bars_held: int = 0,
    ) -> None:
        """Record a closed trade for self-awareness feature computation.

        Supports three calling conventions:
        1. record_trade(trade_obj) - Trade object with pnl_pips attribute
        2. record_trade(trade_dict) - Dict with 'pnl_pips' key
        3. record_trade(pnl_pips=10.0) - Direct keyword argument

        Args:
            trade: Trade object or dict (optional).
            pnl_pips: Direct PnL in pips (keyword-only, optional).
            exit_reason: Exit reason string.
            bars_held: Number of bars the position was held.
        """
        # Resolve pnl_pips from the various calling conventions
        resolved_pnl: float = 0.0
        resolved_reason: str = exit_reason
        resolved_bars: int = bars_held

        if pnl_pips is not None:
            # Direct keyword call: record_trade(pnl_pips=10.0)
            resolved_pnl = float(pnl_pips)
        elif trade is not None:
            if hasattr(trade, "pnl_pips"):
                resolved_pnl = float(trade.pnl_pips)
                resolved_reason = str(getattr(trade, "exit_reason", "unknown"))
                resolved_bars = getattr(trade, "bars_held", 0)
                if resolved_bars == 0 and hasattr(trade, "entry_time") and hasattr(trade, "exit_time"):
                    delta = (trade.exit_time - trade.entry_time).total_seconds()
                    resolved_bars = int(delta / 1800)  # M30 bars
            elif isinstance(trade, dict):
                resolved_pnl = float(trade.get("pnl_pips", 0.0))
                resolved_reason = str(trade.get("exit_reason", "unknown"))
                resolved_bars = int(trade.get("bars_held", 0))
            else:
                return
        else:
            return

        is_win = resolved_pnl > 0
        is_sl = "stop_loss" in resolved_reason.lower() or "sl" in resolved_reason.lower()
        is_tp = "take_profit" in resolved_reason.lower() or "tp" in resolved_reason.lower()

        self._trade_history.append({
            "pnl": resolved_pnl,
            "outcome": "take_profit" if is_tp else ("stop_loss" if is_sl else ("win" if is_win else "loss")),
            "bars_held": resolved_bars,
            "exit_bar": self._current_bar_counter,
        })

        # Update streaks
        if is_win:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0

        self._last_exit_bar = self._current_bar_counter
        self._current_bar_counter += 1

        # Update profit factor
        total_wins = sum(t["pnl"] for t in self._trade_history if t.get("pnl", 0) > 0)
        total_losses = abs(sum(t["pnl"] for t in self._trade_history if t.get("pnl", 0) < 0))
        self._profit_factor = total_wins / total_losses if total_losses > 0 else total_wins if total_wins > 0 else 0.0

    def _fetch_indicators(
        self, symbol: str, timeframe: str, limit: int = 50
    ) -> Optional[pd.DataFrame]:
        """Fetch indicator data from database with caching.

        Args:
            symbol: Trading symbol (e.g., "GBPUSD")
            timeframe: Timeframe string (e.g., "M30", "H1", "D1")
            limit: Max rows to fetch

        Returns:
            DataFrame with indicators or None if unavailable
        """
        cache_key = f"{symbol}_{timeframe}"
        now = time()

        # Check cache
        if cache_key in self._indicator_cache:
            if now - self._cache_timestamps.get(cache_key, 0) < INDICATOR_CACHE_TTL:
                return self._indicator_cache[cache_key]

        if self.db_manager is None:
            return None

        table = INDICATOR_TABLES.get(
            symbol.upper(), f"technical_indicator_{symbol.lower()}"
        )

        query = f"""
            SELECT {INDICATOR_COLUMNS}
            FROM {table}
            WHERE timeframe = :timeframe AND symbol = :symbol
            ORDER BY timestamp DESC
            LIMIT :limit
        """

        try:
            result = self.db_manager.execute_query(
                "ai_model", query,
                {"timeframe": timeframe, "symbol": symbol.upper(), "limit": limit}
            )
            if not result:
                return None

            df = pd.DataFrame(result)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Convert Decimal columns to float
            numeric_cols = [c for c in df.columns if c != "timestamp"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.sort_values("timestamp").reset_index(drop=True)

            # Cache result
            self._indicator_cache[cache_key] = df
            self._cache_timestamps[cache_key] = now

            return df

        except Exception as e:
            logger.debug(f"Indicator fetch failed for {symbol} {timeframe}: {e}")
            return None

    def _load_tracking_data(self) -> Optional[Dict]:
        """Load pipeline_tracking.json for Phase 5 validation.

        Issue #577: Defense-in-depth validation gate.
        Returns tracking dict or None if unavailable.
        """
        tracking_paths = [
            Path("/app/results/pipeline_tracking.json"),  # K8s pod path
            Path("results/pipeline_tracking.json"),  # Local dev path
        ]

        for path in tracking_paths:
            if path.exists():
                try:
                    with open(path) as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load tracking from {path}: {e}")

        logger.warning(
            "pipeline_tracking.json not found; "
            "fail-closed mode will reject all models without tracking data"
        )
        return None

    def _validate_model_approved(
        self, model_key: str, tracking_data: Optional[Dict]
    ) -> bool:
        """Check if a model passed Phase 5 validation.

        Issue #598: FAIL-CLOSED validation gate.
        Returns True ONLY when phase5_results.passed is explicitly True.
        All other cases return False (reject).

        When _FAIL_CLOSED_ENABLED is False, reverts to legacy fail-open
        behavior (Issue #577) for gradual rollout.
        """
        fail_closed = self.__class__._FAIL_CLOSED_ENABLED

        # --- Gate 1: No tracking data at all ---
        if tracking_data is None:
            if fail_closed:
                logger.warning(
                    f"REJECT '{model_key}': No tracking data available "
                    f"(fail-closed mode — cannot validate without pipeline_tracking.json)"
                )
                return False
            # Legacy fail-open
            return True

        # --- Gate 2: Parse model key ---
        parts = model_key.split("_")
        if len(parts) < 4:
            if fail_closed:
                logger.warning(
                    f"REJECT '{model_key}': Cannot parse model key "
                    f"(expected {{symbol}}_{{direction}}_{{signal}}_{{timeframe}})"
                )
                return False
            logger.warning(f"Cannot parse model_key '{model_key}'; allowing load")
            return True

        symbol = parts[0]  # e.g. "gbpusd"
        direction = parts[1]  # e.g. "long"
        signal_name = "_".join(parts[2:-1])  # e.g. "Stoch_RSI_long_15_25"

        # --- Gate 3: Model lookup in tracking ---
        symbols = tracking_data.get("symbols", {})
        sym_data = symbols.get(symbol, {})
        dir_data = sym_data.get(direction, {})
        signals = dir_data.get("signals", {})
        signal_entry = signals.get(signal_name)

        if signal_entry is None:
            if fail_closed:
                logger.warning(
                    f"REJECT '{model_key}': Not found in tracking data "
                    f"(fail-closed mode — rejecting unknown model)"
                )
                return False
            logger.warning(
                f"Model '{model_key}' not found in tracking; allowing load (untracked)"
            )
            return True

        # --- Gate 4: Phase 5 results existence ---
        phase5 = signal_entry.get("phase5_results")
        if phase5 is None:
            if fail_closed:
                logger.warning(
                    f"REJECT '{model_key}': No Phase 5 results in tracking "
                    f"(fail-closed mode — model has not completed Phase 5)"
                )
                return False
            return True

        # --- Gate 5: Phase 5 passed check (must be explicitly True) ---
        passed = phase5.get("passed")

        if passed is True:
            logger.info(f"APPROVED '{model_key}': Phase 5 passed")
            return True

        # Any non-True value (False, None, missing) -> check fail_closed mode
        if fail_closed:
            logger.warning(
                f"REJECT '{model_key}': Phase 5 not passed "
                f"(passed={passed!r}); skipping load"
            )
            return False

        # Legacy fail-open: Allow load even if Phase 5 not passed in tracking
        logger.info(
            f"APPROVED '{model_key}' (fail-open mode): "
            f"Phase 5 status in tracking is {passed!r}, but trusting init container filtering"
        )
        return True

    def _load_models_v2(self) -> None:
        """Load PPO models for hybrid_v2 (single model per symbol)."""
        try:
            from stable_baselines3 import PPO
        except ImportError:
            logger.error("stable_baselines3 not installed")
            return

        for symbol in self.symbols:
            symbol_upper = symbol.upper()
            symbol_lower = symbol.lower()
            seed = SYMBOL_SEEDS.get(symbol_upper, 1234)

            # Model path pattern: {symbol}/fold_001_seed_{seed}.zip
            model_path = self.model_dir / symbol_lower / f"fold_001_seed_{seed}.zip"

            if not model_path.exists():
                # Try alternative path
                model_path = self.model_dir / symbol_lower / f"fold_000_seed_{seed}.zip"

            if not model_path.exists():
                logger.warning(f"Model not found for {symbol}: {model_path}")
                continue

            try:
                model = PPO.load(str(model_path), device="cpu")
                self.models[symbol_upper] = model
                self.model_paths[symbol_upper] = [str(model_path)]
                self.model_hashes[symbol_upper] = self._compute_file_hash(model_path)
                logger.info(f"Loaded v2 model for {symbol}: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load v2 model for {symbol}: {e}")

    def _load_single_model_ensemble(
        self,
        signal_dir: Path,
        tracking_data: Optional[Dict],
    ) -> Optional[Tuple[str, List[Any], List[str], str]]:
        """Load a single model's 30-fold ensemble (Issue #623: Parallel loading).

        This method is designed to be called in parallel via ThreadPoolExecutor.

        Args:
            signal_dir: Path to model directory (e.g., gbpusd_long_Stoch_RSI_long_15_25_H1/)
            tracking_data: Pipeline tracking data for Phase 5 validation

        Returns:
            Tuple of (model_key, ensemble_models, model_paths, combined_hash) or None if failed
        """
        try:
            from stable_baselines3 import PPO
        except ImportError:
            return None

        model_key = signal_dir.name

        # Issue #577: Validate model passed Phase 5 before loading
        if not self._validate_model_approved(model_key, tracking_data):
            return None

        # Load all available folds (up to V4_NUM_FOLDS)
        ensemble_models = []
        model_paths_list = []
        hashes = []

        for fold_idx in range(V4_NUM_FOLDS):
            # Try both naming patterns: fold_00.zip and fold_000.zip
            fold_path = signal_dir / f"fold_{fold_idx:02d}.zip"
            if not fold_path.exists():
                fold_path = signal_dir / f"fold_{fold_idx:03d}.zip"

            if not fold_path.exists():
                continue

            try:
                model = PPO.load(str(fold_path), device="cpu")
                ensemble_models.append(model)
                model_paths_list.append(str(fold_path))
                hashes.append(self._compute_file_hash(fold_path))
            except Exception as e:
                logger.warning(f"Failed to load fold {fold_idx} for {model_key}: {e}")

        if ensemble_models:
            # Combined hash from all folds
            combined_hash = hashlib.sha256("".join(hashes).encode()).hexdigest()[:16]
            logger.info(
                f"Loaded v4 ensemble for {model_key}: {len(ensemble_models)} folds"
            )
            return (model_key, ensemble_models, model_paths_list, combined_hash)

        logger.warning(f"No fold models found in {signal_dir}")
        return None

    def _load_models_v4(self) -> None:
        """Load PPO models for hybrid_v4 (signal-specific 30-fold ensemble).

        Issue #560: Updated to discover signal-specific model directories.
        Issue #577: Added Phase 5 validation gate (defense-in-depth).
        Issue #623: Parallel model loading for faster startup.
        Model directory pattern: {symbol}_{direction}_{signal}_{timeframe}/
        Example: gbpusd_long_Stoch_RSI_long_15_25_H1/
        """
        try:
            from stable_baselines3 import PPO
        except ImportError:
            logger.error("stable_baselines3 not installed")
            return

        # Issue #577: Load tracking data for Phase 5 validation
        tracking_data = self._load_tracking_data()

        # Issue #623: Discover all model directories first (fast)
        all_signal_dirs: List[Path] = []

        for symbol in self.symbols:
            symbol_lower = symbol.lower()

            for direction in self.directions:
                direction_lower = direction.lower()
                base_pattern = f"{symbol_lower}_{direction_lower}"

                # Issue #560: Discover signal-specific model directories using glob
                # Pattern: {symbol}_{direction}_*/ (e.g., gbpusd_long_Stoch_RSI_long_15_25_H1/)
                signal_dirs = list(self.model_dir.glob(f"{base_pattern}_*"))

                if not signal_dirs:
                    # Fallback: Try legacy path {symbol}_{direction}/
                    legacy_dir = self.model_dir / base_pattern
                    if legacy_dir.exists() and legacy_dir.is_dir():
                        signal_dirs = [legacy_dir]
                    else:
                        logger.debug(f"No model directories found for {base_pattern}")
                        continue

                for signal_dir in signal_dirs:
                    if signal_dir.is_dir():
                        all_signal_dirs.append(signal_dir)

        total_dirs = len(all_signal_dirs)
        logger.info(
            f"Discovered {total_dirs} model directories, "
            f"loading with {PARALLEL_MODEL_WORKERS} parallel workers"
        )

        # Issue #623: Load models in parallel using ThreadPoolExecutor
        loaded_count = 0
        skipped_count = 0

        if PARALLEL_MODEL_WORKERS > 1 and total_dirs > 1:
            # Parallel loading
            with ThreadPoolExecutor(max_workers=PARALLEL_MODEL_WORKERS) as executor:
                # Submit all loading tasks
                future_to_dir = {
                    executor.submit(
                        self._load_single_model_ensemble, signal_dir, tracking_data
                    ): signal_dir
                    for signal_dir in all_signal_dirs
                }

                # Collect results as they complete
                for future in as_completed(future_to_dir):
                    signal_dir = future_to_dir[future]
                    try:
                        result = future.result()
                        if result is not None:
                            model_key, ensemble_models, model_paths_list, combined_hash = result
                            self.models[model_key] = ensemble_models
                            self.model_paths[model_key] = model_paths_list
                            self.model_hashes[model_key] = combined_hash
                            loaded_count += 1
                        else:
                            skipped_count += 1
                    except Exception as e:
                        logger.error(f"Failed to load model from {signal_dir}: {e}")
                        skipped_count += 1
        else:
            # Sequential loading (PARALLEL_MODEL_WORKERS=1 or single model)
            for signal_dir in all_signal_dirs:
                result = self._load_single_model_ensemble(signal_dir, tracking_data)
                if result is not None:
                    model_key, ensemble_models, model_paths_list, combined_hash = result
                    self.models[model_key] = ensemble_models
                    self.model_paths[model_key] = model_paths_list
                    self.model_hashes[model_key] = combined_hash
                    loaded_count += 1
                else:
                    skipped_count += 1

        # Issue #577: Log validation summary
        logger.info(
            f"Phase 5 validation: loaded {loaded_count} approved, "
            f"skipped {skipped_count} rejected"
        )

    def _find_model_key(
        self,
        symbol: str,
        direction: str,
        signal_name: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> str:
        """Find the best matching model key for a position (Issue #560).

        Lookup order:
        1. Full match: {symbol}_{direction}_{signal}_{timeframe}
        2. Signal match without timeframe: {symbol}_{direction}_{signal}_*
        3. Legacy fallback: {symbol}_{direction}

        Args:
            symbol: Symbol (lowercase)
            direction: Direction ("long" or "short")
            signal_name: Signal name (e.g., "Stoch_RSI_long_15_25")
            timeframe: Timeframe (e.g., "H1", "H4", "M30")

        Returns:
            Best matching model key, or legacy key if no match found
        """
        # Try full match first
        if signal_name and timeframe:
            full_key = f"{symbol}_{direction}_{signal_name}_{timeframe}"
            if full_key in self.models:
                return full_key

        # Try signal match without timeframe
        if signal_name:
            # Search for any model with this signal
            for model_key in self.models.keys():
                if model_key.startswith(f"{symbol}_{direction}_{signal_name}_"):
                    logger.debug(f"Found signal-specific model: {model_key}")
                    return model_key

        # Legacy fallback: {symbol}_{direction}
        legacy_key = f"{symbol}_{direction}"
        if legacy_key in self.models:
            return legacy_key

        # Return the intended key for error reporting
        if signal_name and timeframe:
            return f"{symbol}_{direction}_{signal_name}_{timeframe}"
        elif signal_name:
            return f"{symbol}_{direction}_{signal_name}"
        else:
            return legacy_key

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of model file."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except Exception:
            return "unknown"

    def evaluate_exit(
        self,
        position: PositionState,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> HybridExitSignal:
        """Evaluate exit decision for a position.

        Args:
            position: Current position state (includes signal_name, timeframe for #560)
            market_data: Optional additional market data for observation

        Returns:
            HybridExitSignal with action and confidence
        """
        symbol = position.symbol.upper()
        symbol_lower = position.symbol.lower()
        direction_str = "long" if position.direction == 1 else "short"

        # Determine model key based on version
        if self.model_version == MODEL_VERSION_V4:
            # Issue #560: Try signal-specific model first
            model_key = self._find_model_key(
                symbol_lower, direction_str,
                position.signal_name, position.timeframe
            )
        else:
            model_key = symbol

        # Check if model exists
        if model_key not in self.models:
            # Issue #588: Log warning so we know which positions lack RL coverage
            logger.warning(
                f"No exit model for {symbol} {direction_str} "
                f"(signal={position.signal_name}, tf={position.timeframe}, "
                f"key={model_key}). Position will ride to SL/TP."
            )
            return HybridExitSignal(
                symbol=symbol,
                action=ACTION_HOLD,
                confidence=0.0,
                timestamp=datetime.now(timezone.utc),
                position_bars=position.bars_held,
                unrealized_pnl_pips=position.unrealized_pnl_pips,
                reason=f"no_model_available_{model_key}",
                direction=direction_str,
            )

        # Build observation
        obs = self._build_observation(position, market_data)

        # Get model prediction
        ensemble_meta = None
        try:
            if self.model_version == MODEL_VERSION_V4:
                # Ensemble prediction for v4
                action, confidence, ensemble_meta = self._ensemble_predict(model_key, obs)
            else:
                # Single model prediction for v2
                action, confidence = self._single_predict(model_key, obs)

        except Exception as e:
            logger.error(f"Model prediction failed for {model_key}: {e}")
            return HybridExitSignal(
                symbol=symbol,
                action=ACTION_HOLD,
                confidence=0.0,
                timestamp=datetime.now(timezone.utc),
                position_bars=position.bars_held,
                unrealized_pnl_pips=position.unrealized_pnl_pips,
                reason=f"prediction_error: {e}",
                direction=direction_str,
            )

        # Determine reason
        if action == ACTION_CLOSE:
            if position.unrealized_pnl_pips > 0:
                reason = "rl_close_profit"
            else:
                reason = "rl_close_loss"
        else:
            if position.unrealized_pnl_pips > 0:
                reason = "rl_hold_profit"
            else:
                reason = "rl_hold_loss"

        model_version_str = (
            f"{self.model_version}_{self.model_hashes.get(model_key, 'unknown')}"
        )

        signal = HybridExitSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            position_bars=position.bars_held,
            unrealized_pnl_pips=position.unrealized_pnl_pips,
            reason=reason,
            model_version=model_version_str,
            direction=direction_str,
            ensemble_meta=ensemble_meta,
        )

        action_str = "CLOSE" if action == ACTION_CLOSE else "HOLD"
        logger.info(
            f"Exit eval: {symbol} {direction_str} {action_str} "
            f"(conf={confidence:.2f}, pnl={position.unrealized_pnl_pips:.1f} pips, "
            f"bars={position.bars_held}, model={model_key})"
        )

        return signal

    def _single_predict(self, model_key: str, obs: np.ndarray) -> tuple:
        """Make prediction with a single model (v2).

        Args:
            model_key: Key for model lookup
            obs: Observation array

        Returns:
            Tuple of (action, confidence)
        """
        model = self.models[model_key]

        action, _states = model.predict(obs, deterministic=True)
        action = int(action)

        # Get action probabilities for confidence
        obs_tensor = model.policy.obs_to_tensor(obs.reshape(1, -1))[0]
        with __import__("torch").no_grad():
            distribution = model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.cpu().numpy()[0]
            confidence = float(probs[action])

        return action, confidence

    def _ensemble_predict(self, model_key: str, obs: np.ndarray) -> tuple:
        """Make ensemble prediction across multiple folds (v4).

        Args:
            model_key: Key for model ensemble lookup
            obs: Observation array

        Returns:
            Tuple of (action, confidence, vote_meta) where vote_meta contains
            per-fold vote breakdown for decision logging.
        """
        import torch

        ensemble = self.models[model_key]
        all_probs = []

        for model in ensemble:
            obs_tensor = model.policy.obs_to_tensor(obs.reshape(1, -1))[0]
            with torch.no_grad():
                distribution = model.policy.get_distribution(obs_tensor)
                probs = distribution.distribution.probs.cpu().numpy()[0]
                all_probs.append(probs)

        # Stack all probabilities: shape (num_folds, num_actions)
        all_probs = np.array(all_probs)

        threshold_override = False

        if self.ensemble_method == "mean":
            # Average probabilities across all folds
            mean_probs = np.mean(all_probs, axis=0)
            action = int(np.argmax(mean_probs))
            confidence = float(mean_probs[action])
            # Apply confidence threshold: require minimum consensus for CLOSE
            if action == ACTION_CLOSE and mean_probs[ACTION_CLOSE] < self.confidence_threshold:
                action = ACTION_HOLD
                confidence = float(mean_probs[ACTION_HOLD])
                threshold_override = True
        else:
            # Majority voting
            actions = np.argmax(all_probs, axis=1)
            action = int(np.bincount(actions).argmax())
            # Confidence = fraction of models that agreed
            confidence = float(np.sum(actions == action) / len(actions))
            if action == ACTION_CLOSE and confidence < self.confidence_threshold:
                action = ACTION_HOLD
                confidence = float(np.sum(actions == ACTION_HOLD) / len(actions))
                threshold_override = True

        # Build vote metadata for decision logging
        per_fold_actions = np.argmax(all_probs, axis=1)
        vote_meta = {
            "hold_votes": int(np.sum(per_fold_actions == ACTION_HOLD)),
            "close_votes": int(np.sum(per_fold_actions == ACTION_CLOSE)),
            "total_folds": len(ensemble),
            "mean_hold_prob": float(np.mean(all_probs[:, ACTION_HOLD])),
            "mean_close_prob": float(np.mean(all_probs[:, ACTION_CLOSE])),
            "threshold_applied": self.confidence_threshold,
            "threshold_override": threshold_override,
        }

        return action, confidence, vote_meta

    def _build_primary_features(self, df: Optional[pd.DataFrame]) -> np.ndarray:
        """Build primary M30 indicator features (indices 0-15).

        Normalizations match training (EntryDataLoader._build_features):
        - Price-based (SMA, EMA, BB): percentage deviation from close
        - RSI/Stoch: (value - 50) / 50 -> [-1, 1]
        - ATR: value / 0.01
        - MACD: value / 0.01

        Args:
            df: DataFrame with indicator columns (at least 1 row)

        Returns:
            numpy array of shape (16,)
        """
        features = np.zeros(16, dtype=np.float32)
        if df is None or len(df) == 0:
            return features

        latest = df.iloc[-1]
        close = float(latest.get("close", 0))
        if close == 0:
            return features

        for i, col in enumerate(PRIMARY_INDICATOR_COLS):
            val = latest.get(col)
            if val is None or pd.isna(val):
                continue
            val = float(val)

            if "rsi" in col or "stoch" in col:
                features[i] = (val - 50.0) / 50.0
            elif "atr" in col:
                features[i] = val / 0.01
            elif "macd" in col:
                features[i] = val / 0.01
            else:
                # Price-based: percentage deviation from close
                features[i] = (val - close) / close * 100.0 if close != 0 else 0.0

        return np.clip(features, -10.0, 10.0).astype(np.float32)

    def _build_price_features(self, df: Optional[pd.DataFrame]) -> np.ndarray:
        """Build price-based features (indices 16-31).

        Features:
        [0] Return (bar-to-bar % change)
        [1] Range (high-low as % of close)
        [2] Body (close-open as % of close)
        [3] Position in range (0=bottom, 0.5=middle, 1=top)
        [4-15] Rolling stats: 3 windows (5,10,20) x 4 features

        Args:
            df: DataFrame with OHLCV columns (multiple rows needed for rolling)

        Returns:
            numpy array of shape (16,)
        """
        features = np.zeros(16, dtype=np.float32)
        if df is None or len(df) < 2:
            return features

        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        open_p = df["open"].values.astype(np.float64)

        c = close[-1]
        if c == 0:
            return features

        # [0] Return
        features[0] = (close[-1] - close[-2]) / close[-2] * 100 if close[-2] != 0 else 0

        # [1] Range
        features[1] = (high[-1] - low[-1]) / c * 100

        # [2] Body
        features[2] = (close[-1] - open_p[-1]) / c * 100

        # [3] Position in range
        r = high[-1] - low[-1]
        features[3] = (close[-1] - low[-1]) / r if r > 0 else 0.5

        # [4-15] Rolling stats: 3 windows x 4 features
        returns = np.diff(close) / close[:-1] * 100
        returns = np.concatenate([[0], returns])

        for j, window in enumerate([5, 10, 20]):
            idx = 4 + j * 4
            if len(returns) >= window:
                r_window = returns[-window:]
                # Mean return
                features[idx] = np.mean(r_window)
                # Volatility
                features[idx + 1] = np.std(r_window) if len(r_window) > 1 else 0
                # Trend (price vs SMA)
                sma = np.mean(close[-window:])
                features[idx + 2] = (close[-1] - sma) / sma * 100 if sma > 0 else 0
                # Momentum
                if len(close) >= window + 1:
                    features[idx + 3] = (close[-1] - close[-window - 1]) / close[-1] * 100

        return np.clip(features, -10.0, 10.0).astype(np.float32)

    def _resolve_tf_data(
        self, fetch_result: Any, timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Resolve fetch result to a DataFrame.

        Handles both direct DataFrame returns and dict-keyed-by-timeframe returns
        (the latter used in tests that mock _fetch_indicators with a dict).

        Args:
            fetch_result: Return from _fetch_indicators (DataFrame, dict, or None)
            timeframe: Timeframe key for dict lookup

        Returns:
            DataFrame or None
        """
        if fetch_result is None:
            return None
        if isinstance(fetch_result, pd.DataFrame):
            return fetch_result if len(fetch_result) > 0 else None
        if isinstance(fetch_result, dict):
            df = fetch_result.get(timeframe)
            if df is not None and isinstance(df, pd.DataFrame) and len(df) > 0:
                return df
            return None
        return None

    def _build_multi_tf_context(self, symbol: str) -> np.ndarray:
        """Build multi-TF context features (indices 32-159, 8 TFs x 16 features).

        Each TF gets the same 16 primary indicator features.

        Args:
            symbol: Trading symbol

        Returns:
            numpy array of shape (128,)
        """
        features = np.zeros(128, dtype=np.float32)

        for i, tf in enumerate(BASE_CONTEXT_TIMEFRAMES):
            raw = self._fetch_indicators(symbol, tf, limit=5)
            df = self._resolve_tf_data(raw, tf)
            if df is not None:
                tf_features = self._build_primary_features(df)
                features[i * 16 : (i + 1) * 16] = tf_features

        return features

    def _build_cross_tf_alignment(
        self, primary: np.ndarray, tf_context: np.ndarray
    ) -> np.ndarray:
        """Build cross-TF alignment features (indices 160-175).

        Args:
            primary: Primary M30 features (16,)
            tf_context: Multi-TF context (128,) = 8 TFs x 16

        Returns:
            numpy array of shape (16,)
        """
        features = np.zeros(16, dtype=np.float32)

        # RSI is at index 6 in primary features
        rsi_m30 = primary[6] if len(primary) > 6 else 0
        rsi_h1 = tf_context[6] if len(tf_context) > 6 else 0
        rsi_h2 = tf_context[22] if len(tf_context) > 22 else 0

        # [0] Average RSI across M30, H1, H2
        features[0] = (rsi_m30 + rsi_h1 + rsi_h2) / 3.0

        # MACD is at index 11
        macd_m30 = primary[11] if len(primary) > 11 else 0
        macd_h1 = tf_context[11] if len(tf_context) > 11 else 0

        # [1] MACD alignment (M30 vs H1)
        features[1] = np.sign(macd_m30) * np.sign(macd_h1)

        # SMA is at index 0
        sma_m30 = primary[0] if len(primary) > 0 else 0
        sma_h1 = tf_context[0] if len(tf_context) > 0 else 0

        # [2] SMA alignment
        features[2] = np.sign(sma_m30) * np.sign(sma_h1)

        return features

    def _build_derived_features(
        self,
        primary: np.ndarray,
        price_feats: np.ndarray,
        df_m30: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Build derived interaction features (indices 176-191).

        Matches training layout:
        [0] RSI * Return (interaction product)
        [1] ATR * Range (interaction product)
        [2] MACD * Body (interaction product)
        [3:8] Lagged bar-to-bar returns (lags 1-5)
        [8:13] Lagged normalized RSI (lags 1-5)
        [13:16] Reserved (zeros)

        Args:
            primary: Primary indicator features (16,)
            price_feats: Price-based features (16,)
            df_m30: M30 DataFrame for computing lagged features

        Returns:
            numpy array of shape (16,)
        """
        features = np.zeros(16, dtype=np.float32)

        # [0] RSI x Return
        features[0] = primary[6] * price_feats[0] if len(primary) > 6 else 0
        # [1] ATR x Range
        features[1] = primary[7] * price_feats[1] if len(primary) > 7 else 0
        # [2] MACD x Body
        features[2] = primary[11] * price_feats[2] if len(primary) > 11 else 0

        # [3:8] Lagged bar-to-bar returns (lags 1-5)
        # Training: features[:,16] = (close[i]-close[i-1])/close[i-1]*100
        # Lag k means the return k bars ago
        if df_m30 is not None and len(df_m30) > 2:
            close_vals = df_m30["close"].values.astype(np.float64)
            for lag in range(1, 6):
                if len(close_vals) > lag + 1:
                    c_curr = close_vals[-(lag + 1)]
                    c_prev = close_vals[-(lag + 2)]
                    if c_prev > 0:
                        features[2 + lag] = (c_curr - c_prev) / c_prev * 100
                    # else stays 0

            # [8:13] Lagged normalized RSI (lags 1-5)
            # Training: features[:,6] = (rsi-50)/50
            if "rsi_14" in df_m30.columns:
                for lag in range(1, 6):
                    if len(df_m30) > lag:
                        rsi_val = df_m30.iloc[-(lag + 1)].get("rsi_14")
                        if rsi_val is not None and not pd.isna(rsi_val):
                            features[7 + lag] = (float(rsi_val) - 50.0) / 50.0

        return np.clip(features, -10.0, 10.0).astype(np.float32)

    def _build_position_context(
        self,
        position: PositionState,
        df_m30: Optional[pd.DataFrame],
        primary: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build position context features (indices 192-196).

        Matches training layout from entry_env_v2.py _calculate_confluence():
        [0] Bars since signal normalized (0-1, max=MAX_POSITION_BARS=50)
        [1] Current ATR (raw value)
        [2] Confluence (4-component weighted: alignment 40%, trend 25%, strength 20%, volatility 15%)
        [3] Session indicator (sine wave for daily cycle)
        [4] Momentum ((close[0]-close[1])/close[1] * 10000, matching training)

        Args:
            position: Current position state
            df_m30: M30 indicator data
            primary: Primary normalized features (16,) for training-matched confluence

        Returns:
            numpy array of shape (5,)
        """
        features = np.zeros(5, dtype=np.float32)

        # [0] Bars since signal normalized (0-1)
        features[0] = min(1.0, position.bars_held / MAX_POSITION_BARS)

        # [1] Current ATR (raw value)
        if df_m30 is not None and len(df_m30) > 0 and "atr_14" in df_m30.columns:
            features[1] = float(df_m30.iloc[-1].get("atr_14", 0))

        # [2] Confluence: exact replica of training's _calculate_confluence()
        # Uses normalized primary features (same indices as training's self.features[idx])
        if primary is not None and len(primary) >= 16:
            # Training indices: [6]=RSI norm, [11]=MACD norm, [14]=Stoch_K norm,
            # [3]=ema_12 pct dev (labeled "ADX" in training), [5]=ema_50 pct dev (labeled "ATR")
            rsi = float(primary[6])
            macd = float(primary[11])
            stoch = float(primary[14])

            # Component 1: Alignment (40%) — consensus of indicators
            signals = []
            if abs(rsi) > 0.2:
                signals.append(float(np.sign(rsi)))
            if abs(macd) > 0.001:
                signals.append(float(np.sign(macd)))
            if abs(stoch) > 0.2:
                signals.append(float(np.sign(stoch)))

            alignment_score = 0.0
            if len(signals) >= 2:
                if all(s == signals[0] for s in signals):
                    alignment_score = 1.0
                elif sum(signals) != 0:
                    alignment_score = abs(sum(signals)) / len(signals)

            # Component 2: Strength (20%) — average magnitude of active indicators
            magnitudes = []
            if abs(rsi) > 0.2:
                magnitudes.append(min(abs(rsi), 1.0))
            if abs(macd) > 0.001:
                magnitudes.append(min(abs(macd) * 100, 1.0))
            if abs(stoch) > 0.2:
                magnitudes.append(min(abs(stoch), 1.0))
            strength_score = float(np.mean(magnitudes)) if magnitudes else 0.0

            # Component 3: Trend (25%) — features[3] = ema_12 pct deviation
            adx = float(primary[3])
            trend_score = float(np.clip(adx * 2, 0, 1))

            # Component 4: Volatility (15%) — piecewise on features[5] = ema_50 pct dev
            atr = float(primary[5])
            if atr < 0.1:
                volatility_score = atr * 5
            elif atr > 0.8:
                volatility_score = max(0.0, 1.0 - (atr - 0.8) * 2)
            else:
                volatility_score = 0.7 + 0.3 * (1.0 - abs(atr - 0.45) / 0.35)

            features[2] = float(np.clip(
                0.40 * alignment_score
                + 0.25 * trend_score
                + 0.20 * strength_score
                + 0.15 * volatility_score,
                0.0, 1.0,
            ))
        elif df_m30 is not None and len(df_m30) > 0:
            features[2] = 0.5

        # [3] Session indicator (sine wave for daily cycle)
        now = datetime.now(timezone.utc)
        features[3] = float(np.sin(now.timestamp() / 86400.0 * 2 * np.pi))

        # [4] Momentum: training uses (close[i]-close[i-1])/close[i-1] * 10000
        if df_m30 is not None and len(df_m30) >= 2:
            c1 = float(df_m30.iloc[-1]["close"])
            c2 = float(df_m30.iloc[-2]["close"])
            features[4] = ((c1 - c2) / c2) * 10000 if c2 > 0 else 0.0

        return features.astype(np.float32)

    def _build_volatility_features(self, df_m30: Optional[pd.DataFrame]) -> np.ndarray:
        """Build volatility features (indices 197-199).

        Args:
            df_m30: M30 indicator data with atr_14 column

        Returns:
            numpy array of shape (3,) with [percentile, z-score, regime]
        """
        features = np.zeros(3, dtype=np.float32)

        if df_m30 is None or len(df_m30) == 0 or "atr_14" not in df_m30.columns:
            return features

        atr_values = df_m30["atr_14"].dropna().values.astype(np.float64)
        if len(atr_values) < 2:
            return features

        current_atr = atr_values[-1]

        # [0] ATR percentile (fraction of window below current ATR)
        percentile = np.sum(atr_values < current_atr) / len(atr_values)
        features[0] = float(percentile)

        # [1] ATR z-score
        mean_atr = np.mean(atr_values)
        std_atr = np.std(atr_values)
        if std_atr > 0:
            features[1] = np.clip((current_atr - mean_atr) / std_atr, -5.0, 5.0)

        # [2] Volatility regime
        # When all ATR values are near-identical (std ~0), classify based on
        # absolute ATR level. This handles uniform data where percentile
        # is uninformative. Use a relative epsilon for the std check.
        has_variation = std_atr > mean_atr * 1e-6 if mean_atr > 0 else std_atr > 1e-12
        if has_variation:
            if percentile < VOLATILITY_REGIME_LOW:
                features[2] = 0.0   # Low volatility
            elif percentile > VOLATILITY_REGIME_HIGH:
                features[2] = 1.0   # High volatility
            else:
                features[2] = 0.5   # Medium volatility
        else:
            # Near-uniform ATR: use absolute thresholds for forex M30 ATR
            # Typical M30 ATR ~0.0008-0.0015 for major pairs
            if current_atr < 0.0005:
                features[2] = 0.0   # Low volatility
            elif current_atr > 0.0030:
                features[2] = 1.0   # High volatility
            else:
                features[2] = 0.5   # Medium volatility

        return features.astype(np.float32)

    def _build_self_awareness_features(self) -> np.ndarray:
        """Build self-awareness features (indices 200-214).

        Tracks the agent's own recent performance to enable awareness of
        recent stop-losses, win/loss streaks, and performance trends.

        Training layout (entry_env_v2.py line 261-278):
        [0]  bars_since_exit_norm
        [1]  sl_count_20_norm
        [2]  tp_count_20_norm
        [3]  win_streak_norm
        [4]  loss_streak_norm
        [5]  avg_holding_norm
        [6]  recent_wr (win rate last 20 trades)
        [7]  recent_pf_norm (profit factor)
        [8]  drawdown_norm
        [9]  bars_since_open_norm (time in week)
        [10] immediately_after_sl
        [11-13] pnl_trends 5/10/20
        [14] reserved (0.0)

        Returns:
            numpy array of shape (15,)
        """
        features = np.zeros(15, dtype=np.float32)

        # [0] Bars since last exit (normalized 0-1, clip at 50)
        if self._last_exit_bar is None:
            features[0] = 1.0  # Large = no recent exit
        else:
            bars_since = self._current_bar_counter - self._last_exit_bar
            features[0] = min(bars_since, 50) / 50.0

        # [1] Recent SL count (last 20 bars) — training index 1
        recent = [t for t in self._trade_history
                  if self._current_bar_counter - t.get("exit_bar", 0) <= 20]
        sl_count = sum(1 for t in recent if t.get("outcome") == "stop_loss")
        features[1] = min(sl_count, 10) / 10.0

        # [2] Recent TP count (last 20 bars) — training index 2
        tp_count = sum(1 for t in recent if t.get("outcome") == "take_profit")
        features[2] = min(tp_count, 10) / 10.0

        # [3] Win streak (normalized /10) — training index 3
        features[3] = min(self._consecutive_wins, 10) / 10.0

        # [4] Loss streak (normalized /10) — training index 4
        features[4] = min(self._consecutive_losses, 10) / 10.0

        # [5] Average holding time (recent 10 trades) — training index 5
        if len(self._trade_history) >= 10:
            recent_10 = self._trade_history[-10:]
            avg_holding = np.mean([t.get("bars_held", 0) for t in recent_10])
            features[5] = min(avg_holding, 20) / 20.0

        # [6] Win rate (last 20 trades or all if <20) — training index 6
        if len(self._trade_history) > 0:
            recent_trades = self._trade_history[-20:] if len(self._trade_history) >= 20 else self._trade_history
            wins = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
            features[6] = wins / len(recent_trades)
        else:
            features[6] = 0.5  # Neutral default

        # [7] Profit factor (last 20 trades)
        if len(self._trade_history) >= 20:
            recent_20 = self._trade_history[-20:]
            total_wins = sum(t["pnl"] for t in recent_20 if t.get("pnl", 0) > 0)
            total_losses = abs(sum(t["pnl"] for t in recent_20 if t.get("pnl", 0) < 0))
            pf = total_wins / total_losses if total_losses > 0 else 2.0
            features[7] = min(pf, 3.0) / 3.0
        else:
            features[7] = 0.5

        # [8] Drawdown (last 20 trades)
        if len(self._trade_history) >= 20:
            pnls = [t.get("pnl", 0) for t in self._trade_history[-20:]]
            cumulative = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative)
            dd = np.max(running_max - cumulative) if len(cumulative) > 0 else 0
            features[8] = min(dd, 100) / 100.0

        # [9] Time in trading week
        now = datetime.now(timezone.utc)
        week_progress = (now.weekday() * 24 + now.hour) / 120.0
        features[9] = min(week_progress, 1.0)

        # [10] Immediately after SL (binary)
        if (self._last_exit_bar is not None and
            self._current_bar_counter - self._last_exit_bar <= 2 and
            len(self._trade_history) > 0 and
            self._trade_history[-1].get("outcome") == "stop_loss"):
            features[10] = 1.0

        # [11-13] PnL trends (tanh normalized)
        def pnl_trend(n: int) -> float:
            if len(self._trade_history) >= n:
                total = sum(t.get("pnl", 0) for t in self._trade_history[-n:])
                return float(np.tanh(total / 50.0))
            return 0.0

        features[11] = pnl_trend(5)
        features[12] = pnl_trend(10)
        features[13] = pnl_trend(20)

        # [14] Reserved
        features[14] = 0.0

        return features.astype(np.float32)

    def _build_regime_features(self, df_m30: Optional[pd.DataFrame]) -> np.ndarray:
        """Build market regime features (indices 215-224).

        Args:
            df_m30: M30 indicator data

        Returns:
            numpy array of shape (10,) with:
            [0] bull, [1] bear, [2] ranging, [3] trend_strength, [4] vol_high
            [5-9] reserved
        """
        features = np.zeros(10, dtype=np.float32)
        features[2] = 1.0  # Default: ranging

        if df_m30 is None or len(df_m30) < 30:
            return features

        close = df_m30["close"].values.astype(np.float64)

        # SMA20 from recent 20 bars
        sma_20 = np.mean(close[-20:])
        current_price = close[-1]

        # SMA20 slope (compare recent 20 vs previous 20)
        if len(close) >= 40:
            sma_20_prev = np.mean(close[-40:-20])
            sma_slope = (sma_20 - sma_20_prev) / sma_20_prev if sma_20_prev > 0 else 0
        else:
            sma_slope = 0

        # Trend strength (tanh normalized)
        trend_strength = float(np.tanh(sma_slope * 100))
        features[3] = trend_strength

        # Regime classification
        slope_threshold = 0.001
        if abs(sma_slope) > slope_threshold:
            if sma_slope > 0 and current_price > sma_20:
                features[0] = 1.0  # Bull
                features[2] = 0.0  # Not ranging
            elif sma_slope < 0 and current_price < sma_20:
                features[1] = 1.0  # Bear
                features[2] = 0.0  # Not ranging

        # Volatility flag
        if "atr_14" in df_m30.columns:
            atr_values = df_m30["atr_14"].dropna().values
            if len(atr_values) > 0:
                current_atr = float(atr_values[-1])
                p75 = float(np.percentile(atr_values, 75))
                if current_atr > p75:
                    features[4] = 1.0

        return features.astype(np.float32)

    def _build_rsi_pattern_features(
        self,
        df_m30: Optional[pd.DataFrame] = None,
        *,
        rsi_value: Optional[float] = None,
    ) -> np.ndarray:
        """Build RSI pattern features (indices 220-224).

        Supports two calling conventions:
        1. _build_rsi_pattern_features(df_m30) - extract RSI from DataFrame
        2. _build_rsi_pattern_features(rsi_value=50.0) - direct RSI value

        Args:
            df_m30: M30 indicator data (optional)
            rsi_value: Direct RSI value (keyword-only, optional)

        Returns:
            numpy array of shape (5,) with [oversold, overbought, rsi_norm, rsi_slope, reserved]
        """
        features = np.zeros(5, dtype=np.float32)

        # Resolve RSI value
        rsi: Optional[float] = rsi_value
        if rsi is None:
            if df_m30 is not None and len(df_m30) > 0 and "rsi_14" in df_m30.columns:
                val = df_m30.iloc[-1]["rsi_14"]
                if val is not None and not pd.isna(val):
                    rsi = float(val)

        if rsi is None:
            return features

        # [0] Oversold
        if rsi < 30:
            features[0] = 1.0

        # [1] Overbought
        if rsi > 70:
            features[1] = 1.0

        # [2] Normalized RSI (0-1)
        features[2] = np.clip(rsi / 100.0, 0.0, 1.0)

        # [3] RSI slope (if df available with >1 row)
        if df_m30 is not None and len(df_m30) >= 2 and "rsi_14" in df_m30.columns:
            rsi_prev = df_m30.iloc[-2].get("rsi_14")
            if rsi_prev is not None and not pd.isna(rsi_prev):
                features[3] = np.clip((rsi - float(rsi_prev)) / 50.0, -1.0, 1.0)

        return features.astype(np.float32)

    def _build_d1_features(self, symbol: str) -> np.ndarray:
        """Build D1 daily timeframe context features (indices 222-236).

        Matches the 15-dim D1 feature vector from training (d1_features.py).
        Uses the most recent D1 bar from the indicator database.

        Features (15 dims):
        [0]  SMA20 > SMA50 (binary)
        [1]  SMA20/50 crossover (-1/0/+1)
        [2]  EMA12 > EMA26 (binary)
        [3]  EMA12/26 crossover (-1/0/+1)
        [4]  Price vs SMA20 (tanh normalized)
        [5]  Price vs SMA50 (tanh normalized)
        [6]  RSI / 100 (0-1)
        [7]  RSI oversold (binary, <30)
        [8]  RSI overbought (binary, >70)
        [9]  MACD > Signal (binary)
        [10] MACD histogram (tanh normalized)
        [11] Near bullish OB (0 - not available)
        [12] Near bearish OB (0 - not available)
        [13] BB position (-1 to 1)
        [14] Combined trend strength (-1 to 1)

        Args:
            symbol: Trading symbol

        Returns:
            numpy array of shape (15,)
        """
        features = np.zeros(D1_FEATURES_DIM, dtype=np.float32)

        raw = self._fetch_indicators(symbol, "D1", limit=5)
        df = self._resolve_tf_data(raw, "D1")
        if df is None or len(df) < 2:
            return features

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Extract values with safe access
        close = latest.get("close")
        sma20 = latest.get("sma_20")
        sma50 = latest.get("sma_50")
        ema12 = latest.get("ema_12")
        ema26 = latest.get("ema_26")
        rsi = latest.get("rsi_14")
        macd_line = latest.get("macd_line")
        macd_signal = latest.get("macd_signal")
        macd_hist = latest.get("macd_histogram")
        bb_upper = latest.get("bb_upper_20")
        bb_lower = latest.get("bb_lower_20")

        prev_sma20 = prev.get("sma_20")
        prev_sma50 = prev.get("sma_50")
        prev_ema12 = prev.get("ema_12")
        prev_ema26 = prev.get("ema_26")

        def _valid(*vals):
            return all(v is not None and not pd.isna(v) for v in vals)

        # [0] SMA20 > SMA50
        if _valid(sma20, sma50):
            features[0] = 1.0 if float(sma20) > float(sma50) else 0.0

        # [1] SMA20/50 crossover
        if _valid(sma20, sma50, prev_sma20, prev_sma50):
            curr_above = float(sma20) > float(sma50)
            prev_above = float(prev_sma20) > float(prev_sma50)
            if curr_above and not prev_above:
                features[1] = 1.0  # Cross up
            elif not curr_above and prev_above:
                features[1] = -1.0  # Cross down

        # [2] EMA12 > EMA26
        if _valid(ema12, ema26):
            features[2] = 1.0 if float(ema12) > float(ema26) else 0.0

        # [3] EMA12/26 crossover
        if _valid(ema12, ema26, prev_ema12, prev_ema26):
            curr_above = float(ema12) > float(ema26)
            prev_above = float(prev_ema12) > float(prev_ema26)
            if curr_above and not prev_above:
                features[3] = 1.0
            elif not curr_above and prev_above:
                features[3] = -1.0

        # [4] Price vs SMA20
        if _valid(close, sma20) and float(sma20) > 1e-8:
            features[4] = float(np.tanh((float(close) - float(sma20)) / float(sma20) * 100))

        # [5] Price vs SMA50
        if _valid(close, sma50) and float(sma50) > 1e-8:
            features[5] = float(np.tanh((float(close) - float(sma50)) / float(sma50) * 100))

        # [6] RSI normalized (0-1)
        if _valid(rsi):
            features[6] = float(np.clip(float(rsi) / 100.0, 0, 1))

        # [7] RSI oversold
        if _valid(rsi):
            features[7] = 1.0 if float(rsi) < 30 else 0.0

        # [8] RSI overbought
        if _valid(rsi):
            features[8] = 1.0 if float(rsi) > 70 else 0.0

        # [9] MACD > Signal
        if _valid(macd_line, macd_signal):
            features[9] = 1.0 if float(macd_line) > float(macd_signal) else 0.0

        # [10] MACD histogram (tanh scaled for forex)
        if _valid(macd_hist):
            features[10] = float(np.tanh(float(macd_hist) * 10000))

        # [11-12] OB proximity (not available in paper trading DB)
        # Left as 0.0

        # [13] BB position (-1 at lower, 0 at middle, 1 at upper)
        if _valid(close, bb_upper, bb_lower):
            bb_width = float(bb_upper) - float(bb_lower)
            if bb_width > 0:
                features[13] = float(np.clip(
                    2 * (float(close) - float(bb_lower)) / bb_width - 1, -1, 1
                ))

        # [14] Combined trend strength
        trend_sma = features[0] * 2 - 1  # -1 or 1
        trend_ema = features[2] * 2 - 1  # -1 or 1
        trend_rsi = (float(rsi) - 50) / 50 if _valid(rsi) else 0.0
        features[14] = float(np.clip((trend_sma + trend_ema + trend_rsi) / 3, -1, 1))

        return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    def _build_v2_multi_tf_features(self, symbol: str) -> np.ndarray:
        """Build V2 multi-TF features (indices 222-321).

        8 timeframes x 12 features + 4 global OB features = 100 dimensions.

        Per-TF features (12):
        [0] SMA20 > SMA50 (binary)
        [1] EMA12 > EMA26 (binary)
        [2] RSI / 100 (0-1)
        [3] MACD > Signal (binary)
        [4] Trend strength (-1 to 1)
        [5] BB position (0=lower, 0.5=mid, 1=upper)
        [6] Stoch K / 100 (0-1)
        [7] Stoch D / 100 (0-1)
        [8] ATR normalized
        [9] OB bullish proximity (0 - not available)
        [10] OB bearish proximity (0 - not available)
        [11] Price vs SMA200 (binary)

        Args:
            symbol: Trading symbol

        Returns:
            numpy array of shape (100,)
        """
        features = np.zeros(100, dtype=np.float32)

        for i, tf in enumerate(V2_TIMEFRAMES):
            raw = self._fetch_indicators(symbol, tf, limit=5)
            df = self._resolve_tf_data(raw, tf)
            if df is None:
                continue

            latest = df.iloc[-1]
            offset = i * V2_FEATURES_PER_TF

            # [0] SMA20 > SMA50
            sma20 = latest.get("sma_20")
            sma50 = latest.get("sma_50")
            if sma20 is not None and sma50 is not None and not pd.isna(sma20) and not pd.isna(sma50):
                features[offset + 0] = 1.0 if float(sma20) > float(sma50) else 0.0

            # [1] EMA12 > EMA26
            ema12 = latest.get("ema_12")
            ema26 = latest.get("ema_26")
            if ema12 is not None and ema26 is not None and not pd.isna(ema12) and not pd.isna(ema26):
                features[offset + 1] = 1.0 if float(ema12) > float(ema26) else 0.0

            # [2] RSI normalized (0-1)
            rsi = latest.get("rsi_14")
            if rsi is not None and not pd.isna(rsi):
                features[offset + 2] = np.clip(float(rsi) / 100.0, 0, 1)

            # [3] MACD > Signal
            macd = latest.get("macd_line")
            signal = latest.get("macd_signal")
            if macd is not None and signal is not None and not pd.isna(macd) and not pd.isna(signal):
                features[offset + 3] = 1.0 if float(macd) > float(signal) else 0.0

            # [4] Trend strength
            if sma20 is not None and sma50 is not None and not pd.isna(sma20) and not pd.isna(sma50):
                s20, s50 = float(sma20), float(sma50)
                if s50 > 0:
                    features[offset + 4] = np.clip((s20 - s50) / s50 * 100, -1, 1)

            # [5] BB position
            close = latest.get("close")
            bb_upper = latest.get("bb_upper_20")
            bb_lower = latest.get("bb_lower_20")
            if (close is not None and bb_upper is not None and bb_lower is not None and
                not pd.isna(close) and not pd.isna(bb_upper) and not pd.isna(bb_lower)):
                bb_range = float(bb_upper) - float(bb_lower)
                if bb_range > 0:
                    features[offset + 5] = np.clip(
                        (float(close) - float(bb_lower)) / bb_range, 0, 1
                    )
                else:
                    features[offset + 5] = 0.5

            # [6] Stoch K normalized
            stoch_k = latest.get("stoch_k")
            if stoch_k is not None and not pd.isna(stoch_k):
                features[offset + 6] = np.clip(float(stoch_k) / 100.0, 0, 1)

            # [7] Stoch D normalized
            stoch_d = latest.get("stoch_d")
            if stoch_d is not None and not pd.isna(stoch_d):
                features[offset + 7] = np.clip(float(stoch_d) / 100.0, 0, 1)

            # [8] ATR normalized (value / (median_atr * 3), capped at 1)
            atr = latest.get("atr_14")
            if atr is not None and not pd.isna(atr):
                atr_col = df["atr_14"].dropna()
                if len(atr_col) > 0:
                    median_atr = float(atr_col.median())
                    if median_atr > 0:
                        features[offset + 8] = np.clip(float(atr) / (median_atr * 3), 0, 1)

            # [9-10] OB proximity (not available from simple query)

            # [11] Price vs SMA200
            sma200 = latest.get("sma_200")
            if close is not None and sma200 is not None and not pd.isna(close) and not pd.isna(sma200):
                features[offset + 11] = 1.0 if float(close) > float(sma200) else 0.0

        # Global OB features [96-99] not available

        return features.astype(np.float32)

    def _build_observation(
        self,
        position: PositionState,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Build 242-dimension observation vector for model prediction.

        Matches trained model observation space (use_d1_features=True, use_v2_features=False):
        [0:16]    Primary M30 indicators
        [16:32]   Price-based features
        [32:160]  Multi-TF context (8 TFs x 16 features)
        [160:176] Cross-TF alignment
        [176:192] Derived interaction features
        [192:197] Position context
        [197:200] Volatility features
        [200:215] Self-awareness features
        [215:222] Regime features (first 7 of 10, capped at BASE_OBS_DIM)
        [222:237] D1 daily timeframe context features (15 dims)
        [237:242] Position features (direction, bars_held, pnl, sl_dist, tp_dist)

        Args:
            position: Current position state
            market_data: Optional additional market data (legacy, kept for compat)

        Returns:
            242-dimensional observation array
        """
        obs = np.zeros(TOTAL_OBS_DIM, dtype=np.float32)
        symbol = position.symbol.upper()
        pip_value = PIP_VALUES.get(symbol, 0.0001)

        # Fetch M30 indicator data (used by multiple feature builders)
        # Fix 7: Use limit=260 to provide 252-bar window for volatility features
        raw_m30 = self._fetch_indicators(symbol, "M30", limit=260)
        df_m30 = self._resolve_tf_data(raw_m30, "M30")

        # === Section 1: Primary M30 indicators [0:16] ===
        primary = self._build_primary_features(df_m30)
        obs[0:16] = primary

        # === Section 2: Price-based features [16:32] ===
        price_feats = self._build_price_features(df_m30)
        obs[16:32] = price_feats

        # === Section 3: Multi-TF context [32:160] ===
        # Fix 1: Training uses DataFrame mode where context_timeframes=[] (empty),
        # so all 128 dims are zeros. Do NOT call _build_multi_tf_context().
        # obs[32:160] stays zeros from initialization.

        # === Section 4: Cross-TF alignment [160:176] ===
        # Fix 2: Training computes (primaryRSI + 0 + 0)/3 since tf_context is all zeros.
        # Indices 161:176 = 0 because sign(0)*sign(0) = 0.
        obs[160] = primary[6] / 3.0  # RSI/3, matching training behavior
        # obs[161:176] stays zero from initialization

        # === Section 5: Derived features [176:192] ===
        # Fix 3: Pass df_m30 to compute lagged returns and RSI
        obs[176:192] = self._build_derived_features(primary, price_feats, df_m30)

        # === Section 6: Position context [192:197] ===
        # Fix 4: Pass primary features for training-matched confluence calculation
        obs[192:197] = self._build_position_context(position, df_m30, primary)

        # === Section 7: Volatility features [197:200] ===
        obs[197:200] = self._build_volatility_features(df_m30)

        # === Section 8: Self-awareness features [200:215] ===
        obs[200:215] = self._build_self_awareness_features()

        # === Section 9: Regime features [215:222] ===
        # Only write up to BASE_OBS_DIM (222), D1 features start there
        regime = self._build_regime_features(df_m30)
        regime_end = min(215 + len(regime), BASE_OBS_DIM)
        obs[215:regime_end] = regime[:regime_end - 215]

        # === Section 9b: RSI pattern features [220:222] ===
        # Fix 6: _build_rsi_pattern_features() exists but was never called.
        # Training uses 2-dim RSI patterns (oversold, overbought) at indices 220-221.
        rsi_patterns = self._build_rsi_pattern_features(df_m30)
        obs[220] = rsi_patterns[0]  # oversold
        obs[221] = rsi_patterns[1]  # overbought

        # === Section 10: D1 daily context features [222:237] ===
        obs[BASE_OBS_DIM:BASE_OBS_DIM + D1_FEATURES_DIM] = self._build_d1_features(symbol)

        # === Section 11: Position features [237:242] ===
        pos_start = BASE_OBS_DIM + D1_FEATURES_DIM
        obs[pos_start + 0] = float(position.direction)
        obs[pos_start + 1] = min(1.0, position.bars_held / MAX_POSITION_BARS)
        obs[pos_start + 2] = np.clip(position.unrealized_pnl_pips / 30.0, -1.0, 1.0)

        sl_dist_pips = position.distance_to_sl / pip_value
        obs[pos_start + 3] = np.clip(sl_dist_pips / SL_PIPS, 0.0, 2.0) / 2.0

        tp_dist_pips = position.distance_to_tp / pip_value
        obs[pos_start + 4] = np.clip(tp_dist_pips / TP_PIPS, 0.0, 2.0) / 2.0

        # Sanitize: replace any NaN/Inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        # Log observation coverage periodically
        nonzero = np.count_nonzero(obs)
        if nonzero > 10:
            logger.debug(f"Observation coverage: {nonzero}/{TOTAL_OBS_DIM} non-zero features")

        return obs

    def check_position(
        self,
        symbol: str,
        direction: int,
        entry_price: float,
        current_price: float,
        sl_price: float,
        tp_price: float,
        bars_held: int,
        market_data: Optional[Dict[str, Any]] = None,
        signal_name: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> HybridExitSignal:
        """Convenience method to check position without building PositionState.

        Args:
            symbol: Trading symbol
            direction: 1 for LONG, -1 for SHORT
            entry_price: Entry price
            current_price: Current market price
            sl_price: Stop loss price
            tp_price: Take profit price
            bars_held: Number of bars since entry
            market_data: Optional additional market data
            signal_name: Signal that opened the position (Issue #560)
            timeframe: Timeframe of the signal model (Issue #560)

        Returns:
            HybridExitSignal
        """
        position = PositionState(
            symbol=symbol.upper(),
            direction=direction,
            entry_price=entry_price,
            entry_time=datetime.now(timezone.utc),
            current_price=current_price,
            sl_price=sl_price,
            tp_price=tp_price,
            bars_held=bars_held,
            signal_name=signal_name,
            timeframe=timeframe,
        )
        return self.evaluate_exit(position, market_data)

    def get_status(self) -> Dict[str, Any]:
        """Get evaluator status for monitoring.

        Returns:
            Status dictionary
        """
        # Get fold counts for v4 models
        fold_counts = {}
        if self.model_version == MODEL_VERSION_V4:
            for key, models in self.models.items():
                fold_counts[key] = len(models) if isinstance(models, list) else 1

        return {
            "evaluator_type": "hybrid_exit",
            "model_version": self.model_version,
            "models_loaded": list(self.models.keys()),
            "model_dir": str(self.model_dir),
            "model_hashes": self.model_hashes,
            "confidence_threshold": self.confidence_threshold,
            "ensemble_method": self.ensemble_method if self.model_version == MODEL_VERSION_V4 else None,
            "fold_counts": fold_counts if fold_counts else None,
        }

    def sync_models_incremental(
        self,
        approved_models: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Incrementally sync models without full reload (Issue #612).

        Compares currently loaded models against approved_models list and:
        - Loads NEW models (in approved but not loaded)
        - Unloads REMOVED models (loaded but not in approved)
        - Skips UNCHANGED models (already loaded with same hash)

        This eliminates the 4-hour startup time on daily reconcile by only
        loading/unloading changed models.

        Args:
            approved_models: List of approved model dicts from approved_models.yaml
                Each dict has: model_dir, symbol, direction, signal_name, timeframe

        Returns:
            Sync result with counts: loaded, unloaded, unchanged, errors
        """
        if self.model_version != MODEL_VERSION_V4:
            logger.warning("Incremental sync only supported for v4 models")
            return {"error": "unsupported_version"}

        try:
            from stable_baselines3 import PPO
        except ImportError:
            logger.error("stable_baselines3 not installed")
            return {"error": "import_error"}

        # Build set of approved model keys
        approved_keys = set()
        for model in approved_models:
            model_key = model.get("model_dir", "")
            if model_key:
                approved_keys.add(model_key)

        currently_loaded = set(self.models.keys())

        # Calculate delta
        to_load = approved_keys - currently_loaded
        to_unload = currently_loaded - approved_keys
        potentially_unchanged = approved_keys & currently_loaded

        # Check for retrained models (same key, different hash)
        to_reload = set()
        unchanged = set()

        for model_key in potentially_unchanged:
            signal_dir = self.model_dir / model_key
            if signal_dir.exists():
                # Compute current disk hash
                disk_hashes = []
                for fold_idx in range(V4_NUM_FOLDS):
                    fold_path = signal_dir / f"fold_{fold_idx:02d}.zip"
                    if not fold_path.exists():
                        fold_path = signal_dir / f"fold_{fold_idx:03d}.zip"
                    if fold_path.exists():
                        disk_hashes.append(self._compute_file_hash(fold_path))

                if disk_hashes:
                    disk_hash = hashlib.sha256("".join(disk_hashes).encode()).hexdigest()[:16]
                    loaded_hash = self.model_hashes.get(model_key, "")

                    if disk_hash != loaded_hash:
                        logger.info(
                            f"Retrained model detected: {model_key} "
                            f"(hash {loaded_hash} -> {disk_hash})"
                        )
                        to_reload.add(model_key)
                    else:
                        unchanged.add(model_key)
                else:
                    unchanged.add(model_key)
            else:
                unchanged.add(model_key)

        # Add retrained models to load list (they'll be unloaded first)
        to_load = to_load | to_reload
        to_unload = to_unload | to_reload

        logger.info(
            f"Incremental sync: {len(to_load)} to load, "
            f"{len(to_unload)} to unload, {len(unchanged)} unchanged, "
            f"{len(to_reload)} retrained"
        )

        result = {
            "loaded": [],
            "unloaded": [],
            "reloaded": [],
            "unchanged": list(unchanged),
            "errors": [],
        }

        # Unload removed models (instant)
        for model_key in to_unload:
            try:
                del self.models[model_key]
                self.model_paths.pop(model_key, None)
                self.model_hashes.pop(model_key, None)
                result["unloaded"].append(model_key)
                logger.info(f"Unloaded model: {model_key}")
            except Exception as e:
                logger.error(f"Failed to unload {model_key}: {e}")
                result["errors"].append({"model": model_key, "error": str(e)})

        # Load new models
        tracking_data = self._load_tracking_data()

        for model_key in to_load:
            signal_dir = self.model_dir / model_key

            if not signal_dir.exists() or not signal_dir.is_dir():
                logger.warning(f"Model directory not found: {signal_dir}")
                result["errors"].append({"model": model_key, "error": "dir_not_found"})
                continue

            # Validate Phase 5 passed
            if not self._validate_model_approved(model_key, tracking_data):
                logger.warning(f"Model not approved (Phase 5): {model_key}")
                result["errors"].append({"model": model_key, "error": "not_approved"})
                continue

            # Load ensemble
            ensemble_models = []
            model_paths_list = []
            hashes = []

            for fold_idx in range(V4_NUM_FOLDS):
                fold_path = signal_dir / f"fold_{fold_idx:02d}.zip"
                if not fold_path.exists():
                    fold_path = signal_dir / f"fold_{fold_idx:03d}.zip"

                if not fold_path.exists():
                    continue

                try:
                    model = PPO.load(str(fold_path), device="cpu")
                    ensemble_models.append(model)
                    model_paths_list.append(str(fold_path))
                    hashes.append(self._compute_file_hash(fold_path))
                except Exception as e:
                    logger.warning(f"Failed to load fold {fold_idx} for {model_key}: {e}")

            if ensemble_models:
                self.models[model_key] = ensemble_models
                self.model_paths[model_key] = model_paths_list
                combined_hash = hashlib.sha256("".join(hashes).encode()).hexdigest()[:16]
                self.model_hashes[model_key] = combined_hash

                # Track if this was a reload (retrained) vs new load
                if model_key in to_reload:
                    result["reloaded"].append(model_key)
                    logger.info(f"Reloaded retrained model: {model_key} ({len(ensemble_models)} folds)")
                else:
                    result["loaded"].append(model_key)
                    logger.info(f"Loaded new model: {model_key} ({len(ensemble_models)} folds)")
            else:
                logger.warning(f"No fold models found for {model_key}")
                result["errors"].append({"model": model_key, "error": "no_folds"})

        logger.info(
            f"Incremental sync complete: "
            f"new={len(result['loaded'])}, reloaded={len(result['reloaded'])}, "
            f"unloaded={len(result['unloaded'])}, unchanged={len(result['unchanged'])}, "
            f"errors={len(result['errors'])}"
        )

        return result

    def get_loaded_model_keys(self) -> set:
        """Get set of currently loaded model keys.

        Returns:
            Set of model keys (e.g., 'gbpusd_long_Stoch_RSI_long_15_25_H1')
        """
        return set(self.models.keys())
