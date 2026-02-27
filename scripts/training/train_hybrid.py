#!/usr/bin/env python3
"""
Hybrid Architecture Training Script (Issue #494, #495, #522).

This script trains hybrid trading models that use:
- Rule-based entries from D1 signals (59.8% WR)
- RL-managed exits (Discrete(2) action space)
- Optional V2 multi-TF features (327 dims) for enhanced context

The hybrid approach solves the under-trading problem where pure RL
models learn to HOLD rather than TRADE.

Usage:
    python scripts/train_hybrid.py --symbol eurusd --fold 1
    python scripts/train_hybrid.py --symbol eurusd --all-folds
    python scripts/train_hybrid.py --all-symbols --all-folds

    # With V2 features (Issue #495)
    python scripts/train_hybrid.py --all-symbols --all-folds --use-v2-features

    # With Optuna hyperparameters (Issue #522)
    python scripts/train_hybrid.py --symbol eurusd --direction long --use-optuna

Input Files:
    - config/symbol_seeds.json: Contains stable seeds per symbol
    - config/folds/all_symbols_folds.json: Contains 30 walk-forward folds

Output Structure:
    models/hybrid/{symbol}/     (without V2 features, 242 dims)
    models/hybrid_v2/{symbol}/  (with V2 features, 327 dims)
    ├── fold_01_seed_1234.zip
    ├── fold_02_seed_1234.zip
    ...
    └── fold_30_seed_1234.zip

V2 Features (Issue #495):
    - Adds 100 dimensions from MultiTFFeatureProviderV2
    - 8 timeframes × 12 features + 4 order block features
    - Total observation: 327 dims (vs 242 without V2)

Issue #522 Updates:
    - Optuna hyperparameter integration: load_optuna_hyperparameters()
    - Adaptive timesteps: calculate_optimal_timesteps() from data size
    - Early stopping: Stop after 3 consecutive unprofitable folds
    - Training metadata: Save hyperparameters and data sizes

Issue: #494, #495, #522
Epic: track7d-hybrid-v1-v2, hybrid-v4
"""

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# PostgreSQL data path
POSTGRES_DATA_PATH = Path("data/rates")

# Default directories
DEFAULT_OUTPUT_DIR = "models/hybrid"
DEFAULT_SEEDS_PATH = Path(__file__).parent.parent / "config" / "symbol_seeds.json"
DEFAULT_FOLDS_PATH = (
    Path(__file__).parent.parent / "config" / "folds" / "all_symbols_folds.json"
)

# Supported symbols
SUPPORTED_SYMBOLS = ["eurusd", "gbpusd", "usdjpy", "eurjpy", "xagusd", "usdcad", "eurcad"]

# Supported directions (Issue #512: Signal-Specific Training)
SUPPORTED_DIRECTIONS = ["long", "short", "both"]

# PPO Configuration for Hybrid (simpler task = can use lower LR)
PPO_CONFIG_HYBRID = {
    "learning_rate": 1e-4,  # Higher LR for simpler task (Discrete(2))
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.02,  # Slightly higher entropy for exploration
    "device": "cpu",  # MLP policies more stable on CPU
}

# Training configuration
DEFAULT_TIMESTEPS = 50000  # Can train longer since task is simpler

# Default Optuna results directory (Issue #522)
DEFAULT_OPTUNA_DIR = Path(__file__).parent.parent / "results" / "optuna_hybrid"

# Default rollouts for adaptive timesteps (Issue #522)
DEFAULT_ROLLOUTS = 2.5

# Early stopping configuration (Issue #522)
DEFAULT_EARLY_STOP_LIMIT = 3  # Stop after N consecutive unprofitable folds


@dataclass
class FoldResult:
    """
    Result from training and validating a single fold.

    Includes full cost accounting for realistic profit factor calculations.

    Issue: #522 - 30-Fold Training Updates
    Epic: hybrid-v4

    Attributes:
        fold_idx: Fold number (1-30)
        profit_factor: Gross profit / Gross loss (after costs)
        win_rate: Winning trades / Total trades
        trades: Total number of trades
        timesteps: Training timesteps used
        training_time_seconds: Wall clock time for training
        hyperparameters: PPO hyperparameters used
        gross_profit: Sum of all winning trades (pips)
        gross_loss: Sum of all losing trades (pips, positive value)
        total_costs: Total trading costs (pips)
    """

    fold_idx: int
    profit_factor: float
    win_rate: float
    trades: int
    timesteps: int
    training_time_seconds: float
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_costs: float = 0.0

    @property
    def is_profitable(self) -> bool:
        """Check if fold is profitable (PF > 1.0)."""
        return self.profit_factor > 1.0

    @property
    def net_profit(self) -> float:
        """Calculate net profit after costs."""
        return self.gross_profit - self.gross_loss - self.total_costs


# Global checkpoint manager for signal handler
_checkpoint_manager = None


class CheckpointManager:
    """Manages checkpoint state for resume functionality."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.completed: dict[str, list[str]] = {}
        self._load()

    def _load(self):
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path) as f:
                    data = json.load(f)
                    self.completed = data.get("completed", {})
            except (json.JSONDecodeError, IOError):
                self.completed = {}

    def save(self):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "completed": self.completed,
            "last_updated": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
        }
        with open(self.checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)

    def mark_completed(self, symbol: str, fold_id: str):
        if symbol not in self.completed:
            self.completed[symbol] = []
        if fold_id not in self.completed[symbol]:
            self.completed[symbol].append(fold_id)

    def is_completed(self, symbol: str, fold_id: str) -> bool:
        return fold_id in self.completed.get(symbol, [])


# ==============================================================================
# Issue #522: Optuna Integration Functions
# ==============================================================================


class EarlyStopTracker:
    """Tracks consecutive unprofitable folds for early stopping (Issue #522).

    Early stopping is triggered when N consecutive folds have profit factor < 1.0.
    A profitable fold (PF >= 1.0) resets the counter.

    Attributes:
        consecutive_unprofitable_limit: Number of consecutive unprofitable folds before stop.
        consecutive_unprofitable: Current count of consecutive unprofitable folds.

    Example:
        tracker = EarlyStopTracker(consecutive_unprofitable_limit=3)
        tracker.add_fold_result(profit_factor=0.8)  # Count: 1
        tracker.add_fold_result(profit_factor=0.9)  # Count: 2
        tracker.add_fold_result(profit_factor=1.2)  # Count: 0 (reset)
        tracker.add_fold_result(profit_factor=0.7)  # Count: 1
    """

    def __init__(self, consecutive_unprofitable_limit: int = DEFAULT_EARLY_STOP_LIMIT):
        """Initialize early stop tracker.

        Args:
            consecutive_unprofitable_limit: Stop after this many consecutive unprofitable folds.
        """
        self.consecutive_unprofitable_limit = consecutive_unprofitable_limit
        self.consecutive_unprofitable = 0
        self._all_results: list[float] = []

    def add_fold_result(self, profit_factor: float) -> None:
        """Add a fold result and update the counter.

        Args:
            profit_factor: Profit factor from the fold validation.
                          PF >= 1.0 is considered profitable (break-even or better).
        """
        self._all_results.append(profit_factor)

        if profit_factor >= 1.0:
            # Profitable or break-even - reset counter
            self.consecutive_unprofitable = 0
        else:
            # Unprofitable - increment counter
            self.consecutive_unprofitable += 1

    def should_stop(self) -> bool:
        """Check if early stopping should be triggered.

        Returns:
            True if consecutive unprofitable folds reached the limit.
        """
        return self.consecutive_unprofitable >= self.consecutive_unprofitable_limit


def should_early_stop(fold_results: list[dict[str, Any]], limit: int = DEFAULT_EARLY_STOP_LIMIT) -> bool:
    """Helper function to check if early stopping should be triggered.

    Args:
        fold_results: List of fold result dicts with 'profit_factor' key.
        limit: Number of consecutive unprofitable folds to trigger stop.

    Returns:
        True if the last N folds all have profit_factor < 1.0.

    Example:
        results = [{'profit_factor': 0.8}, {'profit_factor': 0.9}, {'profit_factor': 0.7}]
        should_early_stop(results, limit=3)  # Returns True
    """
    tracker = EarlyStopTracker(consecutive_unprofitable_limit=limit)

    for result in fold_results:
        pf = result.get("profit_factor", 0.0)
        tracker.add_fold_result(pf)

    return tracker.should_stop()


def load_optuna_hyperparameters(
    symbol: str,
    signal_name: str,
    optuna_dir: str | None = None,
) -> dict[str, Any] | None:
    """Load best hyperparameters from Optuna tuning results (Issue #522).

    Args:
        symbol: Trading symbol (e.g., 'eurusd').
        signal_name: Signal direction ('long' or 'short').
        optuna_dir: Directory containing Optuna results.
                   Defaults to results/optuna_hybrid/.

    Returns:
        Dict with hyperparameters or None if file doesn't exist or is invalid.

    Example:
        params = load_optuna_hyperparameters('eurusd', 'long')
        if params:
            ppo_config.update(params)
    """
    if optuna_dir is None:
        optuna_dir = str(DEFAULT_OPTUNA_DIR)

    optuna_path = Path(optuna_dir)

    # Build filename: {symbol}_{direction}_best_params.json
    params_file = optuna_path / f"{symbol.lower()}_{signal_name.lower()}_best_params.json"

    if not params_file.exists():
        logger.debug(f"Optuna params not found: {params_file}")
        return None

    try:
        with open(params_file) as f:
            params = json.load(f)

        if not params:
            logger.warning(f"Empty Optuna params file: {params_file}")
            return None

        logger.info(f"Loaded Optuna hyperparameters from {params_file}")
        return params

    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in Optuna params file {params_file}: {e}")
        return None
    except IOError as e:
        logger.warning(f"Error reading Optuna params file {params_file}: {e}")
        return None


def save_training_metadata(
    metadata: dict[str, Any],
    output_dir: str,
    fold_id: str,
) -> None:
    """Save training metadata to JSON file (Issue #522).

    Saves hyperparameters, data sizes, and training info for reproducibility.

    Args:
        metadata: Dict with training metadata (hyperparameters, data_size, etc.).
        output_dir: Directory to save metadata.
        fold_id: Fold identifier for filename.

    Example:
        save_training_metadata(
            metadata={'symbol': 'eurusd', 'timesteps': 2500},
            output_dir='models/hybrid/eurusd/long',
            fold_id='fold_001',
        )
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Add timestamp if not present
    if "trained_at" not in metadata:
        metadata["trained_at"] = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    metadata_file = output_path / f"{fold_id}_metadata.json"

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.debug(f"Saved training metadata to {metadata_file}")


def format_eta(seconds: float) -> str:
    """Format ETA in human-readable format."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def load_symbol_seeds(seeds_path: str) -> dict[str, list[int]]:
    """Load stable seeds per symbol from JSON file."""
    with open(seeds_path) as f:
        data = json.load(f)

    result = {}
    symbols_data = data.get("symbols", {})

    for symbol, symbol_data in symbols_data.items():
        stable_seeds = symbol_data.get("stable_seeds", [])
        if stable_seeds:
            result[symbol] = stable_seeds
        else:
            seed_results = symbol_data.get("seed_results", {})
            if seed_results:
                best_seed = max(
                    seed_results.values(),
                    key=lambda x: x.get("win_rate", 0),
                )
                result[symbol] = [best_seed.get("seed")]

    return result


def load_folds_config(folds_path: str) -> dict[str, Any]:
    """Load folds configuration from JSON file."""
    if not Path(folds_path).exists():
        raise FileNotFoundError(f"Folds config not found: {folds_path}")

    with open(folds_path) as f:
        data = json.load(f)

    return data.get("symbols", {})


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators for the dataframe."""
    df = df.copy()

    # Ensure numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # SMA
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["sma_200"] = df["close"].rolling(window=200).mean()

    # EMA
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["atr_14"] = true_range.rolling(window=14).mean()

    # Bollinger Bands
    df["bb_middle_20"] = df["sma_20"]
    bb_std = df["close"].rolling(window=20).std()
    df["bb_upper_20"] = df["bb_middle_20"] + (bb_std * 2)
    df["bb_lower_20"] = df["bb_middle_20"] - (bb_std * 2)

    # MACD
    df["macd_line"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

    # Stochastic
    low_14 = df["low"].rolling(window=14).min()
    high_14 = df["high"].rolling(window=14).max()
    df["stoch_k"] = 100 * ((df["close"] - low_14) / (high_14 - low_14))
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

    # Add datetime column
    if "datetime" not in df.columns:
        if "rate_time" in df.columns:
            df["datetime"] = pd.to_datetime(df["rate_time"], unit="s")
        elif "readable_date" in df.columns:
            df["datetime"] = pd.to_datetime(df["readable_date"])

    # Add time column
    if "time" not in df.columns and "datetime" in df.columns:
        df["time"] = df["datetime"].astype(np.int64) // 10**9

    # Drop NaN rows
    indicator_cols = [
        "open", "high", "low", "close", "volume",
        "sma_20", "sma_50", "sma_200",
        "ema_12", "ema_26", "ema_50",
        "rsi_14", "atr_14",
        "bb_middle_20", "bb_upper_20", "bb_lower_20",
        "macd_line", "macd_signal", "macd_histogram",
        "stoch_k", "stoch_d",
    ]
    existing_cols = [c for c in indicator_cols if c in df.columns]
    if existing_cols:
        df = df.dropna(subset=existing_cols).reset_index(drop=True)

    return df


def load_symbol_data(csv_path: str) -> pd.DataFrame:
    """Load M30/H4/D1 data for a symbol from CSV path."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} candles")

    # Handle D1 data with 'timestamp' column instead of 'time'
    if "timestamp" in df.columns and "time" not in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"])
        df["time"] = df["datetime"].astype(np.int64) // 10**9
    # Handle H4/D1 data where 'time' column is a datetime string, not Unix timestamp
    elif "time" in df.columns:
        # Try to detect if 'time' is already a datetime string
        try:
            sample_value = df["time"].iloc[0]
            if isinstance(sample_value, str) or pd.api.types.is_datetime64_any_dtype(df["time"]):
                # 'time' is a datetime string, convert it
                df["datetime"] = pd.to_datetime(df["time"])
                df["time"] = df["datetime"].astype(np.int64) // 10**9
        except (IndexError, ValueError):
            pass  # 'time' is already Unix timestamp, no conversion needed

    logger.info("Computing technical indicators...")
    df = compute_indicators(df)
    logger.info(f"After indicators: {len(df)} candles")

    return df


def create_hybrid_environment(
    df: pd.DataFrame,
    symbol: str,
    cross_lookback: int = 5,
    signal_duration_bars: int = 48,
    require_ob_confirmation: bool = False,
    max_bars: int = 200,  # Increased from default 50 to allow natural SL/TP exits
    use_v2_features: bool = False,  # Issue #495: V2 multi-TF feature integration
    signal_type: str = "sma",  # Issue #496: Signal type (sma or rsi)
    direction_filter: str = "both",  # Issue #512: Filter by direction
):
    """Create HybridTradingEnv for training.

    Args:
        df: DataFrame with OHLCV data and indicators.
        symbol: Trading symbol.
        cross_lookback: Bars to look back for crossover detection.
        signal_duration_bars: How many M30 bars a signal stays active.
        require_ob_confirmation: If True, require OB proximity for signals.
        max_bars: Episode length in bars (default 200 = ~4 days for M30).
            Increased from 50 to allow trades time to reach SL/TP naturally.
        use_v2_features: If True, enable V2 multi-TF features (327 dims).
            When False, uses backward compatible mode (242 dims).
        signal_type: Signal generator type: 'sma' (D1 crossover) or 'rsi' (H1 30/70).
        direction_filter: Filter signals by direction (Issue #512).
            - "long": Only process LONG signals
            - "short": Only process SHORT signals
            - "both": Process all signals (default)

    Returns:
        HybridTradingEnv instance.
    """
    from pattern_system.rl.hybrid_env import HybridTradingEnv

    return HybridTradingEnv(
        df=df,
        symbol=symbol,
        primary_timeframe="m30",
        cross_lookback=cross_lookback,
        signal_duration_bars=signal_duration_bars,
        require_ob_confirmation=require_ob_confirmation,
        max_bars=max_bars,
        use_v2_features=use_v2_features,
        signal_type=signal_type,
        direction_filter=direction_filter,
    )


def train_single_fold(
    symbol: str,
    fold_id: str,
    fold_config: dict[str, Any],
    seed: int,
    timesteps: int | None,
    output_dir: str,
    data_df: pd.DataFrame,
    cross_lookback: int = 5,
    signal_duration_bars: int = 48,
    require_ob_confirmation: bool = False,
    use_v2_features: bool = False,
    signal_type: str = "sma",
    direction_filter: str = "both",
    ppo_config: dict[str, Any] | None = None,
    use_adaptive_timesteps: bool = False,
    save_metadata: bool = False,
    rollouts: float = DEFAULT_ROLLOUTS,
) -> dict[str, Any] | None:
    """Train a single fold for a symbol using hybrid architecture.

    Args:
        symbol: Trading symbol.
        fold_id: Fold identifier.
        fold_config: Fold configuration with train/val/test indices.
        seed: Random seed for training.
        timesteps: Number of training timesteps. If None and use_adaptive_timesteps=True,
                  will be calculated from data size.
        output_dir: Output directory for models.
        data_df: Full DataFrame with indicators.
        cross_lookback: Bars to look back for crossover detection.
        signal_duration_bars: How many M30 bars a signal stays active.
        require_ob_confirmation: If True, require OB proximity for signals.
        use_v2_features: If True, enable V2 multi-TF features (327 dims).
        signal_type: Signal generator type: 'sma' (D1 crossover) or 'rsi' (H1 30/70).
        direction_filter: Filter signals by direction: 'long', 'short', or 'both'.
        ppo_config: Custom PPO hyperparameters (can be from Optuna).
        use_adaptive_timesteps: If True, calculate timesteps from data size (Issue #522).
        save_metadata: If True, save training metadata to JSON file (Issue #522).
        rollouts: Number of rollouts for adaptive timesteps (default 2.5x).

    Returns:
        Dictionary with training results or None on failure.
    """
    from stable_baselines3 import PPO

    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"HYBRID Training: {symbol.upper()} | {fold_id} | Seed: {seed}")
        logger.info(f"{'='*60}")

        start_time = time.time()

        # Extract training data
        train_start = fold_config["train"]["start"]
        train_end = fold_config["train"]["end"]
        train_df = data_df.iloc[train_start:train_end].copy().reset_index(drop=True)
        data_size = len(train_df)

        logger.info(f"Training data: {data_size} candles")
        logger.info(f"Signal config: lookback={cross_lookback}, duration={signal_duration_bars}")

        # Issue #522: Calculate adaptive timesteps from data size
        actual_timesteps = timesteps
        actual_rollouts = 0.0

        if use_adaptive_timesteps or timesteps is None:
            from pattern_system.rl.optuna_tuning import calculate_optimal_timesteps

            actual_timesteps = calculate_optimal_timesteps(data_size, rollouts=rollouts)
            actual_rollouts = rollouts
            logger.info(
                f"Using adaptive timesteps: {actual_timesteps:,} ({rollouts:.1f}x rollouts)"
            )
        else:
            actual_timesteps = timesteps if timesteps else DEFAULT_TIMESTEPS
            actual_rollouts = actual_timesteps / data_size if data_size > 0 else 0.0
            logger.info(
                f"Using fixed timesteps: {actual_timesteps:,} ({actual_rollouts:.1f}x rollouts)"
            )

        # Create hybrid environment
        env = create_hybrid_environment(
            train_df,
            symbol,
            cross_lookback=cross_lookback,
            signal_duration_bars=signal_duration_bars,
            require_ob_confirmation=require_ob_confirmation,
            use_v2_features=use_v2_features,
            signal_type=signal_type,
            direction_filter=direction_filter,
        )

        # Use custom PPO config or fall back to defaults
        cfg = ppo_config if ppo_config else PPO_CONFIG_HYBRID

        # Create model with hybrid config
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=cfg.get("learning_rate", PPO_CONFIG_HYBRID["learning_rate"]),
            n_steps=cfg.get("n_steps", PPO_CONFIG_HYBRID["n_steps"]),
            batch_size=cfg.get("batch_size", PPO_CONFIG_HYBRID["batch_size"]),
            n_epochs=cfg.get("n_epochs", PPO_CONFIG_HYBRID["n_epochs"]),
            gamma=cfg.get("gamma", PPO_CONFIG_HYBRID["gamma"]),
            gae_lambda=cfg.get("gae_lambda", PPO_CONFIG_HYBRID["gae_lambda"]),
            clip_range=cfg.get("clip_range", PPO_CONFIG_HYBRID["clip_range"]),
            ent_coef=cfg.get("ent_coef", PPO_CONFIG_HYBRID["ent_coef"]),
            verbose=0,
            seed=seed,
            device=cfg.get("device", PPO_CONFIG_HYBRID["device"]),
        )

        # Train
        logger.info(f"Training for {actual_timesteps:,} timesteps...")
        model.learn(total_timesteps=actual_timesteps)

        # Save model (Issue #516: Add direction subdirectory to separate LONG/SHORT models)
        model_path = Path(output_dir) / symbol / direction_filter / f"{fold_id}_seed_{seed}.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path).replace(".zip", ""))

        training_time = time.time() - start_time
        logger.info(
            f"Training complete in {format_eta(training_time)}, saved to {model_path}"
        )

        result = {
            "model_path": str(model_path),
            "training_time": training_time,
            "fold_id": fold_id,
            "seed": seed,
            "timesteps": actual_timesteps,
            "data_size": data_size,
            "rollouts": actual_rollouts,
        }

        # Issue #522: Save training metadata
        if save_metadata:
            metadata = {
                "symbol": symbol,
                "direction": direction_filter,
                "fold_id": fold_id,
                "seed": seed,
                "timesteps": actual_timesteps,
                "data_size": data_size,
                "rollouts": actual_rollouts,
                "hyperparameters": {
                    "learning_rate": cfg.get("learning_rate", PPO_CONFIG_HYBRID["learning_rate"]),
                    "n_steps": cfg.get("n_steps", PPO_CONFIG_HYBRID["n_steps"]),
                    "batch_size": cfg.get("batch_size", PPO_CONFIG_HYBRID["batch_size"]),
                    "n_epochs": cfg.get("n_epochs", PPO_CONFIG_HYBRID["n_epochs"]),
                    "gamma": cfg.get("gamma", PPO_CONFIG_HYBRID["gamma"]),
                    "gae_lambda": cfg.get("gae_lambda", PPO_CONFIG_HYBRID["gae_lambda"]),
                    "clip_range": cfg.get("clip_range", PPO_CONFIG_HYBRID["clip_range"]),
                    "ent_coef": cfg.get("ent_coef", PPO_CONFIG_HYBRID["ent_coef"]),
                },
                "training_duration_seconds": training_time,
                "use_adaptive_timesteps": use_adaptive_timesteps,
            }
            save_training_metadata(
                metadata=metadata,
                output_dir=str(model_path.parent),
                fold_id=fold_id,
            )

        return result

    except Exception as e:
        logger.error(f"Training failed for {symbol} {fold_id}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def validate_fold(
    model_path: str,
    val_df: pd.DataFrame,
    symbol: str,
    cross_lookback: int = 5,
    signal_duration_bars: int = 48,
    require_ob_confirmation: bool = False,
    use_v2_features: bool = False,
    signal_type: str = "sma",
    direction_filter: str = "both",
) -> dict[str, Any]:
    """Validate a trained hybrid model on validation data.

    Issue #522: Enhanced with full cost accounting and profit factor calculation.

    Args:
        model_path: Path to the trained model.
        val_df: Validation DataFrame.
        symbol: Trading symbol.
        cross_lookback: Bars to look back for crossover detection.
        signal_duration_bars: How many M30 bars a signal stays active.
        require_ob_confirmation: If True, require OB proximity for signals.
        use_v2_features: If True, enable V2 multi-TF features (327 dims).
        signal_type: Signal generator type: 'sma' (D1 crossover) or 'rsi' (H1 30/70).
        direction_filter: Filter signals by direction: 'long', 'short', or 'both'.

    Returns:
        Dictionary with validation metrics including:
        - win_rate: Percentage of winning trades
        - trade_rate: Trades per step percentage
        - profit_factor: Gross profit / Gross loss
        - gross_profit: Sum of winning trades (pips)
        - gross_loss: Sum of losing trades (pips, positive)
        - total_costs: Total trading costs (pips)
    """
    from stable_baselines3 import PPO
    from indicator_discovery.tester import TradingCosts
    from indicator_discovery.statistics import compute_profit_factor

    logger.info(f"Validating model: {model_path}")

    if len(val_df) == 0:
        logger.warning("Empty validation data, skipping")
        return {
            "win_rate": 0.0,
            "trade_rate": 0.0,
            "signal_rate": 0.0,
            "total_trades": 0,
            "profit_factor": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "total_costs": 0.0,
            "skipped": True,
        }

    # Load model
    model = PPO.load(model_path.replace(".zip", ""))

    # Create validation environment with direction filter
    env = create_hybrid_environment(
        val_df,
        symbol,
        cross_lookback=cross_lookback,
        signal_duration_bars=signal_duration_bars,
        require_ob_confirmation=require_ob_confirmation,
        use_v2_features=use_v2_features,
        signal_type=signal_type,
        direction_filter=direction_filter,
    )

    # Default trading costs for forex (Issue #522)
    costs = TradingCosts(
        spread=1.5,
        slippage=0.5,
        commission=0.0,
    )

    # Evaluate
    obs, info = env.reset()
    action_counts = {0: 0, 1: 0}
    trades = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    signals_received = 0
    max_steps = min(10000, len(val_df))

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        action_counts[int(action)] += 1
        obs, reward, terminated, truncated, info = env.step(action)

        if info.get("position_closed", False):
            trades += 1
            pnl_pips = info.get("pnl_pips", 0)
            if pnl_pips > 0:
                wins += 1
                gross_profit += pnl_pips
            else:
                gross_loss += abs(pnl_pips)

        if terminated or truncated:
            # Accumulate signal stats
            signals_received += env.signals_received
            obs, info = env.reset()

    # Final episode
    signals_received += env.signals_received

    # Calculate costs (Issue #522: Full cost accounting)
    total_costs = costs.total_per_trade * trades

    # Calculate profit factor after costs
    # Deduct costs from gross profit for realistic PF
    adjusted_gross_profit = max(0.0, gross_profit - total_costs)
    profit_factor = compute_profit_factor(adjusted_gross_profit, gross_loss)

    # Calculate metrics
    total_actions = sum(action_counts.values())
    hold_pct = 100 * action_counts[0] / total_actions if total_actions > 0 else 100
    close_pct = 100 * action_counts[1] / total_actions if total_actions > 0 else 0

    trade_rate = 100 * trades / max_steps if max_steps > 0 else 0
    signal_rate = 100 * signals_received / max_steps if max_steps > 0 else 0
    win_rate = 100 * wins / trades if trades > 0 else 0

    result = {
        "win_rate": round(win_rate, 2),
        "trade_rate": round(trade_rate, 2),
        "signal_rate": round(signal_rate, 2),
        "hold_pct": round(hold_pct, 2),
        "close_pct": round(close_pct, 2),
        "total_trades": trades,
        "winning_trades": wins,
        "losing_trades": trades - wins,
        "signals_received": signals_received,
        # Issue #522: Cost accounting fields
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else float("inf"),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "total_costs": round(total_costs, 2),
    }

    logger.info(
        f"Validation: Win Rate={win_rate:.1f}% | PF={profit_factor:.2f} | "
        f"Trades={trades} | Gross P/L={gross_profit:.1f}/{gross_loss:.1f} pips"
    )

    return result


def train_symbol(
    symbol: str,
    seed: int,
    folds_config: dict[str, Any],
    timesteps: int,
    output_dir: str,
    checkpoint_manager: CheckpointManager,
    specific_fold: int | None = None,
    cross_lookback: int = 5,
    signal_duration_bars: int = 48,
    require_ob_confirmation: bool = False,
    use_v2_features: bool = False,
    signal_type: str = "sma",
    direction_filter: str = "both",
    ppo_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train all folds for a symbol using hybrid architecture.

    Args:
        symbol: Trading symbol.
        seed: Random seed for training.
        folds_config: Folds configuration for the symbol.
        timesteps: Number of training timesteps.
        output_dir: Output directory for models.
        checkpoint_manager: Checkpoint manager for resume support.
        specific_fold: If set, only train this specific fold number.
        cross_lookback: Bars to look back for crossover detection.
        signal_duration_bars: How many M30 bars a signal stays active.
        require_ob_confirmation: If True, require OB proximity for signals.
        signal_type: Signal generator type: 'sma' (D1 crossover) or 'rsi' (H1 30/70).
        direction_filter: Filter signals by direction: 'long', 'short', or 'both'.
        use_v2_features: If True, enable V2 multi-TF features (327 dims).

    Returns:
        Dictionary with training summary.
    """
    logger.info(f"\n{'#'*80}")
    logger.info(f"HYBRID TRAINING: {symbol.upper()}")
    logger.info(f"Seed: {seed} | Timesteps: {timesteps:,}")
    logger.info(f"{'#'*80}")

    # Load symbol data
    csv_path = folds_config.get("csv_path")
    if not csv_path:
        csv_path = str(POSTGRES_DATA_PATH / f"{symbol}_m30_rates.csv")

    # V3 signals use different timeframes per symbol (Issue #516)
    if signal_type == "v3":
        # USDCAD uses D1 timeframe, others use H4
        if symbol.lower() == "usdcad":
            csv_path = str(Path(__file__).parent.parent / "data" / "csv" / f"{symbol}_d1_with_indicators.csv")
            signal_duration_bars = 1  # D1: 1 bar = 1 day
            logger.info(f"Using D1 data for V3 signals: {csv_path}")
        else:
            csv_path = str(Path(__file__).parent.parent / "data" / "csv" / f"{symbol}_h4_with_indicators.csv")
            signal_duration_bars = 6  # H4: 6 bars = 1 day
            logger.info(f"Using H4 data for V3 signals: {csv_path}")

    try:
        data_df = load_symbol_data(csv_path)
    except FileNotFoundError as e:
        logger.error(f"Failed to load data for {symbol}: {e}")
        return {"completed": 0, "failed": 1, "error": str(e)}

    # Get folds
    folds = folds_config.get("folds", {})
    fold_ids = sorted(folds.keys())

    # Filter to specific fold if requested
    if specific_fold is not None:
        fold_id = f"fold_{specific_fold:03d}"
        if fold_id in fold_ids:
            fold_ids = [fold_id]
        else:
            logger.error(f"Fold {fold_id} not found for {symbol}")
            return {"completed": 0, "failed": 1}

    # Skip completed folds
    remaining_folds = [f for f in fold_ids if not checkpoint_manager.is_completed(symbol, f)]

    logger.info(f"Total folds: {len(fold_ids)}")
    logger.info(f"Remaining folds: {len(remaining_folds)}")

    completed = 0
    failed = 0
    results = []

    for fold_id in remaining_folds:
        fold_config = folds[fold_id]

        # Train fold
        result = train_single_fold(
            symbol=symbol,
            fold_id=fold_id,
            fold_config=fold_config,
            seed=seed,
            timesteps=timesteps,
            output_dir=output_dir,
            data_df=data_df,
            cross_lookback=cross_lookback,
            signal_duration_bars=signal_duration_bars,
            require_ob_confirmation=require_ob_confirmation,
            use_v2_features=use_v2_features,
            signal_type=signal_type,
            direction_filter=direction_filter,
            ppo_config=ppo_config,
        )

        if result and "error" not in result:
            # Validate fold
            val_start = fold_config["val"]["start"]
            val_end = fold_config["val"]["end"]
            val_df = data_df.iloc[val_start:val_end].copy().reset_index(drop=True)

            val_result = validate_fold(
                model_path=result["model_path"],
                val_df=val_df,
                symbol=symbol,
                cross_lookback=cross_lookback,
                signal_duration_bars=signal_duration_bars,
                require_ob_confirmation=require_ob_confirmation,
                use_v2_features=use_v2_features,
                signal_type=signal_type,
            )
            result.update(val_result)
            results.append(result)

            # Mark completed
            checkpoint_manager.mark_completed(symbol, fold_id)
            checkpoint_manager.save()
            completed += 1
        else:
            failed += 1

    # Summary statistics
    if results:
        avg_wr = np.mean([r["win_rate"] for r in results])
        avg_tr = np.mean([r["trade_rate"] for r in results])
        avg_sr = np.mean([r["signal_rate"] for r in results])
        total_trades = sum([r["total_trades"] for r in results])

        logger.info(f"\n{'='*60}")
        logger.info(f"SUMMARY: {symbol.upper()}")
        logger.info(f"Avg Win Rate: {avg_wr:.1f}%")
        logger.info(f"Avg Trade Rate: {avg_tr:.2f}%")
        logger.info(f"Avg Signal Rate: {avg_sr:.2f}%")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"{'='*60}")

    return {"completed": completed, "failed": failed, "results": results}


def setup_signal_handlers(checkpoint_manager: CheckpointManager):
    """Setup signal handlers for graceful shutdown."""
    global _checkpoint_manager
    _checkpoint_manager = checkpoint_manager

    def signal_handler(signum, frame):
        logger.info("\nReceived interrupt signal, saving checkpoint...")
        if _checkpoint_manager:
            _checkpoint_manager.save()
        logger.info("Checkpoint saved, exiting.")
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Train Hybrid Architecture models (Issue #494)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train single fold
    python scripts/train_hybrid.py --symbol eurusd --fold 1

    # Train all folds for one symbol
    python scripts/train_hybrid.py --symbol eurusd --all-folds

    # Train all symbols and folds
    python scripts/train_hybrid.py --all-symbols --all-folds

    # Resume interrupted training
    python scripts/train_hybrid.py --symbol eurusd --all-folds --resume
        """,
    )

    # Symbol selection
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument(
        "--symbol",
        type=str,
        choices=SUPPORTED_SYMBOLS,
        help="Train specific symbol",
    )
    symbol_group.add_argument(
        "--all-symbols",
        action="store_true",
        help="Train all symbols",
    )

    # Fold selection
    fold_group = parser.add_mutually_exclusive_group()
    fold_group.add_argument(
        "--fold",
        type=int,
        help="Train specific fold number (1-30)",
    )
    fold_group.add_argument(
        "--all-folds",
        action="store_true",
        help="Train all folds",
    )

    # Training configuration
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"Training timesteps per fold (default: {DEFAULT_TIMESTEPS})",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )

    # PPO hyperparameters (can be tuned with Optuna)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="PPO learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="PPO n_steps (default: 2048)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="PPO batch size (default: 256)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=None,
        help="PPO n_epochs (default: 10)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="PPO discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=None,
        help="PPO GAE lambda (default: 0.95)",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=None,
        help="PPO clip range (default: 0.2)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=None,
        help="PPO entropy coefficient (default: 0.02)",
    )
    parser.add_argument(
        "--ppo-config",
        type=str,
        default=None,
        help="JSON file with PPO hyperparameters",
    )

    # Signal generator configuration
    parser.add_argument(
        "--cross-lookback",
        type=int,
        default=5,
        help="D1 bars to look back for SMA crossover (default: 5)",
    )

    parser.add_argument(
        "--signal-duration",
        type=int,
        default=48,
        help="How many M30 bars a signal stays active (default: 48)",
    )

    parser.add_argument(
        "--require-ob",
        action="store_true",
        help="Require order block confirmation for signals",
    )

    parser.add_argument(
        "--use-v2-features",
        action="store_true",
        help="Enable V2 multi-TF features (327 dims). Issue #495.",
    )

    parser.add_argument(
        "--signal-type",
        type=str,
        default="sma",
        choices=["sma", "rsi", "v3"],
        help="Signal type: 'sma' (D1 SMA crossover), 'rsi' (H1 RSI 30/70), or 'v3' (validated V3 signals). Issue #496, #516.",
    )

    parser.add_argument(
        "--direction",
        type=str,
        default="both",
        choices=SUPPORTED_DIRECTIONS,
        help="Direction filter: 'long' (LONG only), 'short' (SHORT only), 'both' (all). Issue #512.",
    )

    # Configuration paths
    parser.add_argument(
        "--seeds-path",
        type=str,
        default=str(DEFAULT_SEEDS_PATH),
        help="Path to symbol_seeds.json",
    )

    parser.add_argument(
        "--folds-path",
        type=str,
        default=str(DEFAULT_FOLDS_PATH),
        help="Path to all_symbols_folds.json",
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Load configurations
    try:
        symbol_seeds = load_symbol_seeds(args.seeds_path)
        folds_config = load_folds_config(args.folds_path)
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1

    # Determine symbols to train
    if args.all_symbols:
        symbols = SUPPORTED_SYMBOLS
    else:
        symbols = [args.symbol.lower()]

    # Set output directory based on features, signal type, and direction
    output_dir = args.output_dir
    direction = args.direction.lower()

    if args.output_dir == DEFAULT_OUTPUT_DIR:
        if args.signal_type == "rsi":
            # RSI signals go to separate directory
            output_dir = "models/hybrid_rsi" if not args.use_v2_features else "models/hybrid_rsi_v2"
        elif args.use_v2_features:
            output_dir = "models/hybrid_v2"

    # Issue #512: Add direction subdirectory for direction-specific training
    # Structure: models/hybrid_v2/{symbol}/{direction}/ for direction-specific models
    # Note: Symbol subdirectory is added later in train_symbol()

    # Setup checkpoint manager
    checkpoint_path = Path(output_dir) / "checkpoint.json"
    checkpoint_manager = CheckpointManager(str(checkpoint_path))

    # Setup signal handlers
    setup_signal_handlers(checkpoint_manager)

    # Clear checkpoint if not resuming
    if not args.resume:
        checkpoint_manager.completed = {}

    logger.info("=" * 80)
    logger.info("HYBRID ARCHITECTURE TRAINING (Issue #494/#495)")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Timesteps: {args.timesteps:,}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Signal Config: lookback={args.cross_lookback}, duration={args.signal_duration}")
    logger.info(f"Require OB: {args.require_ob}")
    logger.info(f"V2 Features: {args.use_v2_features}")
    logger.info("=" * 80)

    # Build PPO config from args or config file
    ppo_config = None
    if args.ppo_config:
        try:
            with open(args.ppo_config) as f:
                ppo_config = json.load(f)
            logger.info(f"Loaded PPO config from {args.ppo_config}")
        except Exception as e:
            logger.warning(f"Could not load PPO config: {e}, using defaults")

    # Override with command-line args
    if any([args.learning_rate, args.n_steps, args.batch_size, args.n_epochs,
            args.gamma, args.gae_lambda, args.clip_range, args.ent_coef]):
        if ppo_config is None:
            ppo_config = {}
        if args.learning_rate:
            ppo_config["learning_rate"] = args.learning_rate
        if args.n_steps:
            ppo_config["n_steps"] = args.n_steps
        if args.batch_size:
            ppo_config["batch_size"] = args.batch_size
        if args.n_epochs:
            ppo_config["n_epochs"] = args.n_epochs
        if args.gamma:
            ppo_config["gamma"] = args.gamma
        if args.gae_lambda:
            ppo_config["gae_lambda"] = args.gae_lambda
        if args.clip_range:
            ppo_config["clip_range"] = args.clip_range
        if args.ent_coef:
            ppo_config["ent_coef"] = args.ent_coef
        logger.info(f"Custom PPO config: {ppo_config}")

    total_completed = 0
    total_failed = 0

    for symbol in symbols:
        seeds = symbol_seeds.get(symbol, [42])
        seed = seeds[0] if seeds else 42

        symbol_folds = folds_config.get(symbol, {})
        if not symbol_folds:
            logger.warning(f"No folds config for {symbol}, skipping")
            continue

        result = train_symbol(
            symbol=symbol,
            seed=seed,
            folds_config=symbol_folds,
            timesteps=args.timesteps,
            output_dir=output_dir,
            checkpoint_manager=checkpoint_manager,
            specific_fold=args.fold if not args.all_folds else None,
            cross_lookback=args.cross_lookback,
            signal_duration_bars=args.signal_duration,
            require_ob_confirmation=args.require_ob,
            use_v2_features=args.use_v2_features,
            signal_type=args.signal_type,
            direction_filter=args.direction,
            ppo_config=ppo_config,
        )

        total_completed += result.get("completed", 0)
        total_failed += result.get("failed", 0)

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info(f"Total Completed: {total_completed}")
    logger.info(f"Total Failed: {total_failed}")
    logger.info("=" * 80)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
