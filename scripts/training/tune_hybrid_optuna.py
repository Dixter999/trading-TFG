#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization for Hybrid Exit Models.

This script performs hyperparameter optimization for the PPO-based hybrid exit models
that combine rule-based entries with RL-managed exits.

Usage:
    # Tune EURJPY with 50 trials overnight
    python scripts/tune_hybrid_optuna.py --symbol eurjpy --n-trials 50

    # Resume from existing study
    python scripts/tune_hybrid_optuna.py --symbol eurjpy --n-trials 100 --resume

    # Quick test with fewer trials
    python scripts/tune_hybrid_optuna.py --symbol eurjpy --n-trials 10 --timesteps 25000

Key Features:
    - TPE sampler for efficient Bayesian optimization
    - MedianPruner to early-stop bad trials
    - SQLite storage for persistence and resumability
    - Validation on held-out OOS data
    - Win Rate as primary optimization metric (>54% target for EURJPY)
    - Exports best parameters for production training

Issue: Hyperparameter Tuning for EURJPY
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO

# Issue #528: Import bug fix module
from pattern_system.rl.optuna_tuning import (
    calculate_n_episodes,
    calculate_optimal_timesteps,
    get_trial_seed,
    MAX_ROLLOUTS,
    DEFAULT_ROLLOUTS,
)

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Silence noisy loggers
logging.getLogger("stable_baselines3").setLevel(logging.WARNING)

# ==============================================================================
# Constants
# ==============================================================================

# PostgreSQL data path
POSTGRES_DATA_PATH = Path("data/rates")

# Default directories
DEFAULT_OUTPUT_DIR = "results/optuna_hybrid"
# Fix #5: Use symbol-specific fold configurations (not all_symbols_folds.json)
# all_symbols_folds.json has tiny 2.3%/0.5% folds for prototyping
# Symbol-specific files have proper 70/15/15 train/val/test splits
def get_fold_config_path(symbol: str) -> Path:
    """Get symbol-specific fold configuration path.

    Fix #5 (Issue #519): Use symbol-specific fold configs instead of
    all_symbols_folds.json which has tiny prototyping folds.

    Fix #6 (Issue #519): Prefer H4 fold configs over M30 to match actual training data timeframe.
    This prevents validation failures where fold indices are out of bounds.

    Args:
        symbol: Trading symbol (e.g., 'eurusd', 'gbpusd')

    Returns:
        Path to symbol-specific fold configuration file
    """
    symbol_lower = symbol.lower()

    # Prefer H4 fold configs (match H4 training data)
    h4_path = PROJECT_ROOT / "config" / "folds" / f"{symbol_lower}_h4_folds.json"
    if h4_path.exists():
        return h4_path

    # Fallback to base fold config (M30 data - legacy)
    return PROJECT_ROOT / "config" / "folds" / f"{symbol_lower}_folds.json"

DEFAULT_SEEDS_PATH = PROJECT_ROOT / "config" / "symbol_seeds.json"

# Supported symbols (Issue #511: Added USDCAD, EURCAD)
SUPPORTED_SYMBOLS = ["eurusd", "gbpusd", "usdjpy", "eurjpy", "xagusd", "usdcad", "eurcad"]

# Supported directions (Issue #512: Signal-Specific Training)
SUPPORTED_DIRECTIONS = ["long", "short", "both"]

# Default training config
DEFAULT_TIMESTEPS = 50000
DEFAULT_N_TRIALS = 50
DEFAULT_TOTAL_BUDGET = 2_500_000  # Total timesteps across all trials

# ==============================================================================
# Hyperparameter Search Spaces
# ==============================================================================

# Learning rate range (log scale)
LEARNING_RATE_RANGE = (1e-5, 5e-3)

# Discount factor range
GAMMA_RANGE = (0.95, 0.999)

# GAE lambda range
GAE_LAMBDA_RANGE = (0.9, 0.99)

# PPO clip range
CLIP_RANGE_RANGE = (0.1, 0.4)

# Entropy coefficient range
ENT_COEF_RANGE = (0.0, 0.1)

# Number of steps per update
N_STEPS_OPTIONS = [512, 1024, 2048, 4096]

# Batch size options
BATCH_SIZE_OPTIONS = [64, 128, 256, 512]

# Number of epochs per update
N_EPOCHS_RANGE = (3, 15)

# Network architecture options
NET_ARCH_OPTIONS = ["small", "medium", "large"]
NET_ARCH_MAPPING = {
    "small": [64, 64],
    "medium": [128, 64],
    "large": [256, 128],
}


# ==============================================================================
# Helper Functions
# ==============================================================================


def load_folds_config(folds_path: str) -> dict[str, Any]:
    """Load folds configuration from JSON file.

    Returns the raw data dict - caller is responsible for extracting
    the correct structure (symbol-specific files have different structure
    than all_symbols_folds.json).
    """
    if not Path(folds_path).exists():
        raise FileNotFoundError(f"Folds config not found: {folds_path}")

    with open(folds_path) as f:
        data = json.load(f)

    return data  # Return raw data, let caller extract correct structure


def load_symbol_seeds(seeds_path: str) -> dict[str, list[int]]:
    """Load stable seeds per symbol from JSON file."""
    if not Path(seeds_path).exists():
        return {}

    with open(seeds_path) as f:
        data = json.load(f)

    result = {}
    symbols_data = data.get("symbols", {})

    for symbol, symbol_data in symbols_data.items():
        stable_seeds = symbol_data.get("stable_seeds", [])
        if stable_seeds:
            result[symbol] = stable_seeds

    return result


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


def load_symbol_data(symbol: str, signal_type: str = "sma") -> pd.DataFrame:
    """Load data for a symbol (M30 by default, D1 for V3 signals).

    Args:
        symbol: Trading symbol
        signal_type: Signal type - if 'v3', will load D1 data for USDCAD

    Returns:
        DataFrame with OHLCV and indicators
    """
    # V3 signals for USDCAD use D1 timeframe
    if signal_type == "v3" and symbol.lower() == "usdcad":
        csv_path = Path("data/csv") / f"{symbol.lower()}_d1_with_indicators.csv"
        logger.info(f"Loading D1 data for USDCAD V3 from {csv_path}")
    else:
        csv_path = POSTGRES_DATA_PATH / f"{symbol}_m30_rates.csv"
        logger.info(f"Loading M30 data from {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} candles")

    # If D1 data with indicators, rename columns to match expected names
    if signal_type == "v3" and symbol.lower() == "usdcad":
        rename_map = {
            "macd_line": "macd",
            "bb_upper_20": "bb_upper",
            "bb_middle_20": "bb_middle",
            "bb_lower_20": "bb_lower",
        }
        df = df.rename(columns=rename_map)
        logger.info(f"Renamed D1 indicator columns for V3 compatibility")

        # Add datetime and time columns from timestamp
        if "timestamp" in df.columns and "datetime" not in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"])
            df["time"] = df["datetime"].astype(np.int64) // 10**9
            logger.info("Added datetime and time columns from timestamp")
    else:
        logger.info("Computing technical indicators...")
        df = compute_indicators(df)
        logger.info(f"After indicators: {len(df)} candles")

    return df


def create_hybrid_environment(
    df: pd.DataFrame,
    symbol: str,
    cross_lookback: int = 5,
    signal_duration_bars: int = 48,
    max_bars: int = 200,
    use_v2_features: bool = True,
    signal_type: str = "sma",
    direction_filter: str = "both",
    truncate_early: bool = True,  # Issue #528 Bug #2: Set False for validation
    max_entries_per_episode: int | None = 50,  # Issue #528 Bug #3: None = unlimited
    disable_signal_locking: bool = False,  # Issue #528 Bug #3: Disable crossover lock
):
    """Create HybridTradingEnv for training.

    Args:
        df: Price data DataFrame
        symbol: Trading symbol
        cross_lookback: Bars to look back for crossover detection
        signal_duration_bars: How many bars a signal stays active
        max_bars: Maximum bars per episode
        use_v2_features: Use V2 multi-TF features
        signal_type: Signal generator type (sma, rsi, combined, v3)
        direction_filter: Filter signals by direction (long, short, both)
        truncate_early: If True, truncate at 80% of data (training default).
                        Set False for validation to use 100% of data.
        max_entries_per_episode: Max trades per episode. None = unlimited.
        disable_signal_locking: Disable crossover signal locking for validation.

    Issue #528 Bug Fixes:
        - Bug #2: truncate_early=False uses 100% of validation data
        - Bug #3: max_entries_per_episode=None allows unlimited trades
        - Bug #3: disable_signal_locking=True allows multiple entries per crossover
    """
    from pattern_system.rl.hybrid_env import HybridTradingEnv

    # V3 signals for USDCAD use D1 timeframe, adjust parameters accordingly
    if signal_type == "v3" and symbol.lower() == "usdcad":
        primary_timeframe = "d1"
        # D1 signal duration: 1 bar = 1 day (vs M30: 48 bars = 1 day)
        signal_duration_bars = 1
        # Disable V2 features for D1 (datetime type incompatibility)
        use_v2_features = False
    else:
        primary_timeframe = "m30"

    return HybridTradingEnv(
        df=df,
        symbol=symbol,
        primary_timeframe=primary_timeframe,
        cross_lookback=cross_lookback,
        signal_duration_bars=signal_duration_bars,
        require_ob_confirmation=False,
        max_bars=max_bars,
        use_v2_features=use_v2_features,
        signal_type=signal_type,
        direction_filter=direction_filter,
        # Issue #528 Bug Fixes
        truncate_early=truncate_early,
        max_entries_per_episode=max_entries_per_episode,
        disable_signal_locking=disable_signal_locking,
    )


# ==============================================================================
# Evaluation Functions
# ==============================================================================


def evaluate_model(
    model: PPO,
    val_df: pd.DataFrame,
    symbol: str,
    n_episodes: int | None = None,  # Issue #528 Bug #1: Calculate from data size
    use_v2_features: bool = True,
    signal_type: str = "sma",
    direction_filter: str = "both",
    trial_seed: int | None = None,  # Issue #528 Bug #6: Trial-specific seed
    truncate_early: bool = False,  # Issue #528 Bug #2: Use 100% of data
) -> dict[str, float]:
    """Evaluate model on validation data with Profit Factor calculation.

    Issue #528 Bug Fixes Applied:
    - Bug #1: n_episodes calculated from data size (not hardcoded 1000)
    - Bug #2: truncate_early=False uses 100% of validation data
    - Bug #6: trial_seed provides trial-specific randomization
    """
    # Bug #1 Fix: Calculate n_episodes from data size
    if n_episodes is None:
        n_episodes = calculate_n_episodes(len(val_df))

    # Bug #6 Fix: Set trial-specific random seed
    if trial_seed is not None:
        np.random.seed(trial_seed)

    # Bug #2 & #3 Fix: Create environment with validation-specific settings
    env = create_hybrid_environment(
        val_df,
        symbol,
        use_v2_features=use_v2_features,
        signal_type=signal_type,
        direction_filter=direction_filter,
        # Issue #528 Bug Fixes for validation
        truncate_early=truncate_early,  # Bug #2: False = use 100% of data
        max_entries_per_episode=None,  # Bug #3: Unlimited entries
        disable_signal_locking=True,  # Bug #3: Allow multiple entries per crossover
    )

    wins = 0
    losses = 0
    total_trades = 0
    total_pnl = 0.0
    total_wins_pips = 0.0  # Track winning pips separately (Fix #2)
    total_losses_pips = 0.0  # Track losing pips separately (Fix #2)
    signals_received = 0

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_trades = 0
        episode_pnl = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if info.get("position_closed", False):
                episode_trades += 1
                pnl = info.get("pnl_pips", 0)
                episode_pnl += pnl
                if pnl > 0:
                    wins += 1
                    total_wins_pips += pnl  # Accumulate winning pips (Fix #2)
                else:
                    losses += 1
                    total_losses_pips += abs(pnl)  # Accumulate losing pips as positive (Fix #2)

        total_trades += episode_trades
        total_pnl += episode_pnl
        signals_received += env.signals_received

    win_rate = 100 * wins / total_trades if total_trades > 0 else 0.0
    avg_pnl = total_pnl / n_episodes if n_episodes > 0 else 0.0
    trade_rate = total_trades / n_episodes if n_episodes > 0 else 0.0

    # Calculate Profit Factor (Fix #2, #4)
    # Handle three cases correctly:
    if total_losses_pips > 0:
        profit_factor = total_wins_pips / total_losses_pips  # Normal case
    elif total_wins_pips > 0:
        profit_factor = 999.0  # Perfect WR with profits → very high PF
    else:
        profit_factor = 0.0  # No wins, no losses → neutral

    env.close()

    return {
        "win_rate": float(win_rate),
        "total_trades": int(total_trades),
        "avg_pnl_pips": float(avg_pnl),
        "trade_rate": float(trade_rate),
        "signals_received": int(signals_received),
        "profit_factor": float(profit_factor),  # Add Profit Factor to metrics (Fix #2)
        "total_wins_pips": float(total_wins_pips),
        "total_losses_pips": float(total_losses_pips),
    }


# ==============================================================================
# Optuna Objective
# ==============================================================================


def create_objective(
    symbol: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    timesteps: int,
    use_v2_features: bool = True,
    signal_type: str = "sma",
    direction_filter: str = "both",
    verbose: int = 0,
):
    """Create Optuna objective function for hybrid model optimization."""

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: maximize OOS win rate."""
        # Sample hyperparameters
        learning_rate = trial.suggest_float(
            "learning_rate",
            LEARNING_RATE_RANGE[0],
            LEARNING_RATE_RANGE[1],
            log=True,
        )
        gamma = trial.suggest_float(
            "gamma",
            GAMMA_RANGE[0],
            GAMMA_RANGE[1],
        )
        gae_lambda = trial.suggest_float(
            "gae_lambda",
            GAE_LAMBDA_RANGE[0],
            GAE_LAMBDA_RANGE[1],
        )
        clip_range = trial.suggest_float(
            "clip_range",
            CLIP_RANGE_RANGE[0],
            CLIP_RANGE_RANGE[1],
        )
        ent_coef = trial.suggest_float(
            "ent_coef",
            ENT_COEF_RANGE[0],
            ENT_COEF_RANGE[1],
        )
        n_steps = trial.suggest_categorical(
            "n_steps",
            N_STEPS_OPTIONS,
        )
        batch_size = trial.suggest_categorical(
            "batch_size",
            BATCH_SIZE_OPTIONS,
        )
        n_epochs = trial.suggest_int(
            "n_epochs",
            N_EPOCHS_RANGE[0],
            N_EPOCHS_RANGE[1],
        )
        net_arch_name = trial.suggest_categorical(
            "net_arch",
            NET_ARCH_OPTIONS,
        )
        net_arch = NET_ARCH_MAPPING[net_arch_name]

        if verbose > 0:
            logger.info(
                f"Trial {trial.number}: lr={learning_rate:.2e}, "
                f"gamma={gamma:.4f}, ent={ent_coef:.4f}, "
                f"n_steps={n_steps}, batch={batch_size}, "
                f"epochs={n_epochs}, net={net_arch_name}"
            )

        try:
            # Issue #528 Bug #6: Get trial-specific seed for variance
            trial_seed = get_trial_seed(trial.number)

            # Issue #528 Bug #5: Calculate optimal timesteps (2-3x rollouts)
            # Use provided timesteps but warn if too high
            data_size = len(train_df)
            rollouts = timesteps / data_size
            if rollouts > MAX_ROLLOUTS:
                logger.warning(
                    f"Trial {trial.number}: Rollouts={rollouts:.1f}x exceeds max {MAX_ROLLOUTS}x. "
                    f"Consider using timesteps={int(data_size * DEFAULT_ROLLOUTS)}"
                )

            # Create environment (training: truncate_early=True for overfitting prevention)
            env = create_hybrid_environment(
                train_df,
                symbol,
                use_v2_features=use_v2_features,
                signal_type=signal_type,
                direction_filter=direction_filter,
                truncate_early=True,  # Training uses truncation
            )

            # Create model with trial-specific seed (Bug #6)
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                policy_kwargs={"net_arch": net_arch},
                verbose=0,
                device="cpu",
                seed=trial_seed,  # Issue #528 Bug #6: Trial-specific seed
            )

            # Train
            model.learn(total_timesteps=timesteps, progress_bar=False)

            # Evaluate on validation data with all bug fixes
            metrics = evaluate_model(
                model=model,
                val_df=val_df,
                symbol=symbol,
                # Issue #528 Bug Fixes:
                n_episodes=None,  # Bug #1: Calculate from data size
                use_v2_features=use_v2_features,
                signal_type=signal_type,
                direction_filter=direction_filter,
                trial_seed=trial_seed,  # Bug #6: Trial-specific seed
                truncate_early=False,  # Bug #2: Use 100% of validation data
            )

            # Store metrics
            trial.set_user_attr("total_trades", metrics["total_trades"])
            trial.set_user_attr("avg_pnl_pips", metrics["avg_pnl_pips"])
            trial.set_user_attr("trade_rate", metrics["trade_rate"])
            trial.set_user_attr("profit_factor", metrics["profit_factor"])  # Store PF (Fix #2)
            trial.set_user_attr("total_wins_pips", metrics["total_wins_pips"])
            trial.set_user_attr("total_losses_pips", metrics["total_losses_pips"])

            win_rate = metrics["win_rate"]
            profit_factor = metrics["profit_factor"]

            if verbose > 0:
                logger.info(
                    f"Trial {trial.number} complete: "
                    f"PF={profit_factor:.2f}, WR={win_rate:.1f}%, "
                    f"Trades={metrics['total_trades']}, "
                    f"PnL={metrics['avg_pnl_pips']:+.1f} pips"
                )

            env.close()

            # Return negative profit factor (Optuna minimizes, we want to maximize PF) (Fix #2)
            # Penalize unprofitable models (PF < 1.2)
            if profit_factor < 1.2:
                return -10.0  # Very bad score for unprofitable models

            return -profit_factor

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0  # Worst possible

    return objective


# ==============================================================================
# Results Export
# ==============================================================================


def export_best_params(
    study: optuna.Study,
    output_dir: Path,
    symbol: str,
    direction: str = "both",
) -> Path:
    """Export best hyperparameters to JSON."""
    best_trial = study.best_trial
    best_params = best_trial.params.copy()

    # Add network architecture dict
    if "net_arch" in best_params:
        best_params["net_arch_list"] = NET_ARCH_MAPPING[best_params["net_arch"]]

    # Add metrics
    best_params["win_rate"] = float(-best_trial.value) if best_trial.value else 0.0
    best_params["total_trades"] = int(best_trial.user_attrs.get("total_trades", 0))
    best_params["avg_pnl_pips"] = float(best_trial.user_attrs.get("avg_pnl_pips", 0.0))
    best_params["trade_rate"] = float(best_trial.user_attrs.get("trade_rate", 0.0))

    # Metadata
    best_params["symbol"] = symbol
    best_params["direction"] = direction
    best_params["optimized_at"] = datetime.now(timezone.utc).isoformat()
    best_params["trial_number"] = best_trial.number

    # Export - include direction in filename if not "both"
    if direction == "both":
        output_path = output_dir / f"{symbol}_best_params.json"
    else:
        output_path = output_dir / f"{symbol}_{direction}_best_params.json"
    with open(output_path, "w") as f:
        json.dump(best_params, f, indent=2)

    logger.info(f"Best parameters exported to: {output_path}")
    return output_path


def export_trial_history(
    study: optuna.Study,
    output_dir: Path,
    symbol: str,
    direction: str = "both",
) -> Path:
    """Export trial history to CSV."""
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            row = {
                "trial_number": trial.number,
                "win_rate": -trial.value if trial.value else 0.0,
                "total_trades": trial.user_attrs.get("total_trades", 0),
                "avg_pnl_pips": trial.user_attrs.get("avg_pnl_pips", 0.0),
                "trade_rate": trial.user_attrs.get("trade_rate", 0.0),
            }
            row.update(trial.params)
            trials_data.append(row)

    df = pd.DataFrame(trials_data)
    # Include direction in filename if not "both"
    if direction == "both":
        output_path = output_dir / f"{symbol}_trial_history.csv"
    else:
        output_path = output_dir / f"{symbol}_{direction}_trial_history.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"Trial history exported to: {output_path}")
    return output_path


def print_optimization_summary(
    study: optuna.Study,
    start_time: float,
    symbol: str,
    direction: str = "both",
) -> None:
    """Print optimization summary."""
    duration = time.time() - start_time
    best_trial = study.best_trial

    direction_label = f" [{direction.upper()}]" if direction != "both" else ""
    print("\n" + "=" * 70)
    print(f"  Hybrid Model Hyperparameter Optimization Summary - {symbol.upper()}{direction_label}")
    print("=" * 70)
    print(f"Total trials: {len(study.trials)}")
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"Completed trials: {completed}")
    print(f"Pruned trials: {pruned}")
    print(f"Duration: {duration:.1f}s ({duration / 60:.1f} min)")
    print("-" * 70)
    print("Best Trial:")
    print(f"  Trial number: {best_trial.number}")
    print(f"  Win Rate: {-best_trial.value:.1f}%")
    print(f"  Total Trades: {best_trial.user_attrs.get('total_trades', 0)}")
    print(f"  Avg PnL: {best_trial.user_attrs.get('avg_pnl_pips', 0.0):+.1f} pips")
    print(f"  Trade Rate: {best_trial.user_attrs.get('trade_rate', 0.0):.2f}")
    print("-" * 70)
    print("Best Hyperparameters:")
    for key, value in best_trial.params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 70 + "\n")


# ==============================================================================
# CLI
# ==============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for Hybrid exit models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        choices=SUPPORTED_SYMBOLS,
        help="Symbol to optimize",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="Timesteps per trial",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold number to use for train/val split",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing study",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Study name (default: hybrid_{symbol}_optimization)",
    )
    parser.add_argument(
        "--use-v2-features",
        action="store_true",
        default=True,
        help="Use V2 multi-TF features (327 dims)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0=minimal, 1=info, 2=debug)",
    )
    parser.add_argument(
        "--signal-type",
        type=str,
        default="sma",
        choices=["sma", "rsi", "combined", "v3"],
        help="Signal type: sma (D1 SMA), rsi (H1 RSI 30/70), combined (SMA+RSI), v3 (V3 discovered signals Issue #516)",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="both",
        choices=SUPPORTED_DIRECTIONS,
        help="Direction filter: long (LONG signals only), short (SHORT signals only), both (all signals)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    symbol = args.symbol.lower()
    direction = args.direction.lower()

    # Generate study name - include direction if not "both"
    if args.study_name:
        study_name = args.study_name
    elif direction == "both":
        study_name = f"hybrid_{symbol}_optimization"
    else:
        study_name = f"hybrid_{symbol}_{direction}_optimization"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Database path - include direction if not "both"
    if direction == "both":
        db_path = output_dir / f"optuna_{symbol}_study.db"
    else:
        db_path = output_dir / f"optuna_{symbol}_{direction}_study.db"
    storage_url = f"sqlite:///{db_path}"

    direction_label = f" [{direction.upper()}]" if direction != "both" else ""
    logger.info("=" * 70)
    logger.info(f"Hybrid Model Hyperparameter Optimization - {symbol.upper()}{direction_label}")
    logger.info("=" * 70)
    logger.info(f"Study name: {study_name}")
    logger.info(f"Storage: {storage_url}")
    logger.info(f"Trials: {args.n_trials}")
    logger.info(f"Timesteps per trial: {args.timesteps:,}")
    logger.info(f"Fold: {args.fold}")
    logger.info(f"V2 Features: {args.use_v2_features}")
    logger.info(f"Signal Type: {args.signal_type}")
    logger.info(f"Direction Filter: {direction}")
    logger.info("=" * 70)

    # Load data
    try:
        data_df = load_symbol_data(symbol, signal_type=args.signal_type)
    except FileNotFoundError as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    # Load folds config (Fix #5: Use symbol-specific fold config)
    try:
        fold_config_path = get_fold_config_path(symbol)
        logger.info(f"Loading fold config from: {fold_config_path}")

        folds_config = load_folds_config(str(fold_config_path))

        # Symbol-specific files have structure: {"folds": {"fold_001": {...}}}
        # (different from all_symbols_folds.json which had {"symbols": {"eurusd": {...}}})
        folds = folds_config.get("folds", {})
        if not folds:
            logger.error(f"No folds found in {fold_config_path}")
            return 1

        fold_id = f"fold_{args.fold:03d}"
        if fold_id not in folds:
            logger.error(f"Fold {fold_id} not found in {fold_config_path}")
            return 1

        fold_config = folds[fold_id]

        # Extract train/val data
        train_start = fold_config["train"]["start"]
        train_end = fold_config["train"]["end"]
        val_start = fold_config["val"]["start"]
        val_end = fold_config["val"]["end"]

        train_df = data_df.iloc[train_start:train_end].copy().reset_index(drop=True)
        val_df = data_df.iloc[val_start:val_end].copy().reset_index(drop=True)

        logger.info(f"Training data: {len(train_df)} candles")
        logger.info(f"Validation data: {len(val_df)} candles")
        logger.info(f"Expected: ~70% train ({folds_config.get('total_candles', 0) * 0.7:.0f}), ~15% val ({folds_config.get('total_candles', 0) * 0.15:.0f})")

    except Exception as e:
        logger.error(f"Failed to load folds config: {e}")
        return 1

    # Create or load study
    if args.resume:
        logger.info("Resuming from existing study...")
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage_url,
            )
            logger.info(f"Loaded study with {len(study.trials)} existing trials")
        except Exception:
            logger.info("No existing study found, creating new one...")
            args.resume = False

    if not args.resume:
        # Issue #528 Bug #6 Fix: Remove fixed seed=42 to allow trial variance
        # Each trial gets its own seed via get_trial_seed() in objective function
        sampler = TPESampler(n_startup_trials=10)  # No seed - allows variance
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=False,
        )

    # Create objective function
    objective = create_objective(
        symbol=symbol,
        train_df=train_df,
        val_df=val_df,
        timesteps=args.timesteps,
        use_v2_features=args.use_v2_features,
        signal_type=args.signal_type,
        direction_filter=direction,
        verbose=args.verbose,
    )

    # Run optimization
    start_time = time.time()

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            n_jobs=1,  # Sequential for stability
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")

    # Print summary
    print_optimization_summary(study, start_time, symbol, direction)

    # Export results
    export_best_params(study, output_dir, symbol, direction)
    export_trial_history(study, output_dir, symbol, direction)

    logger.info("Optimization complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
