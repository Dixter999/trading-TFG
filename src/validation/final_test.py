"""
Final Test Validation Module - Issue #524.

Phase 5 validation script that evaluates the ensemble of 30 trained models
on the TEST data (20%) that has NEVER been seen before. This is the TRUE
measure of model generalization and determines production readiness.

Issue: #524 - Final Test Validation
Epic: hybrid-v4

Usage:
    from validation.final_test import run_final_validation, FinalValidationResult

    # Run final validation
    result = run_final_validation(
        models=trained_models,  # List of 30 trained models
        full_df=df,
        symbol="EURUSD",
        signal_name="long",
        training_results=fold_results,
    )

    if result.passes_production:
        print("Model is ready for production!")
    else:
        print(f"Rejected: {result.rejection_reasons}")

    # Save results for audit
    save_validation_results(result)
"""

import json
import logging
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from data.splitter import get_test_data
from indicator_discovery.tester import TradingCosts

logger = logging.getLogger(__name__)

# =============================================================================
# Pass Criteria Constants
# =============================================================================

# Minimum PF on test data (20% edge after costs)
MIN_TEST_PF = 1.2

# Maximum PF degradation from training to test (30%)
MAX_PF_DEGRADATION = 0.3

# Minimum trades for statistical validity
MIN_TEST_TRADES = 30

# Default output directory
DEFAULT_OUTPUT_DIR = Path("results/final_validation")


# =============================================================================
# FinalValidationResult Dataclass
# =============================================================================

@dataclass
class FinalValidationResult:
    """
    Result of Phase 5 Final Validation on test data.

    This dataclass captures all metrics from evaluating the ensemble of
    30 trained models on TRUE held-out test data. It includes:
    - Ensemble metrics (average across all models)
    - Model variance (min, max, std of individual model PFs)
    - Comparison to training results
    - Production decision with rejection reasons

    Attributes:
        symbol: Trading symbol (e.g., 'EURUSD')
        signal_name: Signal direction ('long' or 'short')
        timestamp: ISO format timestamp of validation
        ensemble_profit_factor: Average PF across all models
        ensemble_win_rate: Average win rate across all models
        ensemble_trades: Average trades across all models
        ensemble_net_pnl_pips: Average net PnL in pips
        model_pf_min: Minimum PF among all models
        model_pf_max: Maximum PF among all models
        model_pf_std: Standard deviation of model PFs
        training_avg_pf: Average PF from 30-fold training
        pf_degradation: training_avg_pf - ensemble_profit_factor
        passes_production: True if all pass criteria met
        rejection_reasons: List of reasons for rejection (empty if passes)
    """

    symbol: str
    signal_name: str
    timestamp: str
    ensemble_profit_factor: float
    ensemble_win_rate: float
    ensemble_trades: int
    ensemble_net_pnl_pips: float
    model_pf_min: float
    model_pf_max: float
    model_pf_std: float
    training_avg_pf: float
    pf_degradation: float
    passes_production: bool
    rejection_reasons: list[str]

    def to_dict(self) -> dict:
        """
        Convert FinalValidationResult to dictionary for serialization.

        Returns:
            Dictionary with all fields
        """
        return {
            "symbol": self.symbol,
            "signal_name": self.signal_name,
            "timestamp": self.timestamp,
            "ensemble_profit_factor": self.ensemble_profit_factor,
            "ensemble_win_rate": self.ensemble_win_rate,
            "ensemble_trades": self.ensemble_trades,
            "ensemble_net_pnl_pips": self.ensemble_net_pnl_pips,
            "model_pf_min": self.model_pf_min,
            "model_pf_max": self.model_pf_max,
            "model_pf_std": self.model_pf_std,
            "training_avg_pf": self.training_avg_pf,
            "pf_degradation": self.pf_degradation,
            "passes_production": self.passes_production,
            "rejection_reasons": self.rejection_reasons,
        }


# =============================================================================
# Model Evaluation Function
# =============================================================================

def evaluate_model_on_test_data(
    model: Any,
    test_df: pd.DataFrame,
    symbol: str,
    signal_name: str,
    costs: TradingCosts = None,
) -> dict[str, float]:
    """
    Evaluate a single trained model on test data.

    Creates a trading environment with the test data and runs the model
    to collect performance metrics including full cost accounting.

    Args:
        model: Trained model (e.g., PPO from stable_baselines3)
        test_df: Test data DataFrame
        symbol: Trading symbol
        signal_name: Signal direction ('long' or 'short')
        costs: Trading costs (defaults to TradingCosts())

    Returns:
        Dictionary with metrics:
        - profit_factor: Gross profit / Gross loss
        - win_rate: Winning trades / Total trades
        - trades: Total number of trades
        - net_pnl_pips: Net PnL after costs
        - gross_profit: Sum of winning trades
        - gross_loss: Sum of losing trades
        - total_costs: Total trading costs
    """
    if costs is None:
        costs = TradingCosts(spread=1.5, slippage=0.5, commission=0.0)

    if len(test_df) == 0:
        logger.warning("Empty test data")
        return {
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "trades": 0,
            "net_pnl_pips": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "total_costs": 0.0,
        }

    # Import here to avoid circular imports
    try:
        from scripts.train_hybrid import create_hybrid_environment
    except ImportError:
        # Fallback for testing
        logger.warning("Could not import create_hybrid_environment, using fallback")
        return {
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "trades": 0,
            "net_pnl_pips": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "total_costs": 0.0,
        }

    # Check for required columns before creating environment
    required_cols = ["time", "open", "high", "low", "close"]
    missing_cols = [c for c in required_cols if c not in test_df.columns]
    if missing_cols:
        # If 'timestamp' exists, rename to 'time'
        if "timestamp" in test_df.columns and "time" not in test_df.columns:
            test_df = test_df.rename(columns={"timestamp": "time"})
        else:
            logger.warning(f"Missing required columns: {missing_cols}")
            return {
                "profit_factor": 0.0,
                "win_rate": 0.0,
                "trades": 0,
                "net_pnl_pips": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "total_costs": 0.0,
            }

    # Create validation environment
    direction_filter = signal_name if signal_name in ["long", "short"] else "both"

    try:
        env = create_hybrid_environment(
            test_df,
            symbol,
            direction_filter=direction_filter,
        )
    except Exception as e:
        logger.warning(f"Could not create environment: {e}")
        return {
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "trades": 0,
            "net_pnl_pips": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "total_costs": 0.0,
        }

    # Evaluate model
    obs, info = env.reset()
    total_trades = 0
    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Track trades from info
        if "trade_pnl" in info:
            pnl = info["trade_pnl"]
            total_trades += 1
            if pnl > 0:
                gross_profit += pnl
                wins += 1
            else:
                gross_loss += abs(pnl)

    # Calculate costs
    total_costs = total_trades * costs.total_per_trade

    # Calculate metrics
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    adjusted_profit = gross_profit - total_costs
    profit_factor = adjusted_profit / gross_loss if gross_loss > 0 else (
        float("inf") if adjusted_profit > 0 else 0.0
    )
    net_pnl = gross_profit - gross_loss - total_costs

    return {
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "trades": total_trades,
        "net_pnl_pips": net_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "total_costs": total_costs,
    }


# =============================================================================
# Main Validation Function
# =============================================================================

def run_final_validation(
    models: list,
    full_df: pd.DataFrame,
    symbol: str,
    signal_name: str,
    training_results: list[dict],
    costs: TradingCosts = None,
) -> FinalValidationResult:
    """
    Run Phase 5 Final Validation on NEVER-SEEN test data.

    This is the moment of truth - if the model passes here,
    it's ready for production.

    Args:
        models: List of trained models from 30-fold training
        full_df: Full dataset (will be split to get test portion)
        symbol: Symbol name (e.g., 'EURUSD')
        signal_name: Signal direction ('long' or 'short')
        training_results: Results from 30-fold training (list of dicts)
        costs: Trading costs to apply (defaults to TradingCosts())

    Returns:
        FinalValidationResult with production decision

    Raises:
        ValueError: If models list is empty
    """
    if not models:
        raise ValueError("Models list must contain at least one model")

    if costs is None:
        costs = TradingCosts(spread=1.5, slippage=0.5, commission=0.0)

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    print(f"\n{'='*60}")
    print("PHASE 5: FINAL VALIDATION ON TEST DATA")
    print(f"Symbol: {symbol} | Signal: {signal_name}")
    print(f"{'='*60}")

    # Get TEST data - FIRST TIME THIS DATA IS EVER ACCESSED
    print("\n!! Accessing TEST data for the first time !!")
    print("!! This data has NEVER been seen during training !!")
    test_df = get_test_data(full_df, symbol, confirm_phase5=True)
    print(f"Test data: {len(test_df)} candles ({len(test_df)/len(full_df)*100:.1f}% of total)")

    # Evaluate each model on test data
    model_results = []
    for i, model in enumerate(models):
        metrics = evaluate_model_on_test_data(
            model=model,
            test_df=test_df,
            symbol=symbol,
            signal_name=signal_name,
            costs=costs,
        )
        model_results.append(metrics)
        print(f"  Model {i+1}/{len(models)}: PF={metrics['profit_factor']:.2f}, "
              f"WR={metrics['win_rate']:.1%}, Trades={metrics['trades']}")

    # Calculate ensemble metrics (average across models)
    pfs = [m["profit_factor"] for m in model_results if m["profit_factor"] != float("inf")]
    wrs = [m["win_rate"] for m in model_results]
    trades = [m["trades"] for m in model_results]
    pnls = [m["net_pnl_pips"] for m in model_results]

    ensemble_pf = sum(pfs) / len(pfs) if pfs else 0.0
    ensemble_wr = sum(wrs) / len(wrs) if wrs else 0.0
    ensemble_trades = int(sum(trades) / len(trades)) if trades else 0
    ensemble_pnl = sum(pnls) / len(pnls) if pnls else 0.0

    # Model variance
    model_pf_min = min(pfs) if pfs else 0.0
    model_pf_max = max(pfs) if pfs else 0.0
    model_pf_std = statistics.stdev(pfs) if len(pfs) > 1 else 0.0

    # Training comparison
    if training_results:
        training_pfs = [r.get("profit_factor", 0.0) for r in training_results]
        training_avg_pf = sum(training_pfs) / len(training_pfs) if training_pfs else 0.0
    else:
        training_avg_pf = 0.0

    pf_degradation = training_avg_pf - ensemble_pf

    # Production decision
    rejection_reasons = []

    if ensemble_pf < MIN_TEST_PF:
        rejection_reasons.append(
            f"Test PF {ensemble_pf:.2f} < {MIN_TEST_PF} minimum"
        )

    if training_avg_pf > 0 and pf_degradation > MAX_PF_DEGRADATION:
        rejection_reasons.append(
            f"PF degradation {pf_degradation:.2f} > {MAX_PF_DEGRADATION} maximum "
            f"(Training: {training_avg_pf:.2f}, Test: {ensemble_pf:.2f})"
        )

    if ensemble_trades < MIN_TEST_TRADES:
        rejection_reasons.append(
            f"Test trades {ensemble_trades} < {MIN_TEST_TRADES} minimum"
        )

    passes_production = len(rejection_reasons) == 0

    result = FinalValidationResult(
        symbol=symbol,
        signal_name=signal_name,
        timestamp=timestamp,
        ensemble_profit_factor=ensemble_pf,
        ensemble_win_rate=ensemble_wr,
        ensemble_trades=ensemble_trades,
        ensemble_net_pnl_pips=ensemble_pnl,
        model_pf_min=model_pf_min,
        model_pf_max=model_pf_max,
        model_pf_std=model_pf_std,
        training_avg_pf=training_avg_pf,
        pf_degradation=pf_degradation,
        passes_production=passes_production,
        rejection_reasons=rejection_reasons,
    )

    # Print decision
    print(f"\n{'='*60}")
    if passes_production:
        print("  PRODUCTION DECISION: APPROVED")
        print(f"  Ensemble PF: {ensemble_pf:.2f} (>= {MIN_TEST_PF})")
        print(f"  Degradation: {pf_degradation:.2f} (<= {MAX_PF_DEGRADATION})")
    else:
        print("  PRODUCTION DECISION: REJECTED")
        for reason in rejection_reasons:
            print(f"    - {reason}")
    print(f"{'='*60}")

    return result


# =============================================================================
# Results Saving
# =============================================================================

def save_validation_results(
    result: FinalValidationResult,
    output_dir: Path = None,
) -> Path:
    """
    Save validation results to audit log.

    Args:
        result: FinalValidationResult to save
        output_dir: Output directory (defaults to results/final_validation/)

    Returns:
        Path to the saved file
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with symbol, signal, and timestamp
    # Replace colons in timestamp for filename safety
    safe_timestamp = result.timestamp.replace(":", "-")
    filename = f"{result.symbol}_{result.signal_name}_{safe_timestamp}.json"
    output_path = output_dir / filename

    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output_path
