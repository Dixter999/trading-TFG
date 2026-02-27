"""
MLflow Utilities for Experiment Tracking and Model Versioning.

This module provides utilities for:
- Logging backtest results to MLflow
- Versioning models with metadata
- Comparing experiments across runs
- Calculating metric improvements

TDD Phase: GREEN - Minimal implementation to pass tests.
"""

import math
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import pandas as pd


def log_backtest_results(
    backtest_results: Dict[str, float],
    baseline: Optional[Dict[str, float]] = None,
) -> None:
    """
    Log backtest results to MLflow.

    This function logs backtest metrics to the active MLflow run.
    If a baseline is provided, it also logs comparison metrics.

    Args:
        backtest_results: Dictionary of backtest metrics
            Expected keys:
            - total_return_pct: Total return percentage
            - sharpe_ratio: Sharpe ratio
            - max_drawdown_pct: Maximum drawdown percentage
            - win_rate: Win rate (0.0 to 1.0)
            - total_trades: Total number of trades
            Optional keys:
            - avg_trade_duration: Average trade duration in hours
            - profit_factor: Profit factor
            - max_consecutive_losses: Maximum consecutive losses
        baseline: Optional baseline results for comparison

    Example:
        >>> backtest_results = {
        ...     "total_return_pct": 15.5,
        ...     "sharpe_ratio": 1.8,
        ...     "max_drawdown_pct": 12.3,
        ...     "win_rate": 0.45,
        ...     "total_trades": 150,
        ... }
        >>> log_backtest_results(backtest_results)
    """
    # Format metrics for MLflow logging
    metrics = format_metrics_for_mlflow(backtest_results, prefix="backtest")

    # If baseline provided, calculate and log comparison metrics
    if baseline is not None:
        # Calculate improvements
        if "total_return_pct" in baseline:
            return_improvement = calculate_improvement(
                backtest_results.get("total_return_pct", 0.0),
                baseline.get("total_return_pct", 0.0),
            )
            metrics["vs_baseline_return"] = return_improvement

        if "sharpe_ratio" in baseline:
            sharpe_improvement = calculate_improvement(
                backtest_results.get("sharpe_ratio", 0.0),
                baseline.get("sharpe_ratio", 0.0),
            )
            metrics["vs_baseline_sharpe"] = sharpe_improvement

    # Log all metrics to MLflow
    mlflow.log_metrics(metrics)


def version_model(
    model_path: str,
    model_name: str,
    metadata: Dict[str, Any],
    version: Optional[str] = None,
) -> None:
    """
    Version a trained model in MLflow with metadata.

    This function logs a model artifact to MLflow and tags it with metadata.

    Args:
        model_path: Path to the saved model file
        model_name: Name for the model
        metadata: Dictionary of metadata to tag the model with
            Common keys:
            - algorithm: Algorithm name (e.g., "PPO", "DQN")
            - reward_type: Reward function used
            - total_timesteps: Training timesteps
            - backtest_return: Backtest return percentage
            - backtest_sharpe: Backtest Sharpe ratio
        version: Optional version string

    Raises:
        FileNotFoundError: If model_path does not exist

    Example:
        >>> metadata = {
        ...     "algorithm": "PPO",
        ...     "reward_type": "pnl",
        ...     "total_timesteps": 100000,
        ...     "backtest_return": 10.5,
        ... }
        >>> version_model("/path/to/model.zip", "eurusd_ppo", metadata, "1.0.0")
    """
    # Verify model file exists
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Log model artifact to MLflow
    mlflow.log_artifact(model_path, artifact_path="model")

    # Tag model with metadata
    for key, value in metadata.items():
        mlflow.set_tag(key, value)

    # Tag with version if provided
    if version is not None:
        mlflow.set_tag("model_version", version)


def compare_experiments(
    experiment_name: str,
    sort_by: str = "metrics.backtest_return",
    ascending: bool = False,
    export_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare experiment runs and generate comparison report.

    This function loads all runs from a specified experiment,
    compares their metrics, and optionally exports to CSV.

    Args:
        experiment_name: Name of the MLflow experiment
        sort_by: Column to sort results by
        ascending: Sort order (False = descending)
        export_csv: Optional path to export comparison CSV

    Returns:
        DataFrame with experiment run comparisons

    Example:
        >>> df = compare_experiments(
        ...     experiment_name="eurusd_ppo_training",
        ...     sort_by="metrics.backtest_return",
        ...     ascending=False,
        ... )
        >>> print(df.head())
    """
    # Search for runs in the experiment
    runs_df = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"{sort_by} {'ASC' if ascending else 'DESC'}"],
    )

    # Sort results
    if not runs_df.empty and sort_by in runs_df.columns:
        runs_df = runs_df.sort_values(by=sort_by, ascending=ascending)

    # Export to CSV if requested
    if export_csv is not None and not runs_df.empty:
        runs_df.to_csv(export_csv, index=False)

    return runs_df


def calculate_improvement(current: float, baseline: float) -> float:
    """
    Calculate improvement vs baseline.

    This is a simple absolute difference calculation:
    improvement = current - baseline

    Positive values indicate improvement, negative indicate regression.

    Args:
        current: Current metric value
        baseline: Baseline metric value

    Returns:
        Absolute improvement (current - baseline)

    Example:
        >>> calculate_improvement(10.0, 8.0)
        2.0
        >>> calculate_improvement(5.0, 8.0)
        -3.0
    """
    return current - baseline


def format_metrics_for_mlflow(
    raw_metrics: Dict[str, float],
    prefix: str = "backtest",
) -> Dict[str, float]:
    """
    Format metrics for MLflow logging.

    This function:
    1. Filters out invalid metrics (NaN, Inf)
    2. Maps metric names to standardized keys
    3. Adds prefix to metric names

    Args:
        raw_metrics: Raw metrics dictionary
        prefix: Prefix to add to metric names

    Returns:
        Formatted metrics dictionary ready for MLflow

    Example:
        >>> raw = {"total_return_pct": 10.0, "sharpe_ratio": 1.5}
        >>> formatted = format_metrics_for_mlflow(raw, prefix="backtest")
        >>> print(formatted)
        {'backtest_return': 10.0, 'backtest_sharpe': 1.5, ...}
    """
    formatted = {}

    # Define metric name mappings
    metric_mappings = {
        "total_return_pct": "return",
        "sharpe_ratio": "sharpe",
        "max_drawdown_pct": "max_dd",
        "win_rate": "win_rate",
        "total_trades": "trades",
        "avg_trade_duration": "avg_duration",
        "profit_factor": "profit_factor",
        "max_consecutive_losses": "max_consecutive_losses",
    }

    for raw_key, value in raw_metrics.items():
        # Skip invalid values
        if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
            continue

        # Map to standardized key
        mapped_key = metric_mappings.get(raw_key, raw_key)

        # Add prefix
        final_key = f"{prefix}_{mapped_key}"

        formatted[final_key] = float(value)

    return formatted
