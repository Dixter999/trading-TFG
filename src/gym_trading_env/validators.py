"""
Validators for time-series data in gym_trading_env.

Provides validation functions to ensure temporal correctness and data integrity:
- Temporal ordering validation
- Gap detection in time series
- Split boundary validation

TDD Phase: GREEN - Minimal implementation to support DataSplitter.
"""

import logging
from datetime import timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def validate_temporal_order(df: pd.DataFrame, timestamp_col: str = "timestamp") -> None:
    """
    Validate that DataFrame is sorted in ascending chronological order.

    Args:
        df: DataFrame to validate
        timestamp_col: Name of timestamp column (default: 'timestamp')

    Raises:
        ValueError: If timestamps are not in ascending order
        KeyError: If timestamp_col doesn't exist
    """
    if timestamp_col not in df.columns:
        raise KeyError(
            f"Timestamp column '{timestamp_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    if len(df) < 2:
        # Single record or empty DataFrame is considered valid
        return

    timestamps = df[timestamp_col]

    # Check if timestamps are sorted
    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            f"Timestamps are not in ascending chronological order. "
            f"First timestamp: {timestamps.iloc[0]}, "
            f"Last timestamp: {timestamps.iloc[-1]}"
        )

    logger.debug(f"Temporal order validated: {len(df)} records in ascending order")


def validate_no_gaps(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    expected_interval: timedelta = timedelta(hours=1),
    tolerance: timedelta = timedelta(seconds=60),
) -> None:
    """
    Validate that there are no gaps in the time series.

    Args:
        df: DataFrame to validate
        timestamp_col: Name of timestamp column (default: 'timestamp')
        expected_interval: Expected time between consecutive records
        tolerance: Allowed deviation from expected_interval (default: 60 seconds)

    Raises:
        ValueError: If gaps are detected in the time series
        KeyError: If timestamp_col doesn't exist
    """
    if timestamp_col not in df.columns:
        raise KeyError(
            f"Timestamp column '{timestamp_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    if len(df) < 2:
        # Need at least 2 records to check for gaps
        return

    timestamps = df[timestamp_col]

    # Calculate differences between consecutive timestamps
    time_diffs = timestamps.diff()[1:]  # Skip first NaT

    # Check for gaps larger than expected_interval + tolerance
    max_allowed = expected_interval + tolerance
    gaps = time_diffs[time_diffs > max_allowed]

    if len(gaps) > 0:
        gap_indices = gaps.index.tolist()
        first_gap_idx = gap_indices[0]
        gap_start = timestamps.iloc[first_gap_idx - 1]
        gap_end = timestamps.iloc[first_gap_idx]
        gap_size = time_diffs.iloc[first_gap_idx - 1]

        raise ValueError(
            f"Detected {len(gaps)} gaps in time series. "
            f"First gap: {gap_start} -> {gap_end} "
            f"(size: {gap_size}, expected: {expected_interval})"
        )

    logger.debug(
        f"No gaps detected: {len(df)} records with interval {expected_interval}"
    )


def validate_split_boundaries(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> None:
    """
    Validate that train/val/eval splits have proper temporal boundaries.

    Ensures:
    - All splits are non-empty
    - No temporal overlap between splits
    - Chronological ordering: train < val < eval

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        eval_df: Evaluation DataFrame
        timestamp_col: Name of timestamp column (default: 'timestamp')

    Raises:
        ValueError: If split boundaries are invalid
        KeyError: If timestamp_col doesn't exist in any DataFrame
    """
    # Check all dataframes have the timestamp column
    for name, df in [("train", train_df), ("val", val_df), ("eval", eval_df)]:
        if timestamp_col not in df.columns:
            raise KeyError(
                f"Timestamp column '{timestamp_col}' not found in {name} DataFrame"
            )

    # Check all splits are non-empty
    if len(train_df) == 0:
        raise ValueError("Training split is empty")
    if len(val_df) == 0:
        raise ValueError("Validation split is empty")
    if len(eval_df) == 0:
        raise ValueError("Evaluation split is empty")

    # Get boundary timestamps
    train_end = train_df[timestamp_col].max()
    val_start = val_df[timestamp_col].min()
    val_end = val_df[timestamp_col].max()
    eval_start = eval_df[timestamp_col].min()

    # Validate no temporal overlap
    if train_end >= val_start:
        raise ValueError(
            f"Train/Val overlap detected: train ends at {train_end}, "
            f"val starts at {val_start}"
        )

    if val_end >= eval_start:
        raise ValueError(
            f"Val/Eval overlap detected: val ends at {val_end}, "
            f"eval starts at {eval_start}"
        )

    logger.debug(
        f"Split boundaries validated: "
        f"train={len(train_df)}, val={len(val_df)}, eval={len(eval_df)}"
    )
