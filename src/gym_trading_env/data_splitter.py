"""
DataSplitter for time-series trading data.

Implements 50/25/25 split strategy (train/validation/evaluation) with temporal
validation to prevent lookahead bias and data leakage in machine learning models.

This module is critical for maintaining temporal integrity in time-series financial
ML applications. All splits preserve chronological order and are strictly separated.
"""

import logging
from datetime import timedelta

import pandas as pd

from .validators import (
    validate_no_gaps,
    validate_split_boundaries,
    validate_temporal_order,
)

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Splits time-series data into train/validation/evaluation sets.

    Split ratio: 50% train, 25% validation, 25% evaluation

    The split maintains temporal ordering to prevent lookahead bias:
        - Training data: Earliest 50% of records
        - Validation data: Middle 25% of records
        - Evaluation data: Latest 25% of records

    All splits are strictly separated with no temporal overlap to ensure
    that future data never leaks into training or validation.

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from gym_trading_env.data_splitter import DataSplitter
        >>>
        >>> # Create sample data
        >>> df = pd.DataFrame({
        ...     'timestamp': pd.date_range('2025-01-01', periods=100, freq='h'),
        ...     'close': range(100),
        ...     'volume': range(100, 200)
        ... })
        >>>
        >>> # Split data
        >>> splitter = DataSplitter()
        >>> train, val, eval = splitter.split(df, validate=True)
        >>>
        >>> print(f"Train: {len(train)} records")
        >>> print(f"Validation: {len(val)} records")
        >>> print(f"Evaluation: {len(eval)} records")
    """

    def split(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        validate: bool = True,
        expected_interval: timedelta | None = None,
        log_mlflow: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/eval sets using 50/25/25 ratio.

        The split is performed using integer indices based on the total record count:
            - train_end = int(len(df) * 0.50)
            - val_end = train_end + int(len(df) * 0.25)
            - eval = remaining records

        Args:
            df: Input DataFrame (must be sorted by timestamp if validate=True)
            timestamp_col: Name of timestamp column (default: 'timestamp')
            validate: If True, run temporal validation checks (default: True)
            expected_interval: Expected time between records for gap detection.
                              If None, gap validation is skipped. (default: None)
            log_mlflow: If True, log split metadata to MLflow (default: False)

        Returns:
            Tuple of (train_df, val_df, eval_df)

        Raises:
            ValueError: If validation fails or insufficient data (< 12 records when validate=True)
            KeyError: If timestamp_col doesn't exist in DataFrame

        Example:
            >>> df = pd.DataFrame({
            ...     'timestamp': pd.date_range('2025-01-01', periods=20, freq='h'),
            ...     'close': range(20)
            ... })
            >>> splitter = DataSplitter()
            >>> train, val, eval = splitter.split(df, validate=False)
            >>> len(train), len(val), len(eval)
            (10, 5, 5)
        """
        self._validate_minimum_records(df, validate)

        if validate:
            self._run_temporal_validations(df, timestamp_col, expected_interval)

        train_df, val_df, eval_df = self._perform_split(df)

        if validate:
            validate_split_boundaries(train_df, val_df, eval_df, timestamp_col)

        self._log_split_boundaries(train_df, val_df, eval_df, timestamp_col)

        if log_mlflow:
            self._log_to_mlflow(df, train_df, val_df, eval_df, timestamp_col, validate)

        return train_df, val_df, eval_df

    def _validate_minimum_records(self, df: pd.DataFrame, validate: bool) -> None:
        """
        Validate that the DataFrame has sufficient records for splitting.

        Args:
            df: Input DataFrame
            validate: If True, require minimum 12 records (4 per split)

        Raises:
            ValueError: If insufficient data
        """
        if len(df) == 0:
            raise ValueError(
                "Insufficient data: 0 records "
                "(minimum 12 required for 3-way split with validation)"
            )

        if validate and len(df) < 12:
            raise ValueError(
                f"Insufficient data: {len(df)} records "
                f"(minimum 12 required for 3-way split with validation)"
            )

    def _run_temporal_validations(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        expected_interval: timedelta | None,
    ) -> None:
        """
        Run temporal validation checks on the data.

        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            expected_interval: Expected time between records (None to skip gap check)

        Raises:
            ValueError: If temporal validation fails
        """
        validate_temporal_order(df, timestamp_col)

        if expected_interval is not None:
            validate_no_gaps(df, timestamp_col, expected_interval)

    def _perform_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform the actual data split using 50/25/25 ratio.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (train_df, val_df, eval_df)
        """
        total_records = len(df)
        train_end = int(total_records * 0.50)  # 50%
        val_end = train_end + int(total_records * 0.25)  # +25%

        train_df = df.iloc[0:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        eval_df = df.iloc[val_end:].copy()

        return train_df, val_df, eval_df

    def _log_split_boundaries(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        timestamp_col: str,
    ) -> None:
        """
        Log split boundaries with timestamps and record counts.

        Args:
            train_df: Training data
            val_df: Validation data
            eval_df: Evaluation data
            timestamp_col: Name of timestamp column
        """
        logger.info(
            f"Train: {train_df[timestamp_col].min()} to {train_df[timestamp_col].max()} "
            f"({len(train_df)} records)"
        )
        logger.info(
            f"Val: {val_df[timestamp_col].min()} to {val_df[timestamp_col].max()} "
            f"({len(val_df)} records)"
        )
        logger.info(
            f"Eval: {eval_df[timestamp_col].min()} to {eval_df[timestamp_col].max()} "
            f"({len(eval_df)} records)"
        )

    def _log_to_mlflow(
        self,
        df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        timestamp_col: str,
        validate: bool,
    ) -> None:
        """
        Log split metadata to MLflow for experiment tracking.

        Args:
            df: Original DataFrame
            train_df: Training data
            val_df: Validation data
            eval_df: Evaluation data
            timestamp_col: Name of timestamp column
            validate: Whether validation was enabled
        """
        try:
            import mlflow

            # Log split strategy metadata
            mlflow.log_params(
                {
                    "split_strategy": "50/25/25",
                    "total_records": len(df),
                    "validation_enabled": validate,
                    "timestamp_col": timestamp_col,
                }
            )

            # Log train split info
            mlflow.log_params(
                {
                    "train_start": str(train_df[timestamp_col].min()),
                    "train_end": str(train_df[timestamp_col].max()),
                    "train_records": len(train_df),
                }
            )

            # Log validation split info
            mlflow.log_params(
                {
                    "val_start": str(val_df[timestamp_col].min()),
                    "val_end": str(val_df[timestamp_col].max()),
                    "val_records": len(val_df),
                }
            )

            # Log evaluation split info
            mlflow.log_params(
                {
                    "eval_start": str(eval_df[timestamp_col].min()),
                    "eval_end": str(eval_df[timestamp_col].max()),
                    "eval_records": len(eval_df),
                }
            )

            logger.info("Split metadata logged to MLflow")

        except ImportError:
            logger.warning("MLflow not available, skipping experiment tracking")
