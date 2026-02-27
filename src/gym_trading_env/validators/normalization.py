"""
Normalization Validator.

Validates normalized data for:
- NaN (Not a Number) values
- Inf (Infinity) values
- Extreme outliers (beyond threshold standard deviations)

Critical for ensuring numerical stability and preventing
training instability due to invalid or extreme values.
"""

import pandas as pd
import numpy as np

from .base import BaseValidator, ValidationResult


class NormalizationValidator(BaseValidator):
    """
    Validates normalized data for NaN, Inf, and extreme values.

    Checks:
    - No NaN values
    - No infinite values (positive or negative)
    - No extreme outliers beyond threshold_std standard deviations

    Usage:
        validator = NormalizationValidator(threshold_std=3.0)
        result = validator.validate(dataframe)
    """

    def __init__(self, threshold_std: float = 5.0):
        """
        Initialize validator.

        Args:
            threshold_std: Number of standard deviations beyond which
                          values are considered extreme outliers.
                          Default is 5.0 (very lenient).
        """
        self.threshold_std = threshold_std

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate data for NaN, Inf, and extreme values.

        Args:
            data: DataFrame with normalized features

        Returns:
            {
                'passed': bool,
                'violations': List[str],
                'metrics': {
                    'total_values': int,
                    'nan_count': int,
                    'inf_count': int,
                    'extreme_value_count': int,
                }
            }
        """
        violations = []
        metrics = {
            "total_values": data.size,
            "nan_count": 0,
            "inf_count": 0,
            "extreme_value_count": 0,
        }

        # Check for NaN values
        nan_mask = data.isna()
        nan_count = nan_mask.sum().sum()
        if nan_count > 0:
            metrics["nan_count"] = nan_count
            violations.append(
                f"Found {nan_count} NaN values across dataset. "
                f"Columns with NaN: {list(data.columns[nan_mask.any()])}"
            )

        # Check for infinite values
        inf_mask = np.isinf(data.select_dtypes(include=[np.number]))
        inf_count = inf_mask.sum().sum()
        if inf_count > 0:
            metrics["inf_count"] = inf_count
            violations.append(
                f"Found {inf_count} infinite values across dataset. "
                f"Columns with Inf: {list(data.columns[inf_mask.any()])}"
            )

        # Check for extreme outliers (only on numeric columns)
        numeric_data = data.select_dtypes(include=[np.number])

        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()  # Remove NaN for stats

            # Skip if column is constant (std = 0)
            if len(col_data) == 0 or col_data.std() == 0:
                continue

            # Calculate z-scores
            mean = col_data.mean()
            std = col_data.std()

            # Find extreme values
            z_scores = np.abs((col_data - mean) / std)
            extreme_mask = z_scores > self.threshold_std

            if extreme_mask.any():
                extreme_count = extreme_mask.sum()
                metrics["extreme_value_count"] += extreme_count

                extreme_values = col_data[extreme_mask]
                violations.append(
                    f"Column '{col}': Found {extreme_count} extreme outliers "
                    f"(beyond {self.threshold_std} std). "
                    f"Examples: {extreme_values.head(3).tolist()}"
                )

        return self.create_result(violations, metrics)
