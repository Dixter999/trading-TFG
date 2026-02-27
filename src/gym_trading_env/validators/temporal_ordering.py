"""
Temporal Ordering Validator.

Ensures timestamps are strictly monotonically increasing (no duplicates, no backwards).
This is critical to prevent temporal inconsistencies and lookahead bias.
"""

import pandas as pd

from .base import BaseValidator, ValidationResult


class TemporalOrderingValidator(BaseValidator):
    """
    Validates that timestamps are strictly monotonically increasing.

    Checks:
    - No duplicate timestamps
    - No backwards (decreasing) timestamps
    - Strictly ascending temporal order

    Usage:
        validator = TemporalOrderingValidator()
        result = validator.validate(dataframe)
        if not result['passed']:
            print(result['violations'])
    """

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate temporal ordering of timestamp column.

        Args:
            data: DataFrame with 'timestamp' column

        Returns:
            {
                'passed': bool,
                'violations': List[str],
                'metrics': {
                    'total_rows': int,
                    'ordering_violations': int,
                }
            }
        """
        violations = []
        metrics = {
            "total_rows": len(data),
            "ordering_violations": 0,
        }

        # Handle empty dataframe
        if len(data) == 0:
            return self.create_success_result(metrics)

        # Check if timestamp column exists
        if "timestamp" not in data.columns:
            metrics["ordering_violations"] = 1
            return self.create_failure_result(
                "Missing 'timestamp' column in data", metrics
            )

        timestamps = data["timestamp"].values

        # Check for strictly monotonic increasing
        for i in range(1, len(timestamps)):
            prev_ts = timestamps[i - 1]
            curr_ts = timestamps[i]

            # Check for duplicate (equal) timestamps
            if prev_ts == curr_ts:
                violations.append(
                    f"Duplicate timestamp at index {i}: {curr_ts} "
                    f"(same as previous timestamp)"
                )
                metrics["ordering_violations"] += 1

            # Check for backwards (decreasing) timestamps
            elif prev_ts > curr_ts:
                violations.append(
                    f"Backwards timestamp at index {i}: {curr_ts} < {prev_ts} "
                    f"(timestamp decreased)"
                )
                metrics["ordering_violations"] += 1

        return self.create_result(violations, metrics)
