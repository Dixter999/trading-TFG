"""
OHLC Integrity Validator.

Validates OHLC (Open-High-Low-Close) data integrity:
- High >= Low (always)
- Open and Close must be within [Low, High] range
- No negative prices
- No zero prices (invalid for FX trading)

Critical for ensuring data quality and preventing invalid trading scenarios.
"""

import pandas as pd

from .base import BaseValidator, ValidationResult


class OHLCIntegrityValidator(BaseValidator):
    """
    Validates OHLC data integrity.

    Checks:
    - High >= Low
    - Low <= Open <= High
    - Low <= Close <= High
    - All prices > 0 (no negative or zero prices)

    Usage:
        validator = OHLCIntegrityValidator()
        result = validator.validate(dataframe)
    """

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate OHLC data integrity.

        Args:
            data: DataFrame with columns: 'open', 'high', 'low', 'close'

        Returns:
            {
                'passed': bool,
                'violations': List[str],
                'metrics': {
                    'total_rows': int,
                    'integrity_violations': int,
                }
            }
        """
        violations = []
        metrics = {
            "total_rows": len(data),
            "integrity_violations": 0,
        }

        # Check for required columns
        required_cols = ["open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return self.create_failure_result(
                f"Missing required columns: {missing_cols}", metrics
            )

        # Check for empty data
        if len(data) == 0:
            return self.create_success_result(metrics)

        # Check High >= Low
        high_low_violations = data[data["high"] < data["low"]]
        if len(high_low_violations) > 0:
            for idx in high_low_violations.index:
                violations.append(
                    f"Row {idx}: High < Low "
                    f"(high={data.loc[idx, 'high']}, low={data.loc[idx, 'low']})"
                )
                metrics["integrity_violations"] += 1

        # Check Low <= Open <= High
        open_violations = data[
            (data["open"] < data["low"]) | (data["open"] > data["high"])
        ]
        if len(open_violations) > 0:
            for idx in open_violations.index:
                violations.append(
                    f"Row {idx}: Open outside [Low, High] range "
                    f"(open={data.loc[idx, 'open']}, low={data.loc[idx, 'low']}, "
                    f"high={data.loc[idx, 'high']})"
                )
                metrics["integrity_violations"] += 1

        # Check Low <= Close <= High
        close_violations = data[
            (data["close"] < data["low"]) | (data["close"] > data["high"])
        ]
        if len(close_violations) > 0:
            for idx in close_violations.index:
                violations.append(
                    f"Row {idx}: Close outside [Low, High] range "
                    f"(close={data.loc[idx, 'close']}, low={data.loc[idx, 'low']}, "
                    f"high={data.loc[idx, 'high']})"
                )
                metrics["integrity_violations"] += 1

        # Check for negative prices
        for col in required_cols:
            negative_rows = data[data[col] < 0]
            if len(negative_rows) > 0:
                for idx in negative_rows.index:
                    violations.append(
                        f"Row {idx}: Negative price in '{col}' "
                        f"({data.loc[idx, col]})"
                    )
                    metrics["integrity_violations"] += 1

        # Check for zero prices (invalid for FX)
        for col in required_cols:
            zero_rows = data[data[col] == 0]
            if len(zero_rows) > 0:
                for idx in zero_rows.index:
                    violations.append(
                        f"Row {idx}: Zero price in '{col}' " f"(invalid for FX trading)"
                    )
                    metrics["integrity_violations"] += 1

        return self.create_result(violations, metrics)
