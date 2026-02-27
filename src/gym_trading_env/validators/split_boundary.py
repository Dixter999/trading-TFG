"""
Split Boundary Validator.

Ensures train/validation/test splits have no temporal overlap and are
in correct chronological order. Critical for preventing data leakage
and lookahead bias.
"""

from typing import Dict
from datetime import datetime

from .base import BaseValidator, ValidationResult


class SplitBoundaryValidator(BaseValidator):
    """
    Validates that data splits (train/val/test) have no overlap.

    Checks:
    - No overlap between train and validation
    - No overlap between validation and test
    - No overlap between train and test
    - Splits are in chronological order (train -> val -> test)

    Usage:
        validator = SplitBoundaryValidator()
        splits = {
            'train': {'start': ..., 'end': ...},
            'validation': {'start': ..., 'end': ...},
            'test': {'start': ..., 'end': ...},
        }
        result = validator.validate(splits)
    """

    def validate(self, splits: Dict[str, Dict[str, datetime]]) -> ValidationResult:
        """
        Validate split boundaries have no overlap and correct order.

        Args:
            splits: Dictionary with keys 'train', 'validation', 'test',
                   each containing 'start' and 'end' datetime objects

        Returns:
            {
                'passed': bool,
                'violations': List[str],
                'metrics': {
                    'overlaps_detected': int,
                }
            }
        """
        violations = []
        metrics = {
            "overlaps_detected": 0,
        }

        # Validate required keys exist
        required_keys = ["train", "validation", "test"]
        for key in required_keys:
            if key not in splits:
                return self.create_failure_result(
                    f"Missing required split: '{key}'", metrics
                )

            if "start" not in splits[key] or "end" not in splits[key]:
                return self.create_failure_result(
                    f"Split '{key}' missing 'start' or 'end' timestamp", metrics
                )

        # Extract boundaries
        train_start = splits["train"]["start"]
        train_end = splits["train"]["end"]
        val_start = splits["validation"]["start"]
        val_end = splits["validation"]["end"]
        test_start = splits["test"]["start"]
        test_end = splits["test"]["end"]

        # Check train-validation overlap
        if train_end >= val_start:
            violations.append(
                f"Train/Validation overlap detected: "
                f"train ends at {train_end}, validation starts at {val_start}"
            )
            metrics["overlaps_detected"] += 1

        # Check validation-test overlap
        if val_end >= test_start:
            violations.append(
                f"Validation/Test overlap detected: "
                f"validation ends at {val_end}, test starts at {test_start}"
            )
            metrics["overlaps_detected"] += 1

        # Check train-test overlap (should be caught by above, but check anyway)
        if train_end >= test_start:
            violations.append(
                f"Train/Test overlap detected: "
                f"train ends at {train_end}, test starts at {test_start}"
            )
            metrics["overlaps_detected"] += 1

        # Check chronological order: train -> validation -> test
        if not (train_start < train_end < val_start < val_end < test_start < test_end):
            # More detailed check to identify the specific issue
            if train_start >= train_end:
                violations.append(
                    f"Train split: start ({train_start}) >= end ({train_end})"
                )
            if val_start >= val_end:
                violations.append(
                    f"Validation split: start ({val_start}) >= end ({val_end})"
                )
            if test_start >= test_end:
                violations.append(
                    f"Test split: start ({test_start}) >= end ({test_end})"
                )

            # Check correct chronological order of splits
            if not (train_end < val_start < test_start):
                violations.append(
                    f"Splits not in chronological order. Expected: "
                    f"train_end < val_start < test_start, but got "
                    f"{train_end} < {val_start} < {test_start}"
                )

        return self.create_result(violations, metrics)
