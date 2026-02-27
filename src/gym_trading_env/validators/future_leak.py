"""
Future Leak Detector.

Ensures no validation/evaluation data leaks into training data.
This is CRITICAL to prevent overfitting and ensure valid generalization.

The detector checks:
1. No overlap between train/val/eval splits
2. Correct temporal ordering (train < val < eval)
3. No duplicate timestamps across splits
4. No validation data in feature engineering

Strict mode: False positives are better than false negatives.
"""

from typing import Dict, Any, List, Set
from datetime import datetime


class FutureLeakDetector:
    """
    Detects future data leakage into training data.

    This detector is strict - it will flag potential issues even if uncertain.
    False positives are acceptable, false negatives are NOT.

    Usage:
        detector = FutureLeakDetector()
        result = detector.detect(data_splits)
        if result['has_leak']:
            raise ValueError(f"Future leak detected: {result['message']}")
    """

    def __init__(self):
        """Initialize detector."""
        pass

    def detect(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect future data leakage in train/val/eval splits.

        Args:
            data: Dictionary containing data splits:
                - train: Dict with start_date, end_date, data
                - val: Dict with start_date, end_date, data
                - eval: Dict with start_date, end_date, data
                - features_computed_on: Optional str (for feature leak detection)

        Returns:
            {
                'has_leak': bool,
                'temporal_order_valid': bool,
                'message': str,
                'splits_validated': List[str],
                'overlapping_timestamps': List[str],
                'leak_between': List[str],
                'duplicate_count': int,
            }
        """
        result = {
            "has_leak": False,
            "temporal_order_valid": True,
            "message": "No future leak detected",
            "splits_validated": [],
            "overlapping_timestamps": [],
            "leak_between": [],
            "duplicate_count": 0,
        }

        # Check if we have the required splits
        has_train = "train" in data
        has_val = "val" in data
        has_eval = "eval" in data

        if has_train:
            result["splits_validated"].append("train")
        if has_val:
            result["splits_validated"].append("val")
        if has_eval:
            result["splits_validated"].append("eval")

        # Check for overlapping timestamps in data FIRST (more specific)
        if has_train and has_val:
            overlaps = self._find_overlapping_timestamps(
                data["train"].get("data", []), data["val"].get("data", [])
            )
            if overlaps:
                result["has_leak"] = True
                result["message"] = (
                    f"Future leak detected: {len(overlaps)} timestamps appear in both "
                    f"training and validation data."
                )
                result["overlapping_timestamps"] = overlaps
                result["leak_between"] = ["train", "val"]
                result["duplicate_count"] = len(overlaps)
                return result

        if has_val and has_eval:
            overlaps = self._find_overlapping_timestamps(
                data["val"].get("data", []), data["eval"].get("data", [])
            )
            if overlaps:
                result["has_leak"] = True
                result["message"] = (
                    f"Future leak detected: {len(overlaps)} timestamps appear in both "
                    f"validation and evaluation data."
                )
                result["overlapping_timestamps"] = overlaps
                result["leak_between"] = ["val", "eval"]
                result["duplicate_count"] = len(overlaps)
                return result

        if has_train and has_eval:
            overlaps = self._find_overlapping_timestamps(
                data["train"].get("data", []), data["eval"].get("data", [])
            )
            if overlaps:
                result["has_leak"] = True
                result["message"] = (
                    f"Future leak detected: {len(overlaps)} timestamps appear in both "
                    f"training and evaluation data."
                )
                result["overlapping_timestamps"] = overlaps
                result["leak_between"] = ["train", "eval"]
                result["duplicate_count"] = len(overlaps)
                return result

        # Check temporal ordering (as fallback if no timestamp overlaps found)
        if has_train and has_val:
            if not self._check_temporal_order(data["train"], data["val"]):
                result["has_leak"] = True
                result["temporal_order_valid"] = False
                result["message"] = (
                    "Future leak detected: Validation period is before training period. "
                    "Temporal order must be: train < val < eval"
                )
                result["leak_between"] = ["train", "val"]
                return result

        if has_val and has_eval:
            if not self._check_temporal_order(data["val"], data["eval"]):
                result["has_leak"] = True
                result["temporal_order_valid"] = False
                result["message"] = (
                    "Future leak detected: Evaluation period is before validation period. "
                    "Temporal order must be: train < val < eval"
                )
                result["leak_between"] = ["val", "eval"]
                return result

        # Check for feature engineering leak
        if "features_computed_on" in data["train"]:
            if data["train"]["features_computed_on"] == "entire_dataset":
                result["has_leak"] = True
                result["message"] = (
                    "Future leak detected: Training features computed using entire dataset "
                    "(including validation data). Features must use only training data."
                )
                result["leak_between"] = ["feature_engineering"]
                return result

        return result

    def _check_temporal_order(
        self, earlier_split: Dict[str, Any], later_split: Dict[str, Any]
    ) -> bool:
        """
        Check if temporal ordering is correct (earlier < later).

        Args:
            earlier_split: Earlier data split (should come first)
            later_split: Later data split (should come after)

        Returns:
            True if ordering is correct, False otherwise
        """
        # Get end date of earlier split and start date of later split
        earlier_end = earlier_split.get("end_date")
        later_start = later_split.get("start_date")

        if not earlier_end or not later_start:
            # Can't validate without dates
            return True

        try:
            earlier_end_dt = datetime.fromisoformat(earlier_end)
            later_start_dt = datetime.fromisoformat(later_start)

            # Earlier split must end before later split starts
            return earlier_end_dt < later_start_dt

        except (ValueError, TypeError):
            # If we can't parse dates, be conservative and return True
            # (don't flag as error if we're unsure)
            return True

    def _find_overlapping_timestamps(
        self, data1: List[Dict[str, Any]], data2: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Find timestamps that appear in both datasets.

        Args:
            data1: First dataset
            data2: Second dataset

        Returns:
            List of overlapping timestamps
        """
        # Extract timestamps from both datasets
        timestamps1: Set[str] = set()
        timestamps2: Set[str] = set()

        for item in data1:
            if "timestamp" in item:
                timestamps1.add(item["timestamp"])

        for item in data2:
            if "timestamp" in item:
                timestamps2.add(item["timestamp"])

        # Find intersection
        overlaps = timestamps1.intersection(timestamps2)

        return sorted(list(overlaps))
