"""
ValidationSuite - Orchestrates all validators.

Runs all 7 validators together:
- 4 basic validators (temporal, split, OHLC, normalization)
- 3 advanced detectors (lookahead, future leak, suspicious patterns)

Provides unified validation interface and summary reporting.
"""

from typing import Any, Dict, List

import pandas as pd

from .temporal_ordering import TemporalOrderingValidator
from .split_boundary import SplitBoundaryValidator
from .ohlc_integrity import OHLCIntegrityValidator
from .normalization import NormalizationValidator
from .lookahead_bias import LookaheadBiasDetector
from .future_leak import FutureLeakDetector
from .suspicious_patterns import SuspiciousPatternDetector
from .base import ValidationResult


class ValidationSuite:
    """
    Orchestrates all validation checks for trading environment data.

    Runs all 7 validators:
    1. TemporalOrderingValidator - monotonic timestamps
    2. SplitBoundaryValidator - train/val/test boundaries
    3. OHLCIntegrityValidator - OHLC data integrity
    4. NormalizationValidator - NaN, Inf, extreme values
    5. LookaheadBiasDetector - normalization using future data
    6. FutureLeakDetector - validation data in training
    7. SuspiciousPatternDetector - unrealistic patterns

    Usage:
        suite = ValidationSuite()
        results = suite.run_all_checks(env=env, candle_data=data)
        if results['summary']['pass_rate'] < 1.0:
            print("Validation failed!")
            for name, result in results.items():
                if name != 'summary' and not result['passed']:
                    print(f"{name}: {result['violations']}")
    """

    def __init__(self):
        """Initialize all validators."""
        self.temporal_validator = TemporalOrderingValidator()
        self.split_validator = SplitBoundaryValidator()
        self.ohlc_validator = OHLCIntegrityValidator()
        self.normalization_validator = NormalizationValidator()
        self.lookahead_detector = LookaheadBiasDetector()
        self.future_leak_detector = FutureLeakDetector()
        self.suspicious_patterns_detector = SuspiciousPatternDetector()

    def run_all_checks(
        self,
        env: Any,
        candle_data: List[Dict[str, Any]],
        train_split: float = 0.7,
        val_split: float = 0.15,
        log_to_mlflow: bool = False,
    ) -> Dict[str, Any]:
        """
        Run all validation checks on environment and data.

        Args:
            env: Trading environment (EURUSDTradingEnv or similar)
            candle_data: List of candle dictionaries with OHLC data
            train_split: Training data ratio (default 0.7)
            val_split: Validation data ratio (default 0.15)
            log_to_mlflow: If True, log validation results to MLflow (default False)

        Returns:
            Dictionary with validation results:
            {
                'temporal_ordering': ValidationResult,
                'split_boundary': ValidationResult,
                'ohlc_integrity': ValidationResult,
                'normalization': ValidationResult,
                'lookahead_bias': ValidationResult,
                'future_leak': ValidationResult,
                'suspicious_patterns': ValidationResult,
                'summary': {
                    'total_validators': int,
                    'passed_count': int,
                    'failed_count': int,
                    'pass_rate': float,
                }
            }
        """
        # Convert candle_data to DataFrame for validators
        df = pd.DataFrame(candle_data)

        # Ensure timestamp column is datetime type
        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(
            df["timestamp"]
        ):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Create data splits for split boundary and future leak validators
        total_rows = len(df)
        train_end_idx = int(total_rows * train_split)
        val_end_idx = train_end_idx + int(total_rows * val_split)

        train_df = df.iloc[:train_end_idx]
        val_df = df.iloc[train_end_idx:val_end_idx]
        test_df = df.iloc[val_end_idx:]

        # Create split boundaries for SplitBoundaryValidator
        split_boundaries = {
            "train": {
                "start": train_df["timestamp"].iloc[0] if len(train_df) > 0 else None,
                "end": train_df["timestamp"].iloc[-1] if len(train_df) > 0 else None,
            },
            "validation": {
                "start": val_df["timestamp"].iloc[0] if len(val_df) > 0 else None,
                "end": val_df["timestamp"].iloc[-1] if len(val_df) > 0 else None,
            },
            "test": {
                "start": test_df["timestamp"].iloc[0] if len(test_df) > 0 else None,
                "end": test_df["timestamp"].iloc[-1] if len(test_df) > 0 else None,
            },
        }

        # Create data splits for FutureLeakDetector
        data_splits = {
            "train": train_df,
            "val": val_df,
            "test": test_df,
            "full": df,
        }

        # Run all validators
        results: Dict[str, ValidationResult] = {}

        # Basic validators
        results["temporal_ordering"] = self.temporal_validator.validate(df)
        results["split_boundary"] = self.split_validator.validate(split_boundaries)
        results["ohlc_integrity"] = self.ohlc_validator.validate(df)

        # Build observations from environment for normalization check
        # Get observations from current state
        observations_list = []
        if hasattr(env, "_get_observation"):
            # For environments that support observation retrieval
            try:
                obs = env._get_observation()
                observations_list.append(obs)
            except Exception:
                # If observation retrieval fails, use empty list
                pass

        observations_df = (
            pd.DataFrame(observations_list) if observations_list else pd.DataFrame()
        )
        results["normalization"] = self.normalization_validator.validate(
            observations_df
        )

        # Advanced detectors (use detect() method and convert to ValidationResult)
        lookahead_result = self.lookahead_detector.detect(
            {"env": env, "candle_data": candle_data}
        )
        results["lookahead_bias"] = self._convert_lookahead_result(lookahead_result)

        future_leak_result = self.future_leak_detector.detect(data_splits)
        results["future_leak"] = self._convert_future_leak_result(future_leak_result)

        suspicious_result = self.suspicious_patterns_detector.detect(
            {"env": env, "candle_data": candle_data}
        )
        results["suspicious_patterns"] = self._convert_suspicious_patterns_result(
            suspicious_result
        )

        # Compute summary
        passed_count = sum(1 for r in results.values() if r["passed"])
        failed_count = len(results) - passed_count
        pass_rate = passed_count / len(results) if results else 0.0

        results["summary"] = {
            "total_validators": len(results),
            "passed_count": passed_count,
            "failed_count": failed_count,
            "pass_rate": pass_rate,
        }

        # Log to MLflow if requested
        if log_to_mlflow:
            self._log_to_mlflow(results, train_split, val_split)

        return results

    def _convert_lookahead_result(
        self, detect_result: Dict[str, Any]
    ) -> ValidationResult:
        """
        Convert LookaheadBiasDetector.detect() result to ValidationResult.

        Args:
            detect_result: Result from LookaheadBiasDetector.detect()

        Returns:
            ValidationResult with standardized structure
        """
        has_bias = detect_result.get("has_bias", False)
        violations = []

        if has_bias:
            violations.append(detect_result.get("message", "Lookahead bias detected"))

        return {
            "passed": not has_bias,
            "violations": violations,
            "metrics": {
                "bias_type": detect_result.get("bias_type"),
                "normalization_method": detect_result.get("normalization_method"),
                "violating_timestamps": detect_result.get("violating_timestamps", []),
            },
        }

    def _convert_future_leak_result(
        self, detect_result: Dict[str, Any]
    ) -> ValidationResult:
        """
        Convert FutureLeakDetector.detect() result to ValidationResult.

        Args:
            detect_result: Result from FutureLeakDetector.detect()

        Returns:
            ValidationResult with standardized structure
        """
        has_leak = detect_result.get("has_leak", False)
        violations = []

        if has_leak:
            violations.append(detect_result.get("message", "Future data leak detected"))

        return {
            "passed": not has_leak,
            "violations": violations,
            "metrics": {
                "leak_type": detect_result.get("leak_type"),
                "overlapping_timestamps": detect_result.get(
                    "overlapping_timestamps", []
                ),
            },
        }

    def _convert_suspicious_patterns_result(
        self, detect_result: Dict[str, Any]
    ) -> ValidationResult:
        """
        Convert SuspiciousPatternDetector.detect() result to ValidationResult.

        Args:
            detect_result: Result from SuspiciousPatternDetector.detect()

        Returns:
            ValidationResult with standardized structure
        """
        is_suspicious = detect_result.get("is_suspicious", False)
        violations = detect_result.get("suspicious_patterns", [])

        return {
            "passed": not is_suspicious,
            "violations": violations,
            "metrics": {
                "sharpe_ratio": detect_result.get("sharpe_ratio"),
                "win_rate": detect_result.get("win_rate"),
                "return_std": detect_result.get("return_std"),
                "max_drawdown": detect_result.get("max_drawdown"),
                "max_consecutive_wins": detect_result.get("max_consecutive_wins"),
                "action_price_correlation": detect_result.get(
                    "action_price_correlation"
                ),
            },
        }

    def _log_to_mlflow(
        self, results: Dict[str, Any], train_split: float, val_split: float
    ) -> None:
        """
        Log validation results to MLflow.

        Args:
            results: Validation results dictionary
            train_split: Training data ratio
            val_split: Validation data ratio
        """
        try:
            import mlflow

            # Start MLflow run if not already active
            active_run = mlflow.active_run()
            should_end_run = False

            if active_run is None:
                mlflow.start_run()
                should_end_run = True

            # Log summary metrics
            summary = results.get("summary", {})
            mlflow.log_metric("validation_pass_rate", summary.get("pass_rate", 0.0))
            mlflow.log_metric(
                "validation_total_validators", float(summary.get("total_validators", 0))
            )
            mlflow.log_metric(
                "validation_passed_count", float(summary.get("passed_count", 0))
            )
            mlflow.log_metric(
                "validation_failed_count", float(summary.get("failed_count", 0))
            )

            # Log parameters
            mlflow.log_param("train_split", train_split)
            mlflow.log_param("val_split", val_split)

            # Log individual validator results
            for validator_name, result in results.items():
                if validator_name == "summary":
                    continue

                # Log pass/fail status
                passed = result.get("passed", False)
                mlflow.log_metric(f"{validator_name}_passed", 1.0 if passed else 0.0)

                # Log violation count
                violations = result.get("violations", [])
                mlflow.log_metric(
                    f"{validator_name}_violations", float(len(violations))
                )

            # End run if we started it
            if should_end_run:
                mlflow.end_run()

        except ImportError:
            # MLflow not installed, skip logging
            pass
        except Exception as e:
            # Log error but don't fail validation
            import warnings

            warnings.warn(f"Failed to log to MLflow: {e}", RuntimeWarning)
