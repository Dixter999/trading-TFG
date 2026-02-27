"""
Validation module.

Provides TRUE out-of-sample validation using data that was NEVER seen
during Signal Discovery. This eliminates data leakage and provides
genuine validation of discovered signals.

Issues:
- #527 - True Walk-Forward Validation (walk_forward.py)
- #524 - Final Test Validation (final_test.py)

Epic: hybrid-v4
"""

from validation.walk_forward import (
    ValidationResult,
    validate_signal,
    filter_validated_signals,
    MIN_PROFIT_FACTOR,
    MAX_WR_DEGRADATION,
    MIN_TRADES,
)

from validation.final_test import (
    FinalValidationResult,
    run_final_validation,
    save_validation_results,
    evaluate_model_on_test_data,
    MIN_TEST_PF,
    MAX_PF_DEGRADATION,
    MIN_TEST_TRADES,
)

__all__ = [
    # Walk-Forward Validation (#527)
    "ValidationResult",
    "validate_signal",
    "filter_validated_signals",
    "MIN_PROFIT_FACTOR",
    "MAX_WR_DEGRADATION",
    "MIN_TRADES",
    # Final Test Validation (#524)
    "FinalValidationResult",
    "run_final_validation",
    "save_validation_results",
    "evaluate_model_on_test_data",
    "MIN_TEST_PF",
    "MAX_PF_DEGRADATION",
    "MIN_TEST_TRADES",
]
