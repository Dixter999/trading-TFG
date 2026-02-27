"""
Base validator class and common validation utilities.

Provides standard validation result structure and helper methods
for all validators in the suite.
"""

from typing import Dict, List, Any, TypedDict


class ValidationResult(TypedDict):
    """Standard validation result structure."""

    passed: bool
    violations: List[str]
    metrics: Dict[str, Any]


class BaseValidator:
    """
    Base class for all validators.

    Provides common structure and helper methods.
    All validators should follow this interface.
    """

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate data and return standardized result.

        Args:
            data: Data to validate (type varies by validator)

        Returns:
            ValidationResult with 'passed', 'violations', and 'metrics'
        """
        raise NotImplementedError("Subclasses must implement validate()")

    @staticmethod
    def create_result(
        violations: List[str], metrics: Dict[str, Any]
    ) -> ValidationResult:
        """
        Create standardized validation result.

        Args:
            violations: List of violation messages
            metrics: Dictionary of validation metrics

        Returns:
            ValidationResult with 'passed' computed from violations
        """
        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "metrics": metrics,
        }

    @staticmethod
    def create_success_result(metrics: Dict[str, Any]) -> ValidationResult:
        """
        Create validation result for successful validation.

        Args:
            metrics: Dictionary of validation metrics

        Returns:
            ValidationResult with passed=True and no violations
        """
        return {
            "passed": True,
            "violations": [],
            "metrics": metrics,
        }

    @staticmethod
    def create_failure_result(
        violation: str, metrics: Dict[str, Any]
    ) -> ValidationResult:
        """
        Create validation result for single critical failure.

        Args:
            violation: Single violation message (e.g., missing required field)
            metrics: Dictionary of validation metrics

        Returns:
            ValidationResult with passed=False and single violation
        """
        return {
            "passed": False,
            "violations": [violation],
            "metrics": metrics,
        }
