"""
Circuit Breaker Pattern Implementation.

Implements the circuit breaker pattern to prevent cascading failures
in distributed systems. Provides automatic failure detection and recovery.

States:
    CLOSED: Normal operation, requests allowed
    OPEN: Circuit tripped, requests blocked
    HALF_OPEN: Testing if system recovered

Transitions:
    CLOSED -> OPEN: After failure_threshold failures within timeout_seconds
    OPEN -> HALF_OPEN: After timeout_seconds elapsed
    HALF_OPEN -> CLOSED: After half_open_attempts successful requests
    HALF_OPEN -> OPEN: After any failure

Thread Safety:
    All methods are thread-safe using threading.Lock

Usage Example:
    ```python
    cb = CircuitBreaker(failure_threshold=5, timeout_seconds=300)

    if cb.allow_request():
        try:
            result = await risky_operation()
            cb.record_success()
        except Exception:
            cb.record_failure()
    else:
        logger.error("Circuit breaker OPEN - request blocked")
    ```
"""

import threading
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Optional


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.

    Prevents cascading failures by stopping requests when failures
    exceed threshold, then gradually allowing requests to test recovery.

    The circuit breaker uses a sliding time window to track failures,
    ensuring that only recent failures within the configured timeout
    period are counted toward the threshold.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        timeout_seconds: Seconds to wait before attempting recovery
        half_open_attempts: Successful requests needed to close circuit
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 300,
        half_open_attempts: int = 2,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit (default: 5)
            timeout_seconds: Seconds to wait in OPEN state before HALF_OPEN (default: 300)
            half_open_attempts: Successful requests needed in HALF_OPEN to close (default: 2)
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_attempts = half_open_attempts

        # State management
        self._state = CircuitState.CLOSED
        self._lock = threading.Lock()

        # Metrics
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._opened_at: Optional[datetime] = None

        # Failure tracking for time-based threshold
        self._failure_timestamps: list[datetime] = []

    def allow_request(self) -> bool:
        """
        Check if request should be allowed based on circuit state.

        Returns:
            True if request should be allowed, False otherwise

        Side Effects:
            May transition from OPEN to HALF_OPEN if timeout elapsed
        """
        with self._lock:
            self._update_state()
            return self._state != CircuitState.OPEN

    def record_success(self) -> None:
        """
        Record successful request.

        Side Effects:
            - Increments success count
            - Resets failure count
            - May transition from HALF_OPEN to CLOSED if threshold met
        """
        with self._lock:
            self._success_count += 1
            self._failure_count = 0
            self._failure_timestamps.clear()

            # Check if we should close the circuit
            if self._state == CircuitState.HALF_OPEN:
                if self._success_count >= self.half_open_attempts:
                    self._transition_to_closed()

    def record_failure(self) -> None:
        """
        Record failed request.

        Side Effects:
            - Increments failure count
            - Resets success count
            - Updates last failure timestamp
            - May transition to OPEN if threshold exceeded
            - In HALF_OPEN, immediately transitions to OPEN
        """
        with self._lock:
            now = self._get_current_time()
            self._failure_count += 1
            self._success_count = 0
            self._last_failure_time = now
            self._failure_timestamps.append(now)

            # State-specific behavior
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to_open()
            elif self._state == CircuitState.CLOSED:
                self._prune_old_failures(now)
                if len(self._failure_timestamps) >= self.failure_threshold:
                    self._transition_to_open()

    def get_state(self) -> CircuitState:
        """
        Get current circuit breaker state.

        Returns:
            Current CircuitState (CLOSED, OPEN, or HALF_OPEN)

        Note:
            This is an idempotent operation with no side effects
        """
        with self._lock:
            return self._state

    def get_metrics(self) -> dict:
        """
        Get circuit breaker metrics.

        Returns:
            Dictionary containing:
                - state: Current state (str)
                - failure_count: Number of failures (int)
                - success_count: Number of successes (int)
                - last_failure_time: Timestamp of last failure (datetime or None)
                - opened_at: Timestamp when circuit opened (datetime or None)
        """
        with self._lock:
            return {
                'state': self._state.value,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'last_failure_time': self._last_failure_time,
                'opened_at': self._opened_at,
            }

    def _get_current_time(self) -> datetime:
        """
        Get current UTC timestamp.

        Extracted to separate method for better testability.

        Returns:
            Current datetime in UTC timezone
        """
        return datetime.now(UTC)

    def _prune_old_failures(self, current_time: datetime) -> None:
        """
        Remove failures outside the sliding time window.

        Args:
            current_time: Current timestamp to calculate cutoff

        Note:
            For timeout_seconds=0, keeps all failures (no pruning)
        """
        if self.timeout_seconds > 0:
            cutoff = current_time - timedelta(seconds=self.timeout_seconds)
            self._failure_timestamps = [
                ts for ts in self._failure_timestamps if ts >= cutoff
            ]

    def _update_state(self) -> None:
        """
        Update state based on current conditions.

        Called by allow_request() to check for state transitions.
        Must be called with lock held.
        """
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            elapsed = (self._get_current_time() - self._opened_at).total_seconds()
            if elapsed >= self.timeout_seconds:
                self._transition_to_half_open()

    def _transition_to_open(self) -> None:
        """
        Transition to OPEN state.

        Resets success count and records opening timestamp.
        Must be called with lock held.
        """
        self._state = CircuitState.OPEN
        self._opened_at = self._get_current_time()
        self._success_count = 0

    def _transition_to_half_open(self) -> None:
        """
        Transition to HALF_OPEN state.

        Resets all counters to start testing recovery.
        Must be called with lock held.
        """
        self._state = CircuitState.HALF_OPEN
        self._failure_count = 0
        self._success_count = 0
        self._failure_timestamps.clear()

    def _transition_to_closed(self) -> None:
        """
        Transition to CLOSED state.

        Resets all counters and timestamps to clean slate.
        Must be called with lock held.
        """
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at = None
        self._failure_timestamps.clear()
