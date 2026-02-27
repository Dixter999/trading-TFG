"""
Request Tracker for preventing duplicate trading requests.

Tracks trading requests with TTL-based expiration to prevent duplicate
position openings within a configurable time window.

Thread-safe implementation using threading.Lock for concurrent access.
"""

import threading
import time
from datetime import datetime
from typing import Dict, Optional


class RequestTracker:
    """
    Track trading requests to prevent duplicates.

    Stores request data with automatic TTL-based expiration and cleanup.
    Thread-safe for concurrent access from multiple threads.

    Attributes:
        ttl_seconds (int): Time-to-live for requests in seconds
        _requests (dict): Internal storage for request data
        _lock (threading.Lock): Thread safety lock
        _stats (dict): Statistics counters
    """

    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize RequestTracker.

        Args:
            ttl_seconds (int): Time-to-live for requests in seconds (default: 300)
        """
        self.ttl_seconds = ttl_seconds
        self._requests: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "duplicates_prevented": 0,
        }

    def is_duplicate(self, request_id: str) -> bool:
        """
        Check if request ID already exists and is not expired.

        Automatically cleans up expired requests during check.

        Args:
            request_id (str): Request identifier to check

        Returns:
            bool: True if duplicate (exists and not expired), False otherwise
        """
        with self._lock:
            # Auto-cleanup expired requests
            self._cleanup_expired_unsafe()

            # Check if request exists and is not expired
            if request_id in self._requests:
                request = self._requests[request_id]
                if not self._is_expired_unsafe(request):
                    self._stats["duplicates_prevented"] += 1
                    return True

            return False

    def record_request(
        self, request_id: str, symbol: str, direction: str
    ) -> None:
        """
        Store new request as pending.

        Args:
            request_id (str): Unique request identifier
            symbol (str): Trading symbol (e.g., "EURUSD")
            direction (str): Trade direction ("LONG" or "SHORT")
        """
        with self._lock:
            # Auto-cleanup before recording
            self._cleanup_expired_unsafe()

            self._requests[request_id] = {
                "request_id": request_id,
                "symbol": symbol,
                "direction": direction,
                "status": "pending",
                "ticket": None,
                "error_message": None,
                "created_at": datetime.now(),
                "completed_at": None,
            }

            self._stats["total_requests"] += 1

    def record_success(self, request_id: str, ticket: int) -> None:
        """
        Mark request as successful with ticket number.

        Args:
            request_id (str): Request identifier
            ticket (int): MT5 ticket number

        Raises:
            KeyError: If request_id not found
        """
        with self._lock:
            if request_id not in self._requests:
                raise KeyError(f"Request ID not found: {request_id}")

            self._requests[request_id]["status"] = "success"
            self._requests[request_id]["ticket"] = ticket
            self._requests[request_id]["completed_at"] = datetime.now()

            self._stats["successful_requests"] += 1

    def record_failure(self, request_id: str, error: str) -> None:
        """
        Mark request as failed with error message.

        Args:
            request_id (str): Request identifier
            error (str): Error message

        Raises:
            KeyError: If request_id not found
        """
        with self._lock:
            if request_id not in self._requests:
                raise KeyError(f"Request ID not found: {request_id}")

            self._requests[request_id]["status"] = "failed"
            self._requests[request_id]["error_message"] = error
            self._requests[request_id]["completed_at"] = datetime.now()

            self._stats["failed_requests"] += 1

    def cleanup_expired(self) -> None:
        """
        Remove requests older than TTL.

        Thread-safe public method for manual cleanup.
        """
        with self._lock:
            self._cleanup_expired_unsafe()

    def get_request(self, request_id: str) -> Optional[dict]:
        """
        Get request details.

        Args:
            request_id (str): Request identifier

        Returns:
            Optional[dict]: Request data or None if not found
        """
        with self._lock:
            return self._requests.get(request_id)

    def get_active_count(self) -> int:
        """
        Count non-expired requests.

        Automatically cleans up expired requests before counting.

        Returns:
            int: Number of active (non-expired) requests
        """
        with self._lock:
            # Auto-cleanup before counting
            self._cleanup_expired_unsafe()
            return len(self._requests)

    def get_statistics(self) -> dict:
        """
        Get request statistics.

        Returns:
            dict: Statistics including total, success, failed, duplicates, active
        """
        with self._lock:
            # Auto-cleanup for accurate active count
            self._cleanup_expired_unsafe()

            return {
                "total_requests": self._stats["total_requests"],
                "successful_requests": self._stats["successful_requests"],
                "failed_requests": self._stats["failed_requests"],
                "duplicates_prevented": self._stats["duplicates_prevented"],
                "active_requests": len(self._requests),
            }

    def generate_request_id(
        self, symbol: str, direction: str, timestamp: Optional[int] = None
    ) -> str:
        """
        Generate request ID from symbol, direction, and timestamp.

        Args:
            symbol (str): Trading symbol
            direction (str): Trade direction
            timestamp (int, optional): Unix timestamp. Defaults to current time.

        Returns:
            str: Request ID in format: {SYMBOL}_{DIRECTION}_{TIMESTAMP}
        """
        if timestamp is None:
            timestamp = int(time.time())

        return f"{symbol}_{direction}_{timestamp}"

    # Private methods (not thread-safe, use within lock)

    def _is_expired_unsafe(self, request: dict) -> bool:
        """
        Check if request is expired (UNSAFE - must be called within lock).

        Args:
            request (dict): Request data

        Returns:
            bool: True if expired, False otherwise
        """
        if self.ttl_seconds <= 0:
            return True  # Expire immediately for zero/negative TTL

        created_at = request["created_at"]
        age = (datetime.now() - created_at).total_seconds()

        return age > self.ttl_seconds

    def _cleanup_expired_unsafe(self) -> None:
        """
        Remove expired requests (UNSAFE - must be called within lock).

        Internal method for cleanup, called automatically during operations.
        """
        # Use list comprehension for efficiency
        expired_ids = [
            request_id
            for request_id, request in self._requests.items()
            if self._is_expired_unsafe(request)
        ]

        # Remove expired requests
        for request_id in expired_ids:
            del self._requests[request_id]
