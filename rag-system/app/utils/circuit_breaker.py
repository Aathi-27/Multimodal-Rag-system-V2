"""
Circuit Breaker - Protects against cascading failures in worker processes.

Implements the circuit breaker pattern for:
- OCR worker (PaddleOCR hangs/crashes)
- Audio worker (Whisper stalls)
- Qdrant connections

States: CLOSED (normal) → OPEN (tripped) → HALF_OPEN (testing)
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failures exceeded threshold, rejecting calls
    HALF_OPEN = "half_open"    # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit is open and calls are being rejected."""
    pass


class CircuitBreaker:
    """
    Circuit breaker to protect against repeated failures.

    Usage:
        breaker = CircuitBreaker("ocr", failure_threshold=3, recovery_timeout=60)
        result = breaker.call(ocr_function, image_path)
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exceptions: tuple = (Exception,),
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._success_count = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info("Circuit [%s]: OPEN → HALF_OPEN", self.name)
        return self._state

    def call(self, func: Callable, *args, **kwargs):
        """
        Execute a function through the circuit breaker.

        Raises CircuitBreakerError if circuit is OPEN.
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise CircuitBreakerError(
                f"Circuit [{self.name}] is OPEN. "
                f"Failures: {self._failure_count}/{self.failure_threshold}. "
                f"Recovery in {self._time_until_recovery():.0f}s."
            )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exceptions as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= 2:  # Require 2 consecutive successes
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                logger.info("Circuit [%s]: HALF_OPEN → CLOSED", self.name)
        else:
            self._failure_count = max(0, self._failure_count - 1)

    def _on_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        self._success_count = 0

        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit [%s]: → OPEN (failures=%d)",
                self.name,
                self._failure_count,
            )

    def _time_until_recovery(self) -> float:
        elapsed = time.time() - self._last_failure_time
        return max(0.0, self.recovery_timeout - elapsed)

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        logger.info("Circuit [%s]: Manually reset to CLOSED", self.name)

    def status(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
        }
