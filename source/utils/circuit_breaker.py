"""
Circuit Breaker Pattern Implementation.

Provides fault tolerance and resilience for distributed systems.
Part of Phase 3: Scaling & Optimization - Distributed Systems.
"""

import time
import threading
from enum import Enum
from typing import Callable, Any, Optional
from functools import wraps
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    Implements the circuit breaker pattern:
    - CLOSED: Normal operation, passes through requests
    - OPEN: Too many failures, reject requests immediately
    - HALF_OPEN: After timeout, try single request to test recovery
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        name: Optional[str] = None
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
            name: Name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name or "CircuitBreaker"

        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = CircuitState.CLOSED
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    logger.info(f"{self.name}: Circuit transitioning to HALF_OPEN")
            return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return False
        return time.time() - self._last_failure_time >= self.recovery_timeout

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerError(
                    f"{self.name}: Circuit breaker is OPEN"
                )

            # CLOSED or HALF_OPEN: attempt the call
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise

    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logger.info(f"{self.name}: Circuit closed after successful recovery")

    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"{self.name}: Circuit opened after {self._failure_count} failures"
                )

    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        with self._lock:
            self._failure_count = 0
            self._last_failure_time = None
            self._state = CircuitState.CLOSED
            logger.info(f"{self.name}: Circuit manually reset")

    def __call__(self, func: Callable) -> Callable:
        """Decorator usage."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper


class RetryStrategy:
    """
    Retry strategy with exponential backoff.

    Part of Phase 3: Scaling & Optimization - Fault Tolerance.
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize retry strategy.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Add random jitter to prevent thundering herd
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def execute(
        self,
        func: Callable,
        *args,
        retry_on: tuple = (Exception,),
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments
            retry_on: Tuple of exceptions to retry on
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Last exception if all retries failed
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except retry_on as e:
                last_exception = e

                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_retries + 1} attempts failed. "
                        f"Last error: {e}"
                    )

        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff."""
        import random

        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )

        if self.jitter:
            # Add random jitter (±25%)
            delay = delay * (0.75 + 0.5 * random.random())

        return delay

    def __call__(self, *retry_on):
        """Decorator usage."""
        if not retry_on:
            retry_on = (Exception,)

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute(func, *args, retry_on=retry_on, **kwargs)
            return wrapper
        return decorator


# Global circuit breakers for common services
db_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30,
    name="Database"
)

redis_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=20,
    name="Redis"
)

external_api_circuit_breaker = CircuitBreaker(
    failure_threshold=10,
    recovery_timeout=60,
    name="ExternalAPI"
)

# Global retry strategy
default_retry = RetryStrategy(
    max_retries=3,
    initial_delay=1.0,
    max_delay=10.0
)
