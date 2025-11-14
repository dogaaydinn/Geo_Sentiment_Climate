"""
Enterprise-grade Rate Limiting with Redis.

Implements:
- Sliding window algorithm
- Per-user rate limiting
- Per-endpoint rate limiting
- API quota management
- Distributed rate limiting

Part of Week 4 Day 3: Rate Limiting Implementation
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from fastapi import HTTPException, Request, Depends, status
from redis import asyncio as aioredis
import time
import structlog
import os

from source.api.database import get_db, User, UsageRecord
from source.api.auth import get_optional_user
from sqlalchemy.orm import Session

# Configure logger
logger = structlog.get_logger()

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


# ============================================================================
# Rate Limiter Class
# ============================================================================

class RateLimiter:
    """
    Redis-based rate limiter using sliding window algorithm.

    Features:
    - Atomic Redis operations
    - Distributed rate limiting
    - Multiple time windows
    - Per-user and per-endpoint limits
    - Burst protection
    """

    def __init__(
        self,
        redis_url: str = REDIS_URL,
        default_limit: int = 100,
        default_window: int = 60
    ):
        """
        Initialize rate limiter.

        Args:
            redis_url: Redis connection URL
            default_limit: Default requests allowed
            default_window: Default time window in seconds
        """
        self.redis_url = redis_url
        self.default_limit = default_limit
        self.default_window = default_window
        self._redis = None

    async def get_redis(self):
        """Get or create Redis connection."""
        if self._redis is None:
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis

    async def is_allowed(
        self,
        key: str,
        limit: Optional[int] = None,
        window: Optional[int] = None
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Check if request is allowed under rate limit.

        Uses sliding window algorithm with Redis sorted sets:
        1. Remove old entries outside window
        2. Count current requests in window
        3. Check against limit
        4. Add new request if allowed

        Args:
            key: Unique identifier (user_id, ip, etc.)
            limit: Max requests allowed
            window: Time window in seconds

        Returns:
            Tuple of (allowed: bool, metadata: dict)
        """
        limit = limit or self.default_limit
        window = window or self.default_window

        now = time.time()
        window_start = now - window

        # Redis key for this rate limit
        redis_key = f"rate_limit:{key}"

        try:
            redis = await self.get_redis()

            # Pipeline for atomic operations
            pipe = redis.pipeline()

            # Remove old entries
            pipe.zremrangebyscore(redis_key, '-inf', window_start)

            # Count current requests
            pipe.zcard(redis_key)

            # Execute pipeline
            results = await pipe.execute()
            current_count = results[1]

            # Check limit
            if current_count >= limit:
                # Get oldest entry to calculate reset time
                oldest_entries = await redis.zrange(redis_key, 0, 0, withscores=True)

                if oldest_entries:
                    oldest_score = oldest_entries[0][1]
                    reset_time = int(oldest_score + window)
                else:
                    reset_time = int(now + window)

                logger.warning(
                    "rate_limit_exceeded",
                    key=key,
                    limit=limit,
                    current=current_count
                )

                return False, {
                    "allowed": False,
                    "limit": limit,
                    "remaining": 0,
                    "reset": reset_time,
                    "retry_after": int(reset_time - now)
                }

            # Add current request
            pipe = redis.pipeline()
            pipe.zadd(redis_key, {str(now): now})
            pipe.expire(redis_key, window)
            await pipe.execute()

            remaining = limit - current_count - 1
            reset_time = int(now + window)

            return True, {
                "allowed": True,
                "limit": limit,
                "remaining": remaining,
                "reset": reset_time,
                "retry_after": 0
            }

        except Exception as e:
            logger.error("rate_limiter_error", error=str(e), key=key)
            # Fail open (allow request) on errors
            return True, {
                "allowed": True,
                "error": "rate_limiter_unavailable"
            }

    async def check_quota(
        self,
        user_id: int,
        daily_limit: int
    ) -> Tuple[bool, int]:
        """
        Check daily API quota for user.

        Args:
            user_id: User identifier
            daily_limit: Daily quota limit

        Returns:
            Tuple of (allowed: bool, remaining: int)
        """
        today = datetime.utcnow().strftime("%Y-%m-%d")
        redis_key = f"quota:{user_id}:{today}"

        try:
            redis = await self.get_redis()

            # Increment counter
            count = await redis.incr(redis_key)

            # Set expiration to end of day on first request
            if count == 1:
                # Calculate seconds until end of day
                now = datetime.utcnow()
                end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=999999)
                seconds_until_eod = int((end_of_day - now).total_seconds())

                await redis.expire(redis_key, seconds_until_eod)

            remaining = max(0, daily_limit - count)
            allowed = count <= daily_limit

            if not allowed:
                logger.warning(
                    "daily_quota_exceeded",
                    user_id=user_id,
                    limit=daily_limit,
                    count=count
                )

            return allowed, remaining

        except Exception as e:
            logger.error("quota_check_error", error=str(e), user_id=user_id)
            # Fail open
            return True, daily_limit

    async def get_usage_stats(self, key: str, window: int = 3600) -> Dict[str, int]:
        """
        Get usage statistics for a key.

        Args:
            key: Rate limit key
            window: Time window in seconds

        Returns:
            Dictionary with usage statistics
        """
        redis_key = f"rate_limit:{key}"
        now = time.time()
        window_start = now - window

        try:
            redis = await self.get_redis()

            # Get all requests in window
            requests = await redis.zrangebyscore(
                redis_key,
                window_start,
                now,
                withscores=True
            )

            total_requests = len(requests)

            # Calculate requests per minute
            if total_requests > 0:
                duration_minutes = window / 60
                rpm = total_requests / duration_minutes
            else:
                rpm = 0

            return {
                "total_requests": total_requests,
                "window_seconds": window,
                "requests_per_minute": round(rpm, 2),
                "oldest_request": int(requests[0][1]) if requests else 0,
                "newest_request": int(requests[-1][1]) if requests else 0,
            }

        except Exception as e:
            logger.error("usage_stats_error", error=str(e), key=key)
            return {}

    async def reset_limit(self, key: str):
        """
        Reset rate limit for a key.

        Useful for admin operations or testing.

        Args:
            key: Rate limit key to reset
        """
        redis_key = f"rate_limit:{key}"

        try:
            redis = await self.get_redis()
            await redis.delete(redis_key)
            logger.info("rate_limit_reset", key=key)

        except Exception as e:
            logger.error("rate_limit_reset_error", error=str(e), key=key)

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()


# ============================================================================
# Global Rate Limiter Instance
# ============================================================================

rate_limiter = RateLimiter()


# ============================================================================
# Rate Limiting Tiers
# ============================================================================

RATE_LIMIT_TIERS = {
    "free": {
        "requests_per_minute": 60,
        "daily_quota": 1000,
        "burst_limit": 100,
    },
    "basic": {
        "requests_per_minute": 300,
        "daily_quota": 10000,
        "burst_limit": 500,
    },
    "pro": {
        "requests_per_minute": 1000,
        "daily_quota": 100000,
        "burst_limit": 2000,
    },
    "enterprise": {
        "requests_per_minute": None,  # Unlimited
        "daily_quota": None,  # Unlimited
        "burst_limit": None,  # Unlimited
    },
}


# ============================================================================
# FastAPI Dependencies
# ============================================================================

async def rate_limit_dependency(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """
    FastAPI dependency for rate limiting.

    Checks both per-minute rate limit and daily quota.

    Args:
        request: FastAPI request
        user: Current user (optional)
        db: Database session

    Raises:
        HTTPException: If rate limit exceeded
    """
    # Determine rate limit based on user tier
    if user:
        tier = user.tier
        user_key = f"user:{user.id}"
    else:
        # Rate limit by IP for unauthenticated requests
        tier = "free"
        client_ip = request.client.host if request.client else "unknown"
        user_key = f"ip:{client_ip}"

    # Get tier limits
    tier_config = RATE_LIMIT_TIERS.get(tier, RATE_LIMIT_TIERS["free"])

    # Check if enterprise (unlimited)
    if tier_config["requests_per_minute"] is None:
        return  # No rate limiting for enterprise

    # Check per-minute rate limit
    limit = tier_config["requests_per_minute"]
    window = 60  # 60 seconds

    allowed, metadata = await rate_limiter.is_allowed(
        key=user_key,
        limit=limit,
        window=window
    )

    # Add rate limit headers to response
    request.state.rate_limit = metadata

    if not allowed:
        # Log rate limit event
        if user:
            usage_record = UsageRecord(
                user_id=user.id,
                endpoint=str(request.url.path),
                method=request.method,
                status_code=429,
                timestamp=datetime.utcnow()
            )
            db.add(usage_record)
            db.commit()

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "limit": limit,
                "remaining": 0,
                "reset": metadata["reset"],
                "retry_after": metadata["retry_after"]
            },
            headers={
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(metadata["reset"]),
                "Retry-After": str(metadata["retry_after"])
            }
        )

    # Check daily quota (if user is authenticated)
    if user and tier_config["daily_quota"]:
        quota_allowed, quota_remaining = await rate_limiter.check_quota(
            user_id=user.id,
            daily_limit=tier_config["daily_quota"]
        )

        if not quota_allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Daily quota exceeded",
                    "daily_limit": tier_config["daily_quota"],
                    "remaining": 0
                },
                headers={
                    "X-Daily-Quota-Limit": str(tier_config["daily_quota"]),
                    "X-Daily-Quota-Remaining": "0"
                }
            )

        # Add quota headers
        request.state.daily_quota_remaining = quota_remaining


# ============================================================================
# Middleware for Rate Limit Headers
# ============================================================================

async def add_rate_limit_headers(request: Request, call_next):
    """
    Middleware to add rate limit headers to all responses.

    Args:
        request: FastAPI request
        call_next: Next middleware/handler

    Returns:
        Response with rate limit headers
    """
    response = await call_next(request)

    # Add rate limit headers if available
    if hasattr(request.state, "rate_limit"):
        metadata = request.state.rate_limit

        if "limit" in metadata:
            response.headers["X-RateLimit-Limit"] = str(metadata["limit"])
        if "remaining" in metadata:
            response.headers["X-RateLimit-Remaining"] = str(metadata["remaining"])
        if "reset" in metadata:
            response.headers["X-RateLimit-Reset"] = str(metadata["reset"])

    # Add quota headers if available
    if hasattr(request.state, "daily_quota_remaining"):
        response.headers["X-Daily-Quota-Remaining"] = str(
            request.state.daily_quota_remaining
        )

    return response


# ============================================================================
# Circuit Breaker Pattern
# ============================================================================

class CircuitBreaker:
    """
    Circuit breaker for resilience.

    Prevents cascading failures by stopping requests to failing services.

    States:
    - CLOSED: Normal operation
    - OPEN: Service is failing, reject requests
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening
            timeout: Seconds before trying again
            success_threshold: Successes needed to close
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold

        self.failures = 0
        self.successes = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func, *args, **kwargs):
        """
        Execute function through circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            HTTPException: If circuit is open
        """
        # Check if circuit is open
        if self.state == "OPEN":
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = time.time() - self.last_failure_time

                if elapsed > self.timeout:
                    logger.info("circuit_breaker_half_open")
                    self.state = "HALF_OPEN"
                    self.successes = 0
                else:
                    logger.warning("circuit_breaker_open")
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Service temporarily unavailable"
                    )

        try:
            # Execute function
            result = await func(*args, **kwargs)

            # Record success
            self.failures = 0

            if self.state == "HALF_OPEN":
                self.successes += 1

                if self.successes >= self.success_threshold:
                    logger.info("circuit_breaker_closed")
                    self.state = "CLOSED"
                    self.successes = 0

            return result

        except Exception as e:
            # Record failure
            self.failures += 1
            self.last_failure_time = time.time()

            if self.failures >= self.failure_threshold:
                logger.error("circuit_breaker_opened", failures=self.failures)
                self.state = "OPEN"

            raise e
