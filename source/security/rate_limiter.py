"""
Enterprise Rate Limiter with Redis Backend.

Features:
- Token bucket algorithm
- Sliding window rate limiting
- Per-user and per-IP limiting
- Configurable limits per endpoint
- Redis-backed distributed limiting
- Rate limit headers in responses
"""

import os
import time
import hashlib
from typing import Optional, Dict, Callable
from functools import wraps
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import redis
from source.utils.logger import setup_logger

logger = setup_logger(name="rate_limiter", log_file="../logs/rate_limiter.log")

# Redis connection
redis_client = None
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_RATE_LIMIT_DB", "1")),
        decode_responses=True
    )
    redis_client.ping()
    logger.info("Redis connection established for rate limiting")
except Exception as e:
    logger.warning(f"Redis not available for rate limiting: {e}")


class RateLimitExceeded(HTTPException):
    """Rate limit exceeded exception."""

    def __init__(
        self,
        detail: str = "Rate limit exceeded",
        retry_after: int = 60
    ):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={"Retry-After": str(retry_after)}
        )


class RateLimiter:
    """
    Redis-backed rate limiter using token bucket algorithm.

    Supports:
    - Per-user rate limiting
    - Per-IP rate limiting
    - Per-endpoint rate limiting
    - Custom rate limit configurations
    - Distributed rate limiting across instances
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = redis_client,
        default_limit: int = 100,
        default_window: int = 60
    ):
        """
        Initialize rate limiter.

        Args:
            redis_client: Redis client instance
            default_limit: Default number of requests allowed
            default_window: Default time window in seconds
        """
        self.redis = redis_client
        self.default_limit = default_limit
        self.default_window = default_window
        self.logger = logger

        # In-memory fallback if Redis unavailable
        self.fallback_store: Dict[str, Dict] = {}

    def _get_identifier(self, request: Request, user_id: Optional[str] = None) -> str:
        """
        Get unique identifier for rate limiting.

        Args:
            request: FastAPI request
            user_id: Optional user ID

        Returns:
            Unique identifier string
        """
        if user_id:
            return f"user:{user_id}"

        # Use IP address as fallback
        client_ip = request.client.host if request.client else "unknown"

        # Consider X-Forwarded-For for proxied requests
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        return f"ip:{client_ip}"

    def _create_key(self, identifier: str, endpoint: str) -> str:
        """
        Create Redis key for rate limiting.

        Args:
            identifier: User/IP identifier
            endpoint: API endpoint

        Returns:
            Redis key
        """
        # Hash endpoint to keep key length manageable
        endpoint_hash = hashlib.md5(endpoint.encode()).hexdigest()[:8]
        return f"rate_limit:{identifier}:{endpoint_hash}"

    def check_rate_limit(
        self,
        request: Request,
        limit: int,
        window: int,
        user_id: Optional[str] = None,
        endpoint: Optional[str] = None
    ) -> tuple[bool, Dict[str, int]]:
        """
        Check if request is within rate limit.

        Args:
            request: FastAPI request
            limit: Number of requests allowed
            window: Time window in seconds
            user_id: Optional user ID
            endpoint: Optional endpoint identifier

        Returns:
            Tuple of (allowed: bool, info: dict)
        """
        identifier = self._get_identifier(request, user_id)
        endpoint = endpoint or request.url.path
        key = self._create_key(identifier, endpoint)

        if self.redis:
            return self._check_redis(key, limit, window)
        else:
            return self._check_fallback(key, limit, window)

    def _check_redis(self, key: str, limit: int, window: int) -> tuple[bool, Dict[str, int]]:
        """
        Check rate limit using Redis.

        Uses sliding window algorithm with sorted sets.

        Args:
            key: Redis key
            limit: Request limit
            window: Time window in seconds

        Returns:
            Tuple of (allowed, info)
        """
        try:
            now = time.time()
            window_start = now - window

            # Use Redis pipeline for atomic operations
            pipe = self.redis.pipeline()

            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)

            # Count requests in current window
            pipe.zcard(key)

            # Add current request timestamp
            pipe.zadd(key, {str(now): now})

            # Set expiration
            pipe.expire(key, window)

            results = pipe.execute()
            current_count = results[1]

            # Check if limit exceeded
            allowed = current_count < limit
            remaining = max(0, limit - current_count - (1 if allowed else 0))

            # Get oldest request timestamp for reset calculation
            oldest = self.redis.zrange(key, 0, 0, withscores=True)
            reset_time = int(oldest[0][1] + window) if oldest else int(now + window)

            info = {
                "limit": limit,
                "remaining": remaining,
                "reset": reset_time,
                "retry_after": window if not allowed else 0
            }

            if not allowed:
                # Remove the just-added timestamp since request is rejected
                self.redis.zrem(key, str(now))
                self.logger.warning(f"Rate limit exceeded for {key}")

            return allowed, info

        except Exception as e:
            self.logger.error(f"Redis rate limit check failed: {e}")
            # Fail open - allow request if Redis fails
            return True, {"limit": limit, "remaining": limit, "reset": int(time.time() + window)}

    def _check_fallback(self, key: str, limit: int, window: int) -> tuple[bool, Dict[str, int]]:
        """
        Fallback in-memory rate limiting.

        Args:
            key: Rate limit key
            limit: Request limit
            window: Time window in seconds

        Returns:
            Tuple of (allowed, info)
        """
        now = time.time()

        if key not in self.fallback_store:
            self.fallback_store[key] = {"requests": [], "reset": now + window}

        # Clean old requests
        store = self.fallback_store[key]
        store["requests"] = [ts for ts in store["requests"] if ts > now - window]

        # Check limit
        current_count = len(store["requests"])
        allowed = current_count < limit

        if allowed:
            store["requests"].append(now)

        remaining = max(0, limit - current_count - (1 if allowed else 0))
        reset_time = int(min(store["requests"]) + window) if store["requests"] else int(now + window)

        info = {
            "limit": limit,
            "remaining": remaining,
            "reset": reset_time,
            "retry_after": window if not allowed else 0
        }

        return allowed, info

    async def __call__(
        self,
        request: Request,
        limit: Optional[int] = None,
        window: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Rate limit middleware.

        Args:
            request: FastAPI request
            limit: Request limit (uses default if None)
            window: Time window in seconds (uses default if None)
            user_id: Optional user ID

        Returns:
            Rate limit info

        Raises:
            RateLimitExceeded: If rate limit exceeded
        """
        limit = limit or self.default_limit
        window = window or self.default_window

        allowed, info = self.check_rate_limit(
            request=request,
            limit=limit,
            window=window,
            user_id=user_id
        )

        if not allowed:
            raise RateLimitExceeded(
                detail=f"Rate limit exceeded. Try again in {info['retry_after']} seconds.",
                retry_after=info['retry_after']
            )

        return info


# Decorator for endpoint-specific rate limiting
def rate_limit(
    limit: int = 100,
    window: int = 60,
    per_user: bool = True
):
    """
    Decorator for rate limiting endpoints.

    Args:
        limit: Number of requests allowed
        window: Time window in seconds
        per_user: If True, limit per user; otherwise per IP

    Returns:
        Decorated function

    Example:
        @app.get("/api/endpoint")
        @rate_limit(limit=10, window=60)
        async def my_endpoint():
            return {"message": "success"}
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            user_id = None

            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                request = kwargs.get("request")

            if not request:
                # No request found, skip rate limiting
                return await func(*args, **kwargs)

            # Get user ID if per_user is True
            if per_user:
                # Try to get current user from kwargs
                current_user = kwargs.get("current_user")
                if current_user and hasattr(current_user, "id"):
                    user_id = current_user.id

            # Check rate limit
            limiter = RateLimiter()
            info = await limiter(request, limit=limit, window=window, user_id=user_id)

            # Add rate limit headers to response
            response = await func(*args, **kwargs)

            if hasattr(response, "headers"):
                response.headers["X-RateLimit-Limit"] = str(info["limit"])
                response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
                response.headers["X-RateLimit-Reset"] = str(info["reset"])

            return response

        return wrapper

    return decorator


# FastAPI middleware for global rate limiting
class RateLimitMiddleware:
    """
    Middleware for global rate limiting.

    Add to FastAPI app:
        app.add_middleware(
            RateLimitMiddleware,
            limit=100,
            window=60
        )
    """

    def __init__(
        self,
        app,
        limit: int = 100,
        window: int = 60,
        exempt_paths: Optional[list] = None
    ):
        """
        Initialize middleware.

        Args:
            app: FastAPI app
            limit: Request limit
            window: Time window in seconds
            exempt_paths: List of paths to exempt from rate limiting
        """
        self.app = app
        self.limiter = RateLimiter(default_limit=limit, default_window=window)
        self.exempt_paths = exempt_paths or ["/health", "/docs", "/openapi.json"]

    async def __call__(self, scope, receive, send):
        """Process request through rate limiter."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Check if path is exempt
        path = scope["path"]
        if any(path.startswith(exempt) for exempt in self.exempt_paths):
            await self.app(scope, receive, send)
            return

        # Create request-like object
        class FakeRequest:
            def __init__(self, scope):
                self.scope = scope
                self.url = type('obj', (object,), {'path': scope["path"]})()
                self.client = type('obj', (object,), {
                    'host': scope.get("client", ["unknown"])[0]
                })()
                self.headers = dict(scope.get("headers", []))

        request = FakeRequest(scope)

        try:
            # Check rate limit
            info = await self.limiter(request)

            # Add rate limit headers
            async def send_with_headers(message):
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.extend([
                        (b"x-ratelimit-limit", str(info["limit"]).encode()),
                        (b"x-ratelimit-remaining", str(info["remaining"]).encode()),
                        (b"x-ratelimit-reset", str(info["reset"]).encode()),
                    ])
                    message["headers"] = headers
                await send(message)

            await self.app(scope, receive, send_with_headers)

        except RateLimitExceeded as e:
            # Send 429 response
            response = JSONResponse(
                status_code=429,
                content={"detail": str(e.detail)},
                headers=e.headers
            )
            await response(scope, receive, send)


# Rate limit tiers for different user types
RATE_LIMIT_TIERS = {
    "free": {"limit": 100, "window": 3600},  # 100 requests per hour
    "basic": {"limit": 1000, "window": 3600},  # 1000 requests per hour
    "premium": {"limit": 10000, "window": 3600},  # 10000 requests per hour
    "enterprise": {"limit": 100000, "window": 3600},  # 100000 requests per hour
}


def get_user_rate_limit(user_tier: str = "free") -> Dict[str, int]:
    """
    Get rate limit configuration for user tier.

    Args:
        user_tier: User subscription tier

    Returns:
        Rate limit configuration
    """
    return RATE_LIMIT_TIERS.get(user_tier, RATE_LIMIT_TIERS["free"])
