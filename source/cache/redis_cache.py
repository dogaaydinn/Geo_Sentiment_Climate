"""
Redis caching implementation with enterprise features.

Features:
- Automatic serialization/deserialization
- TTL management
- Cache warming
- Batch operations
- Pattern-based invalidation
"""

import os
import json
import pickle
import hashlib
from typing import Any, Optional, Callable, List
from functools import wraps
import redis
from source.utils.logger import setup_logger

logger = setup_logger(name="cache", log_file="../logs/cache.log")


class RedisCache:
    """
    Redis-backed caching with automatic serialization.

    Supports:
    - JSON and pickle serialization
    - TTL management
    - Batch operations
    - Pattern-based key management
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = None,
        password: str = None,
        default_ttl: int = 3600
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            default_ttl: Default TTL in seconds
        """
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.db = db or int(os.getenv("REDIS_CACHE_DB", "3"))
        self.password = password or os.getenv("REDIS_PASSWORD")
        self.default_ttl = default_ttl
        self.logger = logger

        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False,  # Handle bytes for pickle
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            self.client.ping()
            self.logger.info(f"Redis cache connected: {self.host}:{self.port}/{self.db}")
        except Exception as e:
            self.logger.error(f"Redis cache connection failed: {e}")
            self.client = None

    def _make_key(self, key: str, prefix: str = "cache") -> str:
        """Create cache key with prefix."""
        return f"{prefix}:{key}"

    def _serialize(self, value: Any, use_pickle: bool = False) -> bytes:
        """Serialize value for storage."""
        try:
            if use_pickle:
                return pickle.dumps(value)
            else:
                return json.dumps(value).encode('utf-8')
        except (TypeError, pickle.PicklingError) as e:
            self.logger.warning(f"Serialization failed, using pickle: {e}")
            return pickle.dumps(value)

    def _deserialize(self, value: bytes, use_pickle: bool = False) -> Any:
        """Deserialize value from storage."""
        try:
            if use_pickle:
                return pickle.loads(value)
            else:
                return json.loads(value.decode('utf-8'))
        except (json.JSONDecodeError, pickle.UnpicklingError):
            # Try pickle as fallback
            try:
                return pickle.loads(value)
            except Exception as e:
                self.logger.error(f"Deserialization failed: {e}")
                return None

    def get(self, key: str, prefix: str = "cache", use_pickle: bool = False) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key
            prefix: Key prefix
            use_pickle: Use pickle serialization

        Returns:
            Cached value or None
        """
        if not self.client:
            return None

        try:
            full_key = self._make_key(key, prefix)
            value = self.client.get(full_key)

            if value is None:
                return None

            return self._deserialize(value, use_pickle)

        except Exception as e:
            self.logger.error(f"Cache get error for key {key}: {e}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        prefix: str = "cache",
        use_pickle: bool = False
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            prefix: Key prefix
            use_pickle: Use pickle serialization

        Returns:
            True if successful
        """
        if not self.client:
            return False

        try:
            full_key = self._make_key(key, prefix)
            serialized = self._serialize(value, use_pickle)
            ttl = ttl or self.default_ttl

            if ttl:
                self.client.setex(full_key, ttl, serialized)
            else:
                self.client.set(full_key, serialized)

            return True

        except Exception as e:
            self.logger.error(f"Cache set error for key {key}: {e}")
            return False

    def delete(self, key: str, prefix: str = "cache") -> bool:
        """Delete key from cache."""
        if not self.client:
            return False

        try:
            full_key = self._make_key(key, prefix)
            return self.client.delete(full_key) > 0
        except Exception as e:
            self.logger.error(f"Cache delete error for key {key}: {e}")
            return False

    def exists(self, key: str, prefix: str = "cache") -> bool:
        """Check if key exists in cache."""
        if not self.client:
            return False

        try:
            full_key = self._make_key(key, prefix)
            return self.client.exists(full_key) > 0
        except Exception as e:
            self.logger.error(f"Cache exists check error for key {key}: {e}")
            return False

    def get_many(self, keys: List[str], prefix: str = "cache") -> dict:
        """Get multiple values at once."""
        if not self.client:
            return {}

        try:
            full_keys = [self._make_key(k, prefix) for k in keys]
            values = self.client.mget(full_keys)

            result = {}
            for key, value in zip(keys, values):
                if value:
                    result[key] = self._deserialize(value)

            return result

        except Exception as e:
            self.logger.error(f"Cache get_many error: {e}")
            return {}

    def set_many(self, mapping: dict, ttl: Optional[int] = None, prefix: str = "cache") -> bool:
        """Set multiple values at once."""
        if not self.client:
            return False

        try:
            pipe = self.client.pipeline()
            ttl = ttl or self.default_ttl

            for key, value in mapping.items():
                full_key = self._make_key(key, prefix)
                serialized = self._serialize(value)

                if ttl:
                    pipe.setex(full_key, ttl, serialized)
                else:
                    pipe.set(full_key, serialized)

            pipe.execute()
            return True

        except Exception as e:
            self.logger.error(f"Cache set_many error: {e}")
            return False

    def delete_pattern(self, pattern: str, prefix: str = "cache") -> int:
        """
        Delete all keys matching pattern.

        Args:
            pattern: Pattern to match (e.g., "user:*")
            prefix: Key prefix

        Returns:
            Number of deleted keys
        """
        if not self.client:
            return 0

        try:
            full_pattern = self._make_key(pattern, prefix)
            keys = list(self.client.scan_iter(match=full_pattern, count=100))

            if keys:
                return self.client.delete(*keys)
            return 0

        except Exception as e:
            self.logger.error(f"Cache delete_pattern error: {e}")
            return 0

    def flush(self) -> bool:
        """Flush entire cache database."""
        if not self.client:
            return False

        try:
            self.client.flushdb()
            self.logger.info("Cache flushed")
            return True
        except Exception as e:
            self.logger.error(f"Cache flush error: {e}")
            return False

    def ttl(self, key: str, prefix: str = "cache") -> int:
        """Get remaining TTL for key."""
        if not self.client:
            return -1

        try:
            full_key = self._make_key(key, prefix)
            return self.client.ttl(full_key)
        except Exception as e:
            self.logger.error(f"Cache TTL error: {e}")
            return -1

    def extend_ttl(self, key: str, additional_seconds: int, prefix: str = "cache") -> bool:
        """Extend TTL for existing key."""
        if not self.client:
            return False

        try:
            full_key = self._make_key(key, prefix)
            current_ttl = self.client.ttl(full_key)

            if current_ttl > 0:
                new_ttl = current_ttl + additional_seconds
                self.client.expire(full_key, new_ttl)
                return True

            return False

        except Exception as e:
            self.logger.error(f"Cache extend_ttl error: {e}")
            return False


# Global cache instance
cache = RedisCache()


def cache_key_generator(*args, **kwargs) -> str:
    """Generate cache key from function arguments."""
    key_parts = []

    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])

    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool)):
            key_parts.append(f"{k}={v}")
        else:
            key_parts.append(f"{k}={hashlib.md5(str(v).encode()).hexdigest()[:8]}")

    return ":".join(key_parts)


def cache_result(
    ttl: int = 3600,
    prefix: str = "func",
    use_pickle: bool = False,
    key_func: Optional[Callable] = None
):
    """
    Decorator to cache function results.

    Args:
        ttl: Time to live in seconds
        prefix: Cache key prefix
        use_pickle: Use pickle serialization
        key_func: Custom function to generate cache key

    Example:
        @cache_result(ttl=300, prefix="predictions")
        def predict(model_id, data):
            return model.predict(data)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                func_name = f"{func.__module__}.{func.__name__}"
                args_key = cache_key_generator(*args[1:], **kwargs)  # Skip self
                cache_key = f"{func_name}:{args_key}"

            # Try to get from cache
            cached_value = cache.get(cache_key, prefix=prefix, use_pickle=use_pickle)
            if cached_value is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_value

            # Execute function
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result, ttl=ttl, prefix=prefix, use_pickle=use_pickle)
            logger.debug(f"Cache miss for {cache_key}, stored result")

            return result

        return wrapper

    return decorator


def invalidate_cache(pattern: str, prefix: str = "func"):
    """
    Invalidate cache entries matching pattern.

    Args:
        pattern: Pattern to match
        prefix: Cache key prefix
    """
    count = cache.delete_pattern(pattern, prefix=prefix)
    logger.info(f"Invalidated {count} cache entries matching {pattern}")
    return count
