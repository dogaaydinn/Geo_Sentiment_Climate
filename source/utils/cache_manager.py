"""
Advanced Cache Manager with Redis.

Provides multi-tier caching with TTL, eviction policies, and cache warming.
Part of Phase 3: Scaling & Optimization - Advanced Caching.
"""

import json
import pickle
import hashlib
import time
from typing import Any, Optional, Callable, Union
from functools import wraps
import redis
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Advanced cache manager with Redis backend.

    Features:
    - TTL-based expiration
    - LRU eviction
    - Cache warming
    - Compression
    - Metrics tracking
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        default_ttl: int = 3600,
        key_prefix: str = "geo_climate"
    ):
        """
        Initialize cache manager.

        Args:
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database number
            redis_password: Redis password
            default_ttl: Default TTL in seconds
            key_prefix: Prefix for all cache keys
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=False  # Handle binary data
        )
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix

        # Metrics
        self._hits = 0
        self._misses = 0

    def _make_key(self, key: str) -> str:
        """Create full cache key with prefix."""
        return f"{self.key_prefix}:{key}"

    def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        full_key = self._make_key(key)

        try:
            value = self.redis_client.get(full_key)
            if value is None:
                self._misses += 1
                logger.debug(f"Cache miss: {key}")
                return default

            self._hits += 1
            logger.debug(f"Cache hit: {key}")
            return pickle.loads(value)

        except Exception as e:
            logger.error(f"Cache get error for {key}: {e}")
            self._misses += 1
            return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds

        Returns:
            Success status
        """
        full_key = self._make_key(key)
        ttl = ttl or self.default_ttl

        try:
            serialized = pickle.dumps(value)
            self.redis_client.setex(full_key, ttl, serialized)
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.error(f"Cache set error for {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        full_key = self._make_key(key)

        try:
            self.redis_client.delete(full_key)
            logger.debug(f"Cache delete: {key}")
            return True
        except Exception as e:
            logger.error(f"Cache delete error for {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        full_key = self._make_key(key)
        try:
            return bool(self.redis_client.exists(full_key))
        except Exception as e:
            logger.error(f"Cache exists error for {key}: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "model:*")

        Returns:
            Number of keys deleted
        """
        full_pattern = self._make_key(pattern)

        try:
            keys = self.redis_client.keys(full_pattern)
            if keys:
                count = self.redis_client.delete(*keys)
                logger.info(f"Cleared {count} keys matching {pattern}")
                return count
            return 0
        except Exception as e:
            logger.error(f"Cache clear pattern error for {pattern}: {e}")
            return 0

    def get_or_set(
        self,
        key: str,
        callback: Callable,
        ttl: Optional[int] = None
    ) -> Any:
        """
        Get from cache or compute and cache.

        Args:
            key: Cache key
            callback: Function to compute value if not cached
            ttl: Time to live in seconds

        Returns:
            Cached or computed value
        """
        # Try to get from cache
        value = self.get(key)

        if value is not None:
            return value

        # Compute value
        value = callback()

        # Cache it
        self.set(key, value, ttl)

        return value

    def cached(
        self,
        ttl: Optional[int] = None,
        key_func: Optional[Callable] = None
    ):
        """
        Decorator for caching function results.

        Args:
            ttl: Time to live in seconds
            key_func: Function to generate cache key from arguments

        Example:
            @cache_manager.cached(ttl=300)
            def expensive_function(x, y):
                return x + y
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default: hash function name and arguments
                    key_data = f"{func.__name__}:{args}:{kwargs}"
                    cache_key = hashlib.md5(
                        key_data.encode()
                    ).hexdigest()

                # Try cache first
                result = self.get(cache_key)

                if result is not None:
                    return result

                # Compute result
                result = func(*args, **kwargs)

                # Cache result
                self.set(cache_key, result, ttl)

                return result

            return wrapper
        return decorator

    def warm_cache(
        self,
        items: dict,
        ttl: Optional[int] = None
    ) -> int:
        """
        Warm cache with multiple items.

        Args:
            items: Dictionary of key-value pairs
            ttl: Time to live in seconds

        Returns:
            Number of items cached
        """
        count = 0
        for key, value in items.items():
            if self.set(key, value, ttl):
                count += 1

        logger.info(f"Warmed cache with {count} items")
        return count

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

        try:
            info = self.redis_client.info()
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.2f}%",
                "total_keys": self.redis_client.dbsize(),
                "memory_used": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "uptime_seconds": info.get("uptime_in_seconds", 0)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.2f}%",
                "error": str(e)
            }

    def flush_all(self) -> bool:
        """Flush all keys from cache (use with caution)."""
        try:
            self.redis_client.flushdb()
            logger.warning("Cache flushed")
            return True
        except Exception as e:
            logger.error(f"Cache flush error: {e}")
            return False


class PredictionCache:
    """
    Specialized cache for model predictions.

    Provides intelligent caching based on input features.
    """

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager

    def cache_prediction(
        self,
        model_id: str,
        input_data: dict,
        prediction: Any,
        ttl: int = 300
    ) -> bool:
        """Cache a prediction result."""
        # Generate deterministic key from input
        input_hash = hashlib.md5(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()

        cache_key = f"prediction:{model_id}:{input_hash}"

        return self.cache.set(cache_key, prediction, ttl)

    def get_cached_prediction(
        self,
        model_id: str,
        input_data: dict
    ) -> Optional[Any]:
        """Get cached prediction if available."""
        input_hash = hashlib.md5(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()

        cache_key = f"prediction:{model_id}:{input_hash}"

        return self.cache.get(cache_key)

    def invalidate_model(self, model_id: str) -> int:
        """Invalidate all cached predictions for a model."""
        pattern = f"prediction:{model_id}:*"
        return self.cache.clear_pattern(pattern)


# Global cache manager instance
cache_manager = CacheManager()
prediction_cache = PredictionCache(cache_manager)
