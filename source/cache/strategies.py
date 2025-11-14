"""Cache strategies and policies."""

from enum import Enum
from collections import OrderedDict
from typing import Any, Optional
from datetime import datetime, timedelta


class CacheStrategy(str, Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


class LRUCache:
    """In-memory LRU cache (fallback when Redis unavailable)."""

    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def delete(self, key: str) -> bool:
        return self.cache.pop(key, None) is not None


class TTLCache:
    """In-memory TTL cache."""

    def __init__(self, default_ttl: int = 3600):
        self.cache = {}
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None

        value, expiry = self.cache[key]
        if datetime.now() > expiry:
            del self.cache[key]
            return None

        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        ttl = ttl or self.default_ttl
        expiry = datetime.now() + timedelta(seconds=ttl)
        self.cache[key] = (value, expiry)

    def delete(self, key: str) -> bool:
        return self.cache.pop(key, None) is not None
