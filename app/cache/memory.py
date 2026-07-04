"""In-memory TTL cache (Phase 1). Redis planned for production."""

import time
from dataclasses import dataclass
from typing import Generic, TypeVar

from loguru import logger

T = TypeVar("T")


@dataclass
class _CacheEntry(Generic[T]):
    value: T
    expires_at: float


class InMemoryCache(Generic[T]):
    """Thread-unsafe in-memory cache with TTL support."""

    def __init__(self, default_ttl_seconds: int = 300) -> None:
        self._default_ttl_seconds = default_ttl_seconds
        self._store: dict[str, _CacheEntry[T]] = {}

    def get(self, key: str) -> T | None:
        """Return cached value if present and not expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.time() >= entry.expires_at:
            del self._store[key]
            return None
        return entry.value

    def set(self, key: str, value: T, ttl_seconds: int | None = None) -> None:
        """Store a value in cache."""
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        self._store[key] = _CacheEntry(value=value, expires_at=time.time() + ttl)
        logger.debug("Cached key={} ttl={}s", key, ttl)

    def delete(self, key: str) -> None:
        """Remove a value from cache."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Clear all cached values."""
        self._store.clear()
