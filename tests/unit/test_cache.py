"""Tests for in-memory cache."""

from app.cache.memory import InMemoryCache


def test_cache_set_and_get() -> None:
    """Cache should store and retrieve values."""
    cache = InMemoryCache[str](default_ttl_seconds=60)
    cache.set("key", "value")
    assert cache.get("key") == "value"


def test_cache_miss() -> None:
    """Cache should return None for missing keys."""
    cache = InMemoryCache[str]()
    assert cache.get("missing") is None
