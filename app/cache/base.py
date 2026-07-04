"""Cache protocol."""

from typing import Protocol, TypeVar

T = TypeVar("T")


class Cache(Protocol[T]):
    """Generic cache contract."""

    def get(self, key: str) -> T | None:
        """Return cached value if present and not expired."""
        ...

    def set(self, key: str, value: T, ttl_seconds: int | None = None) -> None:
        """Store a value in cache."""
        ...

    def delete(self, key: str) -> None:
        """Remove a value from cache."""
        ...

    def clear(self) -> None:
        """Clear all cached values."""
        ...
