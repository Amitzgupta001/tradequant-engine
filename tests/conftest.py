"""Shared pytest fixtures."""

import os

import pytest


def _set_test_env() -> None:
    os.environ.setdefault("DHAN_CLIENT_ID", "test-client-id")
    os.environ.setdefault("DHAN_ACCESS_TOKEN", "test-access-token")
    os.environ.setdefault("LOG_LEVEL", "DEBUG")
    os.environ.setdefault("STORAGE_PATH", "storage")
    os.environ.setdefault("APP_NAME", "tradequant-engine")


def pytest_configure() -> None:
    """Set environment variables before test collection."""
    _set_test_env()


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """Clear cached settings between tests."""
    from app.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
