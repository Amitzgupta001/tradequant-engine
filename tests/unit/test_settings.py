"""Tests for application settings."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings


def test_settings_load_from_env() -> None:
    """Settings should load required Dhan credentials from environment."""
    settings = get_settings()
    assert settings.dhan_client_id == "test-client-id"
    assert settings.dhan_access_token == "test-access-token"
    assert settings.app_name == "tradequant-engine"


def test_settings_missing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should fail when required credentials are missing."""
    monkeypatch.delenv("DHAN_CLIENT_ID", raising=False)
    monkeypatch.delenv("DHAN_ACCESS_TOKEN", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
