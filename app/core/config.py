"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration for the trading platform."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="tradequant-engine", alias="APP_NAME")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    dhan_client_id: str = Field(alias="DHAN_CLIENT_ID")
    dhan_access_token: str = Field(alias="DHAN_ACCESS_TOKEN")
    storage_path: Path = Field(default=Path("storage"), alias="STORAGE_PATH")
    dhan_api_base_url: str = Field(
        default="https://api.dhan.co/v2",
        alias="DHAN_API_BASE_URL",
    )
    cache_ttl_seconds: int = Field(default=300, alias="CACHE_TTL_SECONDS")
    batch_sleep_seconds: float = Field(
        default=3.0,
        ge=0.0,
        alias="BATCH_SLEEP_SECONDS",
        description="Pause between symbols during batch download (avoids API overload)",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
