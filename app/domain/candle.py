"""OHLCV candle entity."""

from datetime import datetime

from pydantic import BaseModel, Field


class Candle(BaseModel):
    """Single OHLCV market candle."""

    timestamp: datetime
    open: float = Field(ge=0)
    high: float = Field(ge=0)
    low: float = Field(ge=0)
    close: float = Field(ge=0)
    volume: int = Field(ge=0)
    open_interest: int | None = None
