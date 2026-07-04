"""Computed technical indicator snapshot for a candle."""

from datetime import datetime

from pydantic import BaseModel


class IndicatorSnapshot(BaseModel):
    """All computed indicators at a single timestamp."""

    timestamp: datetime
    ema_20: float | None = None
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    atr_14: float | None = None
    vwap: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
