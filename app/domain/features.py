"""ML-ready feature vector for a single candle."""

from datetime import datetime

from pydantic import BaseModel, Field


class FeatureVector(BaseModel):
    """Engineered features derived from OHLCV and indicators."""

    timestamp: datetime
    close: float = Field(ge=0)
    volume: int = Field(ge=0)

    return_1d: float | None = None
    return_3d: float | None = None
    return_5d: float | None = None
    log_return_1d: float | None = None
    high_low_range_pct: float | None = None
    body_pct: float | None = None
    volume_change_pct: float | None = None

    ema_gap_pct: float | None = None
    rsi_14: float | None = None
    rsi_change_3d: float | None = None
    macd_histogram: float | None = None
    atr_pct: float | None = None
    bb_width_pct: float | None = None
    bb_position: float | None = None
    vwap_gap_pct: float | None = None

    return_lag_1: float | None = None
    return_lag_2: float | None = None
    return_lag_3: float | None = None
    volatility_5d: float | None = None
    volatility_10d: float | None = None
    volume_ratio_5d: float | None = None
    up_ratio_20d: float | None = None
    trend_20d: float | None = None

    forward_return_1d: float | None = None
    forward_return_5d: float | None = None
    forward_return_20b: float | None = None
