"""Historical data value objects."""

from datetime import date

from pydantic import BaseModel, Field, model_validator

from app.domain.candle import Candle
from app.domain.enums.market import Timeframe
from app.domain.instrument import Instrument


class HistoricalRequest(BaseModel):
    """Request parameters for historical OHLCV data."""

    instrument: Instrument
    from_date: date
    to_date: date
    timeframe: Timeframe = Timeframe.DAILY
    include_oi: bool = False
    expiry_code: int = Field(default=0, ge=0, le=3)

    @model_validator(mode="after")
    def validate_date_range(self) -> "HistoricalRequest":
        """Ensure from_date is not after to_date."""
        if self.from_date > self.to_date:
            msg = "from_date must be on or before to_date"
            raise ValueError(msg)
        return self


class HistoricalResponse(BaseModel):
    """Normalized historical OHLCV response."""

    instrument: Instrument
    timeframe: Timeframe
    candles: list[Candle]
    source: str = "dhan"
