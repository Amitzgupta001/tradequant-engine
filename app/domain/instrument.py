"""Instrument entity."""

from pydantic import BaseModel, Field

from app.domain.enums.market import ExchangeSegment, InstrumentType


class Instrument(BaseModel):
    """Tradable instrument identifier."""

    security_id: str = Field(min_length=1)
    exchange_segment: ExchangeSegment
    instrument_type: InstrumentType
    symbol: str | None = None
