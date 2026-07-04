"""Tests for Dhan response mapping."""

from datetime import date

import pytest

from app.brokers.dhan.mapper import map_historical_response
from app.brokers.exceptions import DhanAPIError
from app.domain.enums.market import ExchangeSegment, InstrumentType, Timeframe
from app.domain.historical import HistoricalRequest
from app.domain.instrument import Instrument


def _build_request() -> HistoricalRequest:
    return HistoricalRequest(
        instrument=Instrument(
            security_id="1333",
            exchange_segment=ExchangeSegment.NSE_EQ,
            instrument_type=InstrumentType.EQUITY,
            symbol="RELIANCE",
        ),
        from_date=date(2024, 1, 1),
        to_date=date(2024, 1, 31),
        timeframe=Timeframe.DAILY,
    )


def test_map_historical_response_success() -> None:
    """Mapper should zip parallel arrays into candle models."""
    request = _build_request()
    sdk_response = {
        "status": "success",
        "remarks": "",
        "data": {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000, 1100],
            "timestamp": [1704067200, 1704153600],
            "open_interest": [None, None],
        },
    }

    response = map_historical_response(request, sdk_response)

    assert len(response.candles) == 2
    assert response.candles[0].open == 100.0
    assert response.candles[0].volume == 1000
    assert response.candles[0].timestamp.tzinfo is not None
    assert response.source == "dhan"


def test_map_historical_response_failure() -> None:
    """Mapper should raise when SDK response status is failure."""
    request = _build_request()
    sdk_response = {"status": "failure", "remarks": "Invalid token", "data": ""}

    with pytest.raises(DhanAPIError):
        map_historical_response(request, sdk_response)
