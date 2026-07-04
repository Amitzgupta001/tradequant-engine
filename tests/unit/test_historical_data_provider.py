"""Tests for historical data provider."""

from datetime import date
from unittest.mock import Mock

from app.cache.memory import InMemoryCache
from app.data.providers.historical_data_provider import HistoricalDataProvider
from app.domain.candle import Candle
from app.domain.enums.market import ExchangeSegment, InstrumentType, Timeframe
from app.domain.historical import HistoricalRequest, HistoricalResponse
from app.domain.instrument import Instrument


def _build_request(timeframe: Timeframe = Timeframe.DAILY) -> HistoricalRequest:
    return HistoricalRequest(
        instrument=Instrument(
            security_id="1333",
            exchange_segment=ExchangeSegment.NSE_EQ,
            instrument_type=InstrumentType.EQUITY,
        ),
        from_date=date(2024, 1, 1),
        to_date=date(2024, 1, 31),
        timeframe=timeframe,
    )


def _build_response(request: HistoricalRequest) -> HistoricalResponse:
    return HistoricalResponse(
        instrument=request.instrument,
        timeframe=request.timeframe,
        candles=[
            Candle(
                timestamp=request.from_date,
                open=1.0,
                high=2.0,
                low=0.5,
                close=1.5,
                volume=100,
            )
        ],
    )


def test_provider_delegates_to_broker() -> None:
    """Provider should delegate daily requests directly to broker."""
    request = _build_request()
    broker = Mock()
    broker.get_historical_data.return_value = _build_response(request)
    provider = HistoricalDataProvider(broker=broker)

    response = provider.fetch(request)

    broker.get_historical_data.assert_called_once_with(request)
    assert len(response.candles) == 1


def test_provider_chunks_intraday_requests() -> None:
    """Provider should split long intraday ranges into multiple broker calls."""
    request = _build_request(timeframe=Timeframe.MIN_5)
    request = request.model_copy(
        update={
            "from_date": date(2024, 1, 1),
            "to_date": date(2024, 6, 1),
        }
    )
    broker = Mock()
    broker.get_historical_data.side_effect = lambda req: _build_response(req)
    provider = HistoricalDataProvider(broker=broker)

    response = provider.fetch(request)

    assert broker.get_historical_data.call_count >= 2
    assert len(response.candles) >= 2


def test_provider_uses_cache() -> None:
    """Provider should return cached response on second fetch."""
    request = _build_request()
    broker = Mock()
    broker.get_historical_data.return_value = _build_response(request)
    cache = InMemoryCache[HistoricalResponse]()
    provider = HistoricalDataProvider(broker=broker, cache=cache)

    provider.fetch(request)
    provider.fetch(request)

    broker.get_historical_data.assert_called_once()
