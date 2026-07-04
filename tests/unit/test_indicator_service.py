"""Tests for indicator service."""

from datetime import datetime, timezone
from pathlib import Path

from app.data.repositories.csv_historical_repository import CSVHistoricalRepository
from app.domain.candle import Candle
from app.domain.enums.market import ExchangeSegment, InstrumentType, Timeframe
from app.domain.historical import HistoricalResponse
from app.domain.instrument import Instrument
from app.services.indicator_service import IndicatorService


def _build_raw_data(tmp_path: Path) -> Instrument:
    instrument = Instrument(
        security_id="1333",
        exchange_segment=ExchangeSegment.NSE_EQ,
        instrument_type=InstrumentType.EQUITY,
    )
    candles = [
        Candle(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=100 + index,
            high=101 + index,
            low=99 + index,
            close=100 + index,
            volume=1000 + index,
        )
        for index in range(30)
    ]
    repository = CSVHistoricalRepository(base_path=tmp_path)
    repository.save(
        HistoricalResponse(
            instrument=instrument,
            timeframe=Timeframe.DAILY,
            candles=candles,
        )
    )
    return instrument


def test_indicator_service_compute_and_store(tmp_path: Path) -> None:
    """Service should load raw data and write processed indicators."""
    instrument = _build_raw_data(tmp_path)
    repository = CSVHistoricalRepository(base_path=tmp_path)
    service = IndicatorService(
        repository=repository,
        processed_path=tmp_path / "processed",
    )

    snapshots, path = service.compute_and_store(instrument, Timeframe.DAILY)

    assert len(snapshots) == 30
    assert path.exists()
    assert "processed" in str(path)
    assert snapshots[-1].rsi_14 is not None
