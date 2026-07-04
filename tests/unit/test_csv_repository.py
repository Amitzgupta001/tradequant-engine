"""Tests for CSV historical repository."""

from datetime import datetime, timezone
from pathlib import Path

from app.data.repositories.csv_historical_repository import CSVHistoricalRepository
from app.domain.candle import Candle
from app.domain.enums.market import ExchangeSegment, InstrumentType, Timeframe
from app.domain.historical import HistoricalResponse
from app.domain.instrument import Instrument


def _build_response() -> HistoricalResponse:
    instrument = Instrument(
        security_id="1333",
        exchange_segment=ExchangeSegment.NSE_EQ,
        instrument_type=InstrumentType.EQUITY,
    )
    candles = [
        Candle(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000,
            open_interest=None,
        )
    ]
    return HistoricalResponse(
        instrument=instrument,
        timeframe=Timeframe.DAILY,
        candles=candles,
    )


def test_csv_repository_round_trip(tmp_path: Path) -> None:
    """Repository should save and load candles without data loss."""
    repository = CSVHistoricalRepository(base_path=tmp_path)
    response = _build_response()

    saved_path = repository.save(response)
    loaded = repository.load(response.instrument, response.timeframe)

    assert "raw" in str(saved_path)
    assert saved_path.exists()
    assert len(loaded.candles) == 1
    assert loaded.candles[0].close == 100.5
    assert loaded.source == "csv"


def test_csv_repository_skip_existing(tmp_path: Path) -> None:
    """Repository should not overwrite when overwrite=False."""
    repository = CSVHistoricalRepository(base_path=tmp_path)
    response = _build_response()

    first_path = repository.save(response, overwrite=True)
    second_path = repository.save(response, overwrite=False)

    assert first_path == second_path
