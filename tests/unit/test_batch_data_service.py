"""Tests for batch universe download service."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from app.data.universe.registry import Universe
from app.domain.enums.market import ExchangeSegment, InstrumentType, Timeframe
from app.domain.instrument import Instrument
from app.services.batch_data_service import BatchDataService


@pytest.fixture
def universe() -> Universe:
    instruments = [
        Instrument(
            security_id="111",
            exchange_segment=ExchangeSegment.NSE_EQ,
            instrument_type=InstrumentType.EQUITY,
            symbol="AAA",
        ),
        Instrument(
            security_id="222",
            exchange_segment=ExchangeSegment.NSE_EQ,
            instrument_type=InstrumentType.EQUITY,
            symbol="BBB",
        ),
    ]
    return Universe(universe_id="test", name="Test", instruments=instruments)


def test_download_universe_pauses_between_symbols(universe: Universe) -> None:
    training = MagicMock()
    training.has_stored_features.return_value = False
    training.prepare_data.return_value = (100, 80)
    service = BatchDataService(training)

    with patch("app.services.batch_data_service.time.sleep") as sleep_mock:
        results = service.download_universe(
            universe,
            Timeframe.MIN_5,
            days=90,
            to_date=date(2026, 1, 1),
            sleep_seconds=2.5,
        )

    assert len(results) == 2
    assert all(result.status == "ok" for result in results)
    sleep_mock.assert_called_once_with(2.5)


def test_download_universe_skips_pause_when_sleep_zero(universe: Universe) -> None:
    training = MagicMock()
    training.has_stored_features.return_value = False
    training.prepare_data.return_value = (100, 80)
    service = BatchDataService(training)

    with patch("app.services.batch_data_service.time.sleep") as sleep_mock:
        service.download_universe(
            universe,
            Timeframe.MIN_5,
            days=90,
            to_date=date(2026, 1, 1),
            sleep_seconds=0,
        )

    sleep_mock.assert_not_called()
