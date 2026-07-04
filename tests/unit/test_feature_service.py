"""Tests for feature repository and service."""

from datetime import datetime, timezone
from pathlib import Path

from app.data.repositories.csv_historical_repository import CSVHistoricalRepository
from app.domain.candle import Candle
from app.domain.enums.market import ExchangeSegment, InstrumentType, Timeframe
from app.domain.features import FeatureVector
from app.domain.historical import HistoricalResponse
from app.domain.instrument import Instrument
from app.ml.datasets.builder import FeatureDatasetBuilder
from app.ml.feature_store.repository import CSVFeatureRepository
from app.services.feature_service import FeatureService


def _seed_raw_data(tmp_path: Path) -> Instrument:
    instrument = Instrument(
        security_id="1333",
        exchange_segment=ExchangeSegment.NSE_EQ,
        instrument_type=InstrumentType.EQUITY,
    )
    candles = [
        Candle(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=100 + index,
            high=102 + index,
            low=98 + index,
            close=100 + index,
            volume=1000 + index * 10,
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


def test_feature_repository_round_trip(tmp_path: Path) -> None:
    """Feature repository should save and load feature vectors."""
    instrument = Instrument(
        security_id="1333",
        exchange_segment=ExchangeSegment.NSE_EQ,
        instrument_type=InstrumentType.EQUITY,
    )
    repository = CSVFeatureRepository(base_path=tmp_path / "features")
    features = [
        FeatureVector(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            close=100.0,
            volume=1000,
            return_1d=0.01,
            rsi_14=50.0,
        )
    ]

    path = repository.save(instrument, Timeframe.DAILY, features)
    loaded = repository.load(instrument, Timeframe.DAILY)

    assert path.exists()
    assert len(loaded) == 1
    assert loaded[0].return_1d == 0.01


def test_feature_service_build_and_store(tmp_path: Path) -> None:
    """Feature service should build features from raw data and persist them."""
    instrument = _seed_raw_data(tmp_path)
    historical_repository = CSVHistoricalRepository(base_path=tmp_path)
    feature_repository = CSVFeatureRepository(base_path=tmp_path / "features")
    builder = FeatureDatasetBuilder(repository=historical_repository)
    service = FeatureService(
        repository=historical_repository,
        feature_repository=feature_repository,
        builder=builder,
    )

    features, path = service.build_and_store(instrument, Timeframe.DAILY)

    assert len(features) == 30
    assert path.exists()
    assert features[1].return_1d is not None
    assert features[-1].forward_return_1d is None
