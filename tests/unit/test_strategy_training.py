"""Tests for strategy dataset builder and trainer."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from app.domain.candle import Candle
from app.domain.enums.market import ExchangeSegment, InstrumentType, Timeframe
from app.domain.historical import HistoricalResponse
from app.domain.instrument import Instrument
from app.domain.training import TrainingConfig
from app.ml.datasets.strategy_builder import StrategyDatasetBuilder
from app.ml.experiments.store import ExperimentStore
from app.ml.labels.base import LabelConfig
from app.ml.registry.strategy_registry import StrategyModelRegistry
from app.ml.trainer.strategy_trainer import StrategyTrainer
from app.strategies.registry import get_strategy


class InMemoryRepository:
    """Minimal repository for strategy dataset tests."""

    def __init__(self, candles: list[Candle]) -> None:
        self._candles = candles

    def load(self, instrument: Instrument, timeframe: Timeframe) -> HistoricalResponse:
        return HistoricalResponse(
            instrument=instrument,
            timeframe=timeframe,
            candles=self._candles,
        )


def _candles(count: int = 120) -> list[Candle]:
    candles: list[Candle] = []
    for index in range(count):
        close = 100 + index * 0.2 + (index % 7) * 0.1
        candles.append(
            Candle(
                timestamp=datetime(2024, 1, 1, 9, 15, tzinfo=timezone.utc)
                + timedelta(minutes=5 * index),
                open=close - 0.2,
                high=close + 0.5,
                low=close - 0.5,
                close=close,
                volume=1000 + index * 10,
            )
        )
    return candles


@pytest.fixture
def instrument() -> Instrument:
    return Instrument(
        security_id="TEST001",
        exchange_segment=ExchangeSegment.NSE_EQ,
        instrument_type=InstrumentType.EQUITY,
        symbol="TEST",
    )


def test_build_and_train_strategy_dataset(tmp_path: Path, instrument: Instrument) -> None:
    repository = InMemoryRepository(_candles())
    builder = StrategyDatasetBuilder(repository, tmp_path)
    frame, metadata = builder.build("ema_crossover", instrument, Timeframe.MIN_5)

    assert len(frame) == 120
    assert metadata.strategy_id == "ema_crossover"
    assert metadata.row_count == 120

    loaded_frame, loaded_metadata = builder.load(
        "ema_crossover",
        instrument.exchange_segment.value,
        instrument.security_id,
        Timeframe.MIN_5.value,
    )
    assert len(loaded_frame) == len(frame)
    assert loaded_metadata.strategy_id == metadata.strategy_id

    registry = StrategyModelRegistry(tmp_path)
    trainer = StrategyTrainer(registry, ExperimentStore(tmp_path))
    strategy = get_strategy("ema_crossover")
    model_metadata, importance = trainer.train(
        strategy,
        frame,
        metadata,
        config=TrainingConfig(min_train_rows=20, n_estimators=20, early_stopping_rounds=5),
    )

    assert model_metadata.version == 1
    assert registry.latest_version("ema_crossover") == 1
    assert importance.lightgbm_gain
    assert ExperimentStore(tmp_path).list_experiments("ema_crossover")
