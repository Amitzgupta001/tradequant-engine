"""Tests for LightGBM training pipeline."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.domain.enums.market import ExchangeSegment, InstrumentType, Timeframe
from app.domain.features import FeatureVector
from app.domain.instrument import Instrument
from app.domain.training import SetupType, TrainingConfig, TrainingTask
from app.ml.registry.model_registry import ModelRegistry
from app.ml.trainer.lightgbm_trainer import LightGBMTrainer
from app.ml.datasets.preparation import build_training_frame


def _synthetic_feature(index: int, forward_1d: float | None, forward_5d: float | None) -> FeatureVector:
    rsi = 35.0 if index % 3 == 0 else 55.0
    bb_position = 0.1 if index % 3 == 0 else 0.5
    return FeatureVector(
        timestamp=datetime(2022, 1, 1, tzinfo=timezone.utc) + timedelta(days=index),
        close=100.0 + index * 0.1,
        volume=1000 + index,
        return_1d=0.001,
        return_3d=0.003,
        return_5d=0.005,
        log_return_1d=0.0009,
        high_low_range_pct=0.02,
        body_pct=0.003,
        volume_change_pct=0.05,
        ema_gap_pct=-0.01,
        rsi_14=rsi,
        rsi_change_3d=1.5,
        macd_histogram=0.05,
        atr_pct=0.015,
        bb_width_pct=0.04,
        bb_position=bb_position,
        vwap_gap_pct=0.002,
        return_lag_1=0.001,
        return_lag_2=0.002,
        return_lag_3=0.003,
        volatility_5d=0.012,
        volatility_10d=0.011,
        volume_ratio_5d=1.05,
        up_ratio_20d=0.52,
        trend_20d=0.04,
        forward_return_1d=forward_1d,
        forward_return_5d=forward_5d,
        forward_return_20b=0.025 if forward_5d is not None else None,
    )


def _synthetic_features(count: int = 200) -> list[FeatureVector]:
    features: list[FeatureVector] = []
    for index in range(count):
        forward_5d = 0.03 if index % 2 == 0 else -0.01
        forward_1d = 0.01 if index % 2 == 0 else -0.005
        features.append(
            _synthetic_feature(
                index,
                forward_1d if index < count - 1 else None,
                forward_5d if index < count - 6 else None,
            )
        )
    return features


@pytest.fixture
def instrument() -> Instrument:
    return Instrument(
        security_id="1333",
        exchange_segment=ExchangeSegment.NSE_EQ,
        instrument_type=InstrumentType.EQUITY,
    )


def test_lightgbm_trainer_swing_setup(tmp_path: Path, instrument: Instrument) -> None:
    """Trainer should fit swing setup model and persist artifacts."""
    registry = ModelRegistry(base_path=tmp_path / "models")
    trainer = LightGBMTrainer(registry=registry)
    config = TrainingConfig(
        task=TrainingTask.CLASSIFICATION,
        setup_type=SetupType.LONG,
        move_threshold=0.02,
        test_size=0.2,
        n_estimators=50,
        min_train_rows=50,
    )

    result = trainer.train(
        instrument,
        Timeframe.DAILY,
        _synthetic_features(),
        config=config,
    )

    assert Path(result.model_path).exists()
    assert Path(result.metadata_path).exists()
    assert result.metrics.test_rows > 0
    assert result.metrics.accuracy is not None
    assert result.metrics.setup_rows > 0


def test_lightgbm_trainer_panel(tmp_path: Path) -> None:
    """Trainer should fit panel model and persist to panels/ directory."""
    registry = ModelRegistry(base_path=tmp_path / "models")
    trainer = LightGBMTrainer(registry=registry)
    config = TrainingConfig(
        task=TrainingTask.CLASSIFICATION,
        setup_type=SetupType.LONG,
        move_threshold=0.02,
        test_size=0.2,
        n_estimators=50,
        min_train_rows=50,
    )
    frame, label_column = build_training_frame(_synthetic_features(), config)

    result = trainer.train_panel(
        "nifty50",
        ExchangeSegment.NSE_EQ.value,
        Timeframe.DAILY,
        frame,
        label_column,
        config=config,
        constituent_count=50,
    )

    assert Path(result.model_path).exists()
    assert "panels" in result.model_path
    assert "nifty50" in result.model_path
    metadata = registry.load_metadata(
        ExchangeSegment.NSE_EQ.value,
        "PANEL_NIFTY50",
        Timeframe.DAILY.value,
        universe_id="nifty50",
    )
    assert metadata.universe_id == "nifty50"
    assert metadata.constituent_count == 50
