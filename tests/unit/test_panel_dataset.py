"""Tests for panel feature loading."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.domain.enums.market import ExchangeSegment, InstrumentType, Timeframe
from app.domain.features import FeatureVector
from app.domain.instrument import Instrument
from app.domain.training import SetupType, TrainingConfig
from app.ml.datasets.panel import build_panel_training_frame, load_panel_features
from app.ml.feature_store.repository import CSVFeatureRepository


def _feature(index: int) -> FeatureVector:
    return FeatureVector(
        timestamp=datetime(2022, 1, 1, tzinfo=timezone.utc) + timedelta(days=index),
        close=100.0,
        volume=1000,
        return_1d=0.001,
        return_3d=0.003,
        return_5d=0.005,
        log_return_1d=0.0009,
        high_low_range_pct=0.02,
        body_pct=0.003,
        volume_change_pct=0.05,
        ema_gap_pct=-0.01,
        rsi_14=35.0,
        rsi_change_3d=1.5,
        macd_histogram=0.05,
        atr_pct=0.015,
        bb_width_pct=0.04,
        bb_position=0.1,
        vwap_gap_pct=0.002,
        return_lag_1=0.001,
        return_lag_2=0.002,
        return_lag_3=0.003,
        volatility_5d=0.012,
        volatility_10d=0.011,
        volume_ratio_5d=1.05,
        up_ratio_20d=0.52,
        trend_20d=0.04,
        forward_return_1d=0.01,
        forward_return_5d=0.02,
        forward_return_20b=0.025,
    )


@pytest.fixture
def instruments() -> list[Instrument]:
    return [
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


def test_load_panel_features_merges_instruments(
    tmp_path: Path,
    instruments: list[Instrument],
) -> None:
    repo = CSVFeatureRepository(base_path=tmp_path)
    repo.save(instruments[0], Timeframe.DAILY, [_feature(0), _feature(1)])
    repo.save(instruments[1], Timeframe.DAILY, [_feature(2)])

    merged, loaded_ids = load_panel_features(repo, instruments, Timeframe.DAILY)

    assert len(merged) == 3
    assert loaded_ids == ["111", "222"]


def test_load_panel_features_skips_missing(tmp_path: Path, instruments: list[Instrument]) -> None:
    repo = CSVFeatureRepository(base_path=tmp_path)
    repo.save(instruments[0], Timeframe.DAILY, [_feature(0)])

    merged, loaded_ids = load_panel_features(repo, instruments, Timeframe.DAILY)

    assert len(merged) == 1
    assert loaded_ids == ["111"]


def test_build_panel_training_frame_reads_csv_incrementally(
    tmp_path: Path,
    instruments: list[Instrument],
) -> None:
    repo = CSVFeatureRepository(base_path=tmp_path)
    repo.save(instruments[0], Timeframe.DAILY, [_feature(0), _feature(1)])
    repo.save(instruments[1], Timeframe.DAILY, [_feature(2)])

    config = TrainingConfig(setup_type=SetupType.LONG, move_threshold=0.001)
    frame, _, loaded_ids = build_panel_training_frame(
        repo,
        instruments,
        Timeframe.DAILY,
        config,
    )

    assert loaded_ids == ["111", "222"]
    assert len(frame) >= 1
    assert "label" in frame.columns
