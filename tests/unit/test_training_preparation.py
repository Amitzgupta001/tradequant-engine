"""Tests for ML dataset preparation."""

from datetime import datetime, timezone

import pandas as pd

from app.domain.enums.market import Timeframe
from app.domain.features import FeatureVector
from app.domain.training import SetupType, TrainingConfig
from app.strategy.presets import REWARD_PCT_5M, RISK_PCT_5M
from app.ml.datasets.preparation import (
    FEATURE_COLUMNS,
    build_training_frame,
    classification_label,
    feature_matches_setup,
    features_to_rows,
    is_long_setup,
    swing_setup_label,
    training_config_for_timeframe,
)


def _feature(index: int, forward_1d: float | None, forward_5d: float | None) -> FeatureVector:
    return FeatureVector(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        close=100.0 + index,
        volume=1000,
        return_1d=0.01,
        return_3d=0.02,
        return_5d=0.03,
        log_return_1d=0.009,
        high_low_range_pct=0.02,
        body_pct=0.005,
        volume_change_pct=0.1,
        ema_gap_pct=-0.01,
        rsi_14=35.0,
        rsi_change_3d=2.0,
        macd_histogram=0.1,
        atr_pct=0.02,
        bb_width_pct=0.05,
        bb_position=0.1,
        vwap_gap_pct=0.001,
        return_lag_1=0.008,
        return_lag_2=0.007,
        return_lag_3=0.006,
        volatility_5d=0.012,
        volatility_10d=0.011,
        volume_ratio_5d=1.1,
        up_ratio_20d=0.5,
        trend_20d=0.03,
        forward_return_1d=forward_1d,
        forward_return_5d=forward_5d,
        forward_return_20b=0.025 if forward_5d is not None else None,
    )


def test_features_to_rows() -> None:
    """Rows should contain all feature columns and labels."""
    rows = features_to_rows([_feature(0, 0.02, 0.03), _feature(1, -0.01, -0.02)])
    assert len(rows) == 2
    assert "forward_return_5d" in rows[0]


def test_classification_label() -> None:
    """Label should be 1 for positive forward returns."""
    assert classification_label(0.01) == 1
    assert classification_label(-0.01) == 0


def test_swing_setup_label() -> None:
    """Swing labels should reflect setup success."""
    assert swing_setup_label(0.03, 0.02, SetupType.LONG) == 1
    assert swing_setup_label(0.01, 0.02, SetupType.LONG) == 0
    assert swing_setup_label(-0.03, 0.02, SetupType.SHORT) == 1


def test_build_training_frame_filters_setups() -> None:
    """Training frame should keep only configured setup rows."""
    features = [_feature(index, 0.01, 0.03 if index % 2 == 0 else -0.01) for index in range(30)]
    config = TrainingConfig(setup_type=SetupType.LONG, move_threshold=0.02)
    frame, _ = build_training_frame(features, config)
    assert len(frame) > 0
    assert "label" in frame.columns


def test_feature_matches_long_setup() -> None:
    """Long setup detector should match oversold features."""
    feature = _feature(0, 0.01, 0.03)
    assert feature_matches_setup(feature, SetupType.LONG) is True
    row = pd.Series({"rsi_14": 70.0, "bb_position": 0.9, "macd_histogram": -0.1})
    assert is_long_setup(row) is False


def test_intraday_training_defaults() -> None:
    """Intraday timeframes should use bar-based swing defaults."""
    config_15 = training_config_for_timeframe(Timeframe.MIN_15)
    assert config_15.forward_horizon_bars == 20
    assert config_15.move_threshold == RISK_PCT_5M * 4

    config_5 = training_config_for_timeframe(Timeframe.MIN_5)
    assert config_5.forward_horizon_bars == 20
    assert config_5.move_threshold == REWARD_PCT_5M
