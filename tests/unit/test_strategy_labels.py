"""Tests for strategy label generators."""

import pandas as pd
import pytest

from app.ml.labels.base import LabelConfig, LabelType, SignalLabel
from app.ml.labels.generator import apply_labels


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
            "open": [100 + index for index in range(10)],
            "high": [101 + index for index in range(10)],
            "low": [99 + index for index in range(10)],
            "close": [100 + index for index in range(10)],
            "volume": [1000] * 10,
        }
    )


def test_classification_labels(sample_frame: pd.DataFrame) -> None:
    config = LabelConfig(
        label_type=LabelType.CLASSIFICATION,
        forward_horizon_bars=2,
        regression_threshold=0.005,
    )
    labeled = apply_labels(sample_frame, config)
    assert labeled["label_classification"].iloc[0] == SignalLabel.BUY.value
    assert labeled["label"].iloc[-1] is None or pd.isna(labeled["label"].iloc[-1])


def test_regression_labels(sample_frame: pd.DataFrame) -> None:
    config = LabelConfig(label_type=LabelType.REGRESSION, forward_horizon_bars=2)
    labeled = apply_labels(sample_frame, config)
    assert labeled["label_regression"].iloc[0] == pytest.approx(0.02)


def test_triple_barrier_labels(sample_frame: pd.DataFrame) -> None:
    config = LabelConfig(
        label_type=LabelType.TRIPLE_BARRIER,
        take_profit_pct=0.01,
        stop_loss_pct=0.005,
        time_barrier_bars=3,
    )
    labeled = apply_labels(sample_frame, config)
    assert "label_barrier" in labeled.columns
    assert labeled["label_barrier"].iloc[0] in (-1, 0, 1)
