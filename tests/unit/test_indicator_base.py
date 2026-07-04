"""Tests for indicator math helpers."""

import pytest

from app.indicators.base import ema, rolling_std, sma


def test_sma_basic() -> None:
    """SMA should compute rolling mean after warmup."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = sma(values, period=3)
    assert result[:2] == [None, None]
    assert result[2] == pytest.approx(2.0)
    assert result[4] == pytest.approx(4.0)


def test_ema_basic() -> None:
    """EMA should seed with SMA and continue smoothing."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = ema(values, period=3)
    assert result[2] == pytest.approx(2.0)
    assert result[4] is not None
    assert result[4] > result[2]


def test_rolling_std_basic() -> None:
    """Rolling std should be None until enough samples exist."""
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    result = rolling_std(values, period=3)
    assert result[1] is None
    assert result[2] is not None
