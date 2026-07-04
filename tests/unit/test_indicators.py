"""Tests for individual indicators."""

from datetime import datetime, timezone

import pytest

from app.domain.candle import Candle
from app.indicators.atr import compute_atr
from app.indicators.bollinger import compute_bollinger_bands
from app.indicators.ema import compute_ema
from app.indicators.engine import IndicatorEngine
from app.indicators.macd import compute_macd
from app.indicators.rsi import compute_rsi
from app.indicators.vwap import compute_vwap


def _sample_closes(count: int = 30) -> list[float]:
    return [float(100 + index) for index in range(count)]


def test_compute_ema() -> None:
    """EMA should return aligned series with warmup None values."""
    result = compute_ema(_sample_closes(), period=5)
    assert len(result.values) == 30
    assert result.values[3] is None
    assert result.values[4] is not None


def test_compute_rsi() -> None:
    """RSI should stay within 0-100 range."""
    result = compute_rsi(_sample_closes(), period=14)
    valid = [value for value in result.values if value is not None]
    assert valid
    assert all(0 <= value <= 100 for value in valid)


def test_compute_macd() -> None:
    """MACD should produce line, signal, and histogram arrays."""
    result = compute_macd(_sample_closes(60))
    assert len(result.macd) == 60
    assert any(value is not None for value in result.macd)
    assert any(value is not None for value in result.histogram)


def test_compute_atr() -> None:
    """ATR should compute from highs, lows, and closes."""
    closes = _sample_closes()
    highs = [value + 1 for value in closes]
    lows = [value - 1 for value in closes]
    result = compute_atr(highs, lows, closes, period=14)
    assert len(result.values) == 30
    assert result.values[13] is not None


def test_compute_vwap_resets_by_day() -> None:
    """VWAP should reset on a new trading date."""
    candles = [
        Candle(
            timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1000,
        ),
        Candle(
            timestamp=datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
            open=200,
            high=201,
            low=199,
            close=200,
            volume=1000,
        ),
    ]
    result = compute_vwap(candles)
    assert result.values[0] == pytest.approx(100.0)
    assert result.values[1] == pytest.approx(200.0)


def test_compute_bollinger_bands() -> None:
    """Bollinger bands should bracket the moving average."""
    closes = _sample_closes(25)
    result = compute_bollinger_bands(closes, period=20, std_dev=2.0)
    index = 24
    assert result.middle[index] is not None
    assert result.upper[index] > result.middle[index]
    assert result.lower[index] < result.middle[index]


def test_indicator_engine() -> None:
    """Engine should compute all indicators for candle input."""
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
    snapshots = IndicatorEngine().compute(candles)
    assert len(snapshots) == 30
    assert snapshots[-1].ema_20 is not None
    assert snapshots[-1].rsi_14 is not None
