"""Tests for feature engineering engine."""

from datetime import datetime, timezone

import pytest

from app.domain.candle import Candle
from app.domain.indicators import IndicatorSnapshot
from app.ml.feature_store.engine import FeatureEngine


def _candle(day: int, close: float, volume: int = 1000) -> Candle:
    return Candle(
        timestamp=datetime(2024, 1, day, tzinfo=timezone.utc),
        open=close - 1,
        high=close + 2,
        low=close - 2,
        close=close,
        volume=volume,
    )


def _indicator(close: float) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ema_20=close - 5,
        rsi_14=55.0,
        macd_histogram=0.5,
        atr_14=close * 0.02,
        vwap=close - 1,
        bb_upper=close + 10,
        bb_middle=close,
        bb_lower=close - 10,
    )


def test_feature_engine_builds_vectors() -> None:
    """Engine should combine candle and indicator data into features."""
    candles = [_candle(1, 100.0), _candle(2, 105.0, 1200)]
    indicators = [_indicator(100.0), _indicator(105.0)]
    indicators[0].timestamp = candles[0].timestamp
    indicators[1].timestamp = candles[1].timestamp

    features = FeatureEngine().build(candles, indicators)

    assert len(features) == 2
    assert features[0].return_1d is None
    assert features[1].return_1d == pytest.approx(0.05)
    assert features[0].forward_return_1d == pytest.approx(0.05)
    assert features[1].forward_return_1d is None
    assert features[1].bb_position == pytest.approx(0.5)
    assert features[1].volume_change_pct == pytest.approx(0.2)


def test_feature_engine_requires_equal_lengths() -> None:
    """Engine should reject mismatched input lengths."""
    with pytest.raises(ValueError):
        FeatureEngine().build([_candle(1, 100.0)], [])
