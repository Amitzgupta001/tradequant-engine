"""Tests for backtest engine with synthetic data."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from app.backtest.engine import BacktestEngine
from app.domain.backtest import BacktestConfig, SignalAction
from app.domain.candle import Candle
from app.domain.enums.market import ExchangeSegment, InstrumentType, Timeframe
from app.domain.features import FeatureVector
from app.domain.instrument import Instrument
from app.domain.signal import Signal


def _candle(day: int, price: float) -> Candle:
    return Candle(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=day),
        open=price,
        high=price + 1,
        low=price - 1,
        close=price,
        volume=1000,
    )


def _feature(day: int) -> FeatureVector:
    return FeatureVector(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=day),
        close=100.0,
        volume=1000,
        return_1d=0.01,
        log_return_1d=0.009,
        high_low_range_pct=0.02,
        body_pct=0.005,
        volume_change_pct=0.05,
        ema_gap_pct=-0.01,
        rsi_14=50.0,
        macd_histogram=0.1,
        atr_pct=0.02,
        bb_width_pct=0.05,
        bb_position=0.5,
        vwap_gap_pct=0.001,
    )


@pytest.fixture
def instrument() -> Instrument:
    return Instrument(
        security_id="1333",
        exchange_segment=ExchangeSegment.NSE_EQ,
        instrument_type=InstrumentType.EQUITY,
    )


def test_backtest_engine_runs(instrument: Instrument) -> None:
    """Engine should produce trades and equity curve."""
    candles = [_candle(i, 100 + i) for i in range(20)]
    features = [_feature(i) for i in range(20)]

    strategy = Mock()
    buy = Signal(action=SignalAction.BUY, confidence=0.7)
    hold = Signal(action=SignalAction.HOLD, confidence=0.4)
    strategy.generate_signal.side_effect = [buy] * 5 + [hold] * 15

    result = BacktestEngine().run(
        instrument,
        Timeframe.DAILY,
        candles,
        features,
        strategy,
        config=BacktestConfig(initial_capital=100_000.0, commission_pct=0.0),
    )

    assert len(result.equity_curve) > 0
    assert result.metrics.final_equity > 0
