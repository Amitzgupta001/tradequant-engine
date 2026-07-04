"""Tests for trading strategy implementations."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.strategies import list_strategy_ids
from app.strategies.dataframe import candles_to_dataframe
from app.domain.candle import Candle
from app.domain.indicators import IndicatorSnapshot
from app.strategies.registry import get_strategy


def _sample_candles(count: int = 60) -> list[Candle]:
    candles: list[Candle] = []
    for index in range(count):
        price = 100 + index * 0.5
        candles.append(
            Candle(
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=index),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price + (0.2 if index % 5 == 0 else -0.1),
                volume=1000 + index,
            )
        )
    return candles


def _sample_indicators(candles: list[Candle]) -> list[IndicatorSnapshot]:
    snapshots: list[IndicatorSnapshot] = []
    for index, candle in enumerate(candles):
        snapshots.append(
            IndicatorSnapshot(
                timestamp=candle.timestamp,
                ema_20=candle.close - 0.5,
                rsi_14=35.0 if index % 4 == 0 else 55.0,
                macd=0.1,
                macd_signal=0.05,
                macd_histogram=0.05 if index % 2 == 0 else -0.05,
                atr_14=1.5,
                vwap=candle.close - 0.2,
                bb_upper=candle.close + 2,
                bb_middle=candle.close,
                bb_lower=candle.close - 2,
            )
        )
    return snapshots


@pytest.fixture
def market_frame() -> pd.DataFrame:
    candles = _sample_candles()
    return candles_to_dataframe(candles, _sample_indicators(candles))


def test_all_builtin_strategies_registered() -> None:
    expected = {
        "ema_crossover",
        "ema_pullback",
        "vwap_breakout",
        "orb",
        "cpr_breakout",
        "supertrend",
        "macd_momentum",
        "rsi_reversal",
        "bollinger_mean_reversion",
        "price_action_breakout",
        "breakout",
        "breakdown",
    }
    assert expected.issubset(set(list_strategy_ids()))


@pytest.mark.parametrize("strategy_id", list_strategy_ids())
def test_strategy_pipeline(strategy_id: str, market_frame: pd.DataFrame) -> None:
    strategy = get_strategy(strategy_id)
    assert strategy.name
    assert strategy.description
    assert strategy.required_indicators()

    featured = strategy.generate_features(market_frame)
    signaled = strategy.generate_signals(featured)
    labeled = strategy.generate_labels(signaled)

    assert "strategy_signal" in labeled.columns
    assert "label" in labeled.columns
    assert len(strategy.feature_columns(labeled)) > 0
