"""Tests for cross validation, ensemble, regime, and recommendation."""

import pandas as pd

from app.ml.ensembles.engine import EnsembleEngine, EnsembleMethod, StrategyPrediction
from app.ml.evaluator.cross_validation import SplitConfig, rolling_window_splits, walk_forward_splits
from app.ml.inference.recommendation import StrategyRecommendationEngine
from app.ml.inference.regime import MarketRegime, RegimeClassifier


def _frame(rows: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "close": [100 + index * 0.1 for index in range(rows)],
            "ema_20": [100 + index * 0.08 for index in range(rows)],
            "strategy_signal": [0] * rows,
        }
    )


def test_walk_forward_splits() -> None:
    frame = _frame()
    splits = list(walk_forward_splits(frame, SplitConfig(train_window=50, step_size=20)))
    assert len(splits) >= 3
    train, val = splits[0]
    assert len(train) == 50
    assert len(val) == 20


def test_rolling_window_splits() -> None:
    frame = _frame()
    splits = list(rolling_window_splits(frame, SplitConfig(train_window=40, step_size=10)))
    assert len(splits) >= 5


def test_ensemble_voting() -> None:
    engine = EnsembleEngine()
    result = engine.combine(
        [
            StrategyPrediction(strategy_id="a", signal=1, confidence=0.8),
            StrategyPrediction(strategy_id="b", signal=1, confidence=0.7),
            StrategyPrediction(strategy_id="c", signal=-1, confidence=0.6),
        ],
        method=EnsembleMethod.VOTING,
    )
    assert result.signal == 1
    assert result.confidence > 0.5


def test_regime_classifier() -> None:
    snapshot = RegimeClassifier().classify(_frame())
    assert snapshot.primary in MarketRegime
    assert snapshot.tags


def test_recommendation_engine() -> None:
    recommendation = StrategyRecommendationEngine().recommend(_frame())
    assert recommendation.strategy_id
    assert recommendation.confidence > 0
    assert recommendation.reason
