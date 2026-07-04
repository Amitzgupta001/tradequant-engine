"""Tests for strategy selector training and recommendation."""

from datetime import datetime, timezone
import json

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from app.ml.datasets.strategy_selector_builder import (
    SELECTOR_FEATURE_COLUMNS,
    SelectionObjective,
    StrategySelectorBuilderConfig,
    StrategySelectorDatasetMetadata,
)
from app.ml.inference.recommendation import StrategyRecommendationEngine
from app.ml.inference.regime import MarketRegime
from app.ml.registry.strategy_selector_registry import StrategySelectorRegistry
from app.ml.trainer.strategy_selector_trainer import StrategySelectorTrainer


def _benchmark_row(strategy_id: str, index: int) -> dict[str, object]:
    features = {column: 0.1 * (index + 1) for column in SELECTOR_FEATURE_COLUMNS}
    returns = {"ema_crossover": 0.2, "breakout": 0.5, "rsi_reversal": -0.1}
    scores = {"ema_crossover": 0.8, "breakout": 1.2, "rsi_reversal": 0.4}
    return {
        "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
        "best_strategy_id": strategy_id,
        "strategy_scores_json": json.dumps(scores),
        "strategy_returns_json": json.dumps(returns),
        **features,
    }


def _metadata() -> StrategySelectorDatasetMetadata:
    return StrategySelectorDatasetMetadata(
        security_id="25",
        exchange_segment="NSE_EQ",
        timeframe="MIN_5",
        row_count=120,
        feature_columns=SELECTOR_FEATURE_COLUMNS,
        strategy_ids=["ema_crossover", "breakout", "rsi_reversal"],
        objective=SelectionObjective.PROFIT_FACTOR,
        builder_config=StrategySelectorBuilderConfig(),
        built_at=datetime.now(timezone.utc),
    )


def test_selector_trainer_trains_multiclass_model(tmp_path) -> None:
    rows = [
        _benchmark_row("breakout", index)
        if index % 3 == 0
        else _benchmark_row("ema_crossover", index)
        if index % 3 == 1
        else _benchmark_row("rsi_reversal", index)
        for index in range(120)
    ]
    frame = pd.DataFrame(rows)
    registry = StrategySelectorRegistry(tmp_path)
    metadata = _metadata()
    trained = StrategySelectorTrainer(registry).train(frame, metadata)
    assert trained.version == 1
    assert trained.metrics.accuracy is not None
    assert trained.min_margin >= 0.0


def test_selector_evaluate_backtest_selection() -> None:
    frame = pd.DataFrame([_benchmark_row("breakout", 0), _benchmark_row("breakout", 1)])
    encoder = LabelEncoder()
    encoder.fit(["ema_crossover", "breakout", "rsi_reversal"])
    probabilities = [
        [0.1, 0.8, 0.1],
        [0.1, 0.7, 0.2],
    ]
    metrics = StrategySelectorTrainer._evaluate_backtest_selection(
        frame,
        probabilities,
        encoder,
        min_margin=0.0,
    )
    assert metrics["selected_windows"] == 2.0
    assert metrics["simulated_return_pct"] == 0.4


def test_recommendation_engine_regime_fallback() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="D"),
            "close": [100 + index * 0.1 for index in range(200)],
            "ema_20": [100 + index * 0.08 for index in range(200)],
            "strategy_signal": [0] * 200,
        }
    )
    recommendations = StrategyRecommendationEngine().recommend_all(frame, top_n=3)
    assert recommendations
    assert recommendations[0].strategy_id
    assert recommendations[0].regime in MarketRegime


def test_recommendation_engine_uses_selector_model(tmp_path) -> None:
    rows = [
        _benchmark_row("breakout", index)
        if index % 3 == 0
        else _benchmark_row("ema_crossover", index)
        if index % 3 == 1
        else _benchmark_row("rsi_reversal", index)
        for index in range(120)
    ]
    frame = pd.DataFrame(rows)
    registry = StrategySelectorRegistry(tmp_path)
    metadata = _metadata()
    StrategySelectorTrainer(registry).train(frame, metadata)

    latest = frame.copy()
    for column in SELECTOR_FEATURE_COLUMNS:
        if column not in latest.columns:
            latest[column] = 0.5
    latest["close"] = 120.0
    latest["ema_20"] = 119.0

    engine = StrategyRecommendationEngine(selector_registry=registry)
    recommendations = engine.recommend_all(
        latest,
        exchange_segment="NSE_EQ",
        security_id="25",
        timeframe="MIN_5",
        top_n=3,
    )
    assert recommendations
    assert recommendations[0].strategy_id in metadata.strategy_ids
    assert "selector" in recommendations[0].reason.lower()


def test_panel_selector_training(tmp_path) -> None:
    rows = []
    for source in ("25", "1333"):
        for index in range(60):
            strategy_id = (
                "breakout"
                if index % 3 == 0
                else "ema_crossover"
                if index % 3 == 1
                else "rsi_reversal"
            )
            row = _benchmark_row(strategy_id, index)
            row["source_security_id"] = source
            rows.append(row)
    frame = pd.DataFrame(rows)
    registry = StrategySelectorRegistry(tmp_path)
    metadata = StrategySelectorDatasetMetadata(
        security_id=StrategySelectorRegistry.panel_security_id("nifty50"),
        exchange_segment="NSE_EQ",
        timeframe="MIN_5",
        row_count=len(frame),
        feature_columns=SELECTOR_FEATURE_COLUMNS,
        strategy_ids=["ema_crossover", "breakout", "rsi_reversal"],
        objective=SelectionObjective.PROFIT_FACTOR,
        builder_config=StrategySelectorBuilderConfig(),
        built_at=datetime.now(timezone.utc),
        universe_id="nifty50",
        constituent_count=2,
    )
    trained = StrategySelectorTrainer(registry).train(frame, metadata)
    assert trained.universe_id == "nifty50"
    assert trained.constituent_count == 2
    assert (
        registry.latest_version(
            "NSE_EQ",
            StrategySelectorRegistry.panel_security_id("nifty50"),
            "MIN_5",
            universe_id="nifty50",
        )
        == 1
    )


def test_panel_recommendation_engine(tmp_path) -> None:
    rows = []
    for index in range(120):
        strategy_id = (
            "breakout"
            if index % 3 == 0
            else "ema_crossover"
            if index % 3 == 1
            else "rsi_reversal"
        )
        row = _benchmark_row(strategy_id, index)
        row["source_security_id"] = "25"
        rows.append(row)
    frame = pd.DataFrame(rows)
    registry = StrategySelectorRegistry(tmp_path)
    metadata = StrategySelectorDatasetMetadata(
        security_id=StrategySelectorRegistry.panel_security_id("nifty50"),
        exchange_segment="NSE_EQ",
        timeframe="MIN_5",
        row_count=len(frame),
        feature_columns=SELECTOR_FEATURE_COLUMNS,
        strategy_ids=["ema_crossover", "breakout", "rsi_reversal"],
        objective=SelectionObjective.PROFIT_FACTOR,
        builder_config=StrategySelectorBuilderConfig(),
        built_at=datetime.now(timezone.utc),
        universe_id="nifty50",
        constituent_count=1,
    )
    StrategySelectorTrainer(registry).train(frame, metadata)

    latest = frame.tail(1).copy()
    latest["close"] = 120.0
    latest["ema_20"] = 119.0
    engine = StrategyRecommendationEngine(selector_registry=registry)
    recommendations = engine.recommend_all(
        latest,
        exchange_segment="NSE_EQ",
        security_id=StrategySelectorRegistry.panel_security_id("nifty50"),
        timeframe="MIN_5",
        universe_id="nifty50",
        top_n=3,
    )
    assert recommendations
    assert recommendations[0].strategy_id in metadata.strategy_ids
