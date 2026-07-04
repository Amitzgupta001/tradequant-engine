"""Build strategy-selector training data from walk-forward backtest benchmarks."""

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from app.backtest.strategy_engine import StrategyBacktestEngine
from app.data.repositories.base import HistoricalRepository
from app.data.universe.registry import Universe
from app.domain.backtest import BacktestConfig, BacktestResult
from app.domain.enums.market import Timeframe
from app.domain.instrument import Instrument
from app.indicators.engine import IndicatorEngine
from app.ml.datasets.preparation import FEATURE_COLUMNS
from app.ml.evaluator.cross_validation import SplitConfig, rolling_window_splits
from app.ml.feature_store.repository import CSVFeatureRepository
from app.ml.inference.regime import RegimeClassifier
from app.ml.labels.base import LabelConfig
from app.ml.registry.strategy_registry import StrategyModelRegistry
from app.services.strategy_training_service import StrategyTrainingService
from app.strategy.presets import REWARD_PCT_5M, RISK_PCT_5M
from app.strategy.strategy_model_bridge import StrategyModelBridge
from app.strategies.dataframe import candles_to_dataframe
from app.strategies.registry import list_strategy_ids


SELECTOR_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "trend_strength",
    "volatility_pct",
    "regime_trending",
    "regime_sideways",
    "regime_bullish",
    "regime_bearish",
    "regime_high_volatility",
    "regime_low_volatility",
]


class SelectionObjective(str, Enum):
    """Metric used to pick the best strategy per walk-forward window."""

    PROFIT_FACTOR = "profit_factor"
    TOTAL_RETURN = "total_return"
    SHARPE = "sharpe"


class StrategySelectorBuilderConfig(BaseModel):
    """Walk-forward benchmark configuration."""

    train_window: int = Field(default=400, ge=50)
    step_size: int = Field(default=50, ge=5)
    min_trades: int = Field(default=2, ge=0)
    objective: SelectionObjective = SelectionObjective.PROFIT_FACTOR
    strategy_ids: list[str] | None = None


class StrategySelectorDatasetMetadata(BaseModel):
    """Persisted metadata for a strategy-selector benchmark dataset."""

    security_id: str
    exchange_segment: str
    timeframe: str
    row_count: int
    feature_columns: list[str]
    strategy_ids: list[str]
    objective: SelectionObjective
    builder_config: StrategySelectorBuilderConfig
    built_at: datetime
    universe_id: str | None = None
    constituent_count: int | None = None


class StrategySelectorDatasetBuilder:
    """Label each market window with the best backtested strategy."""

    def __init__(
        self,
        repository: HistoricalRepository,
        feature_repository: CSVFeatureRepository,
        model_registry: StrategyModelRegistry,
        training_service: StrategyTrainingService,
        storage_path: Path,
        indicator_engine: IndicatorEngine | None = None,
    ) -> None:
        self._repository = repository
        self._feature_repository = feature_repository
        self._model_registry = model_registry
        self._training_service = training_service
        self._storage_path = storage_path
        self._indicator_engine = indicator_engine or IndicatorEngine()
        self._engine = StrategyBacktestEngine()
        self._regime_classifier = RegimeClassifier()

    def dataset_dir(
        self,
        exchange_segment: str,
        security_id: str,
        timeframe: str,
        universe_id: str | None = None,
    ) -> Path:
        """Return storage path for selector benchmark datasets."""
        if universe_id:
            return (
                self._storage_path
                / "datasets"
                / "strategy_selector"
                / "panels"
                / universe_id
                / timeframe.lower()
            )
        return (
            self._storage_path
            / "datasets"
            / "strategy_selector"
            / exchange_segment
            / security_id
            / timeframe.lower()
        )

    def build_panel(
        self,
        universe: Universe,
        timeframe: Timeframe,
        backtest_config: BacktestConfig,
        label_config: LabelConfig | None = None,
        builder_config: StrategySelectorBuilderConfig | None = None,
        ensure_models: bool = True,
    ) -> tuple[pd.DataFrame, StrategySelectorDatasetMetadata]:
        """Build pooled selector dataset across all universe symbols."""
        builder_config = builder_config or StrategySelectorBuilderConfig()
        parts: list[pd.DataFrame] = []
        loaded_ids: list[str] = []
        exchange_segment = universe.instruments[0].exchange_segment.value

        for instrument in universe.instruments:
            label = instrument.symbol or instrument.security_id
            try:
                frame, _ = self.build(
                    instrument,
                    timeframe,
                    backtest_config,
                    label_config=label_config,
                    builder_config=builder_config,
                    ensure_models=ensure_models,
                )
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("Skipping {} in panel selector build: {}", label, exc)
                continue
            symbol_frame = frame.copy()
            symbol_frame["source_security_id"] = instrument.security_id
            parts.append(symbol_frame)
            loaded_ids.append(instrument.security_id)
            logger.info(
                "Added {} windows from {} to panel selector dataset",
                len(symbol_frame),
                label,
            )

        if not parts:
            msg = f"No selector benchmark rows generated for universe '{universe.id}'"
            raise ValueError(msg)

        dataset = pd.concat(parts, ignore_index=True)
        metadata = StrategySelectorDatasetMetadata(
            security_id=f"PANEL_{universe.id.upper()}",
            exchange_segment=exchange_segment,
            timeframe=timeframe.value,
            row_count=len(dataset),
            feature_columns=SELECTOR_FEATURE_COLUMNS,
            strategy_ids=builder_config.strategy_ids or list_strategy_ids(),
            objective=builder_config.objective,
            builder_config=builder_config,
            built_at=datetime.now(timezone.utc),
            universe_id=universe.id,
            constituent_count=len(loaded_ids),
        )
        self.save(dataset, metadata)
        logger.info(
            "Built panel strategy selector dataset for {} ({} rows from {}/{} symbols)",
            universe.id,
            len(dataset),
            len(loaded_ids),
            len(universe.instruments),
        )
        return dataset, metadata

    def build(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        backtest_config: BacktestConfig,
        label_config: LabelConfig | None = None,
        builder_config: StrategySelectorBuilderConfig | None = None,
        ensure_models: bool = True,
    ) -> tuple[pd.DataFrame, StrategySelectorDatasetMetadata]:
        """Run walk-forward backtests and label each window with the best strategy."""
        builder_config = builder_config or StrategySelectorBuilderConfig()
        label_config = label_config or LabelConfig(
            forward_horizon_bars=20,
            regression_threshold=REWARD_PCT_5M,
            take_profit_pct=REWARD_PCT_5M,
            stop_loss_pct=RISK_PCT_5M,
        )
        strategy_ids = builder_config.strategy_ids or list_strategy_ids()
        if not strategy_ids:
            msg = "No registered strategies available for selector training"
            raise ValueError(msg)

        features = self._feature_repository.load(instrument, timeframe)
        feature_frame = pd.DataFrame([feature.model_dump(mode="json") for feature in features])
        feature_frame["timestamp"] = pd.to_datetime(feature_frame["timestamp"])
        feature_frame = feature_frame.sort_values("timestamp").reset_index(drop=True)

        response = self._repository.load(instrument, timeframe)
        indicators = self._indicator_engine.compute(response.candles)
        market_frame = candles_to_dataframe(response.candles, indicators)
        market_frame["timestamp"] = pd.to_datetime(market_frame["timestamp"])
        market_frame = market_frame.sort_values("timestamp").reset_index(drop=True)
        aligned = feature_frame.merge(
            market_frame[["timestamp", "close"]],
            on="timestamp",
            how="inner",
        )
        if len(aligned) < builder_config.train_window + builder_config.step_size:
            msg = (
                f"Need at least {builder_config.train_window + builder_config.step_size} "
                f"aligned rows, got {len(aligned)}"
            )
            raise ValueError(msg)

        strategy_frames: dict[str, pd.DataFrame] = {}
        for strategy_id in strategy_ids:
            if ensure_models and self._model_registry.latest_version(strategy_id) is None:
                logger.info("Training missing strategy model: {}", strategy_id)
                self._training_service.train_strategy(
                    strategy_id,
                    instrument,
                    timeframe,
                    label_config=label_config,
                )
            frame, _ = self._training_service.build_dataset(
                strategy_id,
                instrument,
                timeframe,
                label_config,
            )
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            strategy_frames[strategy_id] = frame.sort_values("timestamp").reset_index(drop=True)

        split_config = SplitConfig(
            train_window=builder_config.train_window,
            step_size=builder_config.step_size,
        )
        rows: list[dict[str, object]] = []
        for train_slice, eval_slice in rolling_window_splits(aligned, split_config):
            anchor_ts = train_slice.iloc[-1]["timestamp"]
            eval_start = eval_slice.iloc[0]["timestamp"]
            eval_end = eval_slice.iloc[-1]["timestamp"]

            feature_row = self._selector_features(aligned, train_slice, market_frame)
            if feature_row is None:
                continue

            strategy_scores: dict[str, float] = {}
            strategy_returns: dict[str, float] = {}
            for strategy_id in strategy_ids:
                eval_frame = self._slice_strategy_frame(
                    strategy_frames[strategy_id],
                    eval_start,
                    eval_end,
                )
                if len(eval_frame) < 3:
                    strategy_scores[strategy_id] = float("-inf")
                    strategy_returns[strategy_id] = 0.0
                    continue
                result = self._backtest_strategy_slice(
                    strategy_id,
                    eval_frame,
                    instrument,
                    timeframe,
                    backtest_config,
                )
                strategy_scores[strategy_id] = self._score_result(
                    result,
                    builder_config.objective,
                    builder_config.min_trades,
                )
                strategy_returns[strategy_id] = result.metrics.total_return_pct

            best_strategy_id = max(strategy_scores, key=strategy_scores.get)
            if strategy_scores[best_strategy_id] == float("-inf"):
                continue

            row: dict[str, object] = {
                "timestamp": anchor_ts,
                "eval_start": eval_start,
                "eval_end": eval_end,
                "best_strategy_id": best_strategy_id,
                "strategy_scores_json": json.dumps(strategy_scores),
                "strategy_returns_json": json.dumps(strategy_returns),
            }
            row.update(feature_row)
            rows.append(row)

        if not rows:
            msg = "No selector benchmark rows generated; check data coverage and strategy models"
            raise ValueError(msg)

        dataset = pd.DataFrame(rows)
        metadata = StrategySelectorDatasetMetadata(
            security_id=instrument.security_id,
            exchange_segment=instrument.exchange_segment.value,
            timeframe=timeframe.value,
            row_count=len(dataset),
            feature_columns=SELECTOR_FEATURE_COLUMNS,
            strategy_ids=strategy_ids,
            objective=builder_config.objective,
            builder_config=builder_config,
            built_at=datetime.now(timezone.utc),
        )
        self.save(dataset, metadata)
        logger.info(
            "Built strategy selector dataset for {} {} ({} windows)",
            instrument.security_id,
            timeframe.value,
            len(dataset),
        )
        return dataset, metadata

    def save(self, frame: pd.DataFrame, metadata: StrategySelectorDatasetMetadata) -> Path:
        """Persist selector benchmark dataset."""
        directory = self.dataset_dir(
            metadata.exchange_segment,
            metadata.security_id,
            metadata.timeframe,
            universe_id=metadata.universe_id,
        )
        directory.mkdir(parents=True, exist_ok=True)
        dataset_path = directory / "dataset.csv"
        metadata_path = directory / "metadata.json"
        frame.to_csv(dataset_path, index=False)
        metadata_path.write_text(
            json.dumps(metadata.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        return directory

    def load(
        self,
        exchange_segment: str,
        security_id: str,
        timeframe: str,
        universe_id: str | None = None,
    ) -> tuple[pd.DataFrame, StrategySelectorDatasetMetadata]:
        """Load a persisted selector benchmark dataset."""
        directory = self.dataset_dir(
            exchange_segment,
            security_id,
            timeframe,
            universe_id=universe_id,
        )
        dataset_path = directory / "dataset.csv"
        metadata_path = directory / "metadata.json"
        if not dataset_path.exists() or not metadata_path.exists():
            msg = f"Strategy selector dataset not found: {directory}"
            raise FileNotFoundError(msg)
        frame = pd.read_csv(dataset_path, parse_dates=["timestamp"])
        metadata = StrategySelectorDatasetMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        return frame, metadata

    def _selector_features(
        self,
        aligned: pd.DataFrame,
        train_slice: pd.DataFrame,
        market_frame: pd.DataFrame,
    ) -> dict[str, float] | None:
        """Extract selector features at the end of a train window."""
        anchor_ts = train_slice.iloc[-1]["timestamp"]
        row = aligned.loc[aligned["timestamp"] == anchor_ts]
        if row.empty:
            return None
        row = row.iloc[0]
        market_upto = market_frame.loc[market_frame["timestamp"] <= anchor_ts]
        if market_upto.empty:
            return None
        regime = self._regime_classifier.classify(market_upto.tail(max(30, len(market_upto))))

        features: dict[str, float] = {}
        for column in FEATURE_COLUMNS:
            value = row.get(column)
            if value is None or pd.isna(value):
                return None
            features[column] = float(value)

        features["trend_strength"] = regime.trend_strength
        features["volatility_pct"] = regime.volatility_pct
        features["regime_trending"] = 1.0 if regime.primary.value == "trending" else 0.0
        features["regime_sideways"] = 1.0 if regime.primary.value == "sideways" else 0.0
        features["regime_bullish"] = 1.0 if "bullish" in {tag.value for tag in regime.tags} else 0.0
        features["regime_bearish"] = 1.0 if "bearish" in {tag.value for tag in regime.tags} else 0.0
        features["regime_high_volatility"] = (
            1.0 if "high_volatility" in {tag.value for tag in regime.tags} else 0.0
        )
        features["regime_low_volatility"] = (
            1.0 if "low_volatility" in {tag.value for tag in regime.tags} else 0.0
        )
        return features

    def _slice_strategy_frame(
        self,
        frame: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return strategy rows within an evaluation window."""
        mask = (frame["timestamp"] >= start) & (frame["timestamp"] <= end)
        return frame.loc[mask].copy()

    def _backtest_strategy_slice(
        self,
        strategy_id: str,
        eval_frame: pd.DataFrame,
        instrument: Instrument,
        timeframe: Timeframe,
        config: BacktestConfig,
    ) -> BacktestResult:
        """Backtest one strategy on an evaluation slice."""
        signal_lookup = {}
        for _, row in eval_frame.iterrows():
            timestamp = row["timestamp"]
            if isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()
            signal_lookup[timestamp] = int(row.get("strategy_signal", 0) or 0)
        bridge = StrategyModelBridge.load(
            self._model_registry,
            strategy_id,
            signal_lookup,
            config=config,
        )
        return self._engine.run(instrument, timeframe, eval_frame, bridge, config=config)

    def run_window_backtest(
        self,
        strategy_id: str,
        instrument: Instrument,
        timeframe: Timeframe,
        eval_start: pd.Timestamp,
        eval_end: pd.Timestamp,
        config: BacktestConfig,
        label_config: LabelConfig | None = None,
    ) -> BacktestResult:
        """Backtest one strategy on a single walk-forward evaluation window."""
        label_config = label_config or LabelConfig(
            forward_horizon_bars=20,
            regression_threshold=REWARD_PCT_5M,
            take_profit_pct=REWARD_PCT_5M,
            stop_loss_pct=RISK_PCT_5M,
        )
        frame, _ = self._training_service.build_dataset(
            strategy_id,
            instrument,
            timeframe,
            label_config,
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        eval_frame = self._slice_strategy_frame(frame, eval_start, eval_end)
        return self._backtest_strategy_slice(
            strategy_id,
            eval_frame,
            instrument,
            timeframe,
            config,
        )

    @staticmethod
    def _score_result(
        result: BacktestResult,
        objective: SelectionObjective,
        min_trades: int,
    ) -> float:
        """Score a backtest slice for strategy selection."""
        if result.metrics.total_trades < min_trades:
            return float("-inf")
        if objective == SelectionObjective.PROFIT_FACTOR:
            return result.metrics.profit_factor or float("-inf")
        if objective == SelectionObjective.SHARPE:
            return result.metrics.sharpe_ratio or float("-inf")
        return result.metrics.total_return_pct
