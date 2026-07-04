"""Train and apply the strategy-selector meta model."""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from app.core.config import Settings, get_settings
from app.data.repositories.base import HistoricalRepository
from app.data.universe.registry import Universe
from app.domain.backtest import BacktestConfig, BacktestResult
from app.domain.enums.market import Timeframe
from app.domain.instrument import Instrument
from app.indicators.engine import IndicatorEngine
from app.ml.datasets.strategy_selector_builder import (
    SelectionObjective,
    StrategySelectorBuilderConfig,
    StrategySelectorDatasetBuilder,
)
from app.ml.feature_store.repository import CSVFeatureRepository
from app.ml.inference.recommendation import StrategyRecommendation, StrategyRecommendationEngine
from app.ml.labels.base import LabelConfig
from app.ml.registry.strategy_registry import StrategyModelRegistry
from app.ml.registry.strategy_selector_registry import StrategySelectorMetadata, StrategySelectorRegistry
from app.ml.trainer.strategy_selector_trainer import StrategySelectorTrainer
from app.ml.selector.picker import compute_strategy_priors, pick_strategy, strategy_returns_from_row
from app.services.strategy_backtest_service import StrategyBacktestService
from app.services.strategy_training_service import StrategyTrainingService
from app.strategy.presets import BEST_5M_BACKTEST, REWARD_PCT_5M, RISK_PCT_5M
from app.strategies.dataframe import candles_to_dataframe


class MetaBacktestResult:
    """Simulated rolling strategy selection backtest."""

    def __init__(
        self,
        windows: int,
        selected_windows: int,
        cumulative_return_pct: float,
        selector_hit_rate: float | None,
        avg_window_return_pct: float,
        top_strategy_counts: dict[str, int],
    ) -> None:
        self.windows = windows
        self.selected_windows = selected_windows
        self.cumulative_return_pct = cumulative_return_pct
        self.selector_hit_rate = selector_hit_rate
        self.avg_window_return_pct = avg_window_return_pct
        self.top_strategy_counts = top_strategy_counts


class RollingBacktestResult:
    """Real rolling backtest with strategy switching per window."""

    def __init__(
        self,
        windows: int,
        traded_windows: int,
        skipped_low_confidence: int,
        compounded_return_pct: float,
        total_trades: int,
        strategy_counts: dict[str, int],
        initial_capital: float,
        final_equity: float,
    ) -> None:
        self.windows = windows
        self.traded_windows = traded_windows
        self.skipped_low_confidence = skipped_low_confidence
        self.compounded_return_pct = compounded_return_pct
        self.total_trades = total_trades
        self.strategy_counts = strategy_counts
        self.initial_capital = initial_capital
        self.final_equity = final_equity


class StrategySelectorService:
    """Orchestrate selector training, recommendation, and auto backtests."""

    def __init__(
        self,
        repository: HistoricalRepository,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        storage_path = Path(self._settings.storage_path)
        self._repository = repository
        self._feature_repository = CSVFeatureRepository(storage_path / "features")
        self._model_registry = StrategyModelRegistry(storage_path)
        self._selector_registry = StrategySelectorRegistry(storage_path)
        self._training_service = StrategyTrainingService(repository, self._settings)
        self._dataset_builder = StrategySelectorDatasetBuilder(
            repository,
            self._feature_repository,
            self._model_registry,
            self._training_service,
            storage_path,
        )
        self._selector_trainer = StrategySelectorTrainer(self._selector_registry)
        self._backtest_service = StrategyBacktestService(repository, self._settings)
        self._recommendation_engine = StrategyRecommendationEngine(
            selector_registry=self._selector_registry,
            feature_repository=self._feature_repository,
        )
        self._indicator_engine = IndicatorEngine()

    def train(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        backtest_config: BacktestConfig | None = None,
        label_config: LabelConfig | None = None,
        builder_config: StrategySelectorBuilderConfig | None = None,
        rebuild_dataset: bool = False,
        train_strategy_models: bool = True,
    ) -> StrategySelectorMetadata:
        """Train all strategy models, build benchmark dataset, train selector."""
        backtest_config = backtest_config or BEST_5M_BACKTEST.model_copy(
            update={"require_strategy_signal": True}
        )
        label_config = label_config or LabelConfig(
            forward_horizon_bars=20,
            regression_threshold=REWARD_PCT_5M,
            take_profit_pct=REWARD_PCT_5M,
            stop_loss_pct=RISK_PCT_5M,
        )
        builder_config = builder_config or StrategySelectorBuilderConfig()

        try:
            if not rebuild_dataset:
                frame, metadata = self._dataset_builder.load(
                    instrument.exchange_segment.value,
                    instrument.security_id,
                    timeframe.value,
                )
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            frame, metadata = self._dataset_builder.build(
                instrument,
                timeframe,
                backtest_config,
                label_config=label_config,
                builder_config=builder_config,
                ensure_models=train_strategy_models,
            )

        return self._selector_trainer.train(frame, metadata)

    def train_panel(
        self,
        universe: Universe,
        timeframe: Timeframe,
        backtest_config: BacktestConfig | None = None,
        label_config: LabelConfig | None = None,
        builder_config: StrategySelectorBuilderConfig | None = None,
        rebuild_dataset: bool = False,
        train_strategy_models: bool = True,
    ) -> StrategySelectorMetadata:
        """Train a pooled selector model across a stock universe."""
        backtest_config = backtest_config or BEST_5M_BACKTEST.model_copy(
            update={"require_strategy_signal": True}
        )
        label_config = label_config or LabelConfig(
            forward_horizon_bars=20,
            regression_threshold=REWARD_PCT_5M,
            take_profit_pct=REWARD_PCT_5M,
            stop_loss_pct=RISK_PCT_5M,
        )
        builder_config = builder_config or StrategySelectorBuilderConfig()
        panel_security_id = StrategySelectorRegistry.panel_security_id(universe.id)
        exchange_segment = universe.instruments[0].exchange_segment.value

        try:
            if not rebuild_dataset:
                frame, metadata = self._dataset_builder.load(
                    exchange_segment,
                    panel_security_id,
                    timeframe.value,
                    universe_id=universe.id,
                )
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            frame, metadata = self._dataset_builder.build_panel(
                universe,
                timeframe,
                backtest_config,
                label_config=label_config,
                builder_config=builder_config,
                ensure_models=train_strategy_models,
            )

        return self._selector_trainer.train(frame, metadata)

    def recommend(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        selector_version: int | None = None,
        universe_id: str | None = None,
        top_n: int = 5,
    ) -> list[StrategyRecommendation]:
        """Recommend strategies for the latest market snapshot."""
        merged = self._market_feature_frame(instrument, timeframe)
        scope = self._resolve_selector_scope(
            instrument,
            timeframe,
            universe_id=universe_id,
            selector_version=selector_version,
        )
        strategy_priors = None
        if scope["version"] is not None:
            try:
                benchmark_frame, _ = self._dataset_builder.load(
                    instrument.exchange_segment.value,
                    scope["scope_security_id"],
                    timeframe.value,
                    universe_id=scope["universe_id"],
                )
                if scope["universe_id"] and "source_security_id" in benchmark_frame.columns:
                    benchmark_frame = benchmark_frame.loc[
                        benchmark_frame["source_security_id"] == instrument.security_id
                    ]
                strategy_priors = compute_strategy_priors(benchmark_frame)
            except FileNotFoundError:
                strategy_priors = None

        return self._recommendation_engine.recommend_all(
            merged,
            exchange_segment=instrument.exchange_segment.value,
            security_id=scope["scope_security_id"],
            timeframe=timeframe.value,
            selector_version=scope["version"],
            universe_id=scope["universe_id"],
            strategy_priors=strategy_priors,
            top_n=top_n,
        )

    def simulate_meta_backtest(
        self,
        instrument: Instrument | None = None,
        timeframe: Timeframe | None = None,
        selector_version: int | None = None,
        universe_id: str | None = None,
        universe: Universe | None = None,
    ) -> MetaBacktestResult:
        """Simulate rolling strategy selection using stored benchmark returns."""
        if universe is not None:
            return self._simulate_panel_meta_backtest(
                universe,
                timeframe,
                selector_version=selector_version,
            )
        if instrument is None or timeframe is None:
            msg = "instrument and timeframe required when universe is not provided"
            raise ValueError(msg)

        scope = self._resolve_selector_scope(
            instrument,
            timeframe,
            universe_id=universe_id,
            selector_version=selector_version,
        )
        if scope["version"] is None:
            msg = "Train strategy selector before running meta backtest"
            raise FileNotFoundError(msg)

        metadata = self._selector_registry.load_metadata(
            instrument.exchange_segment.value,
            scope["scope_security_id"],
            timeframe.value,
            scope["version"],
            universe_id=scope["universe_id"],
        )
        model = self._selector_registry.load_model(
            instrument.exchange_segment.value,
            scope["scope_security_id"],
            timeframe.value,
            scope["version"],
            universe_id=scope["universe_id"],
        )
        encoder = self._selector_registry.load_label_encoder(
            instrument.exchange_segment.value,
            scope["scope_security_id"],
            timeframe.value,
            scope["version"],
            universe_id=scope["universe_id"],
        )
        frame, _ = self._dataset_builder.load(
            instrument.exchange_segment.value,
            scope["scope_security_id"],
            timeframe.value,
            universe_id=scope["universe_id"],
        )
        if scope["universe_id"] and "source_security_id" in frame.columns:
            frame = frame.loc[frame["source_security_id"] == instrument.security_id].copy()

        return self._simulate_on_frame(frame, metadata, model, encoder)

    def _simulate_panel_meta_backtest(
        self,
        universe: Universe,
        timeframe: Timeframe,
        selector_version: int | None = None,
    ) -> MetaBacktestResult:
        """Simulate meta backtest on the full pooled benchmark dataset."""
        exchange_segment = universe.instruments[0].exchange_segment.value
        panel_security_id = StrategySelectorRegistry.panel_security_id(universe.id)
        version = selector_version or self._selector_registry.latest_version(
            exchange_segment,
            panel_security_id,
            timeframe.value,
            universe_id=universe.id,
        )
        if version is None:
            msg = f"Train panel strategy selector for universe '{universe.id}' first"
            raise FileNotFoundError(msg)

        metadata = self._selector_registry.load_metadata(
            exchange_segment,
            panel_security_id,
            timeframe.value,
            version,
            universe_id=universe.id,
        )
        model = self._selector_registry.load_model(
            exchange_segment,
            panel_security_id,
            timeframe.value,
            version,
            universe_id=universe.id,
        )
        encoder = self._selector_registry.load_label_encoder(
            exchange_segment,
            panel_security_id,
            timeframe.value,
            version,
            universe_id=universe.id,
        )
        frame, _ = self._dataset_builder.load(
            exchange_segment,
            panel_security_id,
            timeframe.value,
            universe_id=universe.id,
        )
        return self._simulate_on_frame(frame, metadata, model, encoder)

    def backtest_recommended(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        backtest_config: BacktestConfig | None = None,
        selector_version: int | None = None,
        universe_id: str | None = None,
        retrain: bool = False,
    ) -> tuple[str, BacktestResult, list[StrategyRecommendation]]:
        """Recommend a strategy and run a full backtest with it."""
        recommendations = self.recommend(
            instrument,
            timeframe,
            selector_version=selector_version,
            universe_id=universe_id,
            top_n=3,
        )
        if not recommendations:
            msg = "No strategy recommendation available"
            raise ValueError(msg)

        top = recommendations[0]
        config = backtest_config or BEST_5M_BACKTEST.model_copy(
            update={"require_strategy_signal": True}
        )
        logger.info(
            "Auto backtest using recommended strategy {} ({:.0%})",
            top.strategy_id,
            top.confidence,
        )
        result, _ = self._backtest_service.run(
            top.strategy_id,
            instrument,
            timeframe,
            config=config,
            retrain=retrain,
        )
        return top.strategy_id, result, recommendations

    def backtest_rolling(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        backtest_config: BacktestConfig | None = None,
        selector_version: int | None = None,
        universe_id: str | None = None,
    ) -> RollingBacktestResult:
        """Switch strategies each walk-forward window using the selector model."""
        config = backtest_config or BEST_5M_BACKTEST.model_copy(
            update={"require_strategy_signal": True}
        )
        scope = self._resolve_selector_scope(
            instrument,
            timeframe,
            universe_id=universe_id,
            selector_version=selector_version,
        )
        if scope["version"] is None:
            msg = "Train strategy selector before running rolling backtest"
            raise FileNotFoundError(msg)

        metadata = self._selector_registry.load_metadata(
            instrument.exchange_segment.value,
            scope["scope_security_id"],
            timeframe.value,
            scope["version"],
            universe_id=scope["universe_id"],
        )
        model = self._selector_registry.load_model(
            instrument.exchange_segment.value,
            scope["scope_security_id"],
            timeframe.value,
            scope["version"],
            universe_id=scope["universe_id"],
        )
        encoder = self._selector_registry.load_label_encoder(
            instrument.exchange_segment.value,
            scope["scope_security_id"],
            timeframe.value,
            scope["version"],
            universe_id=scope["universe_id"],
        )
        frame, dataset_meta = self._dataset_builder.load(
            instrument.exchange_segment.value,
            scope["scope_security_id"],
            timeframe.value,
            universe_id=scope["universe_id"],
        )
        if scope["universe_id"] and "source_security_id" in frame.columns:
            frame = frame.loc[frame["source_security_id"] == instrument.security_id].copy()

        working = frame.dropna(subset=metadata.feature_columns).copy()
        if working.empty:
            msg = "No selector benchmark windows available for rolling backtest"
            raise ValueError(msg)

        strategy_priors = compute_strategy_priors(working)
        probabilities = model.predict_proba(working[metadata.feature_columns])
        strategy_ids = list(encoder.classes_)
        equity = config.initial_capital
        traded = 0
        skipped = 0
        total_trades = 0
        counts: dict[str, int] = {}

        for index, (_, row) in enumerate(working.iterrows()):
            probs = probabilities[index]
            strategy_id, _, _, _ = pick_strategy(
                probs,
                strategy_ids,
                min_confidence=metadata.min_confidence,
                min_margin=getattr(metadata, "min_margin", 0.0) or 0.0,
                window_returns=strategy_returns_from_row(row),
                strategy_priors=strategy_priors,
            )
            if strategy_id is None:
                skipped += 1
                continue

            eval_start, eval_end = self._window_bounds(row, dataset_meta)
            try:
                result = self._dataset_builder.run_window_backtest(
                    strategy_id,
                    instrument,
                    timeframe,
                    eval_start,
                    eval_end,
                    config,
                )
            except ValueError:
                skipped += 1
                continue

            traded += 1
            total_trades += result.metrics.total_trades
            counts[strategy_id] = counts.get(strategy_id, 0) + 1
            equity *= 1 + result.metrics.total_return_pct / 100

        compounded_return = ((equity / config.initial_capital) - 1) * 100
        logger.info(
            "Rolling selector backtest windows={} traded={} return={:.2f}%",
            len(working),
            traded,
            compounded_return,
        )
        return RollingBacktestResult(
            windows=len(working),
            traded_windows=traded,
            skipped_low_confidence=skipped,
            compounded_return_pct=compounded_return,
            total_trades=total_trades,
            strategy_counts=counts,
            initial_capital=config.initial_capital,
            final_equity=equity,
        )

    @staticmethod
    def _window_bounds(
        row: pd.Series,
        dataset_meta,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Resolve evaluation window timestamps for a benchmark row."""
        if "eval_start" in row and "eval_end" in row and pd.notna(row["eval_start"]):
            return pd.Timestamp(row["eval_start"]), pd.Timestamp(row["eval_end"])

        step_size = dataset_meta.builder_config.step_size
        anchor = pd.Timestamp(row["timestamp"])
        bar_minutes = {
            Timeframe.MIN_5.value: 5,
            Timeframe.MIN_15.value: 15,
        }.get(dataset_meta.timeframe, 5)
        eval_start = anchor + pd.Timedelta(minutes=bar_minutes)
        eval_end = anchor + pd.Timedelta(minutes=bar_minutes * step_size)
        return eval_start, eval_end

    def _market_feature_frame(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
    ) -> pd.DataFrame:
        """Load merged market features for recommendation."""
        response = self._repository.load(instrument, timeframe)
        indicators = self._indicator_engine.compute(response.candles)
        market_frame = candles_to_dataframe(response.candles, indicators)
        features = self._feature_repository.load(instrument, timeframe)
        feature_frame = pd.DataFrame([feature.model_dump(mode="json") for feature in features])
        feature_frame["timestamp"] = pd.to_datetime(feature_frame["timestamp"])
        market_frame["timestamp"] = pd.to_datetime(market_frame["timestamp"])

        merged = feature_frame.sort_values("timestamp")
        if "close" not in merged.columns:
            merged = merged.merge(
                market_frame[["timestamp", "close"]],
                on="timestamp",
                how="inner",
            )
        if "ema_20" in market_frame.columns and "ema_20" not in merged.columns:
            merged = merged.merge(
                market_frame[["timestamp", "ema_20"]],
                on="timestamp",
                how="left",
            )
        return merged.sort_values("timestamp")

    def _resolve_selector_scope(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        universe_id: str | None = None,
        selector_version: int | None = None,
    ) -> dict[str, str | int | None]:
        """Resolve per-stock or panel selector model scope."""
        exchange_segment = instrument.exchange_segment.value
        if universe_id:
            scope_security_id = StrategySelectorRegistry.panel_security_id(universe_id)
            version = selector_version or self._selector_registry.latest_version(
                exchange_segment,
                scope_security_id,
                timeframe.value,
                universe_id=universe_id,
            )
            return {
                "scope_security_id": scope_security_id,
                "universe_id": universe_id,
                "version": version,
            }

        version = selector_version or self._selector_registry.latest_version(
            exchange_segment,
            instrument.security_id,
            timeframe.value,
        )
        return {
            "scope_security_id": instrument.security_id,
            "universe_id": None,
            "version": version,
        }

    @staticmethod
    def _simulate_on_frame(frame, metadata, model, encoder) -> MetaBacktestResult:
        """Run meta-backtest simulation on a benchmark dataframe."""
        working = frame.dropna(subset=metadata.feature_columns).copy()
        probabilities = model.predict_proba(working[metadata.feature_columns])
        strategy_ids = list(encoder.classes_)

        cumulative = 0.0
        selected = 0
        hits = 0
        counts: dict[str, int] = {}
        for index, (_, row) in enumerate(working.iterrows()):
            probs = probabilities[index]
            strategy_id, _, _, _ = pick_strategy(
                probs,
                strategy_ids,
                min_confidence=metadata.min_confidence,
                min_margin=getattr(metadata, "min_margin", 0.0) or 0.0,
                window_returns=strategy_returns_from_row(row),
            )
            if strategy_id is None:
                continue
            predicted_id = strategy_id
            returns_map = json.loads(row["strategy_returns_json"])
            cumulative += float(returns_map.get(predicted_id, 0.0))
            selected += 1
            counts[predicted_id] = counts.get(predicted_id, 0) + 1
            if predicted_id == row["best_strategy_id"]:
                hits += 1

        return MetaBacktestResult(
            windows=len(working),
            selected_windows=selected,
            cumulative_return_pct=cumulative,
            selector_hit_rate=(hits / selected) if selected else None,
            avg_window_return_pct=(cumulative / selected) if selected else 0.0,
            top_strategy_counts=counts,
        )
