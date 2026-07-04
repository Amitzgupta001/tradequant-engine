"""ML training orchestration service."""

import gc
from datetime import date, timedelta

from loguru import logger

from app.core.events import EventType, event_bus
from app.domain.enums.market import Timeframe
from app.domain.instrument import Instrument
from app.domain.training import TrainingConfig, TrainingResult
from app.data.universe.registry import Universe
from app.ml.datasets.panel import build_panel_training_frame
from app.ml.datasets.preparation import training_config_for_timeframe
from app.ml.feature_store.repository import FeatureRepository
from app.ml.registry.model_registry import ModelRegistry
from app.ml.trainer.lightgbm_trainer import LightGBMTrainer
from app.services.feature_service import FeatureService
from app.services.historical_data_service import HistoricalDataService


class TrainingService:
    """End-to-end pipeline: download → features → train."""

    def __init__(
        self,
        historical_service: HistoricalDataService,
        feature_service: FeatureService,
        feature_repository: FeatureRepository,
        trainer: LightGBMTrainer,
    ) -> None:
        self._historical_service = historical_service
        self._feature_service = feature_service
        self._feature_repository = feature_repository
        self._trainer = trainer

    def train_from_features(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        config: TrainingConfig | None = None,
    ) -> TrainingResult:
        """Train using features already stored on disk."""
        config = training_config_for_timeframe(timeframe, config)
        features = self._feature_repository.load(instrument, timeframe)
        logger.info(
            "Training on {} feature rows for security_id={} timeframe={}",
            len(features),
            instrument.security_id,
            timeframe.value,
        )
        result = self._trainer.train(instrument, timeframe, features, config=config)
        event_bus.publish(EventType.PREDICTION_READY, result.metrics)
        return result

    def prepare_data(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        from_date: date,
        to_date: date,
    ) -> tuple[int, int]:
        """Download raw OHLCV and build features for a date range."""
        response, _ = self._historical_service.download_range_and_store(
            instrument,
            timeframe,
            from_date,
            to_date,
        )
        features, _ = self._feature_service.build_and_store(instrument, timeframe)
        return len(response.candles), len(features)

    def release_batch_memory(self) -> None:
        """Drop cached candles and encourage GC between batch symbols."""
        self._historical_service.clear_cache()
        gc.collect()

    def train_with_history(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        years: int = 3,
        days: int | None = None,
        config: TrainingConfig | None = None,
        to_date: date | None = None,
    ) -> TrainingResult:
        """Download historical data, build features, and train."""
        config = training_config_for_timeframe(timeframe, config)
        end = to_date or date.today()
        if timeframe.is_daily:
            start = end - timedelta(days=years * 365)
            window_label = f"{years} years"
        else:
            lookback_days = days or 90
            start = end - timedelta(days=lookback_days)
            window_label = f"{lookback_days} days"

        candle_count, feature_count = self.prepare_data(
            instrument, timeframe, start, end
        )
        logger.info(
            "Prepared {} candles and {} feature rows over {}",
            candle_count,
            feature_count,
            window_label,
        )
        return self.train_from_features(instrument, timeframe, config=config)

    def has_stored_features(self, instrument: Instrument, timeframe: Timeframe) -> bool:
        """Return True if feature CSV exists for an instrument."""
        path = self._feature_repository.build_path(instrument, timeframe)
        return path.exists()

    def count_stored_features(self, instrument: Instrument, timeframe: Timeframe) -> int:
        """Count feature rows on disk without loading them into memory."""
        path = self._feature_repository.build_path(instrument, timeframe)
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return max(0, sum(1 for _ in handle) - 1)

    def train_panel_from_features(
        self,
        universe: Universe,
        timeframe: Timeframe,
        config: TrainingConfig | None = None,
    ) -> TrainingResult:
        """Train a pooled model using stored features for all universe instruments."""
        config = training_config_for_timeframe(timeframe, config)
        frame, label_column, loaded_ids = build_panel_training_frame(
            self._feature_repository,
            universe.instruments,
            timeframe,
            config,
        )

        logger.info(
            "Panel training on {} setup rows from {}/{} symbols",
            len(frame),
            len(loaded_ids),
            len(universe.instruments),
        )
        return self._trainer.train_panel(
            universe.id,
            universe.instruments[0].exchange_segment.value,
            timeframe,
            frame,
            label_column,
            config=config,
            constituent_count=len(loaded_ids),
        )

    def train_panel_with_history(
        self,
        universe: Universe,
        timeframe: Timeframe,
        years: int = 3,
        days: int | None = None,
        config: TrainingConfig | None = None,
        to_date: date | None = None,
        skip_existing: bool = False,
        sleep_seconds: float = 0.0,
    ) -> TrainingResult:
        """Download data for all universe symbols, then train a pooled model."""
        from app.services.batch_data_service import BatchDataService

        batch = BatchDataService(self)
        results = batch.download_universe(
            universe,
            timeframe,
            years=years,
            days=days,
            to_date=to_date,
            skip_existing=skip_existing,
            sleep_seconds=sleep_seconds,
        )
        ok_count = sum(
            1 for result in results if result.status in ("ok", "skipped")
        )
        logger.info(
            "Batch prepare complete: {}/{} symbols succeeded",
            ok_count,
            len(results),
        )
        if ok_count == 0:
            msg = f"All downloads failed for universe '{universe.id}'"
            raise RuntimeError(msg)
        return self.train_panel_from_features(universe, timeframe, config=config)
