"""Feature engineering orchestration service."""

from pathlib import Path

from loguru import logger

from app.core.events import EventType, event_bus
from app.data.repositories.base import HistoricalRepository
from app.domain.enums.market import Timeframe
from app.domain.features import FeatureVector
from app.domain.instrument import Instrument
from app.ml.datasets.builder import FeatureDatasetBuilder
from app.ml.feature_store.repository import CSVFeatureRepository, FeatureRepository


class FeatureService:
    """Build and persist ML-ready feature datasets."""

    def __init__(
        self,
        repository: HistoricalRepository,
        feature_repository: FeatureRepository,
        builder: FeatureDatasetBuilder | None = None,
    ) -> None:
        self._repository = repository
        self._feature_repository = feature_repository
        self._builder = builder or FeatureDatasetBuilder(repository=repository)

    def build(self, instrument: Instrument, timeframe: Timeframe) -> list[FeatureVector]:
        """Build feature vectors without persisting."""
        logger.info(
            "Building features for security_id={} timeframe={}",
            instrument.security_id,
            timeframe.value,
        )
        features = self._builder.build(instrument, timeframe)
        if features:
            event_bus.publish(EventType.FEATURES_READY, features[-1])
        return features

    def build_and_store(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        overwrite: bool = True,
    ) -> tuple[list[FeatureVector], Path]:
        """Build features and save to feature storage."""
        features = self.build(instrument, timeframe)
        path = self._feature_repository.save(
            instrument,
            timeframe,
            features,
            overwrite=overwrite,
        )
        return features, path
