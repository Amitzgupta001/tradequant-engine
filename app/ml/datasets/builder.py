"""Feature dataset builder for ML pipelines."""

from app.data.repositories.base import HistoricalRepository
from app.domain.enums.market import Timeframe
from app.domain.features import FeatureVector
from app.domain.instrument import Instrument
from app.indicators.engine import IndicatorEngine
from app.ml.feature_store.engine import FeatureEngine


class FeatureDatasetBuilder:
    """Build ML feature datasets from raw OHLCV data."""

    def __init__(
        self,
        repository: HistoricalRepository,
        indicator_engine: IndicatorEngine | None = None,
        feature_engine: FeatureEngine | None = None,
    ) -> None:
        self._repository = repository
        self._indicator_engine = indicator_engine or IndicatorEngine()
        self._feature_engine = feature_engine or FeatureEngine()

    def build(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
    ) -> list[FeatureVector]:
        """Load candles, compute indicators, and engineer features."""
        response = self._repository.load(instrument, timeframe)
        indicators = self._indicator_engine.compute(response.candles)
        return self._feature_engine.build(response.candles, indicators)
