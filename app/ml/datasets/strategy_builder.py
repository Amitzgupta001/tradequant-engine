"""Build per-strategy ML datasets from historical market data."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from app.data.repositories.base import HistoricalRepository
from app.domain.enums.market import Timeframe
from app.domain.instrument import Instrument
from app.indicators.engine import IndicatorEngine
from app.ml.labels.base import LabelConfig
from app.strategies.base import TradingStrategy
from app.strategies.dataframe import candles_to_dataframe
from app.strategies.registry import get_strategy


class StrategyDatasetMetadata(BaseModel):
    """Metadata persisted alongside a strategy dataset."""

    strategy_id: str
    security_id: str
    exchange_segment: str
    timeframe: str
    row_count: int
    feature_columns: list[str]
    label_config: LabelConfig
    dataset_version: str = "v1"
    feature_version: str = "v1"
    built_at: datetime


class StrategyDatasetBuilder:
    """Build strategy-specific datasets with features, signals, and labels."""

    def __init__(
        self,
        repository: HistoricalRepository,
        storage_path: Path,
        indicator_engine: IndicatorEngine | None = None,
    ) -> None:
        self._repository = repository
        self._storage_path = storage_path
        self._indicator_engine = indicator_engine or IndicatorEngine()

    def dataset_dir(
        self,
        strategy_id: str,
        exchange_segment: str,
        security_id: str,
        timeframe: str,
    ) -> Path:
        """Return storage path for a strategy dataset."""
        return (
            self._storage_path
            / "datasets"
            / strategy_id
            / exchange_segment
            / security_id
            / timeframe.lower()
        )

    def build(
        self,
        strategy_id: str,
        instrument: Instrument,
        timeframe: Timeframe,
        label_config: LabelConfig | None = None,
        strategy: TradingStrategy | None = None,
    ) -> tuple[pd.DataFrame, StrategyDatasetMetadata]:
        """Build and persist a strategy dataset."""
        label_config = label_config or LabelConfig()
        strategy = strategy or get_strategy(strategy_id)

        response = self._repository.load(instrument, timeframe)
        indicators = self._indicator_engine.compute(response.candles)
        frame = candles_to_dataframe(response.candles, indicators)
        frame = strategy.generate_features(frame)
        frame = strategy.generate_signals(frame)
        frame = strategy.generate_labels(frame, label_config)

        feature_columns = strategy.feature_columns(frame)
        metadata = StrategyDatasetMetadata(
            strategy_id=strategy.strategy_id,
            security_id=instrument.security_id,
            exchange_segment=instrument.exchange_segment.value,
            timeframe=timeframe.value,
            row_count=len(frame),
            feature_columns=feature_columns,
            label_config=label_config,
            feature_version=label_config.version,
            built_at=datetime.now(timezone.utc),
        )
        self.save(frame, metadata)
        logger.info(
            "Built {} dataset for {} {} ({} rows, {} features)",
            strategy.strategy_id,
            instrument.security_id,
            timeframe.value,
            len(frame),
            len(feature_columns),
        )
        return frame, metadata

    def save(self, frame: pd.DataFrame, metadata: StrategyDatasetMetadata) -> Path:
        """Persist dataset parquet and metadata JSON."""
        directory = self.dataset_dir(
            metadata.strategy_id,
            metadata.exchange_segment,
            metadata.security_id,
            metadata.timeframe,
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
        strategy_id: str,
        exchange_segment: str,
        security_id: str,
        timeframe: str,
    ) -> tuple[pd.DataFrame, StrategyDatasetMetadata]:
        """Load a persisted strategy dataset."""
        directory = self.dataset_dir(strategy_id, exchange_segment, security_id, timeframe)
        dataset_path = directory / "dataset.csv"
        metadata_path = directory / "metadata.json"
        if not dataset_path.exists() or not metadata_path.exists():
            msg = f"Strategy dataset not found: {directory}"
            raise FileNotFoundError(msg)
        frame = pd.read_csv(dataset_path, parse_dates=["timestamp"])
        metadata = StrategyDatasetMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        return frame, metadata

    @staticmethod
    def training_frame(
        frame: pd.DataFrame,
        feature_columns: list[str],
        label_column: str = "label",
    ) -> pd.DataFrame:
        """Return cleaned ML frame with numeric features and labels."""
        columns = feature_columns + [label_column, "timestamp", "strategy_signal"]
        available = [column for column in columns if column in frame.columns]
        subset = frame[available].copy()
        for column in feature_columns:
            subset[column] = pd.to_numeric(subset[column], errors="coerce")
        subset = subset.dropna(subset=feature_columns + [label_column])
        return subset
