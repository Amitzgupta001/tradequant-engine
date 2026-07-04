"""Feature persistence repository."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Protocol

from loguru import logger

from app.domain.enums.market import Timeframe
from app.domain.features import FeatureVector
from app.domain.instrument import Instrument


class FeatureRepository(Protocol):
    """Contract for feature vector persistence."""

    def save(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        features: list[FeatureVector],
        overwrite: bool = True,
    ) -> Path:
        """Persist feature vectors."""
        ...

    def load(self, instrument: Instrument, timeframe: Timeframe) -> list[FeatureVector]:
        """Load feature vectors."""
        ...


FEATURE_COLUMNS = list(FeatureVector.model_fields.keys())


class CSVFeatureRepository:
    """CSV implementation of feature storage under storage/features/."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def build_path(self, instrument: Instrument, timeframe: Timeframe) -> Path:
        """Build CSV path for feature vectors."""
        return (
            self._base_path
            / instrument.exchange_segment.value
            / instrument.security_id
            / f"{timeframe.value.lower()}_features.csv"
        )

    def save(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        features: list[FeatureVector],
        overwrite: bool = True,
    ) -> Path:
        """Write feature vectors to CSV."""
        path = self.build_path(instrument, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite:
            logger.info("Skipping existing feature file at {}", path)
            return path

        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FEATURE_COLUMNS)
            writer.writeheader()
            for feature in features:
                writer.writerow(feature.model_dump(mode="json"))

        logger.info("Saved {} feature rows to {}", len(features), path)
        return path

    def load(self, instrument: Instrument, timeframe: Timeframe) -> list[FeatureVector]:
        """Load feature vectors from CSV."""
        path = self.build_path(instrument, timeframe)
        if not path.exists():
            msg = f"Feature file not found: {path}"
            raise FileNotFoundError(msg)

        features: list[FeatureVector] = []
        with path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                parsed = {key: _parse_value(key, value) for key, value in row.items()}
                features.append(FeatureVector(**parsed))
        return features


def _parse_value(column: str, value: str) -> object:
    """Parse CSV cell into typed value."""
    if value == "" or value is None:
        return None
    if column == "timestamp":
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    if column == "volume":
        return int(value)
    if column in {"close"}:
        return float(value)
    return float(value)
