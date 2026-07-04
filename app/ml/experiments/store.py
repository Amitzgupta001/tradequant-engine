"""Experiment tracking for strategy training runs."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from app.domain.training import TrainingMetrics
from app.ml.labels.base import LabelConfig


class ExperimentRecord(BaseModel):
    """Persisted experiment metadata for a strategy training run."""

    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: str
    dataset_version: str
    feature_version: str
    label_version: str
    model_version: int
    parameters: dict[str, float | int | str | bool]
    training_start: datetime
    training_end: datetime
    validation_metrics: TrainingMetrics
    test_metrics: TrainingMetrics
    git_commit_hash: str | None = None
    security_id: str
    exchange_segment: str
    timeframe: str


class ExperimentStore:
    """Append-only experiment log under storage/experiments."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path / "experiments"

    def save(self, record: ExperimentRecord) -> Path:
        """Persist an experiment record as JSON."""
        strategy_dir = self._base_path / record.strategy_id
        strategy_dir.mkdir(parents=True, exist_ok=True)
        timestamp = record.training_end.strftime("%Y%m%dT%H%M%S")
        path = strategy_dir / f"{timestamp}_{record.experiment_id[:8]}.json"
        path.write_text(json.dumps(record.model_dump(mode="json"), indent=2), encoding="utf-8")
        logger.info("Saved experiment {} for {}", record.experiment_id, record.strategy_id)
        return path

    def list_experiments(self, strategy_id: str) -> list[ExperimentRecord]:
        """Load all experiments for a strategy."""
        strategy_dir = self._base_path / strategy_id
        if not strategy_dir.exists():
            return []
        records: list[ExperimentRecord] = []
        for path in sorted(strategy_dir.glob("*.json")):
            records.append(ExperimentRecord.model_validate_json(path.read_text(encoding="utf-8")))
        return records
