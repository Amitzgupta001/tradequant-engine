"""Versioned model registry for per-strategy ML artifacts."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
from loguru import logger
from pydantic import BaseModel

from app.domain.training import TrainingMetrics
from app.ml.labels.base import LabelConfig


class StrategyModelMetadata(BaseModel):
    """Metadata for a versioned strategy model."""

    strategy_id: str
    version: int
    security_id: str
    exchange_segment: str
    timeframe: str
    feature_columns: list[str]
    label_config: LabelConfig
    metrics: TrainingMetrics
    parameters: dict[str, float | int | str | bool]
    dataset_version: str = "v1"
    feature_version: str = "v1"
    label_version: str = "v1"
    git_commit_hash: str | None = None
    trained_at: datetime


class StrategyModelRegistry:
    """Persist and load versioned strategy models."""

    MODEL_FILENAME = "model.pkl"

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def strategy_root(self, strategy_id: str) -> Path:
        """Return root directory for a strategy's models."""
        return self._base_path / "models" / strategy_id

    def version_dir(self, strategy_id: str, version: int) -> Path:
        """Return directory for a specific model version."""
        return self.strategy_root(strategy_id) / f"v{version}"

    def next_version(self, strategy_id: str) -> int:
        """Return the next model version number."""
        root = self.strategy_root(strategy_id)
        if not root.exists():
            return 1
        versions = []
        for path in root.iterdir():
            if path.is_dir() and path.name.startswith("v") and path.name[1:].isdigit():
                versions.append(int(path.name[1:]))
        return max(versions, default=0) + 1

    @staticmethod
    def git_commit_hash() -> str | None:
        """Return current git commit hash when available."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
        return result.stdout.strip() or None

    def save(
        self,
        model: object,
        metadata: StrategyModelMetadata,
    ) -> Path:
        """Persist model and sidecar JSON artifacts."""
        directory = self.version_dir(metadata.strategy_id, metadata.version)
        directory.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, directory / self.MODEL_FILENAME)
        (directory / "metrics.json").write_text(
            json.dumps(metadata.metrics.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        (directory / "features.json").write_text(
            json.dumps(
                {
                    "feature_columns": metadata.feature_columns,
                    "feature_version": metadata.feature_version,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (directory / "params.json").write_text(
            json.dumps(metadata.parameters, indent=2),
            encoding="utf-8",
        )
        training_report = metadata.model_dump(mode="json")
        (directory / "training_report.json").write_text(
            json.dumps(training_report, indent=2),
            encoding="utf-8",
        )
        logger.info("Registered strategy model {} v{}", metadata.strategy_id, metadata.version)
        return directory

    def load_model(self, strategy_id: str, version: int) -> object:
        """Load a persisted strategy model."""
        path = self.version_dir(strategy_id, version) / self.MODEL_FILENAME
        if not path.exists():
            msg = f"Strategy model not found: {path}"
            raise FileNotFoundError(msg)
        return joblib.load(path)

    def load_metadata(self, strategy_id: str, version: int) -> StrategyModelMetadata:
        """Load training report metadata for a model version."""
        path = self.version_dir(strategy_id, version) / "training_report.json"
        if not path.exists():
            msg = f"Strategy model metadata not found: {path}"
            raise FileNotFoundError(msg)
        return StrategyModelMetadata.model_validate_json(path.read_text(encoding="utf-8"))

    def latest_version(self, strategy_id: str) -> int | None:
        """Return latest registered version for a strategy."""
        root = self.strategy_root(strategy_id)
        if not root.exists():
            return None
        versions = []
        for path in root.iterdir():
            if path.is_dir() and path.name.startswith("v") and path.name[1:].isdigit():
                if (path / self.MODEL_FILENAME).exists():
                    versions.append(int(path.name[1:]))
        return max(versions) if versions else None
