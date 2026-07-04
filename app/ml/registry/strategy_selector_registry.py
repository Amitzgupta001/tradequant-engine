"""Versioned registry for strategy-selector meta models."""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import joblib
from loguru import logger
from pydantic import BaseModel

from app.domain.training import TrainingMetrics
from app.ml.datasets.strategy_selector_builder import (
    SelectionObjective,
    StrategySelectorBuilderConfig,
)


class StrategySelectorMetadata(BaseModel):
    """Metadata for a trained strategy-selector model."""

    version: int
    security_id: str
    exchange_segment: str
    timeframe: str
    feature_columns: list[str]
    strategy_ids: list[str]
    objective: SelectionObjective
    builder_config: StrategySelectorBuilderConfig
    metrics: TrainingMetrics
    backtest_metrics: TrainingMetrics | None = None
    min_confidence: float = 0.0
    min_margin: float = 0.0
    parameters: dict[str, float | int | str | bool] = {}
    universe_id: str | None = None
    constituent_count: int | None = None
    git_commit_hash: str | None = None
    trained_at: datetime


class StrategySelectorRegistry:
    """Persist and load per-stock or pooled universe selector models."""

    MODEL_FILENAME = "model.pkl"
    ENCODER_FILENAME = "label_encoder.pkl"

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    @staticmethod
    def panel_security_id(universe_id: str) -> str:
        """Synthetic security id used in panel selector metadata."""
        return f"PANEL_{universe_id.upper()}"

    def selector_root(
        self,
        exchange_segment: str,
        security_id: str,
        timeframe: str,
        universe_id: str | None = None,
    ) -> Path:
        """Return root directory for selector models."""
        if universe_id:
            return (
                self._base_path
                / "models"
                / "strategy_selector"
                / "panels"
                / universe_id
                / timeframe.lower()
            )
        return (
            self._base_path
            / "models"
            / "strategy_selector"
            / exchange_segment
            / security_id
            / timeframe.lower()
        )

    def version_dir(
        self,
        exchange_segment: str,
        security_id: str,
        timeframe: str,
        version: int,
        universe_id: str | None = None,
    ) -> Path:
        """Return directory for a specific selector model version."""
        return self.selector_root(
            exchange_segment,
            security_id,
            timeframe,
            universe_id=universe_id,
        ) / f"v{version}"

    def next_version(
        self,
        exchange_segment: str,
        security_id: str,
        timeframe: str,
        universe_id: str | None = None,
    ) -> int:
        """Return the next selector model version number."""
        root = self.selector_root(
            exchange_segment,
            security_id,
            timeframe,
            universe_id=universe_id,
        )
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
        label_encoder: object,
        metadata: StrategySelectorMetadata,
    ) -> Path:
        """Persist selector model and metadata."""
        directory = self.version_dir(
            metadata.exchange_segment,
            metadata.security_id,
            metadata.timeframe,
            metadata.version,
            universe_id=metadata.universe_id,
        )
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, directory / self.MODEL_FILENAME)
        joblib.dump(label_encoder, directory / self.ENCODER_FILENAME)
        (directory / "training_report.json").write_text(
            json.dumps(metadata.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        scope = metadata.universe_id or metadata.security_id
        logger.info(
            "Registered strategy selector v{} for {} {}",
            metadata.version,
            scope,
            metadata.timeframe,
        )
        return directory

    def load_model(
        self,
        exchange_segment: str,
        security_id: str,
        timeframe: str,
        version: int,
        universe_id: str | None = None,
    ) -> object:
        """Load a persisted selector model."""
        path = (
            self.version_dir(
                exchange_segment,
                security_id,
                timeframe,
                version,
                universe_id=universe_id,
            )
            / self.MODEL_FILENAME
        )
        if not path.exists():
            msg = f"Strategy selector model not found: {path}"
            raise FileNotFoundError(msg)
        return joblib.load(path)

    def load_label_encoder(
        self,
        exchange_segment: str,
        security_id: str,
        timeframe: str,
        version: int,
        universe_id: str | None = None,
    ) -> object:
        """Load label encoder for strategy ids."""
        path = (
            self.version_dir(
                exchange_segment,
                security_id,
                timeframe,
                version,
                universe_id=universe_id,
            )
            / self.ENCODER_FILENAME
        )
        if not path.exists():
            msg = f"Strategy selector label encoder not found: {path}"
            raise FileNotFoundError(msg)
        return joblib.load(path)

    def load_metadata(
        self,
        exchange_segment: str,
        security_id: str,
        timeframe: str,
        version: int,
        universe_id: str | None = None,
    ) -> StrategySelectorMetadata:
        """Load selector training metadata."""
        path = (
            self.version_dir(
                exchange_segment,
                security_id,
                timeframe,
                version,
                universe_id=universe_id,
            )
            / "training_report.json"
        )
        if not path.exists():
            msg = f"Strategy selector metadata not found: {path}"
            raise FileNotFoundError(msg)
        return StrategySelectorMetadata.model_validate_json(path.read_text(encoding="utf-8"))

    def latest_version(
        self,
        exchange_segment: str,
        security_id: str,
        timeframe: str,
        universe_id: str | None = None,
    ) -> int | None:
        """Return latest registered selector version."""
        root = self.selector_root(
            exchange_segment,
            security_id,
            timeframe,
            universe_id=universe_id,
        )
        if not root.exists():
            return None
        versions = []
        for path in root.iterdir():
            if path.is_dir() and path.name.startswith("v") and path.name[1:].isdigit():
                if (path / self.MODEL_FILENAME).exists():
                    versions.append(int(path.name[1:]))
        return max(versions) if versions else None
