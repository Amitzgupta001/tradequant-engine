"""Model registry for trained artifacts."""

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from app.domain.training import TrainingConfig, TrainingMetrics, TrainingTask


class ModelMetadata(BaseModel):
    """Persisted metadata for a trained model."""

    security_id: str
    exchange_segment: str
    timeframe: str
    task: TrainingTask
    feature_columns: list[str]
    label_column: str
    metrics: TrainingMetrics
    config: TrainingConfig
    trained_at: datetime
    model_file: str
    universe_id: str | None = None
    constituent_count: int | None = None


class ModelRegistry:
    """Save and load trained models with metadata."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def model_dir(
        self,
        exchange_segment: str,
        security_id: str,
        timeframe: str,
        universe_id: str | None = None,
    ) -> Path:
        """Return directory for a model artifact set."""
        if universe_id:
            return self._base_path / "panels" / universe_id / timeframe.lower()
        return self._base_path / exchange_segment / security_id / timeframe.lower()

    def panel_model_dir(self, universe_id: str, timeframe: str) -> Path:
        """Return directory for a panel (multi-stock) model."""
        return self._base_path / "panels" / universe_id / timeframe.lower()

    @staticmethod
    def panel_security_id(universe_id: str) -> str:
        """Synthetic security id used in panel model metadata."""
        return f"PANEL_{universe_id.upper()}"

    def load_panel_metadata(self, universe_id: str, timeframe: str) -> ModelMetadata:
        """Load metadata for a panel model."""
        return self.load_metadata(
            "NSE_EQ",
            self.panel_security_id(universe_id),
            timeframe,
            universe_id=universe_id,
        )

    def load_panel_model_path(self, universe_id: str, timeframe: str) -> Path:
        """Return path to a saved panel LightGBM model."""
        return self.load_model_path(
            "NSE_EQ",
            self.panel_security_id(universe_id),
            timeframe,
            universe_id=universe_id,
        )

    def save(
        self,
        model: object,
        metadata: ModelMetadata,
    ) -> tuple[Path, Path]:
        """Persist LightGBM model and metadata JSON."""
        directory = self.model_dir(
            metadata.exchange_segment,
            metadata.security_id,
            metadata.timeframe,
            universe_id=metadata.universe_id,
        )
        directory.mkdir(parents=True, exist_ok=True)

        model_path = directory / "model.txt"
        metadata_path = directory / "metadata.json"
        model.booster_.save_model(str(model_path))

        metadata.model_file = model_path.name
        metadata_path.write_text(
            json.dumps(metadata.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        logger.info("Registered model at {}", directory)
        return model_path, metadata_path

    def load_metadata(
        self,
        exchange_segment: str,
        security_id: str,
        timeframe: str,
        universe_id: str | None = None,
    ) -> ModelMetadata:
        """Load model metadata JSON."""
        metadata_path = (
            self.model_dir(exchange_segment, security_id, timeframe, universe_id=universe_id)
            / "metadata.json"
        )
        if not metadata_path.exists():
            msg = f"Model metadata not found: {metadata_path}"
            raise FileNotFoundError(msg)
        return ModelMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))

    def load_model_path(
        self,
        exchange_segment: str,
        security_id: str,
        timeframe: str,
        universe_id: str | None = None,
    ) -> Path:
        """Return path to saved LightGBM model file."""
        metadata = self.load_metadata(
            exchange_segment,
            security_id,
            timeframe,
            universe_id=universe_id,
        )
        return self.model_dir(
            exchange_segment,
            security_id,
            timeframe,
            universe_id=universe_id,
        ) / metadata.model_file
