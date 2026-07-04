"""LightGBM inference."""

from pathlib import Path

import lightgbm as lgb
import pandas as pd

from app.ml.registry.model_registry import ModelRegistry


class LightGBMPredictor:
    """Run inference with a registered LightGBM model."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._model: lgb.Booster | None = None
        self._task: str | None = None
        self._feature_columns: list[str] = []
        self._prediction_threshold: float = 0.5

    def load(self, exchange_segment: str, security_id: str, timeframe: str) -> None:
        """Load model from registry."""
        metadata = self._registry.load_metadata(exchange_segment, security_id, timeframe)
        model_path = self._registry.load_model_path(exchange_segment, security_id, timeframe)
        self._model = lgb.Booster(model_file=str(model_path))
        self._task = metadata.task.value
        self._feature_columns = metadata.feature_columns
        if metadata.metrics.prediction_threshold is not None:
            self._prediction_threshold = metadata.metrics.prediction_threshold

    def load_panel(self, universe_id: str, timeframe: str) -> None:
        """Load a pooled panel model from registry."""
        metadata = self._registry.load_panel_metadata(universe_id, timeframe)
        model_path = self._registry.load_panel_model_path(universe_id, timeframe)
        self._model = lgb.Booster(model_file=str(model_path))
        self._task = metadata.task.value
        self._feature_columns = metadata.feature_columns
        if metadata.metrics.prediction_threshold is not None:
            self._prediction_threshold = metadata.metrics.prediction_threshold

    def predict(self, features: object) -> dict[str, float | int]:
        """Predict direction or return for a single feature vector."""
        if self._model is None:
            msg = "Model not loaded — call load() first"
            raise RuntimeError(msg)

        row = {column: getattr(features, column) for column in self._feature_columns}
        frame = pd.DataFrame([row])
        if frame.isna().any(axis=1).iloc[0]:
            msg = "Feature vector contains null values required for inference"
            raise ValueError(msg)

        if self._task == "classification":
            probability = float(self._model.predict(frame)[0])
            direction = 1 if probability >= self._prediction_threshold else 0
            return {
                "direction": direction,
                "probability_up": probability,
                "threshold": self._prediction_threshold,
            }

        predicted_return = float(self._model.predict(frame)[0])
        return {"predicted_return_1d": predicted_return}
