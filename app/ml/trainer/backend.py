"""Abstract trainer backend for future model engine compatibility."""

from typing import Protocol

import pandas as pd

from app.domain.training import TrainingConfig, TrainingMetrics


class TrainerBackend(Protocol):
    """Contract for swappable ML training backends."""

    def train_classifier(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        config: TrainingConfig,
    ) -> tuple[object, TrainingMetrics]:
        """Train a classification model."""

    def train_regressor(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        config: TrainingConfig,
    ) -> tuple[object, TrainingMetrics]:
        """Train a regression model."""

    def predict_proba(self, model: object, features: pd.DataFrame) -> list[float]:
        """Return positive-class probabilities for classifiers."""

    def predict(self, model: object, features: pd.DataFrame) -> list[float]:
        """Return point predictions."""
