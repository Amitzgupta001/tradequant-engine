"""Strategy-agnostic trading strategy interface for ML pipelines."""

from abc import ABC, abstractmethod

import pandas as pd

from app.ml.labels.base import LabelConfig


class TradingStrategy(ABC):
    """Contract for per-strategy feature, signal, and label generation."""

    @property
    @abstractmethod
    def strategy_id(self) -> str:
        """Unique identifier used for dataset and model paths."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of strategy logic."""

    @abstractmethod
    def required_indicators(self) -> list[str]:
        """Indicator column names required from the market dataframe."""

    @abstractmethod
    def generate_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific feature columns to the market dataframe."""

    @abstractmethod
    def generate_signals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Add strategy signal columns (e.g. strategy_signal)."""

    def generate_labels(
        self,
        dataframe: pd.DataFrame,
        label_config: LabelConfig | None = None,
    ) -> pd.DataFrame:
        """Add ML label columns using the configured label generator."""
        from app.ml.labels.generator import apply_labels

        return apply_labels(dataframe, label_config)

    def feature_columns(self, dataframe: pd.DataFrame) -> list[str]:
        """Return ML feature column names excluding market and label columns."""
        excluded = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "session_date",
            "strategy_signal",
            "label_classification",
            "label_regression",
            "label_barrier",
            "label",
        }
        return [
            column
            for column in dataframe.columns
            if column not in excluded and not column.startswith("forward_")
        ]
