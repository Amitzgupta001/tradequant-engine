"""Bridge Phase 3 strategy models to the backtest engine."""

import pandas as pd

from app.domain.backtest import BacktestConfig, SignalAction
from app.domain.features import FeatureVector
from app.domain.signal import Signal
from app.ml.registry.strategy_registry import StrategyModelMetadata, StrategyModelRegistry


class StrategyModelBridge:
    """Generate backtest signals from a trained per-strategy model."""

    def __init__(
        self,
        model: object,
        metadata: StrategyModelMetadata,
        signal_lookup: dict[object, int],
        config: BacktestConfig | None = None,
    ) -> None:
        self._model = model
        self._metadata = metadata
        self._signal_lookup = signal_lookup
        self._feature_columns = metadata.feature_columns
        if config and config.probability_threshold is not None:
            self._threshold = config.probability_threshold
        else:
            self._threshold = metadata.metrics.prediction_threshold or 0.5
        self._config = config

    @classmethod
    def load(
        cls,
        registry: StrategyModelRegistry,
        strategy_id: str,
        signal_lookup: dict[object, int],
        version: int | None = None,
        config: BacktestConfig | None = None,
    ) -> "StrategyModelBridge":
        """Load the latest or specific strategy model version."""
        resolved = version or registry.latest_version(strategy_id)
        if resolved is None:
            msg = f"No trained model found for strategy '{strategy_id}'"
            raise FileNotFoundError(msg)
        model = registry.load_model(strategy_id, resolved)
        metadata = registry.load_metadata(strategy_id, resolved)
        return cls(model, metadata, signal_lookup, config=config)

    def generate_signal(self, features: FeatureVector) -> Signal:
        """Predict using strategy features aligned by timestamp."""
        raw_signal = self._signal_lookup.get(features.timestamp, 0)
        if self._config and self._config.require_strategy_signal and raw_signal <= 0:
            return Signal(action=SignalAction.HOLD, confidence=0.0)

        row = self._feature_row(features)
        if row is None:
            return Signal(action=SignalAction.HOLD, confidence=0.0)

        frame = pd.DataFrame([row])
        probability = float(self._model.predict_proba(frame)[0][1])
        if probability < self._threshold:
            return Signal(action=SignalAction.HOLD, confidence=1.0 - probability)

        if self._config and self._config.min_expected_value is not None:
            expected_value = (
                probability * self._config.expected_win_pct
                - (1 - probability) * self._config.expected_loss_pct
            )
            if expected_value < self._config.min_expected_value:
                return Signal(action=SignalAction.HOLD, confidence=1.0 - probability)

        return Signal(action=SignalAction.BUY, confidence=probability)

    def _feature_row(self, features: FeatureVector) -> dict[str, float] | None:
        """Build a strategy feature row keyed by timestamp lookup."""
        del features
        return None

    def generate_signal_from_row(self, row: pd.Series) -> Signal:
        """Predict directly from a strategy dataframe row."""
        frame = pd.DataFrame([{column: row[column] for column in self._feature_columns}])
        if frame.isna().any(axis=1).iloc[0]:
            return Signal(action=SignalAction.HOLD, confidence=0.0)
        probability = float(self._model.predict_proba(frame)[0][1])
        raw_signal = int(row.get("strategy_signal", 0))
        if self._config and self._config.require_strategy_signal and raw_signal <= 0:
            return Signal(action=SignalAction.HOLD, confidence=0.0)
        if probability < self._threshold:
            return Signal(action=SignalAction.HOLD, confidence=1.0 - probability)
        return Signal(action=SignalAction.BUY, confidence=probability)
