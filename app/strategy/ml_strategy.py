"""ML-driven trading strategy."""

from app.domain.backtest import BacktestConfig, SignalAction
from app.domain.features import FeatureVector
from app.domain.signal import Signal
from app.domain.training import SetupType
from app.ml.datasets.preparation import FEATURE_COLUMNS, feature_matches_setup
from app.ml.inference.predictor import LightGBMPredictor
from app.ml.registry.model_registry import ModelMetadata


class MLStrategy:
    """Generate signals from a trained swing-setup classifier."""

    def __init__(
        self,
        predictor: LightGBMPredictor,
        metadata: ModelMetadata | None = None,
        probability_threshold: float | None = None,
        config: BacktestConfig | None = None,
    ) -> None:
        self._predictor = predictor
        self._metadata = metadata
        self._config = config
        if probability_threshold is not None:
            self._threshold = probability_threshold
        elif config and config.probability_threshold is not None:
            self._threshold = config.probability_threshold
        elif metadata and metadata.metrics.prediction_threshold is not None:
            self._threshold = metadata.metrics.prediction_threshold
        else:
            self._threshold = 0.5
        self._setup_type = metadata.config.setup_type if metadata else None

    def generate_signal(self, features: FeatureVector) -> Signal:
        """Return BUY when a long setup is confirmed by the model."""
        if not self._features_ready(features):
            return Signal(action=SignalAction.HOLD, confidence=0.0)

        if self._setup_type is not None and not feature_matches_setup(features, self._setup_type):
            return Signal(action=SignalAction.HOLD, confidence=0.0)

        prediction = self._predictor.predict(features)
        probability = float(prediction["probability_up"])
        if probability < self._threshold:
            return Signal(action=SignalAction.HOLD, confidence=1.0 - probability)

        if self._config and self._config.min_expected_value is not None:
            expected_value = (
                probability * self._config.expected_win_pct
                - (1 - probability) * self._config.expected_loss_pct
            )
            if expected_value < self._config.min_expected_value:
                return Signal(action=SignalAction.HOLD, confidence=1.0 - probability)

        if self._setup_type == SetupType.SHORT:
            return Signal(action=SignalAction.HOLD, confidence=1.0 - probability)
        return Signal(action=SignalAction.BUY, confidence=probability)

    @staticmethod
    def _features_ready(features: FeatureVector) -> bool:
        """Check all model features are present."""
        return all(getattr(features, column) is not None for column in FEATURE_COLUMNS)
