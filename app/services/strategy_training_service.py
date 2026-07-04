"""Orchestrate per-strategy dataset build and model training."""

from pathlib import Path

from loguru import logger

from app.core.config import Settings, get_settings
from app.data.repositories.base import HistoricalRepository
from app.domain.enums.market import Timeframe
from app.domain.instrument import Instrument
from app.domain.training import TrainingConfig
from app.ml.datasets.strategy_builder import StrategyDatasetBuilder
from app.ml.experiments.store import ExperimentStore
from app.ml.labels.base import LabelConfig
from app.ml.registry.strategy_registry import StrategyModelRegistry
from app.ml.trainer.hyperparameter_optimizer import HyperparameterOptimizer, HyperparameterSearchConfig
from app.ml.trainer.strategy_trainer import StrategyTrainer
from app.strategies.registry import get_strategy


class StrategyTrainingService:
    """Build datasets and train models independently per strategy."""

    def __init__(
        self,
        repository: HistoricalRepository,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        storage_path = Path(self._settings.storage_path)
        self._dataset_builder = StrategyDatasetBuilder(repository, storage_path)
        self._model_registry = StrategyModelRegistry(storage_path)
        self._experiment_store = ExperimentStore(storage_path)
        self._trainer = StrategyTrainer(self._model_registry, self._experiment_store)

    def build_dataset(
        self,
        strategy_id: str,
        instrument: Instrument,
        timeframe: Timeframe,
        label_config: LabelConfig | None = None,
    ):
        """Build and persist a strategy dataset."""
        return self._dataset_builder.build(strategy_id, instrument, timeframe, label_config)

    def train_strategy(
        self,
        strategy_id: str,
        instrument: Instrument,
        timeframe: Timeframe,
        label_config: LabelConfig | None = None,
        config: TrainingConfig | None = None,
        optimize_hyperparameters: bool = False,
    ):
        """Build dataset (if needed) and train a strategy model."""
        label_config = label_config or LabelConfig()
        strategy = get_strategy(strategy_id)
        frame, metadata = self.build_dataset(strategy_id, instrument, timeframe, label_config)

        if optimize_hyperparameters:
            encoded = StrategyDatasetBuilder.training_frame(
                frame,
                metadata.feature_columns,
            )
            encoded["label_encoded"] = encoded["label"].map(
                {"BUY": 1, "HOLD": 0, "SELL": 0}
            ).fillna(encoded["label"])
            optimizer = HyperparameterOptimizer()
            config = optimizer.optimize(
                encoded,
                metadata.feature_columns,
                "label_encoded",
                base_config=config,
                search_config=HyperparameterSearchConfig(n_trials=10),
            )
            logger.info("Optimized hyperparameters for {}", strategy_id)

        return self._trainer.train(strategy, frame, metadata, config=config)
