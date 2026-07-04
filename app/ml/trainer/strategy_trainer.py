"""Per-strategy LightGBM training pipeline."""

from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from app.domain.training import TrainingConfig, TrainingMetrics, TrainingTask
from app.ml.datasets.strategy_builder import StrategyDatasetBuilder, StrategyDatasetMetadata
from app.ml.evaluator.cross_validation import SplitConfig, time_series_split
from app.ml.evaluator.feature_importance import FeatureImportanceReport, compute_feature_importance
from app.ml.evaluator.metrics import (
    apply_threshold,
    evaluate_classification,
    evaluate_regression,
    finalize_metrics,
    find_best_threshold,
    signal_hit_rate,
)
from app.ml.experiments.store import ExperimentRecord, ExperimentStore
from app.ml.labels.base import LabelConfig, LabelType, SignalLabel
from app.ml.registry.strategy_registry import StrategyModelMetadata, StrategyModelRegistry
from app.ml.trainer.backend import TrainerBackend
from app.strategies.base import TradingStrategy


class LightGBMStrategyBackend:
    """LightGBM implementation of TrainerBackend."""

    def train_classifier(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        config: TrainingConfig,
    ) -> tuple[lgb.LGBMClassifier, TrainingMetrics]:
        """Train binary classifier with early stopping."""
        pos_count = int((y_train == 1).sum())
        neg_count = len(y_train) - pos_count
        scale_pos_weight = (neg_count / pos_count) if pos_count > 0 and neg_count > 0 else 1.0
        model = lgb.LGBMClassifier(
            num_leaves=config.num_leaves,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            n_estimators=config.n_estimators,
            min_child_samples=config.min_child_samples,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=config.random_state,
            verbose=-1,
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[
                lgb.early_stopping(config.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        return model, TrainingMetrics(train_rows=len(x_train), test_rows=0)

    def train_regressor(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        config: TrainingConfig,
    ) -> tuple[lgb.LGBMRegressor, TrainingMetrics]:
        """Train regressor with early stopping."""
        model = lgb.LGBMRegressor(
            num_leaves=config.num_leaves,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            n_estimators=config.n_estimators,
            min_child_samples=config.min_child_samples,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            random_state=config.random_state,
            verbose=-1,
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(config.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        return model, TrainingMetrics(train_rows=len(x_train), test_rows=0)

    def predict_proba(self, model: object, features: pd.DataFrame) -> list[float]:
        """Return positive-class probabilities."""
        classifier = model
        return classifier.predict_proba(features)[:, 1].tolist()

    def predict(self, model: object, features: pd.DataFrame) -> list[float]:
        """Return point predictions."""
        return model.predict(features).tolist()


class StrategyTrainer:
    """Train one model per strategy without mixing datasets."""

    def __init__(
        self,
        model_registry: StrategyModelRegistry,
        experiment_store: ExperimentStore | None = None,
        backend: TrainerBackend | None = None,
    ) -> None:
        self._model_registry = model_registry
        self._experiment_store = experiment_store or ExperimentStore(model_registry._base_path)
        self._backend = backend or LightGBMStrategyBackend()

    def train(
        self,
        strategy: TradingStrategy,
        frame: pd.DataFrame,
        metadata: StrategyDatasetMetadata,
        config: TrainingConfig | None = None,
        split_config: SplitConfig | None = None,
    ) -> tuple[StrategyModelMetadata, FeatureImportanceReport]:
        """Train, evaluate, register, and track a strategy model."""
        config = config or TrainingConfig()
        split_config = split_config or SplitConfig(
            test_size=config.test_size,
            validation_size=config.validation_size,
        )
        training_frame = StrategyDatasetBuilder.training_frame(
            frame,
            metadata.feature_columns,
            label_column="label",
        )
        if "strategy_signal" in training_frame.columns:
            signal_rows = training_frame[training_frame["strategy_signal"] != 0]
            if len(signal_rows) >= config.min_train_rows:
                training_frame = signal_rows.copy()
        if len(training_frame) < config.min_train_rows:
            msg = f"Need at least {config.min_train_rows} rows, got {len(training_frame)}"
            raise ValueError(msg)

        encoded = self._encode_labels(training_frame, metadata.label_config)
        feature_columns = metadata.feature_columns
        train, val, test = time_series_split(encoded, split_config)

        training_start = datetime.now(timezone.utc)
        if metadata.label_config.label_type == LabelType.REGRESSION:
            config = config.model_copy(update={"task": TrainingTask.REGRESSION})
            model, _ = self._backend.train_regressor(
                train[feature_columns],
                train["label_encoded"],
                val[feature_columns],
                val["label_encoded"],
                config,
            )
            y_pred = self._backend.predict(model, test[feature_columns])
            test_metrics = finalize_metrics(
                evaluate_regression(test["label_encoded"].tolist(), y_pred),
                train_rows=len(train),
            )
            val_metrics = TrainingMetrics(
                train_rows=len(train),
                validation_rows=len(val),
                test_rows=len(val),
                rmse=test_metrics.rmse,
            )
        else:
            config = config.model_copy(update={"task": TrainingTask.CLASSIFICATION})
            model, _ = self._backend.train_classifier(
                train[feature_columns],
                train["label_encoded"],
                val[feature_columns],
                val["label_encoded"],
                config,
            )
            val_prob = self._backend.predict_proba(model, val[feature_columns])
            test_prob = self._backend.predict_proba(model, test[feature_columns])
            threshold = 0.5
            if config.tune_threshold:
                threshold, _, _, _ = find_best_threshold(
                    val["label_encoded"].astype(int).tolist(),
                    val_prob,
                )
            y_pred = apply_threshold(test_prob, threshold)
            test_signals, test_hit_rate = signal_hit_rate(
                test["label_encoded"].astype(int).tolist(),
                test_prob,
                threshold,
            )
            test_metrics = finalize_metrics(
                evaluate_classification(
                    test["label_encoded"].astype(int).tolist(),
                    y_pred,
                    test_prob,
                ),
                train_rows=len(train),
            )
            test_metrics = test_metrics.model_copy(
                update={
                    "validation_rows": len(val),
                    "prediction_threshold": threshold,
                    "signal_count": test_signals,
                    "signal_hit_rate": test_hit_rate,
                }
            )
            val_metrics = test_metrics.model_copy(update={"test_rows": len(val)})

        importance = compute_feature_importance(
            model,
            test[feature_columns],
            test["label_encoded"],
            feature_columns,
            metadata.label_config.label_type == LabelType.REGRESSION,
        )
        training_end = datetime.now(timezone.utc)
        version = self._model_registry.next_version(strategy.strategy_id)
        model_metadata = StrategyModelMetadata(
            strategy_id=strategy.strategy_id,
            version=version,
            security_id=metadata.security_id,
            exchange_segment=metadata.exchange_segment,
            timeframe=metadata.timeframe,
            feature_columns=feature_columns,
            label_config=metadata.label_config,
            metrics=test_metrics,
            parameters=config.model_dump(mode="json"),
            dataset_version=metadata.dataset_version,
            feature_version=metadata.feature_version,
            label_version=metadata.label_config.version,
            git_commit_hash=StrategyModelRegistry.git_commit_hash(),
            trained_at=training_end,
        )
        directory = self._model_registry.save(model, model_metadata)
        importance.save(directory / "feature_importance.json")

        self._experiment_store.save(
            ExperimentRecord(
                strategy_id=strategy.strategy_id,
                dataset_version=metadata.dataset_version,
                feature_version=metadata.feature_version,
                label_version=metadata.label_config.version,
                model_version=version,
                parameters=config.model_dump(mode="json"),
                training_start=training_start,
                training_end=training_end,
                validation_metrics=val_metrics,
                test_metrics=test_metrics,
                git_commit_hash=model_metadata.git_commit_hash,
                security_id=metadata.security_id,
                exchange_segment=metadata.exchange_segment,
                timeframe=metadata.timeframe,
            )
        )
        logger.info(
            "Trained {} model v{} on {} rows",
            strategy.strategy_id,
            version,
            len(training_frame),
        )
        return model_metadata, importance

    @staticmethod
    def _encode_labels(frame: pd.DataFrame, label_config: LabelConfig) -> pd.DataFrame:
        """Convert labels to numeric values for model training."""
        result = frame.copy()
        if label_config.label_type == LabelType.REGRESSION:
            result["label_encoded"] = pd.to_numeric(result["label"], errors="coerce")
            return result.dropna(subset=["label_encoded"])

        if label_config.label_type == LabelType.TRIPLE_BARRIER:
            result["label_encoded"] = pd.to_numeric(result["label"], errors="coerce")
            result["label_encoded"] = (result["label_encoded"] > 0).astype(int)
            return result.dropna(subset=["label_encoded"])

        mapping = {
            SignalLabel.BUY.value: 1,
            SignalLabel.HOLD.value: 0,
            SignalLabel.SELL.value: 0,
        }
        result["label_encoded"] = result["label"].map(mapping)
        return result.dropna(subset=["label_encoded"])
