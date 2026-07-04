"""LightGBM training pipeline."""

from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import balanced_accuracy_score

from app.domain.enums.market import Timeframe
from app.domain.features import FeatureVector
from app.domain.instrument import Instrument
from app.domain.training import TrainingConfig, TrainingMetrics, TrainingResult, TrainingTask
from app.ml.datasets.preparation import FEATURE_COLUMNS, build_training_frame
from app.ml.evaluator.metrics import (
    apply_threshold,
    evaluate_classification,
    evaluate_regression,
    finalize_metrics,
    find_best_threshold,
    signal_hit_rate,
)
from app.ml.registry.model_registry import ModelMetadata, ModelRegistry


class LightGBMTrainer:
    """Train LightGBM models on engineered feature vectors."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def train(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        features: list[FeatureVector],
        config: TrainingConfig | None = None,
    ) -> TrainingResult:
        """Train, evaluate, and register a LightGBM model."""
        config = config or TrainingConfig()
        frame, label_column = build_training_frame(features, config)
        if len(frame) < config.min_train_rows:
            msg = f"Need at least {config.min_train_rows} setup rows after cleaning, got {len(frame)}"
            raise ValueError(msg)

        logger.info(
            "Prepared {} setup rows with {} features (setup={}, horizon={} bars, move={:.1%})",
            len(frame),
            len(FEATURE_COLUMNS),
            config.setup_type.value if config.setup_type else "none",
            config.forward_horizon_bars,
            config.move_threshold,
        )

        x_train, x_val, x_test, y_train, y_val, y_test = self._time_split(frame, config)

        if config.task == TrainingTask.CLASSIFICATION:
            model, metrics = self._train_classifier(
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                config,
                setup_rows=len(frame),
            )
        else:
            model, metrics = self._train_regressor(
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                config,
                setup_rows=len(frame),
            )

        trained_at = datetime.now(timezone.utc)
        metadata = ModelMetadata(
            security_id=instrument.security_id,
            exchange_segment=instrument.exchange_segment.value,
            timeframe=timeframe.value,
            task=config.task,
            feature_columns=FEATURE_COLUMNS,
            label_column=label_column,
            metrics=metrics,
            config=config,
            trained_at=trained_at,
            model_file="model.txt",
        )
        model_path, metadata_path = self._registry.save(model, metadata)

        return TrainingResult(
            model_path=str(model_path),
            metadata_path=str(metadata_path),
            instrument_security_id=instrument.security_id,
            timeframe=timeframe.value,
            feature_columns=FEATURE_COLUMNS,
            metrics=metrics,
            trained_at=trained_at,
        )

    def train_panel(
        self,
        universe_id: str,
        exchange_segment: str,
        timeframe: Timeframe,
        frame: pd.DataFrame,
        label_column: str,
        config: TrainingConfig | None = None,
        constituent_count: int = 0,
    ) -> TrainingResult:
        """Train a pooled model across multiple instruments."""
        config = config or TrainingConfig()
        if len(frame) < config.min_train_rows:
            msg = f"Need at least {config.min_train_rows} setup rows after cleaning, got {len(frame)}"
            raise ValueError(msg)

        logger.info(
            "Panel {}: {} setup rows from {} stocks (horizon={} bars, move={:.1%})",
            universe_id,
            len(frame),
            constituent_count,
            config.forward_horizon_bars,
            config.move_threshold,
        )

        x_train, x_val, x_test, y_train, y_val, y_test = self._time_split(frame, config)

        if config.task == TrainingTask.CLASSIFICATION:
            model, metrics = self._train_classifier(
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                config,
                setup_rows=len(frame),
            )
        else:
            model, metrics = self._train_regressor(
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                config,
                setup_rows=len(frame),
            )

        trained_at = datetime.now(timezone.utc)
        metadata = ModelMetadata(
            security_id=f"PANEL_{universe_id.upper()}",
            exchange_segment=exchange_segment,
            timeframe=timeframe.value,
            task=config.task,
            feature_columns=FEATURE_COLUMNS,
            label_column=label_column,
            metrics=metrics,
            config=config,
            trained_at=trained_at,
            model_file="model.txt",
            universe_id=universe_id,
            constituent_count=constituent_count,
        )
        model_path, metadata_path = self._registry.save(model, metadata)

        return TrainingResult(
            model_path=str(model_path),
            metadata_path=str(metadata_path),
            instrument_security_id=metadata.security_id,
            timeframe=timeframe.value,
            feature_columns=FEATURE_COLUMNS,
            metrics=metrics,
            trained_at=trained_at,
        )

    @staticmethod
    def _time_split(frame: pd.DataFrame, config: TrainingConfig):
        """Split chronologically into train, validation, and test sets."""
        test_start = int(len(frame) * (1 - config.test_size))
        if test_start <= 0 or test_start >= len(frame):
            msg = "Invalid train/test split for dataset size"
            raise ValueError(msg)

        train_val = frame.iloc[:test_start]
        test = frame.iloc[test_start:]
        val_size = max(1, int(len(train_val) * config.validation_size))
        train = train_val.iloc[:-val_size]
        val = train_val.iloc[-val_size:]

        if len(train) == 0:
            msg = "Training split is empty after validation holdout"
            raise ValueError(msg)

        return (
            train[FEATURE_COLUMNS],
            val[FEATURE_COLUMNS],
            test[FEATURE_COLUMNS],
            train["label"],
            val["label"],
            test["label"],
        )

    def _train_classifier(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        config: TrainingConfig,
        setup_rows: int,
    ) -> tuple[lgb.LGBMClassifier, TrainingMetrics]:
        """Train binary setup-success classifier with early stopping."""
        pos_count = int(y_train.sum())
        neg_count = len(y_train) - pos_count
        scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

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

        val_prob = model.predict_proba(x_val)[:, 1].tolist()
        test_prob = model.predict_proba(x_test)[:, 1].tolist()
        threshold = 0.5
        validation_accuracy = None

        if config.tune_threshold:
            threshold, validation_accuracy, val_signals, _ = find_best_threshold(
                y_val.astype(int).tolist(),
                val_prob,
            )
            if val_signals == 0:
                threshold = float(np.percentile(val_prob, 75))
                logger.warning(
                    "No validation signals at tuned threshold — using p75={:.3f}",
                    threshold,
                )
        else:
            validation_accuracy = balanced_accuracy_score(
                y_val.astype(int),
                apply_threshold(val_prob, threshold),
            )
            val_signals = sum(1 for probability in val_prob if probability >= threshold)

        y_pred = apply_threshold(test_prob, threshold)
        test_signals, test_hit_rate = signal_hit_rate(
            y_test.astype(int).tolist(),
            test_prob,
            threshold,
        )
        metrics = finalize_metrics(
            evaluate_classification(y_test.astype(int).tolist(), y_pred, test_prob),
            train_rows=len(x_train),
        )
        return model, metrics.model_copy(
            update={
                "validation_rows": len(x_val),
                "validation_accuracy": validation_accuracy,
                "prediction_threshold": threshold,
                "setup_rows": setup_rows,
                "signal_count": test_signals,
                "signal_hit_rate": test_hit_rate,
            }
        )

    def _train_regressor(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        config: TrainingConfig,
        setup_rows: int,
    ) -> tuple[lgb.LGBMRegressor, TrainingMetrics]:
        """Train forward-return regressor."""
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
        y_pred = model.predict(x_test).tolist()
        metrics = finalize_metrics(
            evaluate_regression(y_test.tolist(), y_pred),
            train_rows=len(x_train),
        )
        return model, metrics.model_copy(
            update={"validation_rows": len(x_val), "setup_rows": setup_rows}
        )
