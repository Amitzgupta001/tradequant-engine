"""Optuna hyperparameter optimization for strategy models."""

from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, Field

from app.domain.training import TrainingConfig, TrainingTask
from app.ml.evaluator.cross_validation import SplitConfig, walk_forward_splits
from app.ml.trainer.strategy_trainer import LightGBMStrategyBackend

if TYPE_CHECKING:
    import optuna


class HyperparameterSearchConfig(BaseModel):
    """Optuna search space configuration."""

    n_trials: int = Field(default=20, ge=5, le=200)
    timeout_seconds: int | None = Field(default=None, ge=60)


class HyperparameterOptimizer:
    """Optimize LightGBM hyperparameters with Optuna."""

    def __init__(self, backend: LightGBMStrategyBackend | None = None) -> None:
        self._backend = backend or LightGBMStrategyBackend()

    def optimize(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        label_column: str,
        base_config: TrainingConfig | None = None,
        search_config: HyperparameterSearchConfig | None = None,
    ) -> TrainingConfig:
        """Return best TrainingConfig from Optuna study."""
        try:
            import optuna
        except ImportError as exc:
            msg = "Optuna is required for hyperparameter optimization"
            raise ImportError(msg) from exc

        base_config = base_config or TrainingConfig()
        search_config = search_config or HyperparameterSearchConfig()
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            config = base_config.model_copy(
                update={
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 8, 64),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "colsample_bytree": trial.suggest_float("feature_fraction", 0.5, 1.0),
                    "subsample": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("lambda_l1", 0.0, 5.0),
                    "reg_lambda": trial.suggest_float("lambda_l2", 0.0, 5.0),
                }
            )
            split_config = SplitConfig(
                test_size=config.test_size,
                validation_size=config.validation_size,
            )
            scores: list[float] = []
            for train, val in walk_forward_splits(frame, split_config):
                if len(train) < config.min_train_rows:
                    continue
                if config.task == TrainingTask.REGRESSION:
                    model, _ = self._backend.train_regressor(
                        train[feature_columns],
                        train[label_column],
                        val[feature_columns],
                        val[label_column],
                        config,
                    )
                    predictions = self._backend.predict(model, val[feature_columns])
                    errors = [
                        (actual - predicted) ** 2
                        for actual, predicted in zip(
                            val[label_column].tolist(),
                            predictions,
                            strict=True,
                        )
                    ]
                    scores.append(-float(sum(errors) / len(errors)))
                else:
                    model, _ = self._backend.train_classifier(
                        train[feature_columns],
                        train[label_column],
                        val[feature_columns],
                        val[label_column],
                        config,
                    )
                    probabilities = self._backend.predict_proba(model, val[feature_columns])
                    hits = sum(
                        1
                        for probability, label in zip(probabilities, val[label_column], strict=True)
                        if (probability >= 0.5 and label == 1) or (probability < 0.5 and label == 0)
                    )
                    scores.append(hits / len(probabilities))
            if not scores:
                return 0.0
            return float(sum(scores) / len(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=search_config.n_trials,
            timeout=search_config.timeout_seconds,
        )
        best = study.best_params
        return base_config.model_copy(
            update={
                "learning_rate": best["learning_rate"],
                "num_leaves": best["num_leaves"],
                "max_depth": best["max_depth"],
                "colsample_bytree": best["feature_fraction"],
                "subsample": best["bagging_fraction"],
                "reg_alpha": best["lambda_l1"],
                "reg_lambda": best["lambda_l2"],
            }
        )
