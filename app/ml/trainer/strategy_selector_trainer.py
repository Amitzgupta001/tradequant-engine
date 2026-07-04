"""Train a meta-model that picks the best strategy from backtest benchmarks."""

import json
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from app.domain.training import TrainingConfig, TrainingMetrics
from app.ml.datasets.strategy_selector_builder import (
    StrategySelectorDatasetMetadata,
)
from app.ml.evaluator.cross_validation import SplitConfig, time_series_split
from app.ml.registry.strategy_selector_registry import (
    StrategySelectorMetadata,
    StrategySelectorRegistry,
)
from app.ml.selector.confidence import passes_selector_gate


class StrategySelectorTrainer:
    """Train a multiclass selector from walk-forward backtest labels."""

    def __init__(self, registry: StrategySelectorRegistry) -> None:
        self._registry = registry

    def train(
        self,
        frame: pd.DataFrame,
        metadata: StrategySelectorDatasetMetadata,
        config: TrainingConfig | None = None,
        split_config: SplitConfig | None = None,
    ) -> StrategySelectorMetadata:
        """Train selector model and tune confidence on validation backtest PnL."""
        config = config or TrainingConfig()
        split_config = split_config or SplitConfig(
            test_size=config.test_size,
            validation_size=config.validation_size,
        )
        working = self._prepare_frame(frame, metadata.feature_columns)
        if len(working) < config.min_train_rows:
            msg = f"Need at least {config.min_train_rows} selector rows, got {len(working)}"
            raise ValueError(msg)

        encoder = LabelEncoder()
        working["label_encoded"] = encoder.fit_transform(working["best_strategy_id"])
        train, val, test = time_series_split(working, split_config)
        feature_columns = metadata.feature_columns

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
            objective="multiclass",
            num_class=len(encoder.classes_),
            random_state=config.random_state,
            verbose=-1,
        )
        model.fit(
            train[feature_columns],
            train["label_encoded"],
            eval_set=[(val[feature_columns], val["label_encoded"])],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(config.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        val_probs = model.predict_proba(val[feature_columns])
        test_probs = model.predict_proba(test[feature_columns])
        min_margin, val_backtest_metrics = self._tune_on_backtest(
            val,
            val_probs,
            encoder,
            metadata.objective.value,
        )
        test_backtest_metrics = self._evaluate_backtest_selection(
            test,
            test_probs,
            encoder,
            min_margin,
        )
        y_pred = np.argmax(test_probs, axis=1)
        test_metrics = TrainingMetrics(
            train_rows=len(train),
            validation_rows=len(val),
            test_rows=len(test),
            accuracy=accuracy_score(test["label_encoded"], y_pred),
            balanced_accuracy=balanced_accuracy_score(test["label_encoded"], y_pred),
            prediction_threshold=min_margin,
            signal_count=int(test_backtest_metrics.get("selected_windows") or 0),
            signal_hit_rate=test_backtest_metrics.get("selector_hit_rate"),
        )

        version = self._registry.next_version(
            metadata.exchange_segment,
            metadata.security_id,
            metadata.timeframe,
            universe_id=metadata.universe_id,
        )
        model_metadata = StrategySelectorMetadata(
            version=version,
            security_id=metadata.security_id,
            exchange_segment=metadata.exchange_segment,
            timeframe=metadata.timeframe,
            feature_columns=feature_columns,
            strategy_ids=metadata.strategy_ids,
            objective=metadata.objective,
            builder_config=metadata.builder_config,
            metrics=test_metrics,
            backtest_metrics=TrainingMetrics(
                train_rows=len(train),
                validation_rows=len(val),
                test_rows=len(test),
                signal_hit_rate=val_backtest_metrics.get("selector_hit_rate"),
                signal_count=int(val_backtest_metrics.get("selected_windows") or 0),
                prediction_threshold=min_margin,
            ),
            min_confidence=0.0,
            min_margin=min_margin,
            parameters=config.model_dump(mode="json"),
            universe_id=metadata.universe_id,
            constituent_count=metadata.constituent_count,
            git_commit_hash=StrategySelectorRegistry.git_commit_hash(),
            trained_at=datetime.now(timezone.utc),
        )
        self._registry.save(model, encoder, model_metadata)
        logger.info(
            "Trained strategy selector v{} accuracy={:.3f} backtest_hit={:.3f}",
            version,
            test_metrics.accuracy or 0.0,
            test_backtest_metrics.get("selector_hit_rate") or 0.0,
        )
        return model_metadata

    @staticmethod
    def _prepare_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        """Clean selector training frame."""
        working = frame.copy()
        for column in feature_columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
        working = working.dropna(subset=feature_columns + ["best_strategy_id"])
        return working

    def _tune_on_backtest(
        self,
        frame: pd.DataFrame,
        probabilities: np.ndarray,
        encoder: LabelEncoder,
        objective: str,
    ) -> tuple[float, dict[str, float | None]]:
        """Pick probability-margin threshold that maximizes simulated backtest return."""
        margins = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15]
        best_margin = 0.0
        best_metrics: dict[str, float | None] = {
            "simulated_return_pct": float("-inf"),
            "simulated_profit_factor": None,
            "selector_hit_rate": None,
        }
        for margin in margins:
            metrics = self._evaluate_backtest_selection(frame, probabilities, encoder, margin)
            score = metrics["simulated_return_pct"]
            if objective == "profit_factor":
                score = metrics["simulated_profit_factor"] or float("-inf")
            if score is None:
                continue
            current_best = best_metrics["simulated_return_pct"] or float("-inf")
            if objective == "profit_factor":
                current_best = best_metrics["simulated_profit_factor"] or float("-inf")
            if score is not None and score > current_best:
                best_margin = margin
                best_metrics = metrics
        return best_margin, best_metrics

    @staticmethod
    def _evaluate_backtest_selection(
        frame: pd.DataFrame,
        probabilities: np.ndarray,
        encoder: LabelEncoder,
        min_margin: float,
    ) -> dict[str, float | None]:
        """Simulate selecting one strategy per window using model margin."""
        selected_returns: list[float] = []
        selected_wins = 0.0
        selected_losses = 0.0
        hits = 0
        selected = 0
        strategy_ids = list(encoder.classes_)

        for index, (_, row) in enumerate(frame.iterrows()):
            best_index, _, _, passes = passes_selector_gate(
                probabilities[index],
                min_margin=min_margin,
            )
            if not passes:
                continue
            predicted_id = strategy_ids[best_index]
            returns_map = json.loads(row["strategy_returns_json"])
            return_pct = float(returns_map.get(predicted_id, 0.0))
            selected_returns.append(return_pct)
            selected += 1
            if predicted_id == row["best_strategy_id"]:
                hits += 1
            if return_pct >= 0:
                selected_wins += return_pct
            else:
                selected_losses += abs(return_pct)

        simulated_return = sum(selected_returns) if selected_returns else 0.0
        profit_factor = None
        if selected_losses > 0:
            profit_factor = selected_wins / selected_losses
        elif selected_wins > 0:
            profit_factor = float("inf")

        return {
            "simulated_return_pct": simulated_return,
            "simulated_profit_factor": profit_factor,
            "selector_hit_rate": (hits / selected) if selected else None,
            "selected_windows": float(selected),
        }
