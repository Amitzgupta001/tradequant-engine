"""Feature importance reporting for strategy models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.inspection import permutation_importance


class FeatureImportanceReport(BaseModel):
    """Feature importance artifacts for a strategy model."""

    lightgbm_gain: dict[str, float] = Field(default_factory=dict)
    permutation: dict[str, float] = Field(default_factory=dict)
    shap_available: bool = False
    shap_mean_abs: dict[str, float] = Field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Persist feature importance JSON report."""
        path.write_text(json.dumps(self.model_dump(), indent=2), encoding="utf-8")


def compute_feature_importance(
    model: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: list[str],
    is_regression: bool,
) -> FeatureImportanceReport:
    """Compute LightGBM gain, permutation, and optional SHAP importances."""
    report = FeatureImportanceReport()
    if hasattr(model, "booster_"):
        gain = model.booster_.feature_importance(importance_type="gain")
        report.lightgbm_gain = {
            feature: float(value)
            for feature, value in zip(feature_columns, gain, strict=True)
        }

    scoring = "neg_mean_squared_error" if is_regression else "balanced_accuracy"
    perm = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring=scoring,
    )
    report.permutation = {
        feature: float(value)
        for feature, value in zip(feature_columns, perm.importances_mean, strict=True)
    }

    try:
        import shap
    except ImportError:
        return report

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    if isinstance(shap_values, list):
        values = np.abs(shap_values[1]).mean(axis=0)
    else:
        values = np.abs(shap_values).mean(axis=0)
    report.shap_available = True
    report.shap_mean_abs = {
        feature: float(value)
        for feature, value in zip(feature_columns, values, strict=True)
    }
    return report
