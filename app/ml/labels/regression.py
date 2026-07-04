"""Regression label generation for expected future return."""

import pandas as pd

from app.ml.labels.base import LabelConfig


def generate_regression_labels(frame: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    """Label rows with forward return over the configured horizon."""
    result = frame.copy()
    horizon = config.forward_horizon_bars
    closes = result["close"].tolist()
    labels: list[float | None] = [None] * len(result)

    for index in range(len(result)):
        if index + horizon >= len(result):
            continue
        entry = closes[index]
        future = closes[index + horizon]
        if entry <= 0:
            continue
        labels[index] = (future - entry) / entry

    result["label_regression"] = labels
    result["forward_return"] = labels
    return result
