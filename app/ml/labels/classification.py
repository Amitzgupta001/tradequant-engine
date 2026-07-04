"""Classification label generation (BUY / SELL / HOLD)."""

import pandas as pd

from app.ml.labels.base import LabelConfig, SignalLabel


def generate_classification_labels(frame: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    """Label rows by forward return relative to a threshold."""
    result = frame.copy()
    horizon = config.forward_horizon_bars
    closes = result["close"].tolist()
    labels: list[str | None] = [None] * len(result)

    for index in range(len(result)):
        if index + horizon >= len(result):
            continue
        entry = closes[index]
        future = closes[index + horizon]
        if entry <= 0:
            continue
        forward_return = (future - entry) / entry
        if forward_return >= config.regression_threshold:
            labels[index] = SignalLabel.BUY.value
        elif forward_return <= -config.regression_threshold:
            labels[index] = SignalLabel.SELL.value
        else:
            labels[index] = SignalLabel.HOLD.value

    result["label_classification"] = labels
    result["forward_return"] = [
        None
        if index + horizon >= len(result) or closes[index] <= 0
        else (closes[index + horizon] - closes[index]) / closes[index]
        for index in range(len(result))
    ]
    return result
