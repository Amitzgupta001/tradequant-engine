"""Triple barrier labeling for event-driven ML."""

import pandas as pd

from app.ml.labels.base import LabelConfig


def generate_triple_barrier_labels(frame: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    """Apply take-profit, stop-loss, and time barrier labeling."""
    result = frame.copy()
    closes = result["close"].tolist()
    highs = result["high"].tolist()
    lows = result["low"].tolist()
    labels: list[int | None] = [None] * len(result)

    for index in range(len(result)):
        entry = closes[index]
        if entry <= 0:
            continue
        take_profit = entry * (1 + config.take_profit_pct)
        stop_loss = entry * (1 - config.stop_loss_pct)
        end = min(len(result), index + config.time_barrier_bars + 1)
        label = 0
        for step in range(index + 1, end):
            if highs[step] >= take_profit:
                label = 1
                break
            if lows[step] <= stop_loss:
                label = -1
                break
        labels[index] = label

    result["label_barrier"] = labels
    return result
