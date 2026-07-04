"""Unified label application for strategy datasets."""

import pandas as pd

from app.ml.labels.base import LabelConfig, LabelType
from app.ml.labels.classification import generate_classification_labels
from app.ml.labels.regression import generate_regression_labels
from app.ml.labels.triple_barrier import generate_triple_barrier_labels


def apply_labels(
    frame: pd.DataFrame,
    config: LabelConfig | None = None,
) -> pd.DataFrame:
    """Apply all configured label types and set primary label column."""
    config = config or LabelConfig()
    result = frame.copy()

    if config.label_type in (LabelType.CLASSIFICATION, LabelType.TRIPLE_BARRIER):
        result = generate_classification_labels(result, config)
    if config.label_type == LabelType.REGRESSION:
        result = generate_regression_labels(result, config)
    if config.label_type == LabelType.TRIPLE_BARRIER:
        result = generate_triple_barrier_labels(result, config)

    if config.label_type == LabelType.CLASSIFICATION:
        result["label"] = result["label_classification"]
    elif config.label_type == LabelType.REGRESSION:
        result["label"] = result["label_regression"]
    else:
        result["label"] = result["label_barrier"]

    return result
