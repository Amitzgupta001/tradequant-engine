"""Time-series cross validation utilities."""

from collections.abc import Iterator

import pandas as pd
from pydantic import BaseModel, Field


class SplitConfig(BaseModel):
    """Configuration for chronological data splits."""

    test_size: float = Field(default=0.2, gt=0.0, lt=0.5)
    validation_size: float = Field(default=0.15, gt=0.0, lt=0.4)
    n_splits: int = Field(default=5, ge=2, le=20)
    train_window: int = Field(default=100, ge=20)
    step_size: int = Field(default=20, ge=1)


def time_series_split(
    frame: pd.DataFrame,
    config: SplitConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically into train, validation, and test sets."""
    config = config or SplitConfig()
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
    return train, val, test


def walk_forward_splits(
    frame: pd.DataFrame,
    config: SplitConfig | None = None,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield expanding-window walk-forward train/validation splits."""
    config = config or SplitConfig()
    minimum = config.train_window + 1
    if len(frame) < minimum:
        msg = f"Need at least {minimum} rows for walk-forward validation"
        raise ValueError(msg)

    start = config.train_window
    while start < len(frame):
        train = frame.iloc[:start]
        end = min(len(frame), start + config.step_size)
        val = frame.iloc[start:end]
        if len(val) == 0:
            break
        yield train, val
        start += config.step_size


def rolling_window_splits(
    frame: pd.DataFrame,
    config: SplitConfig | None = None,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield fixed-size rolling train/validation splits."""
    config = config or SplitConfig()
    minimum = config.train_window + config.step_size
    if len(frame) < minimum:
        msg = f"Need at least {minimum} rows for rolling validation"
        raise ValueError(msg)

    start = 0
    while start + config.train_window + config.step_size <= len(frame):
        train = frame.iloc[start : start + config.train_window]
        val = frame.iloc[
            start + config.train_window : start + config.train_window + config.step_size
        ]
        yield train, val
        start += config.step_size
