"""Prepare feature vectors for ML training."""

import pandas as pd

from app.domain.enums.market import Timeframe
from app.domain.features import FeatureVector
from app.domain.training import SetupType, TrainingConfig

FEATURE_COLUMNS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "log_return_1d",
    "high_low_range_pct",
    "body_pct",
    "volume_change_pct",
    "ema_gap_pct",
    "rsi_14",
    "rsi_change_3d",
    "macd_histogram",
    "atr_pct",
    "bb_width_pct",
    "bb_position",
    "vwap_gap_pct",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "volatility_5d",
    "volatility_10d",
    "volume_ratio_5d",
    "up_ratio_20d",
    "trend_20d",
]

LABEL_COLUMN = "forward_return_1d"
FORWARD_RETURN_COLUMNS = {
    1: "forward_return_1d",
    5: "forward_return_5d",
    20: "forward_return_20b",
}

INTRADAY_DEFAULTS_BY_TIMEFRAME: dict[Timeframe, dict[str, float | int]] = {
    Timeframe.MIN_5: {"forward_horizon_bars": 20, "move_threshold": 0.003},
    Timeframe.MIN_15: {"forward_horizon_bars": 20, "move_threshold": 0.004},
}
DEFAULT_INTRADAY = {"forward_horizon_bars": 20, "move_threshold": 0.004}


def training_config_for_timeframe(timeframe: Timeframe, base: TrainingConfig | None = None) -> TrainingConfig:
    """Apply timeframe-specific defaults for swing setup training."""
    config = base or TrainingConfig()
    if timeframe.is_daily:
        return config
    overrides = INTRADAY_DEFAULTS_BY_TIMEFRAME.get(timeframe, DEFAULT_INTRADAY)
    return config.model_copy(update=overrides)


def label_column_for_horizon(horizon_bars: int) -> str:
    """Return the forward-return column for a prediction horizon in bars."""
    column = FORWARD_RETURN_COLUMNS.get(horizon_bars)
    if column is None:
        msg = f"Unsupported forward horizon: {horizon_bars} bars"
        raise ValueError(msg)
    return column


def features_to_rows(
    features: list[FeatureVector],
    label_column: str = LABEL_COLUMN,
) -> list[dict[str, float]]:
    """Convert feature vectors to flat dict rows for pandas."""
    rows: list[dict[str, float]] = []
    for feature in features:
        row = {column: getattr(feature, column) for column in FEATURE_COLUMNS}
        row[LABEL_COLUMN] = feature.forward_return_1d
        row["forward_return_5d"] = feature.forward_return_5d
        row["forward_return_20b"] = feature.forward_return_20b
        if label_column not in row:
            row[label_column] = getattr(feature, label_column)
        rows.append(row)
    return rows


def classification_label(forward_return: float, threshold: float = 0.0) -> int:
    """Convert forward return into binary direction label."""
    return 1 if forward_return > threshold else 0


def swing_setup_label(
    forward_return: float,
    move_threshold: float,
    setup_type: SetupType,
) -> int:
    """Label whether a technical setup succeeded over the forward horizon."""
    if setup_type == SetupType.LONG:
        return 1 if forward_return > move_threshold else 0
    return 1 if forward_return < -move_threshold else 0


def is_long_setup(row: pd.Series) -> bool:
    """Detect oversold / bounce long setups."""
    rsi = row.get("rsi_14")
    bb_position = row.get("bb_position")
    macd_histogram = row.get("macd_histogram")
    if rsi is None or bb_position is None or macd_histogram is None:
        return False
    return bool(
        rsi < 40
        or bb_position < 0.15
        or (rsi < 45 and macd_histogram > 0)
    )


def is_short_setup(row: pd.Series) -> bool:
    """Detect overbought / reversal short setups."""
    rsi = row.get("rsi_14")
    bb_position = row.get("bb_position")
    macd_histogram = row.get("macd_histogram")
    if rsi is None or bb_position is None or macd_histogram is None:
        return False
    return bool(
        rsi > 60
        or bb_position > 0.85
        or (rsi > 55 and macd_histogram < 0)
    )


def is_setup_row(row: pd.Series, setup_type: SetupType) -> bool:
    """Check if a row matches the configured setup filter."""
    if setup_type == SetupType.LONG:
        return is_long_setup(row)
    return is_short_setup(row)


def feature_matches_setup(feature: FeatureVector, setup_type: SetupType) -> bool:
    """Check if a feature vector matches a technical setup."""
    row = pd.Series({column: getattr(feature, column) for column in FEATURE_COLUMNS})
    return is_setup_row(row, setup_type)


def apply_setup_filter(frame: pd.DataFrame, setup_type: SetupType) -> pd.DataFrame:
    """Keep only rows that match a technical setup."""
    if setup_type == SetupType.LONG:
        mask = (
            (frame["rsi_14"] < 40)
            | (frame["bb_position"] < 0.15)
            | ((frame["rsi_14"] < 45) & (frame["macd_histogram"] > 0))
        )
    else:
        mask = (
            (frame["rsi_14"] > 60)
            | (frame["bb_position"] > 0.85)
            | ((frame["rsi_14"] > 55) & (frame["macd_histogram"] < 0))
        )
    return frame.loc[mask].copy()


def prepare_training_frame_from_df(
    frame: pd.DataFrame,
    config: TrainingConfig,
    label_column: str,
) -> pd.DataFrame:
    """Apply cleaning, setup filter, and labels to a feature dataframe."""
    working = frame.dropna(subset=FEATURE_COLUMNS + [label_column])

    if config.label_threshold > 0:
        working = working[working[label_column].abs() > config.label_threshold]

    if config.setup_type is not None:
        working = apply_setup_filter(working, config.setup_type)
        working = working.copy()
        working["label"] = working[label_column].apply(
            lambda value: swing_setup_label(value, config.move_threshold, config.setup_type)
        )
    elif config.task.value == "classification":
        working = working.copy()
        working["label"] = working[label_column].apply(classification_label)
    else:
        working = working.copy()
        working["label"] = working[label_column]

    return working


def build_training_frame(
    features: list[FeatureVector],
    config: TrainingConfig,
) -> tuple[pd.DataFrame, str]:
    """Build a cleaned dataframe ready for model training."""
    label_column = (
        label_column_for_horizon(config.forward_horizon_bars)
        if config.setup_type is not None
        else LABEL_COLUMN
    )
    frame = pd.DataFrame(features_to_rows(features, label_column=label_column))
    return prepare_training_frame_from_df(frame, config, label_column), label_column
