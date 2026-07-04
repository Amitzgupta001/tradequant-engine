"""Panel dataset loading — merge features across multiple instruments."""

from loguru import logger
import pandas as pd

from app.domain.enums.market import Timeframe
from app.domain.features import FeatureVector
from app.domain.instrument import Instrument
from app.domain.training import TrainingConfig
from app.ml.datasets.preparation import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    label_column_for_horizon,
    prepare_training_frame_from_df,
)
from app.ml.feature_store.repository import FeatureRepository

FORWARD_COLUMNS = ("forward_return_1d", "forward_return_5d", "forward_return_20b")


def load_panel_features(
    feature_repository: FeatureRepository,
    instruments: list[Instrument],
    timeframe: Timeframe,
) -> tuple[list[FeatureVector], list[str]]:
    """Load and concatenate feature vectors from multiple instruments.

    Prefer :func:`build_panel_training_frame` for large universes — this loads
    every row as a Pydantic object and can use several GB of RAM.
    """
    merged: list[FeatureVector] = []
    loaded_ids: list[str] = []

    for instrument in instruments:
        try:
            features = feature_repository.load(instrument, timeframe)
        except FileNotFoundError:
            label = instrument.symbol or instrument.security_id
            logger.warning("Skipping {} — no feature file", label)
            continue
        if not features:
            continue
        merged.extend(features)
        loaded_ids.append(instrument.security_id)
        logger.debug(
            "Loaded {} feature rows for {}",
            len(features),
            instrument.symbol or instrument.security_id,
        )

    return merged, loaded_ids


def build_panel_training_frame(
    feature_repository: FeatureRepository,
    instruments: list[Instrument],
    timeframe: Timeframe,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, str, list[str]]:
    """Build a training dataframe by reading feature CSVs one symbol at a time.

    Only setup-filtered rows are kept in memory, avoiding multi-GB Pydantic loads.
    """
    label_column = (
        label_column_for_horizon(config.forward_horizon_bars)
        if config.setup_type is not None
        else LABEL_COLUMN
    )
    needed_columns = set(FEATURE_COLUMNS) | set(FORWARD_COLUMNS) | {label_column}

    parts: list[pd.DataFrame] = []
    loaded_ids: list[str] = []

    for instrument in instruments:
        path = feature_repository.build_path(instrument, timeframe)
        if not path.exists():
            label = instrument.symbol or instrument.security_id
            logger.warning("Skipping {} — no feature file", label)
            continue

        raw = pd.read_csv(path, usecols=lambda column: column in needed_columns)
        prepared = prepare_training_frame_from_df(raw, config, label_column)
        if prepared.empty:
            continue

        parts.append(prepared)
        loaded_ids.append(instrument.security_id)
        logger.debug(
            "Loaded {} setup rows for {}",
            len(prepared),
            instrument.symbol or instrument.security_id,
        )

    if not parts:
        msg = "No panel training rows found in feature files"
        raise FileNotFoundError(msg)

    frame = pd.concat(parts, ignore_index=True)
    logger.info(
        "Panel frame: {} setup rows from {} symbols",
        len(frame),
        len(loaded_ids),
    )
    return frame, label_column, loaded_ids
