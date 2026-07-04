"""Indicator computation and persistence service."""

import csv
from pathlib import Path

from loguru import logger

from app.data.repositories.base import HistoricalRepository
from app.domain.enums.market import Timeframe
from app.domain.indicators import IndicatorSnapshot
from app.domain.instrument import Instrument
from app.indicators.engine import IndicatorEngine


class IndicatorService:
    """Load candles and compute technical indicators."""

    INDICATOR_COLUMNS = [
        "timestamp",
        "ema_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "atr_14",
        "vwap",
        "bb_upper",
        "bb_middle",
        "bb_lower",
    ]

    def __init__(
        self,
        repository: HistoricalRepository,
        engine: IndicatorEngine | None = None,
        processed_path: Path | None = None,
    ) -> None:
        self._repository = repository
        self._engine = engine or IndicatorEngine()
        self._processed_path = processed_path or Path("storage/processed")

    def compute_for_instrument(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
    ) -> list[IndicatorSnapshot]:
        """Load raw candles and compute indicators."""
        response = self._repository.load(instrument, timeframe)
        logger.info(
            "Computing indicators for security_id={} candles={}",
            instrument.security_id,
            len(response.candles),
        )
        return self._engine.compute(response.candles)

    def compute_and_store(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        overwrite: bool = True,
    ) -> tuple[list[IndicatorSnapshot], Path]:
        """Compute indicators and save to processed storage."""
        snapshots = self.compute_for_instrument(instrument, timeframe)
        path = self._build_path(instrument, timeframe)

        if path.exists() and not overwrite:
            logger.info("Skipping existing indicator file at {}", path)
            return snapshots, path

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.INDICATOR_COLUMNS)
            writer.writeheader()
            for snapshot in snapshots:
                writer.writerow(snapshot.model_dump(mode="json"))

        logger.info("Saved {} indicator rows to {}", len(snapshots), path)
        return snapshots, path

    def _build_path(self, instrument: Instrument, timeframe: Timeframe) -> Path:
        return (
            self._processed_path
            / instrument.exchange_segment.value
            / instrument.security_id
            / f"{timeframe.value.lower()}_indicators.csv"
        )
