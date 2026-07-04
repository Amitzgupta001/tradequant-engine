"""Batch download and feature build for a stock universe."""

import time
from datetime import date, timedelta

from loguru import logger

from app.data.universe.registry import Universe
from app.domain.enums.market import Timeframe
from app.services.training_service import TrainingService


class BatchDownloadResult:
    """Outcome for a single instrument in a batch run."""

    def __init__(
        self,
        security_id: str,
        symbol: str | None,
        candle_count: int = 0,
        feature_count: int = 0,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        self.security_id = security_id
        self.symbol = symbol
        self.candle_count = candle_count
        self.feature_count = feature_count
        self.status = status
        self.error = error

    def to_dict(self) -> dict:
        """Serialize for CLI JSON output."""
        return {
            "security_id": self.security_id,
            "symbol": self.symbol,
            "candle_count": self.candle_count,
            "feature_count": self.feature_count,
            "status": self.status,
            "error": self.error,
        }


class BatchDataService:
    """Download OHLCV and build features for every instrument in a universe."""

    def __init__(self, training_service: TrainingService) -> None:
        self._training = training_service

    def download_universe(
        self,
        universe: Universe,
        timeframe: Timeframe,
        years: int = 5,
        days: int | None = None,
        to_date: date | None = None,
        skip_existing: bool = False,
        sleep_seconds: float = 0.0,
    ) -> list[BatchDownloadResult]:
        """Download and feature-engineer all instruments in a universe."""
        end = to_date or date.today()
        if timeframe.is_daily:
            start = end - timedelta(days=years * 365)
            window_label = f"{years} years"
        else:
            lookback_days = days or 90
            start = end - timedelta(days=lookback_days)
            window_label = f"{lookback_days} days"

        total = len(universe.instruments)
        logger.info(
            "Batch download for {} ({} symbols) over {}",
            universe.name,
            total,
            window_label,
        )

        results: list[BatchDownloadResult] = []
        for index, instrument in enumerate(universe.instruments, start=1):
            label = instrument.symbol or instrument.security_id
            logger.info("[{}/{}] {} ({})", index, total, label, instrument.security_id)
            try:
                if skip_existing and self._training.has_stored_features(instrument, timeframe):
                    feature_count = self._training.count_stored_features(instrument, timeframe)
                    logger.info("Skipping {} — features already stored ({} rows)", label, feature_count)
                    results.append(
                        BatchDownloadResult(
                            security_id=instrument.security_id,
                            symbol=instrument.symbol,
                            feature_count=feature_count,
                            status="skipped",
                        )
                    )
                    continue
                candle_count, feature_count = self._training.prepare_data(
                    instrument,
                    timeframe,
                    start,
                    end,
                )
                self._training.release_batch_memory()
                results.append(
                    BatchDownloadResult(
                        security_id=instrument.security_id,
                        symbol=instrument.symbol,
                        candle_count=candle_count,
                        feature_count=feature_count,
                    )
                )
                self._pause_between_symbols(index, total, label, sleep_seconds)
            except Exception as exc:
                logger.exception("Failed for {}: {}", label, exc)
                results.append(
                    BatchDownloadResult(
                        security_id=instrument.security_id,
                        symbol=instrument.symbol,
                        status="error",
                        error=str(exc),
                    )
                )
                self._training.release_batch_memory()
                self._pause_between_symbols(index, total, label, sleep_seconds)
        return results

    @staticmethod
    def _pause_between_symbols(
        index: int,
        total: int,
        label: str,
        sleep_seconds: float,
    ) -> None:
        """Sleep between symbols to avoid overloading the Dhan API or local machine."""
        if sleep_seconds <= 0 or index >= total:
            return
        logger.info(
            "Waiting {:.1f}s before next symbol ({}/{})",
            sleep_seconds,
            index,
            total,
        )
        time.sleep(sleep_seconds)
