"""Historical data orchestration service."""

from datetime import date
from pathlib import Path

from loguru import logger

from app.core.events import EventType, event_bus
from app.data.providers.historical_data_provider import HistoricalDataProvider
from app.data.repositories.base import HistoricalRepository
from app.domain.enums.market import Timeframe
from app.domain.historical import HistoricalRequest, HistoricalResponse


class HistoricalDataService:
    """Download historical data and persist via repository."""

    def __init__(
        self,
        provider: HistoricalDataProvider,
        repository: HistoricalRepository,
    ) -> None:
        self._provider = provider
        self._repository = repository

    def download(self, request: HistoricalRequest) -> HistoricalResponse:
        """Fetch historical data without persisting."""
        logger.info(
            "Downloading historical data for security_id={}",
            request.instrument.security_id,
        )
        response = self._provider.fetch(request)
        if response.candles:
            event_bus.publish(EventType.NEW_CANDLE, response.candles[-1])
        return response

    def download_range(
        self,
        instrument,
        timeframe: Timeframe,
        from_date: date,
        to_date: date,
        include_oi: bool = False,
    ) -> HistoricalResponse:
        """Fetch historical data for a date range (intraday chunking handled by provider)."""
        request = HistoricalRequest(
            instrument=instrument,
            from_date=from_date,
            to_date=to_date,
            timeframe=timeframe,
            include_oi=include_oi,
        )
        return self.download(request)

    def download_and_store(
        self,
        request: HistoricalRequest,
        overwrite: bool = True,
    ) -> tuple[HistoricalResponse, Path]:
        """Fetch historical data and save via repository."""
        response = self.download(request)
        path = self._repository.save(response, overwrite=overwrite)
        return response, path

    def clear_cache(self) -> None:
        """Clear in-memory historical data cache."""
        if self._provider._cache is not None:
            self._provider._cache.clear()

    def download_range_and_store(
        self,
        instrument,
        timeframe: Timeframe,
        from_date: date,
        to_date: date,
        overwrite: bool = True,
        include_oi: bool = False,
    ) -> tuple[HistoricalResponse, Path]:
        """Fetch a date range and persist to CSV."""
        response = self.download_range(
            instrument,
            timeframe,
            from_date,
            to_date,
            include_oi=include_oi,
        )
        path = self._repository.save(response, overwrite=overwrite)
        return response, path
