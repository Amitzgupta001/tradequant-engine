"""Historical market data provider."""

from datetime import date, timedelta

from loguru import logger

from app.brokers.base import BrokerClient
from app.cache.memory import InMemoryCache
from app.domain.historical import HistoricalRequest, HistoricalResponse

INTRADAY_CHUNK_DAYS = 90


class HistoricalDataProvider:
    """Fetch historical OHLCV data through broker clients with optional caching."""

    def __init__(
        self,
        broker: BrokerClient,
        cache: InMemoryCache[HistoricalResponse] | None = None,
    ) -> None:
        self._broker = broker
        self._cache = cache

    def fetch(self, request: HistoricalRequest) -> HistoricalResponse:
        """Fetch historical candles, using cache and chunking when needed."""
        cache_key = self._cache_key(request)
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for historical request key={}", cache_key)
                return cached

        if request.timeframe.is_daily:
            response = self._broker.get_historical_data(request)
        else:
            response = self._fetch_intraday_chunked(request)

        if self._cache is not None:
            self._cache.set(cache_key, response)

        return response

    def _fetch_intraday_chunked(self, request: HistoricalRequest) -> HistoricalResponse:
        """Fetch intraday data in API-sized date chunks."""
        chunks = self._split_date_range(request.from_date, request.to_date, INTRADAY_CHUNK_DAYS)
        all_candles = []
        response: HistoricalResponse | None = None

        for chunk_start, chunk_end in chunks:
            chunk_request = request.model_copy(
                update={"from_date": chunk_start, "to_date": chunk_end}
            )
            chunk_response = self._broker.get_historical_data(chunk_request)
            all_candles.extend(chunk_response.candles)
            response = chunk_response

        if response is None:
            return HistoricalResponse(
                instrument=request.instrument,
                timeframe=request.timeframe,
                candles=[],
            )

        deduped = self._dedupe_candles(all_candles)
        logger.info(
            "Fetched {} intraday candles for security_id={}",
            len(deduped),
            request.instrument.security_id,
        )
        return response.model_copy(update={"candles": deduped})

    @staticmethod
    def _cache_key(request: HistoricalRequest) -> str:
        instrument = request.instrument
        return (
            f"{instrument.exchange_segment.value}:"
            f"{instrument.security_id}:"
            f"{request.timeframe.value}:"
            f"{request.from_date.isoformat()}:"
            f"{request.to_date.isoformat()}:"
            f"{request.include_oi}"
        )

    @staticmethod
    def _split_date_range(
        start: date,
        end: date,
        chunk_days: int,
    ) -> list[tuple[date, date]]:
        """Split a date range into inclusive chunks."""
        chunks: list[tuple[date, date]] = []
        current = start

        while current <= end:
            chunk_end = min(current + timedelta(days=chunk_days - 1), end)
            chunks.append((current, chunk_end))
            current = chunk_end + timedelta(days=1)

        return chunks

    @staticmethod
    def _dedupe_candles(candles: list) -> list:
        """Remove duplicate candles by timestamp while preserving order."""
        seen: set = set()
        deduped = []
        for candle in candles:
            key = candle.timestamp
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candle)
        return deduped
