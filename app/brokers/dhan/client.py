"""Dhan broker client wrapping the official dhanhq SDK."""

from typing import Any

from dhanhq import DhanContext, dhanhq
from loguru import logger

from app.brokers.dhan.mapper import (
    format_request_date,
    map_historical_response,
    to_sdk_exchange_segment,
    to_sdk_instrument_type,
)
from app.brokers.exceptions import DhanAPIError
from app.domain.historical import HistoricalRequest, HistoricalResponse
from app.domain.instrument import Instrument


class DhanClient:
    """Reusable Dhan client for market data and future trading operations."""

    def __init__(self, client_id: str, access_token: str) -> None:
        self._client_id = client_id
        self._access_token = access_token
        self._context = DhanContext(client_id, access_token)
        self._client = dhanhq(self._context)
        logger.info("Initialized Dhan client for client_id={}", client_id)

    def get_historical_data(self, request: HistoricalRequest) -> HistoricalResponse:
        """Fetch historical OHLCV candles via the Dhan SDK."""
        instrument = request.instrument
        exchange_segment = to_sdk_exchange_segment(instrument.exchange_segment)
        instrument_type = to_sdk_instrument_type(instrument.instrument_type)
        from_date = format_request_date(request.from_date, request.timeframe)
        to_date = format_request_date(request.to_date, request.timeframe)

        logger.info(
            "Fetching historical data security_id={} timeframe={} from={} to={}",
            instrument.security_id,
            request.timeframe.value,
            from_date,
            to_date,
        )

        if request.timeframe.is_daily:
            sdk_response = self._client.historical_daily_data(
                security_id=instrument.security_id,
                exchange_segment=exchange_segment,
                instrument_type=instrument_type,
                from_date=from_date,
                to_date=to_date,
                expiry_code=request.expiry_code,
                oi=request.include_oi,
            )
        else:
            interval = request.timeframe.interval_minutes
            if interval is None:
                raise DhanAPIError(f"Unsupported intraday timeframe: {request.timeframe}")

            sdk_response = self._client.intraday_minute_data(
                security_id=instrument.security_id,
                exchange_segment=exchange_segment,
                instrument_type=instrument_type,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                oi=request.include_oi,
            )

        return map_historical_response(
            request,
            sdk_response,
            timestamp_converter=self._client.convert_to_date_time,
        )

    def connect_market_feed(self, instruments: list[Instrument]) -> None:
        """Connect to live market feed. Available in Phase 7."""
        raise NotImplementedError("Live market feed is available in Phase 7.")

    def place_order(self, order: Any) -> Any:
        """Place a trade order. Available in Phase 7."""
        raise NotImplementedError("Order placement is available in Phase 7.")

    def get_positions(self) -> Any:
        """Retrieve open positions. Available in Phase 7."""
        raise NotImplementedError("Position management is available in Phase 7.")
