"""Map Dhan SDK responses to domain models."""

from datetime import date, datetime, timedelta, timezone
from typing import Callable

from app.brokers.exceptions import DhanAPIError
from app.domain.candle import Candle
from app.domain.enums.market import ExchangeSegment, InstrumentType, Timeframe
from app.domain.historical import HistoricalRequest, HistoricalResponse

IST = timezone(timedelta(hours=5, minutes=30))

EXCHANGE_SEGMENT_TO_SDK: dict[ExchangeSegment, str] = {
    ExchangeSegment.NSE_EQ: "NSE_EQ",
    ExchangeSegment.NSE_FNO: "NSE_FNO",
    ExchangeSegment.BSE_EQ: "BSE_EQ",
    ExchangeSegment.BSE_FNO: "BSE_FNO",
    ExchangeSegment.MCX_COMM: "MCX_COMM",
    ExchangeSegment.IDX_I: "IDX_I",
}


def to_sdk_exchange_segment(segment: ExchangeSegment) -> str:
    """Convert domain exchange segment to SDK value."""
    return EXCHANGE_SEGMENT_TO_SDK[segment]


def to_sdk_instrument_type(instrument_type: InstrumentType) -> str:
    """Convert domain instrument type to SDK value."""
    return instrument_type.value


def epoch_to_datetime(epoch: int) -> datetime:
    """Convert epoch timestamp to timezone-aware IST datetime."""
    converted = datetime.fromtimestamp(epoch, IST)
    if converted.time() == datetime.min.time():
        return converted.replace(hour=0, minute=0, second=0, microsecond=0)
    return converted


def format_request_date(value: date, timeframe: Timeframe) -> str:
    """Format date for Dhan historical API requests."""
    if timeframe.is_daily:
        return value.isoformat()
    return f"{value.isoformat()} 09:30:00"


def map_historical_response(
    request: HistoricalRequest,
    sdk_response: dict,
    timestamp_converter: Callable[[int], datetime | date] | None = None,
) -> HistoricalResponse:
    """Convert Dhan SDK historical response to domain models."""
    if sdk_response.get("status") != "success":
        remarks = sdk_response.get("remarks", "Unknown Dhan API error")
        raise DhanAPIError(f"Dhan historical data request failed: {remarks}", remarks=remarks)

    data = sdk_response.get("data") or {}
    timestamps = data.get("timestamp") or []
    opens = data.get("open") or []
    highs = data.get("high") or []
    lows = data.get("low") or []
    closes = data.get("close") or []
    volumes = data.get("volume") or []
    open_interests = data.get("open_interest") or []

    converter = timestamp_converter or epoch_to_datetime
    candles: list[Candle] = []

    for index, epoch in enumerate(timestamps):
        converted_ts = converter(int(epoch))
        if isinstance(converted_ts, date) and not isinstance(converted_ts, datetime):
            converted_ts = datetime.combine(converted_ts, datetime.min.time(), tzinfo=IST)

        open_interest = None
        if open_interests and index < len(open_interests):
            value = open_interests[index]
            open_interest = int(value) if value is not None else None

        candles.append(
            Candle(
                timestamp=converted_ts,
                open=float(opens[index]),
                high=float(highs[index]),
                low=float(lows[index]),
                close=float(closes[index]),
                volume=int(volumes[index]),
                open_interest=open_interest,
            )
        )

    return HistoricalResponse(
        instrument=request.instrument,
        timeframe=request.timeframe,
        candles=candles,
        source="dhan",
    )
