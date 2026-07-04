"""Volume Weighted Average Price indicator."""

from datetime import date

from app.domain.candle import Candle
from app.indicators.base import IndicatorSeries


def compute_vwap(candles: list[Candle]) -> IndicatorSeries:
    """Compute session VWAP, resetting at each trading date."""
    if not candles:
        return IndicatorSeries(name="vwap", values=[])

    values: list[float | None] = []
    current_date: date | None = None
    cumulative_pv = 0.0
    cumulative_volume = 0

    for candle in candles:
        trading_date = candle.timestamp.date()
        if current_date != trading_date:
            current_date = trading_date
            cumulative_pv = 0.0
            cumulative_volume = 0

        typical_price = (candle.high + candle.low + candle.close) / 3
        cumulative_pv += typical_price * candle.volume
        cumulative_volume += candle.volume

        if cumulative_volume == 0:
            values.append(None)
        else:
            values.append(cumulative_pv / cumulative_volume)

    return IndicatorSeries(name="vwap", values=values)
