"""Average True Range indicator."""

from app.indicators.base import IndicatorSeries, validate_period, wilder_smooth


def compute_atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> IndicatorSeries:
    """Compute ATR using Wilder's smoothing."""
    validate_period(period)
    length = len(closes)
    if not (len(highs) == len(lows) == length):
        msg = "highs, lows, and closes must have equal length"
        raise ValueError(msg)
    if length == 0:
        return IndicatorSeries(name=f"atr_{period}", values=[])

    true_ranges: list[float] = [highs[0] - lows[0]]
    for index in range(1, length):
        high_low = highs[index] - lows[index]
        high_close = abs(highs[index] - closes[index - 1])
        low_close = abs(lows[index] - closes[index - 1])
        true_ranges.append(max(high_low, high_close, low_close))

    atr_values = wilder_smooth(true_ranges, period)
    return IndicatorSeries(name=f"atr_{period}", values=atr_values)
