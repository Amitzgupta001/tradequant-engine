"""MACD indicator."""

from app.indicators.base import MACDResult, ema, validate_period


def compute_macd(
    closes: list[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> MACDResult:
    """Compute MACD line, signal line, and histogram."""
    validate_period(fast_period)
    validate_period(slow_period)
    validate_period(signal_period)
    if slow_period <= fast_period:
        msg = "slow_period must be greater than fast_period"
        raise ValueError(msg)

    fast_ema = ema(closes, fast_period)
    slow_ema = ema(closes, slow_period)

    macd_line: list[float | None] = [None] * len(closes)
    macd_floats: list[float] = []
    macd_indices: list[int] = []

    for index in range(len(closes)):
        fast = fast_ema[index]
        slow = slow_ema[index]
        if fast is None or slow is None:
            continue
        value = fast - slow
        macd_line[index] = value
        macd_floats.append(value)
        macd_indices.append(index)

    signal_line: list[float | None] = [None] * len(closes)
    histogram: list[float | None] = [None] * len(closes)

    if len(macd_floats) >= signal_period:
        signal_values = ema(macd_floats, signal_period)
        for offset, signal_value in enumerate(signal_values):
            if signal_value is None:
                continue
            candle_index = macd_indices[offset]
            signal_line[candle_index] = signal_value
            macd_value = macd_line[candle_index]
            if macd_value is not None:
                histogram[candle_index] = macd_value - signal_value

    return MACDResult(macd=macd_line, signal=signal_line, histogram=histogram)
