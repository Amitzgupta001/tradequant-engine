"""Shared types and math helpers for technical indicators."""

from pydantic import BaseModel, Field


class IndicatorSeries(BaseModel):
    """Named indicator values aligned with input candles."""

    name: str
    values: list[float | None]


class BollingerBands(BaseModel):
    """Bollinger Bands output."""

    upper: list[float | None]
    middle: list[float | None]
    lower: list[float | None]


class MACDResult(BaseModel):
    """MACD indicator output."""

    macd: list[float | None]
    signal: list[float | None]
    histogram: list[float | None]


def validate_period(period: int, minimum: int = 1) -> None:
    """Validate indicator lookback period."""
    if period < minimum:
        msg = f"period must be >= {minimum}, got {period}"
        raise ValueError(msg)


def validate_candles(length: int, period: int) -> None:
    """Validate candle count against period."""
    if length == 0:
        msg = "candles must not be empty"
        raise ValueError(msg)
    if length < period:
        msg = f"need at least {period} candles, got {length}"
        raise ValueError(msg)


def sma(values: list[float], period: int) -> list[float | None]:
    """Simple moving average aligned to input length."""
    validate_period(period)
    result: list[float | None] = [None] * len(values)
    if len(values) < period:
        return result

    window_sum = sum(values[:period])
    result[period - 1] = window_sum / period
    for index in range(period, len(values)):
        window_sum += values[index] - values[index - period]
        result[index] = window_sum / period
    return result


def ema(values: list[float], period: int) -> list[float | None]:
    """Exponential moving average using SMA seed."""
    validate_period(period)
    result: list[float | None] = [None] * len(values)
    if len(values) < period:
        return result

    seed = sum(values[:period]) / period
    result[period - 1] = seed
    multiplier = 2 / (period + 1)

    previous = seed
    for index in range(period, len(values)):
        current = (values[index] - previous) * multiplier + previous
        result[index] = current
        previous = current
    return result


def wilder_smooth(values: list[float], period: int) -> list[float | None]:
    """Wilder's smoothing (used by RSI and ATR)."""
    validate_period(period)
    result: list[float | None] = [None] * len(values)
    if len(values) < period:
        return result

    seed = sum(values[:period]) / period
    result[period - 1] = seed
    previous = seed
    for index in range(period, len(values)):
        previous = ((previous * (period - 1)) + values[index]) / period
        result[index] = previous
    return result


def rolling_std(values: list[float], period: int) -> list[float | None]:
    """Rolling standard deviation aligned to input length."""
    validate_period(period, minimum=2)
    result: list[float | None] = [None] * len(values)
    if len(values) < period:
        return result

    for index in range(period - 1, len(values)):
        window = values[index - period + 1 : index + 1]
        mean = sum(window) / period
        variance = sum((value - mean) ** 2 for value in window) / period
        result[index] = variance**0.5
    return result
