"""Exponential Moving Average indicator."""

from app.indicators.base import IndicatorSeries, ema, validate_period


def compute_ema(closes: list[float], period: int = 20) -> IndicatorSeries:
    """Compute EMA for close prices."""
    validate_period(period)
    return IndicatorSeries(name=f"ema_{period}", values=ema(closes, period))
