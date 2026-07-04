"""Bollinger Bands indicator."""

from app.indicators.base import BollingerBands, rolling_std, sma, validate_period


def compute_bollinger_bands(
    closes: list[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> BollingerBands:
    """Compute upper, middle, and lower Bollinger Bands."""
    validate_period(period, minimum=2)
    if std_dev <= 0:
        msg = "std_dev must be positive"
        raise ValueError(msg)

    middle = sma(closes, period)
    std_values = rolling_std(closes, period)

    upper: list[float | None] = [None] * len(closes)
    lower: list[float | None] = [None] * len(closes)

    for index in range(len(closes)):
        mid = middle[index]
        std = std_values[index]
        if mid is None or std is None:
            continue
        upper[index] = mid + (std_dev * std)
        lower[index] = mid - (std_dev * std)

    return BollingerBands(upper=upper, middle=middle, lower=lower)
