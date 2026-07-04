"""Relative Strength Index indicator."""

from app.indicators.base import IndicatorSeries, validate_period, wilder_smooth


def compute_rsi(closes: list[float], period: int = 14) -> IndicatorSeries:
    """Compute RSI using Wilder's smoothing."""
    validate_period(period)
    if len(closes) < period + 1:
        return IndicatorSeries(name=f"rsi_{period}", values=[None] * len(closes))

    gains: list[float] = [0.0]
    losses: list[float] = [0.0]
    for index in range(1, len(closes)):
        change = closes[index] - closes[index - 1]
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))

    avg_gain = wilder_smooth(gains, period)
    avg_loss = wilder_smooth(losses, period)

    rsi_values: list[float | None] = [None] * len(closes)
    for index in range(len(closes)):
        gain = avg_gain[index]
        loss = avg_loss[index]
        if gain is None or loss is None:
            continue
        if loss == 0:
            rsi_values[index] = 100.0
            continue
        rs = gain / loss
        rsi_values[index] = 100 - (100 / (1 + rs))

    return IndicatorSeries(name=f"rsi_{period}", values=rsi_values)
