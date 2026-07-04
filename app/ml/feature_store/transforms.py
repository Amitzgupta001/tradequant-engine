"""Feature engineering math helpers."""

import math


def safe_pct(numerator: float | None, denominator: float | None) -> float | None:
    """Compute percentage change safely."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return (numerator - denominator) / denominator


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Compute ratio safely."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def safe_log_return(current: float, previous: float) -> float | None:
    """Compute log return between two prices."""
    if previous <= 0 or current <= 0:
        return None
    return math.log(current / previous)
