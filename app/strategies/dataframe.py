"""Market dataframe helpers for strategy ML pipelines."""

from datetime import date

import pandas as pd

from app.domain.candle import Candle
from app.domain.indicators import IndicatorSnapshot
from app.indicators.base import ema
from app.indicators.engine import IndicatorEngine

MARKET_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

INDICATOR_FIELD_MAP: dict[str, str] = {
    "ema_20": "ema_20",
    "rsi_14": "rsi_14",
    "macd": "macd",
    "macd_signal": "macd_signal",
    "macd_histogram": "macd_histogram",
    "atr_14": "atr_14",
    "vwap": "vwap",
    "bb_upper": "bb_upper",
    "bb_middle": "bb_middle",
    "bb_lower": "bb_lower",
}


def candles_to_dataframe(
    candles: list[Candle],
    indicators: list[IndicatorSnapshot] | None = None,
) -> pd.DataFrame:
    """Build a pandas dataframe from OHLCV candles and indicator snapshots."""
    if not candles:
        return pd.DataFrame(columns=MARKET_COLUMNS)

    if indicators is None:
        indicators = IndicatorEngine().compute(candles)

    rows: list[dict[str, object]] = []
    for candle, snapshot in zip(candles, indicators, strict=True):
        row: dict[str, object] = {
            "timestamp": candle.timestamp,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        }
        for column in INDICATOR_FIELD_MAP:
            row[column] = getattr(snapshot, column)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame["session_date"] = frame["timestamp"].apply(
        lambda value: value.date() if hasattr(value, "date") else date.today()
    )
    return frame


def add_ema_columns(frame: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    """Append EMA columns computed from close prices."""
    result = frame.copy()
    closes = result["close"].tolist()
    for period in periods:
        column = f"ema_{period}"
        result[column] = ema(closes, period)
    return result


def compute_supertrend(
    frame: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Compute SuperTrend direction and line values."""
    result = frame.copy()
    if "atr_14" not in result.columns:
        msg = "SuperTrend requires atr_14 in dataframe"
        raise ValueError(msg)

    upper_band: list[float | None] = [None] * len(result)
    lower_band: list[float | None] = [None] * len(result)
    supertrend: list[float | None] = [None] * len(result)
    direction: list[int] = [0] * len(result)

    for index in range(len(result)):
        atr = result.iloc[index]["atr_14"]
        hl2 = (result.iloc[index]["high"] + result.iloc[index]["low"]) / 2
        if atr is None or pd.isna(atr):
            continue
        upper_band[index] = hl2 + multiplier * atr
        lower_band[index] = hl2 - multiplier * atr

    for index in range(1, len(result)):
        if upper_band[index] is None or lower_band[index] is None:
            continue
        close = result.iloc[index]["close"]
        prev_close = result.iloc[index - 1]["close"]
        prev_upper = upper_band[index - 1]
        prev_lower = lower_band[index - 1]

        if prev_upper is not None and upper_band[index] < prev_upper and prev_close > prev_upper:
            upper_band[index] = prev_upper
        if prev_lower is not None and lower_band[index] > prev_lower and prev_close < prev_lower:
            lower_band[index] = prev_lower

        prev_direction = direction[index - 1]
        if prev_direction <= 0 and close > (upper_band[index - 1] or close):
            direction[index] = 1
            supertrend[index] = lower_band[index]
        elif prev_direction >= 0 and close < (lower_band[index - 1] or close):
            direction[index] = -1
            supertrend[index] = upper_band[index]
        else:
            direction[index] = prev_direction
            supertrend[index] = (
                lower_band[index] if direction[index] == 1 else upper_band[index]
            )

    result["supertrend"] = supertrend
    result["supertrend_direction"] = direction
    return result


def opening_range_bounds(
    frame: pd.DataFrame,
    bars: int = 1,
) -> pd.DataFrame:
    """Compute opening range high/low per session."""
    result = frame.copy()
    or_high: list[float | None] = [None] * len(result)
    or_low: list[float | None] = [None] * len(result)

    grouped = result.groupby("session_date", sort=False)
    for _, group in grouped:
        indices = group.index.tolist()
        window = group.head(bars)
        high = window["high"].max()
        low = window["low"].min()
        for index in indices:
            or_high[index] = high
            or_low[index] = low

    result["or_high"] = or_high
    result["or_low"] = or_low
    return result


def central_pivot_range(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute CPR levels from previous session OHLC."""
    result = frame.copy()
    pp: list[float | None] = [None] * len(result)
    bc: list[float | None] = [None] * len(result)
    tc: list[float | None] = [None] * len(result)

    sessions = result.groupby("session_date", sort=False)
    previous_high: float | None = None
    previous_low: float | None = None
    previous_close: float | None = None

    for _, group in sessions:
        indices = group.index.tolist()
        if previous_high is not None and previous_low is not None and previous_close is not None:
            pivot = (previous_high + previous_low + previous_close) / 3
            bottom = (previous_high + previous_low) / 2
            top = (pivot - bottom) + pivot
            for index in indices:
                pp[index] = pivot
                bc[index] = bottom
                tc[index] = top

        previous_high = group["high"].max()
        previous_low = group["low"].min()
        previous_close = group.iloc[-1]["close"]

    result["cpr_pp"] = pp
    result["cpr_bc"] = bc
    result["cpr_tc"] = tc
    return result
