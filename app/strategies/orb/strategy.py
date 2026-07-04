"""Opening Range Breakout strategy."""

import pandas as pd

from app.strategies.base import TradingStrategy
from app.strategies.dataframe import opening_range_bounds


class OpeningRangeBreakoutStrategy(TradingStrategy):
    """Breakout above/below the session opening range."""

    @property
    def strategy_id(self) -> str:
        return "orb"

    @property
    def name(self) -> str:
        return "Opening Range Breakout"

    @property
    def description(self) -> str:
        return "Trades breakouts from the first session bars."

    def required_indicators(self) -> list[str]:
        return ["atr_14"]

    def generate_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = opening_range_bounds(dataframe, bars=1)
        frame["or_width_pct"] = (frame["or_high"] - frame["or_low"]) / frame["close"]
        frame["or_position"] = (frame["close"] - frame["or_low"]) / (
            frame["or_high"] - frame["or_low"]
        ).replace(0, pd.NA)
        return frame

    def generate_signals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = self.generate_features(dataframe)
        signals = [0] * len(frame)
        for index in range(1, len(frame)):
            close = frame.iloc[index]["close"]
            or_high = frame.iloc[index]["or_high"]
            or_low = frame.iloc[index]["or_low"]
            prev_close = frame.iloc[index - 1]["close"]
            if any(value is None or pd.isna(value) for value in (or_high, or_low)):
                continue
            if prev_close <= or_high < close:
                signals[index] = 1
            elif prev_close >= or_low > close:
                signals[index] = -1
        frame["strategy_signal"] = signals
        return frame
