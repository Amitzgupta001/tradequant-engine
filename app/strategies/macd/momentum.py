"""MACD momentum strategy."""

import pandas as pd

from app.strategies.base import TradingStrategy


class MacdMomentumStrategy(TradingStrategy):
    """Trade MACD histogram momentum and signal-line crosses."""

    @property
    def strategy_id(self) -> str:
        return "macd_momentum"

    @property
    def name(self) -> str:
        return "MACD Momentum"

    @property
    def description(self) -> str:
        return "Signals on MACD histogram expansion and zero-line crosses."

    def required_indicators(self) -> list[str]:
        return ["macd", "macd_signal", "macd_histogram"]

    def generate_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = dataframe.copy()
        frame["macd_hist_slope"] = frame["macd_histogram"].diff()
        frame["macd_signal_gap"] = frame["macd"] - frame["macd_signal"]
        return frame

    def generate_signals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = self.generate_features(dataframe)
        signals = [0] * len(frame)
        for index in range(1, len(frame)):
            histogram = frame.iloc[index]["macd_histogram"]
            prev_histogram = frame.iloc[index - 1]["macd_histogram"]
            slope = frame.iloc[index]["macd_hist_slope"]
            if any(value is None or pd.isna(value) for value in (histogram, prev_histogram, slope)):
                continue
            if prev_histogram <= 0 < histogram and slope > 0:
                signals[index] = 1
            elif prev_histogram >= 0 > histogram and slope < 0:
                signals[index] = -1
        frame["strategy_signal"] = signals
        return frame
