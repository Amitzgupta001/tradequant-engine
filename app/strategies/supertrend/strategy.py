"""SuperTrend strategy."""

import pandas as pd

from app.strategies.base import TradingStrategy
from app.strategies.dataframe import compute_supertrend


class SuperTrendStrategy(TradingStrategy):
    """Follow SuperTrend direction flips."""

    @property
    def strategy_id(self) -> str:
        return "supertrend"

    @property
    def name(self) -> str:
        return "SuperTrend"

    @property
    def description(self) -> str:
        return "Signals on SuperTrend direction changes."

    def required_indicators(self) -> list[str]:
        return ["atr_14"]

    def generate_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = compute_supertrend(dataframe)
        frame["supertrend_distance_pct"] = (frame["close"] - frame["supertrend"]) / frame["close"]
        return frame

    def generate_signals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = self.generate_features(dataframe)
        signals = [0] * len(frame)
        for index in range(1, len(frame)):
            direction = frame.iloc[index]["supertrend_direction"]
            prev_direction = frame.iloc[index - 1]["supertrend_direction"]
            if direction == 1 and prev_direction != 1:
                signals[index] = 1
            elif direction == -1 and prev_direction != -1:
                signals[index] = -1
        frame["strategy_signal"] = signals
        return frame
