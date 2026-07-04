"""EMA crossover strategy."""

import pandas as pd

from app.strategies.base import TradingStrategy
from app.strategies.dataframe import add_ema_columns


class EmaCrossoverStrategy(TradingStrategy):
    """Buy when fast EMA crosses above slow EMA; sell on bearish cross."""

    @property
    def strategy_id(self) -> str:
        return "ema_crossover"

    @property
    def name(self) -> str:
        return "EMA Crossover"

    @property
    def description(self) -> str:
        return "Generates signals when EMA(9) crosses EMA(21)."

    def required_indicators(self) -> list[str]:
        return ["ema_20", "atr_14"]

    def generate_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = add_ema_columns(dataframe, [9, 21])
        frame["ema_cross_gap_pct"] = (frame["ema_9"] - frame["ema_21"]) / frame["close"]
        frame["ema_cross_slope"] = frame["ema_cross_gap_pct"].diff()
        return frame

    def generate_signals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = self.generate_features(dataframe)
        signals = [0] * len(frame)
        for index in range(1, len(frame)):
            prev_gap = frame.iloc[index - 1]["ema_cross_gap_pct"]
            gap = frame.iloc[index]["ema_cross_gap_pct"]
            if prev_gap is None or gap is None or pd.isna(prev_gap) or pd.isna(gap):
                continue
            if prev_gap <= 0 < gap:
                signals[index] = 1
            elif prev_gap >= 0 > gap:
                signals[index] = -1
        frame["strategy_signal"] = signals
        return frame
