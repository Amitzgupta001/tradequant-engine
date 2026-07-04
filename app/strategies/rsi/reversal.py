"""RSI reversal strategy."""

import pandas as pd

from app.strategies.base import TradingStrategy


class RsiReversalStrategy(TradingStrategy):
    """Mean-reversion entries from RSI extremes."""

    @property
    def strategy_id(self) -> str:
        return "rsi_reversal"

    @property
    def name(self) -> str:
        return "RSI Reversal"

    @property
    def description(self) -> str:
        return "Buys oversold RSI reversals and sells overbought reversals."

    def required_indicators(self) -> list[str]:
        return ["rsi_14", "atr_14"]

    def generate_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = dataframe.copy()
        frame["rsi_change"] = frame["rsi_14"].diff()
        frame["rsi_distance_from_50"] = frame["rsi_14"] - 50
        return frame

    def generate_signals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = self.generate_features(dataframe)
        signals = [0] * len(frame)
        for index in range(1, len(frame)):
            rsi = frame.iloc[index]["rsi_14"]
            prev_rsi = frame.iloc[index - 1]["rsi_14"]
            if rsi is None or prev_rsi is None or pd.isna(rsi) or pd.isna(prev_rsi):
                continue
            if prev_rsi < 30 <= rsi:
                signals[index] = 1
            elif prev_rsi > 70 >= rsi:
                signals[index] = -1
        frame["strategy_signal"] = signals
        return frame
