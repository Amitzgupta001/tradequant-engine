"""EMA pullback strategy."""

import pandas as pd

from app.strategies.base import TradingStrategy
from app.strategies.dataframe import add_ema_columns


class EmaPullbackStrategy(TradingStrategy):
    """Buy pullbacks to EMA in an uptrend; sell rallies in a downtrend."""

    @property
    def strategy_id(self) -> str:
        return "ema_pullback"

    @property
    def name(self) -> str:
        return "EMA Pullback"

    @property
    def description(self) -> str:
        return "Enters on pullbacks to EMA(20) aligned with EMA(50) trend."

    def required_indicators(self) -> list[str]:
        return ["ema_20", "rsi_14", "atr_14"]

    def generate_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = add_ema_columns(dataframe, [50])
        frame["ema_pullback_distance_pct"] = (frame["close"] - frame["ema_20"]) / frame["close"]
        frame["trend_bias"] = (frame["ema_20"] - frame["ema_50"]) / frame["close"]
        return frame

    def generate_signals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = self.generate_features(dataframe)
        signals = [0] * len(frame)
        for index in range(1, len(frame)):
            distance = frame.iloc[index]["ema_pullback_distance_pct"]
            trend = frame.iloc[index]["trend_bias"]
            rsi = frame.iloc[index]["rsi_14"]
            if any(value is None or pd.isna(value) for value in (distance, trend, rsi)):
                continue
            if trend > 0 and -0.01 <= distance <= 0 and rsi < 45:
                signals[index] = 1
            elif trend < 0 and 0 <= distance <= 0.01 and rsi > 55:
                signals[index] = -1
        frame["strategy_signal"] = signals
        return frame
