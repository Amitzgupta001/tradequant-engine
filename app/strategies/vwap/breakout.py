"""VWAP breakout strategy."""

import pandas as pd

from app.strategies.base import TradingStrategy


class VwapBreakoutStrategy(TradingStrategy):
    """Trade breakouts above/below session VWAP with volume confirmation."""

    @property
    def strategy_id(self) -> str:
        return "vwap_breakout"

    @property
    def name(self) -> str:
        return "VWAP Breakout"

    @property
    def description(self) -> str:
        return "Signals when price breaks away from VWAP with rising volume."

    def required_indicators(self) -> list[str]:
        return ["vwap", "atr_14"]

    def generate_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = dataframe.copy()
        frame["vwap_distance_pct"] = (frame["close"] - frame["vwap"]) / frame["close"]
        frame["volume_ma_5"] = frame["volume"].rolling(5, min_periods=1).mean()
        frame["volume_ratio"] = frame["volume"] / frame["volume_ma_5"]
        return frame

    def generate_signals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = self.generate_features(dataframe)
        signals = [0] * len(frame)
        for index in range(1, len(frame)):
            distance = frame.iloc[index]["vwap_distance_pct"]
            prev_distance = frame.iloc[index - 1]["vwap_distance_pct"]
            volume_ratio = frame.iloc[index]["volume_ratio"]
            if any(value is None or pd.isna(value) for value in (distance, prev_distance, volume_ratio)):
                continue
            if prev_distance <= 0 < distance and volume_ratio >= 1.2:
                signals[index] = 1
            elif prev_distance >= 0 > distance and volume_ratio >= 1.2:
                signals[index] = -1
        frame["strategy_signal"] = signals
        return frame
