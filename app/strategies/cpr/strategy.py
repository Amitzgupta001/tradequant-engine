"""Central Pivot Range breakout strategy."""

import pandas as pd

from app.strategies.base import TradingStrategy
from app.strategies.dataframe import central_pivot_range


class CprBreakoutStrategy(TradingStrategy):
    """Trade breakouts from the Central Pivot Range."""

    @property
    def strategy_id(self) -> str:
        return "cpr_breakout"

    @property
    def name(self) -> str:
        return "CPR Breakout"

    @property
    def description(self) -> str:
        return "Signals when price breaks above TC or below BC."

    def required_indicators(self) -> list[str]:
        return ["atr_14"]

    def generate_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = central_pivot_range(dataframe)
        frame["cpr_width_pct"] = (frame["cpr_tc"] - frame["cpr_bc"]) / frame["close"]
        frame["cpr_position"] = (frame["close"] - frame["cpr_bc"]) / (
            frame["cpr_tc"] - frame["cpr_bc"]
        ).replace(0, pd.NA)
        return frame

    def generate_signals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = self.generate_features(dataframe)
        signals = [0] * len(frame)
        for index in range(1, len(frame)):
            close = frame.iloc[index]["close"]
            tc = frame.iloc[index]["cpr_tc"]
            bc = frame.iloc[index]["cpr_bc"]
            prev_close = frame.iloc[index - 1]["close"]
            if any(value is None or pd.isna(value) for value in (tc, bc)):
                continue
            if prev_close <= tc < close:
                signals[index] = 1
            elif prev_close >= bc > close:
                signals[index] = -1
        frame["strategy_signal"] = signals
        return frame
