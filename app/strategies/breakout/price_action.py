"""Price action breakout strategy."""

import pandas as pd

from app.strategies.base import TradingStrategy


class PriceActionBreakoutStrategy(TradingStrategy):
    """Breakout from recent swing high/low with range compression."""

    @property
    def strategy_id(self) -> str:
        return "price_action_breakout"

    @property
    def name(self) -> str:
        return "Price Action Breakout"

    @property
    def description(self) -> str:
        return "Signals when price breaks a 20-bar range after compression."

    def required_indicators(self) -> list[str]:
        return ["atr_14"]

    def generate_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = dataframe.copy()
        frame["range_high_20"] = frame["high"].rolling(20, min_periods=5).max()
        frame["range_low_20"] = frame["low"].rolling(20, min_periods=5).min()
        frame["range_width_pct"] = (
            frame["range_high_20"] - frame["range_low_20"]
        ) / frame["close"]
        return frame

    def generate_signals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = self.generate_features(dataframe)
        signals = [0] * len(frame)
        for index in range(20, len(frame)):
            close = frame.iloc[index]["close"]
            prev_close = frame.iloc[index - 1]["close"]
            range_high = frame.iloc[index - 1]["range_high_20"]
            range_low = frame.iloc[index - 1]["range_low_20"]
            width = frame.iloc[index]["range_width_pct"]
            if any(value is None or pd.isna(value) for value in (range_high, range_low, width)):
                continue
            if prev_close <= range_high < close and width < 0.05:
                signals[index] = 1
            elif prev_close >= range_low > close and width < 0.05:
                signals[index] = -1
        frame["strategy_signal"] = signals
        return frame
