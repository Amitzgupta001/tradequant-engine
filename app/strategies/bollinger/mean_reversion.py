"""Bollinger mean reversion strategy."""

import pandas as pd

from app.strategies.base import TradingStrategy


class BollingerMeanReversionStrategy(TradingStrategy):
    """Fade touches of Bollinger band extremes."""

    @property
    def strategy_id(self) -> str:
        return "bollinger_mean_reversion"

    @property
    def name(self) -> str:
        return "Bollinger Mean Reversion"

    @property
    def description(self) -> str:
        return "Buys lower-band rejections and sells upper-band rejections."

    def required_indicators(self) -> list[str]:
        return ["bb_upper", "bb_middle", "bb_lower", "rsi_14"]

    def generate_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = dataframe.copy()
        width = frame["bb_upper"] - frame["bb_lower"]
        frame["bb_position"] = (frame["close"] - frame["bb_lower"]) / width.replace(0, pd.NA)
        frame["bb_width_pct"] = width / frame["close"]
        return frame

    def generate_signals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = self.generate_features(dataframe)
        signals = [0] * len(frame)
        for index in range(1, len(frame)):
            position = frame.iloc[index]["bb_position"]
            prev_position = frame.iloc[index - 1]["bb_position"]
            rsi = frame.iloc[index]["rsi_14"]
            if any(value is None or pd.isna(value) for value in (position, prev_position, rsi)):
                continue
            if prev_position < 0.05 and position >= 0.05 and rsi < 40:
                signals[index] = 1
            elif prev_position > 0.95 and position <= 0.95 and rsi > 60:
                signals[index] = -1
        frame["strategy_signal"] = signals
        return frame
