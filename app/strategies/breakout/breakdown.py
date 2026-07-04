"""Range breakdown strategy (bearish setup signals)."""

import pandas as pd

from app.strategies.base import TradingStrategy


class BreakdownStrategy(TradingStrategy):
    """Detect breakdowns below a consolidation range with volume confirmation."""

    @property
    def strategy_id(self) -> str:
        return "breakdown"

    @property
    def name(self) -> str:
        return "Breakdown"

    @property
    def description(self) -> str:
        return "Flags breakdowns below a 20-bar range low on rising volume."

    def required_indicators(self) -> list[str]:
        return ["atr_14"]

    def generate_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = dataframe.copy()
        frame["range_high_20"] = frame["high"].rolling(20, min_periods=10).max()
        frame["range_low_20"] = frame["low"].rolling(20, min_periods=10).min()
        frame["range_width_pct"] = (
            frame["range_high_20"] - frame["range_low_20"]
        ) / frame["close"]
        frame["volume_ratio_10"] = frame["volume"] / frame["volume"].rolling(10, min_periods=1).mean()
        frame["breakdown_distance_pct"] = (
            frame["range_low_20"] - frame["close"]
        ) / frame["close"]
        return frame

    def generate_signals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = self.generate_features(dataframe)
        signals = [0] * len(frame)
        for index in range(20, len(frame)):
            close = frame.iloc[index]["close"]
            prev_close = frame.iloc[index - 1]["close"]
            range_low = frame.iloc[index - 1]["range_low_20"]
            width = frame.iloc[index]["range_width_pct"]
            volume_ratio = frame.iloc[index]["volume_ratio_10"]
            if any(value is None or pd.isna(value) for value in (range_low, width, volume_ratio)):
                continue
            if width < 0.06 and prev_close >= range_low > close and volume_ratio >= 1.15:
                signals[index] = -1
        frame["strategy_signal"] = signals
        return frame
