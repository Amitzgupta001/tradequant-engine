"""Market regime detection for strategy routing."""

from enum import Enum

import pandas as pd
from pydantic import BaseModel, Field


class MarketRegime(str, Enum):
    """Detected market regimes."""

    TRENDING = "trending"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BULLISH = "bullish"
    BEARISH = "bearish"


class RegimeSnapshot(BaseModel):
    """Detected regime labels for a market snapshot."""

    primary: MarketRegime
    tags: list[MarketRegime] = Field(default_factory=list)
    trend_strength: float
    volatility_pct: float


class RegimeClassifier:
    """Rule-based regime classifier from market features."""

    def classify(self, frame: pd.DataFrame) -> RegimeSnapshot:
        """Detect regime from the latest row in a market dataframe."""
        if frame.empty:
            msg = "Cannot classify regime on empty dataframe"
            raise ValueError(msg)

        row = frame.iloc[-1]
        close = float(row["close"])
        trend = 0.0
        if "ema_20" in frame.columns and row.get("ema_20") is not None:
            trend = (close - float(row["ema_20"])) / close

        returns = frame["close"].pct_change().dropna()
        volatility = float(returns.tail(20).std()) if len(returns) >= 5 else 0.0

        tags: list[MarketRegime] = []
        if abs(trend) >= 0.01:
            tags.append(MarketRegime.TRENDING)
            tags.append(MarketRegime.BULLISH if trend > 0 else MarketRegime.BEARISH)
        else:
            tags.append(MarketRegime.SIDEWAYS)

        if volatility >= 0.015:
            tags.append(MarketRegime.HIGH_VOLATILITY)
        else:
            tags.append(MarketRegime.LOW_VOLATILITY)

        primary = MarketRegime.TRENDING if MarketRegime.TRENDING in tags else tags[0]
        return RegimeSnapshot(
            primary=primary,
            tags=tags,
            trend_strength=trend,
            volatility_pct=volatility,
        )
