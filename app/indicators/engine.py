"""Indicator computation engine."""

from app.domain.candle import Candle
from app.domain.indicators import IndicatorSnapshot
from app.indicators.atr import compute_atr
from app.indicators.bollinger import compute_bollinger_bands
from app.indicators.ema import compute_ema
from app.indicators.macd import compute_macd
from app.indicators.rsi import compute_rsi
from app.indicators.vwap import compute_vwap


class IndicatorEngine:
    """Compute standard technical indicators from OHLCV candles."""

    def compute(self, candles: list[Candle]) -> list[IndicatorSnapshot]:
        """Return indicator snapshots aligned with input candles."""
        if not candles:
            return []

        closes = [candle.close for candle in candles]
        highs = [candle.high for candle in candles]
        lows = [candle.low for candle in candles]

        ema_20 = compute_ema(closes, period=20).values
        rsi_14 = compute_rsi(closes, period=14).values
        macd = compute_macd(closes)
        atr_14 = compute_atr(highs, lows, closes, period=14).values
        vwap = compute_vwap(candles).values
        bollinger = compute_bollinger_bands(closes, period=20, std_dev=2.0)

        snapshots: list[IndicatorSnapshot] = []
        for index, candle in enumerate(candles):
            snapshots.append(
                IndicatorSnapshot(
                    timestamp=candle.timestamp,
                    ema_20=ema_20[index],
                    rsi_14=rsi_14[index],
                    macd=macd.macd[index],
                    macd_signal=macd.signal[index],
                    macd_histogram=macd.histogram[index],
                    atr_14=atr_14[index],
                    vwap=vwap[index],
                    bb_upper=bollinger.upper[index],
                    bb_middle=bollinger.middle[index],
                    bb_lower=bollinger.lower[index],
                )
            )
        return snapshots
