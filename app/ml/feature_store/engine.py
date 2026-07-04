"""Feature engineering engine."""

import statistics

from app.domain.candle import Candle
from app.domain.features import FeatureVector
from app.domain.indicators import IndicatorSnapshot
from app.ml.feature_store.transforms import safe_log_return, safe_pct, safe_ratio


class FeatureEngine:
    """Transform OHLCV candles and indicators into ML-ready features."""

    def build(
        self,
        candles: list[Candle],
        indicators: list[IndicatorSnapshot],
    ) -> list[FeatureVector]:
        """Build feature vectors aligned with candle timestamps."""
        if not candles:
            return []
        if len(candles) != len(indicators):
            msg = "candles and indicators must have equal length"
            raise ValueError(msg)

        daily_returns: list[float | None] = []
        for index, candle in enumerate(candles):
            previous_close = candles[index - 1].close if index > 0 else None
            daily_returns.append(safe_pct(candle.close, previous_close))

        features: list[FeatureVector] = []
        for index, (candle, indicator) in enumerate(zip(candles, indicators, strict=True)):
            previous_close = candles[index - 1].close if index > 0 else None
            previous_volume = candles[index - 1].volume if index > 0 else None
            next_close = candles[index + 1].close if index + 1 < len(candles) else None
            forward_close_5 = candles[index + 5].close if index + 5 < len(candles) else None
            forward_close_20 = candles[index + 20].close if index + 20 < len(candles) else None

            bb_position = None
            if (
                indicator.bb_upper is not None
                and indicator.bb_lower is not None
                and indicator.bb_upper != indicator.bb_lower
            ):
                bb_position = (candle.close - indicator.bb_lower) / (
                    indicator.bb_upper - indicator.bb_lower
                )

            volume_change_pct = None
            if previous_volume and previous_volume > 0:
                volume_change_pct = (candle.volume - previous_volume) / previous_volume

            return_1d = daily_returns[index]
            return_3d = safe_pct(candle.close, candles[index - 3].close) if index >= 3 else None
            return_5d = safe_pct(candle.close, candles[index - 5].close) if index >= 5 else None

            rsi_change_3d = None
            if index >= 3 and indicator.rsi_14 is not None:
                prior_rsi = indicators[index - 3].rsi_14
                if prior_rsi is not None:
                    rsi_change_3d = indicator.rsi_14 - prior_rsi

            recent_returns = [value for value in daily_returns[max(0, index - 9) : index + 1] if value is not None]
            volatility_5d = (
                statistics.pstdev(recent_returns[-5:]) if len(recent_returns) >= 5 else None
            )
            volatility_10d = (
                statistics.pstdev(recent_returns[-10:]) if len(recent_returns) >= 10 else None
            )

            recent_volumes = [item.volume for item in candles[max(0, index - 4) : index + 1]]
            volume_ratio_5d = (
                candle.volume / statistics.mean(recent_volumes) if len(recent_volumes) >= 5 else None
            )

            recent_daily_returns = [
                value for value in daily_returns[max(0, index - 19) : index + 1] if value is not None
            ]
            up_ratio_20d = None
            if len(recent_daily_returns) >= 20:
                up_ratio_20d = sum(1 for value in recent_daily_returns if value > 0) / len(
                    recent_daily_returns
                )
            trend_20d = safe_pct(candle.close, candles[index - 20].close) if index >= 20 else None

            features.append(
                FeatureVector(
                    timestamp=candle.timestamp,
                    close=candle.close,
                    volume=candle.volume,
                    return_1d=return_1d,
                    return_3d=return_3d,
                    return_5d=return_5d,
                    log_return_1d=(
                        safe_log_return(candle.close, previous_close)
                        if previous_close is not None
                        else None
                    ),
                    high_low_range_pct=safe_ratio(candle.high - candle.low, candle.close),
                    body_pct=safe_pct(candle.close, candle.open),
                    volume_change_pct=volume_change_pct,
                    ema_gap_pct=safe_pct(candle.close, indicator.ema_20),
                    rsi_14=indicator.rsi_14,
                    rsi_change_3d=rsi_change_3d,
                    macd_histogram=indicator.macd_histogram,
                    atr_pct=safe_ratio(indicator.atr_14, candle.close),
                    bb_width_pct=(
                        safe_ratio(
                            indicator.bb_upper - indicator.bb_lower,
                            indicator.bb_middle,
                        )
                        if indicator.bb_upper is not None
                        and indicator.bb_lower is not None
                        and indicator.bb_middle is not None
                        else None
                    ),
                    bb_position=bb_position,
                    vwap_gap_pct=safe_pct(candle.close, indicator.vwap),
                    return_lag_1=daily_returns[index - 1] if index >= 1 else None,
                    return_lag_2=daily_returns[index - 2] if index >= 2 else None,
                    return_lag_3=daily_returns[index - 3] if index >= 3 else None,
                    volatility_5d=volatility_5d,
                    volatility_10d=volatility_10d,
                    volume_ratio_5d=volume_ratio_5d,
                    up_ratio_20d=up_ratio_20d,
                    trend_20d=trend_20d,
                    forward_return_1d=safe_pct(next_close, candle.close),
                    forward_return_5d=safe_pct(forward_close_5, candle.close),
                    forward_return_20b=safe_pct(forward_close_20, candle.close),
                )
            )

        return features
