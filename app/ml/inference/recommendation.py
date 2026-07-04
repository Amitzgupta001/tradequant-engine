"""Strategy recommendation engine."""

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from app.ml.datasets.strategy_selector_builder import SELECTOR_FEATURE_COLUMNS
from app.ml.feature_store.repository import CSVFeatureRepository
from app.ml.inference.regime import MarketRegime, RegimeClassifier
from app.ml.registry.strategy_selector_registry import StrategySelectorRegistry
from app.ml.selector.picker import pick_strategy
from app.strategies.registry import list_strategies


class StrategyRecommendation(BaseModel):
    """Recommended strategy for current market conditions."""

    strategy_id: str
    strategy_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    reason: str
    regime: MarketRegime
    rank: int = 1


_REGIME_STRATEGY_MAP: dict[MarketRegime, list[str]] = {
    MarketRegime.TRENDING: ["ema_pullback", "supertrend", "macd_momentum"],
    MarketRegime.BULLISH: ["ema_crossover", "ema_pullback", "vwap_breakout"],
    MarketRegime.BEARISH: ["rsi_reversal", "bollinger_mean_reversion"],
    MarketRegime.SIDEWAYS: ["breakout", "bollinger_mean_reversion", "rsi_reversal", "cpr_breakout"],
    MarketRegime.HIGH_VOLATILITY: ["orb", "price_action_breakout", "breakout"],
    MarketRegime.LOW_VOLATILITY: ["bollinger_mean_reversion", "cpr_breakout"],
}


class StrategyRecommendationEngine:
    """Recommend strategies using a trained selector or regime fallback."""

    def __init__(
        self,
        regime_classifier: RegimeClassifier | None = None,
        selector_registry: StrategySelectorRegistry | None = None,
        feature_repository: CSVFeatureRepository | None = None,
    ) -> None:
        self._regime_classifier = regime_classifier or RegimeClassifier()
        self._selector_registry = selector_registry
        self._feature_repository = feature_repository

    def recommend(
        self,
        frame: pd.DataFrame,
        exchange_segment: str | None = None,
        security_id: str | None = None,
        timeframe: str | None = None,
        selector_version: int | None = None,
        universe_id: str | None = None,
        strategy_priors: dict[str, float] | None = None,
    ) -> StrategyRecommendation:
        """Return the top strategy recommendation."""
        ranked = self.recommend_all(
            frame,
            exchange_segment=exchange_segment,
            security_id=security_id,
            timeframe=timeframe,
            selector_version=selector_version,
            universe_id=universe_id,
            strategy_priors=strategy_priors,
        )
        return ranked[0]

    def recommend_all(
        self,
        frame: pd.DataFrame,
        exchange_segment: str | None = None,
        security_id: str | None = None,
        timeframe: str | None = None,
        selector_version: int | None = None,
        universe_id: str | None = None,
        strategy_priors: dict[str, float] | None = None,
        top_n: int = 5,
    ) -> list[StrategyRecommendation]:
        """Return ranked strategy recommendations."""
        regime = self._regime_classifier.classify(frame)
        registered = {strategy.strategy_id: strategy for strategy in list_strategies()}

        if (
            self._selector_registry
            and exchange_segment
            and security_id
            and timeframe
        ):
            model_recs = self._recommend_from_model(
                frame,
                exchange_segment,
                security_id,
                timeframe,
                selector_version,
                universe_id,
                strategy_priors,
                regime,
                registered,
                top_n,
            )
            if model_recs:
                return model_recs

        return self._recommend_from_regime(regime, registered, frame, top_n)

    def _recommend_from_model(
        self,
        frame: pd.DataFrame,
        exchange_segment: str,
        security_id: str,
        timeframe: str,
        selector_version: int | None,
        universe_id: str | None,
        strategy_priors: dict[str, float] | None,
        regime,
        registered: dict,
        top_n: int,
    ) -> list[StrategyRecommendation]:
        """Use trained selector model probabilities."""
        scope_security_id = security_id
        if universe_id:
            scope_security_id = StrategySelectorRegistry.panel_security_id(universe_id)

        version = selector_version or self._selector_registry.latest_version(
            exchange_segment,
            scope_security_id,
            timeframe,
            universe_id=universe_id,
        )
        if version is None:
            return []

        metadata = self._selector_registry.load_metadata(
            exchange_segment,
            scope_security_id,
            timeframe,
            version,
            universe_id=universe_id,
        )
        model = self._selector_registry.load_model(
            exchange_segment,
            scope_security_id,
            timeframe,
            version,
            universe_id=universe_id,
        )
        encoder = self._selector_registry.load_label_encoder(
            exchange_segment,
            scope_security_id,
            timeframe,
            version,
            universe_id=universe_id,
        )
        feature_row = self._build_selector_row(frame)
        if feature_row is None:
            return []

        input_frame = pd.DataFrame([feature_row])
        probabilities = model.predict_proba(input_frame[metadata.feature_columns])[0]
        strategy_ids = list(encoder.classes_)
        picked_id, top_prob, margin, pick_reason = pick_strategy(
            probabilities,
            strategy_ids,
            min_confidence=metadata.min_confidence,
            min_margin=getattr(metadata, "min_margin", 0.0) or 0.0,
            strategy_priors=strategy_priors,
        )
        if picked_id is None:
            return []

        ranked_indices = sorted(
            range(len(probabilities)),
            key=lambda index: probabilities[index],
            reverse=True,
        )

        recommendations: list[StrategyRecommendation] = []
        for rank, index in enumerate(ranked_indices[:top_n], start=1):
            strategy_id = strategy_ids[index]
            strategy = registered.get(strategy_id)
            confidence = float(probabilities[index])
            is_primary = strategy_id == picked_id
            recommendations.append(
                StrategyRecommendation(
                    strategy_id=strategy_id,
                    strategy_name=strategy.name if strategy else strategy_id,
                    confidence=confidence,
                    risk_score=min(1.0, regime.volatility_pct * 20),
                    reason=(
                        f"Backtest-tuned selector v{version} "
                        f"{'recommends' if is_primary else 'candidate'} {strategy_id} "
                        f"({confidence:.0%} prob"
                        f"{', ' + pick_reason if is_primary else ''})"
                        f"{' [panel ' + universe_id + ']' if universe_id else ''} "
                        f"in {regime.primary.value.replace('_', ' ')} regime."
                    ),
                    regime=regime.primary,
                    rank=1 if is_primary else rank + 1,
                )
            )
        recommendations.sort(key=lambda item: item.rank)
        for position, item in enumerate(recommendations, start=1):
            item.rank = position
        return recommendations

    def _recommend_from_regime(
        self,
        regime,
        registered: dict,
        frame: pd.DataFrame,
        top_n: int,
    ) -> list[StrategyRecommendation]:
        """Fallback to static regime map when no selector model exists."""
        candidates = _REGIME_STRATEGY_MAP.get(regime.primary, list(registered))
        recommendations: list[StrategyRecommendation] = []
        signal_strength = 0.0
        if "strategy_signal" in frame.columns:
            signal_strength = abs(float(frame.iloc[-1]["strategy_signal"]))

        for rank, strategy_id in enumerate(candidates[:top_n], start=1):
            strategy = registered.get(strategy_id)
            if strategy is None:
                continue
            confidence = min(
                0.85,
                0.55 + abs(regime.trend_strength) * 5 + signal_strength * 0.1 - (rank - 1) * 0.08,
            )
            recommendations.append(
                StrategyRecommendation(
                    strategy_id=strategy_id,
                    strategy_name=strategy.name,
                    confidence=max(0.1, confidence),
                    risk_score=min(1.0, regime.volatility_pct * 20),
                    reason=(
                        f"Regime fallback: market is {regime.primary.value.replace('_', ' ')} "
                        f"(no trained selector model)."
                    ),
                    regime=regime.primary,
                    rank=rank,
                )
            )

        if not recommendations and registered:
            strategy_id, strategy = next(iter(registered.items()))
            recommendations.append(
                StrategyRecommendation(
                    strategy_id=strategy_id,
                    strategy_name=strategy.name,
                    confidence=0.5,
                    risk_score=min(1.0, regime.volatility_pct * 20),
                    reason="Default fallback strategy.",
                    regime=regime.primary,
                    rank=1,
                )
            )
        return recommendations

    def _build_selector_row(self, frame: pd.DataFrame) -> dict[str, float] | None:
        """Build selector feature row from market frame and regime."""
        if frame.empty:
            return None
        row = frame.iloc[-1]
        regime = self._regime_classifier.classify(frame)
        features: dict[str, float] = {}
        for column in SELECTOR_FEATURE_COLUMNS:
            if column in {
                "trend_strength",
                "volatility_pct",
                "regime_trending",
                "regime_sideways",
                "regime_bullish",
                "regime_bearish",
                "regime_high_volatility",
                "regime_low_volatility",
            }:
                continue
            value = row.get(column)
            if value is None or pd.isna(value):
                return None
            features[column] = float(value)

        features["trend_strength"] = regime.trend_strength
        features["volatility_pct"] = regime.volatility_pct
        features["regime_trending"] = 1.0 if regime.primary.value == "trending" else 0.0
        features["regime_sideways"] = 1.0 if regime.primary.value == "sideways" else 0.0
        features["regime_bullish"] = 1.0 if "bullish" in {tag.value for tag in regime.tags} else 0.0
        features["regime_bearish"] = 1.0 if "bearish" in {tag.value for tag in regime.tags} else 0.0
        features["regime_high_volatility"] = (
            1.0 if "high_volatility" in {tag.value for tag in regime.tags} else 0.0
        )
        features["regime_low_volatility"] = (
            1.0 if "low_volatility" in {tag.value for tag in regime.tags} else 0.0
        )
        return features

    @classmethod
    def from_storage(cls, storage_path: Path) -> "StrategyRecommendationEngine":
        """Build engine with selector registry and feature repository."""
        return cls(
            selector_registry=StrategySelectorRegistry(storage_path),
            feature_repository=CSVFeatureRepository(storage_path / "features"),
        )

    def recommend_for_instrument(
        self,
        instrument,
        timeframe,
        market_frame: pd.DataFrame | None = None,
        selector_version: int | None = None,
        universe_id: str | None = None,
        strategy_priors: dict[str, float] | None = None,
        top_n: int = 5,
    ) -> list[StrategyRecommendation]:
        """Recommend strategies using stored features and market data."""
        if market_frame is None and self._feature_repository is None:
            msg = "Feature repository required when market_frame is not provided"
            raise ValueError(msg)

        if market_frame is None:
            features = self._feature_repository.load(instrument, timeframe)
            market_frame = pd.DataFrame([feature.model_dump(mode="json") for feature in features])
            market_frame["timestamp"] = pd.to_datetime(market_frame["timestamp"])
            market_frame = market_frame.sort_values("timestamp")

        return self.recommend_all(
            market_frame,
            exchange_segment=instrument.exchange_segment.value,
            security_id=instrument.security_id,
            timeframe=timeframe.value,
            selector_version=selector_version,
            universe_id=universe_id,
            strategy_priors=strategy_priors,
            top_n=top_n,
        )
