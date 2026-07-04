"""Ensemble prediction engine for strategy models."""

from enum import Enum

from pydantic import BaseModel, Field


class EnsembleMethod(str, Enum):
    """Supported ensemble aggregation methods."""

    VOTING = "voting"
    WEIGHTED_VOTING = "weighted_voting"
    CONFIDENCE_AVERAGING = "confidence_averaging"


class StrategyPrediction(BaseModel):
    """Prediction from a single strategy model."""

    strategy_id: str
    signal: int = Field(description="1 buy, -1 sell, 0 hold")
    confidence: float = Field(ge=0.0, le=1.0)


class EnsemblePrediction(BaseModel):
    """Aggregated ensemble output."""

    signal: int
    confidence: float
    method: EnsembleMethod
    contributors: list[StrategyPrediction]


class EnsembleEngine:
    """Combine multiple strategy predictions."""

    def combine(
        self,
        predictions: list[StrategyPrediction],
        method: EnsembleMethod = EnsembleMethod.VOTING,
        weights: dict[str, float] | None = None,
    ) -> EnsemblePrediction:
        """Aggregate strategy predictions."""
        if not predictions:
            return EnsemblePrediction(
                signal=0,
                confidence=0.0,
                method=method,
                contributors=[],
            )

        if method == EnsembleMethod.VOTING:
            buy_votes = sum(1 for item in predictions if item.signal > 0)
            sell_votes = sum(1 for item in predictions if item.signal < 0)
            if buy_votes > sell_votes:
                signal = 1
            elif sell_votes > buy_votes:
                signal = -1
            else:
                signal = 0
            confidence = max(buy_votes, sell_votes) / len(predictions)
            return EnsemblePrediction(
                signal=signal,
                confidence=confidence,
                method=method,
                contributors=predictions,
            )

        weights = weights or {item.strategy_id: 1.0 for item in predictions}
        weighted_signal = 0.0
        total_weight = 0.0
        confidence_sum = 0.0
        for item in predictions:
            weight = weights.get(item.strategy_id, 1.0)
            weighted_signal += item.signal * weight * item.confidence
            confidence_sum += item.confidence * weight
            total_weight += weight

        if method == EnsembleMethod.CONFIDENCE_AVERAGING:
            avg_confidence = confidence_sum / total_weight if total_weight else 0.0
            signal = 1 if weighted_signal > 0.1 else -1 if weighted_signal < -0.1 else 0
            return EnsemblePrediction(
                signal=signal,
                confidence=avg_confidence,
                method=method,
                contributors=predictions,
            )

        signal = 1 if weighted_signal > 0 else -1 if weighted_signal < 0 else 0
        confidence = abs(weighted_signal) / total_weight if total_weight else 0.0
        return EnsemblePrediction(
            signal=signal,
            confidence=min(1.0, confidence),
            method=method,
            contributors=predictions,
        )
