"""Confidence gating for multiclass strategy selection."""

import numpy as np


def top_strategy_prediction(probabilities: np.ndarray) -> tuple[int, float, float]:
    """Return best class index, top probability, and margin over runner-up."""
    best_index = int(np.argmax(probabilities))
    top = float(probabilities[best_index])
    if len(probabilities) < 2:
        return best_index, top, top
    ordered = np.sort(probabilities)
    second = float(ordered[-2])
    return best_index, top, top - second


def effective_gate_params(min_confidence: float, min_margin: float) -> tuple[float, float]:
    """Normalize legacy absolute thresholds for multiclass models."""
    if min_margin > 0:
        return min_margin, 0.0
    if min_confidence >= 0.30:
        return 0.02, 0.0
    return 0.0, min_confidence


def passes_selector_gate(
    probabilities: np.ndarray,
    min_confidence: float = 0.0,
    min_margin: float = 0.0,
) -> tuple[int, float, float, bool]:
    """Decide whether selector prediction is strong enough to trade."""
    margin_threshold, confidence_threshold = effective_gate_params(min_confidence, min_margin)
    best_index, top, margin = top_strategy_prediction(probabilities)

    if margin_threshold > 0:
        return best_index, top, margin, margin >= margin_threshold

    if confidence_threshold > 0:
        baseline = 1.0 / max(len(probabilities), 1)
        effective = max(confidence_threshold, baseline * 1.25)
        return best_index, top, margin, top >= effective

    return best_index, top, margin, True
