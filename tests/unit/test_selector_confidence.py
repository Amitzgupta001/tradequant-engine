"""Tests for strategy selector confidence gating."""

import numpy as np

from app.ml.selector.confidence import effective_gate_params, passes_selector_gate


def test_legacy_high_confidence_uses_margin_gate() -> None:
    margin, confidence = effective_gate_params(min_confidence=0.35, min_margin=0.0)
    assert margin == 0.02
    assert confidence == 0.0


def test_multiclass_probabilities_pass_with_legacy_metadata() -> None:
    probabilities = np.array([0.12, 0.18, 0.09, 0.11, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.04])
    _, top, margin, passes = passes_selector_gate(probabilities, min_confidence=0.35)
    assert top == 0.18
    assert margin == 0.06
    assert passes
