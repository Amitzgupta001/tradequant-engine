"""Tests for threshold tuning metrics."""

from app.ml.evaluator.metrics import find_best_threshold, signal_hit_rate


def test_find_best_threshold_with_low_probabilities() -> None:
    """Threshold search should work when model outputs compressed probabilities."""
    y_true = [1, 0, 1, 0, 0, 1, 0, 0, 1, 0] * 5
    y_prob = [0.31, 0.22, 0.28, 0.19, 0.25, 0.30, 0.21, 0.20, 0.29, 0.24] * 5

    threshold, hit_rate, count, _ = find_best_threshold(y_true, y_prob, min_signals=5)

    assert threshold < 0.5
    assert count >= 5
    assert hit_rate is not None


def test_signal_hit_rate_counts_buy_signals() -> None:
    """Signals should count rows above threshold."""
    y_true = [1, 0, 1, 0]
    y_prob = [0.35, 0.25, 0.40, 0.20]

    count, hit_rate = signal_hit_rate(y_true, y_prob, 0.30)

    assert count == 2
    assert hit_rate == 1.0
