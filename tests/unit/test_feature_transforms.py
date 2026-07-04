"""Tests for feature engineering transforms."""

import pytest

from app.ml.feature_store.transforms import safe_log_return, safe_pct, safe_ratio


def test_safe_pct() -> None:
    """Percentage change should handle edge cases."""
    assert safe_pct(110.0, 100.0) == pytest.approx(0.1)
    assert safe_pct(100.0, 0.0) is None
    assert safe_pct(None, 100.0) is None


def test_safe_ratio() -> None:
    """Ratio helper should guard zero denominators."""
    assert safe_ratio(10.0, 2.0) == pytest.approx(5.0)
    assert safe_ratio(10.0, 0.0) is None


def test_safe_log_return() -> None:
    """Log return should use natural log of price ratio."""
    assert safe_log_return(110.0, 100.0) == pytest.approx(0.095310179)
    assert safe_log_return(0.0, 100.0) is None
