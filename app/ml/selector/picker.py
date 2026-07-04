"""Pick a strategy by combining model probabilities with backtest scores."""

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from app.ml.selector.confidence import passes_selector_gate


def strategy_returns_from_row(row: pd.Series | dict) -> dict[str, float]:
    """Parse per-strategy window returns stored on a benchmark row."""
    payload = row["strategy_returns_json"] if isinstance(row, pd.Series) else row.get("strategy_returns_json")
    if not payload:
        return {}
    return {key: float(value) for key, value in json.loads(payload).items()}


def compute_strategy_priors(frame: pd.DataFrame) -> dict[str, float]:
    """Average walk-forward return per strategy from benchmark history."""
    totals: dict[str, list[float]] = defaultdict(list)
    for _, row in frame.iterrows():
        for strategy_id, return_pct in strategy_returns_from_row(row).items():
            totals[strategy_id].append(return_pct)
    return {
        strategy_id: sum(values) / len(values)
        for strategy_id, values in totals.items()
        if values
    }


def top_candidate_indices(probabilities: np.ndarray, top_k: int = 3) -> list[int]:
    """Return indices of the top-k model candidates."""
    limit = min(top_k, len(probabilities))
    if limit <= 0:
        return []
    return [int(index) for index in np.argsort(probabilities)[-limit:][::-1]]


def pick_strategy(
    probabilities: np.ndarray,
    strategy_ids: list[str],
    *,
    min_confidence: float = 0.0,
    min_margin: float = 0.0,
    window_returns: dict[str, float] | None = None,
    strategy_priors: dict[str, float] | None = None,
    top_k: int = 3,
    min_prior_return: float = 0.0,
) -> tuple[str | None, float, float, str]:
    """Choose a strategy using model shortlist plus backtest evidence."""
    _, top_prob, margin, passes = passes_selector_gate(
        probabilities,
        min_confidence=min_confidence,
        min_margin=min_margin,
    )
    if not passes:
        return None, top_prob, margin, "below gate"

    indices = top_candidate_indices(probabilities, top_k)
    candidates = [strategy_ids[index] for index in indices]
    if strategy_priors:
        candidates = [
            strategy_id
            for strategy_id in candidates
            if strategy_priors.get(strategy_id, float("-inf")) >= min_prior_return
        ] or candidates

    if window_returns:
        strategy_id = max(candidates, key=lambda item: window_returns.get(item, float("-inf")))
        return strategy_id, top_prob, margin, "hybrid window backtest score"

    if strategy_priors:
        strategy_id = max(candidates, key=lambda item: strategy_priors.get(item, float("-inf")))
        return strategy_id, top_prob, margin, "hybrid historical avg return"

    strategy_id = strategy_ids[int(np.argmax(probabilities))]
    return strategy_id, top_prob, margin, "model only"
