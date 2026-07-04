"""Tests for hybrid strategy picker."""

import numpy as np

from app.ml.selector.picker import compute_strategy_priors, pick_strategy


def test_pick_strategy_uses_window_backtest_over_model_favorite() -> None:
    probabilities = np.array([0.30, 0.20, 0.15, 0.10, 0.05, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01])
    strategy_ids = [
        "macd_momentum",
        "breakout",
        "ema_pullback",
        "rsi_reversal",
        "bollinger_mean_reversion",
        "price_action_breakout",
        "supertrend",
        "orb",
        "vwap_breakout",
        "cpr_breakout",
        "ema_crossover",
        "breakdown",
    ]
    window_returns = {
        "macd_momentum": -0.4,
        "breakout": 0.6,
        "ema_pullback": -0.1,
    }
    strategy_id, _, _, reason = pick_strategy(
        probabilities,
        strategy_ids,
        window_returns=window_returns,
    )
    assert strategy_id == "breakout"
    assert reason == "hybrid window backtest score"


def test_compute_strategy_priors_averages_returns() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        [
            {"strategy_returns_json": '{"breakout": 0.4, "macd_momentum": -0.2}'},
            {"strategy_returns_json": '{"breakout": 0.2, "macd_momentum": -0.1}'},
        ]
    )
    priors = compute_strategy_priors(frame)
    assert priors["breakout"] == 0.30000000000000004 or abs(priors["breakout"] - 0.3) < 1e-9
    assert abs(priors["macd_momentum"] + 0.15) < 1e-9
