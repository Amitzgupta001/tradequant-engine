"""Tests for backtest trade filters and improved threshold tuning."""

from datetime import datetime, timezone

from app.backtest.trade_filters import TradeFilterState
from app.domain.backtest import BacktestConfig
from app.ml.evaluator.metrics import find_best_threshold, simulated_profit_factor


def test_trade_filter_blocks_daily_limit() -> None:
    config = BacktestConfig(max_trades_per_day=2, min_bars_between_entries=0)
    state = TradeFilterState(config)
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert state.can_enter(now)
    state.on_entry(now)
    state.on_entry(now)
    assert not state.can_enter(now)


def test_exit_confirmation_requires_multiple_non_buy_bars() -> None:
    config = BacktestConfig(exit_confirmation_bars=2)
    state = TradeFilterState(config)
    assert not state.register_non_buy()
    assert state.register_non_buy()


def test_simulated_profit_factor() -> None:
    y_true = [1, 1, 0, 0, 1, 0]
    y_prob = [0.55, 0.58, 0.51, 0.49, 0.57, 0.46]
    profit_factor, count = simulated_profit_factor(y_true, y_prob, 0.50, 0.003, 0.007)
    assert count == 4
    assert profit_factor is not None
    assert profit_factor > 0


def test_find_best_threshold_profit_objective() -> None:
    y_true = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0] * 5
    y_prob = [0.55, 0.40, 0.58, 0.39, 0.60, 0.35, 0.30, 0.57, 0.38, 0.32] * 5
    threshold, _, count, score = find_best_threshold(
        y_true,
        y_prob,
        objective="profit_factor",
        win_pct=0.003,
        loss_pct=0.007,
    )
    assert count >= 5
    assert threshold >= 0.15
    assert score is not None
