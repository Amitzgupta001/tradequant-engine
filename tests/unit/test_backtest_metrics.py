"""Tests for backtest metrics."""

from datetime import datetime, timezone

from app.backtest.metrics import calculate_max_drawdown, summarize_backtest
from app.domain.backtest import BacktestConfig, BacktestTrade, EquityPoint


def test_calculate_max_drawdown() -> None:
    """Drawdown should reflect peak-to-trough decline."""
    curve = [
        EquityPoint(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            equity=100.0,
            cash=100.0,
            position_value=0.0,
        ),
        EquityPoint(
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
            equity=120.0,
            cash=120.0,
            position_value=0.0,
        ),
        EquityPoint(
            timestamp=datetime(2024, 1, 3, tzinfo=timezone.utc),
            equity=90.0,
            cash=90.0,
            position_value=0.0,
        ),
    ]
    assert calculate_max_drawdown(curve) == 25.0


def test_summarize_backtest() -> None:
    """Summary should compute total return and win rate."""
    config = BacktestConfig(initial_capital=100_000.0)
    trades = [
        BacktestTrade(
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
            entry_price=100.0,
            exit_price=110.0,
            quantity=10,
            pnl=100.0,
            return_pct=0.1,
        ),
        BacktestTrade(
            entry_time=datetime(2024, 1, 3, tzinfo=timezone.utc),
            exit_time=datetime(2024, 1, 4, tzinfo=timezone.utc),
            entry_price=100.0,
            exit_price=95.0,
            quantity=10,
            pnl=-50.0,
            return_pct=-0.05,
        ),
    ]
    curve = [
        EquityPoint(
            timestamp=datetime(2024, 1, 4, tzinfo=timezone.utc),
            equity=100_050.0,
            cash=100_050.0,
            position_value=0.0,
        )
    ]
    metrics = summarize_backtest(config, trades, curve)
    assert metrics.total_trades == 2
    assert metrics.winning_trades == 1
    assert metrics.losing_trades == 1
