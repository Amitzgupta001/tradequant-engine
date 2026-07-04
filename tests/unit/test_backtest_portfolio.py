"""Tests for backtest portfolio."""

from datetime import datetime, timezone

from app.backtest.portfolio import Portfolio
from app.domain.backtest import BacktestConfig
from app.domain.candle import Candle


def test_portfolio_round_trip_trade() -> None:
    """Portfolio should record PnL after buy and sell."""
    portfolio = Portfolio(BacktestConfig(initial_capital=100_000.0, commission_pct=0.0))
    entry_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    exit_time = datetime(2024, 1, 2, tzinfo=timezone.utc)

    assert portfolio.buy(100.0, entry_time, probability=0.6)
    assert portfolio.is_long
    assert portfolio.sell(110.0, exit_time)

    assert not portfolio.is_long
    assert len(portfolio.trades) == 1
    assert portfolio.trades[0].pnl > 0


def test_portfolio_insufficient_capital() -> None:
    """Portfolio should reject buy when price exceeds budget."""
    portfolio = Portfolio(BacktestConfig(initial_capital=10.0))
    assert not portfolio.buy(1000.0, datetime(2024, 1, 1, tzinfo=timezone.utc))


def test_trailing_stop_exits_losing_trade() -> None:
    """Fixed stop should cut a losing position intrabar."""
    config = BacktestConfig(
        initial_capital=100_000.0,
        commission_pct=0.0,
        stop_loss_pct=0.01,
        trailing_stop_pct=0.005,
    )
    portfolio = Portfolio(config)
    entry_time = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)

    assert portfolio.buy(100.0, entry_time)
    portfolio.on_bar(
        Candle(
            timestamp=entry_time,
            open=100.0,
            high=100.5,
            low=98.5,
            close=99.0,
            volume=1000,
        )
    )

    stop_price = portfolio.check_stop_hit(
        Candle(
            timestamp=entry_time,
            open=99.0,
            high=99.5,
            low=98.5,
            close=98.8,
            volume=1000,
        )
    )

    assert stop_price is not None
    assert stop_price >= 99.0
    assert portfolio.sell(stop_price, entry_time, exit_reason="stop_loss")
    assert portfolio.trades[0].exit_reason == "stop_loss"
    assert portfolio.trades[0].pnl < 0
