"""Manage open positions during backtest bar processing."""

from collections.abc import Callable

from app.backtest.portfolio import Portfolio
from app.backtest.trade_filters import TradeFilterState
from app.domain.candle import Candle


def manage_open_position(
    portfolio: Portfolio,
    filters: TradeFilterState,
    current: Candle,
    next_candle: Candle,
    atr_pct: float | None,
    on_signal_check: Callable[[], bool],
) -> None:
    """Process targets, stops, time exits, and signal exits for an open position."""
    portfolio.on_bar(current)
    trades_before = len(portfolio.trades)
    portfolio.process_profit_targets(current)
    if not portfolio.is_long:
        reason = portfolio.trades[-1].exit_reason if len(portfolio.trades) > trades_before else "target_3"
        filters.on_exit(current.timestamp, reason or "target_3")
        return

    stop_price = portfolio.check_stop_hit(current, atr_pct=atr_pct)
    if stop_price is not None:
        portfolio.sell(stop_price, current.timestamp, exit_reason="stop_loss")
        filters.on_exit(current.timestamp, "stop_loss")
        return

    if portfolio.should_time_exit():
        portfolio.sell(next_candle.open, next_candle.timestamp, exit_reason="max_hold")
        filters.on_exit(next_candle.timestamp, "max_hold")
        return

    if on_signal_check():
        portfolio.sell(next_candle.open, next_candle.timestamp, exit_reason="signal")
        filters.on_exit(next_candle.timestamp, "signal")
