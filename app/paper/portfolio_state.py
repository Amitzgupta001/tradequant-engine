"""Serialize Portfolio and TradeFilterState for paper trading persistence."""

from app.backtest.portfolio import Portfolio
from app.backtest.trade_filters import TradeFilterState
from app.domain.backtest import BacktestConfig, BacktestTrade


def portfolio_to_state(portfolio: Portfolio) -> dict:
    """Export portfolio fields to a JSON-safe dict."""
    return {
        "cash": portfolio.cash,
        "quantity": portfolio.quantity,
        "initial_quantity": portfolio.initial_quantity,
        "entry_price": portfolio.entry_price,
        "entry_time": portfolio.entry_time.isoformat() if portfolio.entry_time else None,
        "entry_probability": portfolio.entry_probability,
        "peak_price": portfolio.peak_price,
        "bars_held": portfolio.bars_held,
        "target_1_hit": portfolio._target_1_hit,
        "target_2_hit": portfolio._target_2_hit,
        "target_3_hit": portfolio._target_3_hit,
        "stop_at_breakeven": portfolio._stop_at_breakeven,
    }


def portfolio_from_state(config: BacktestConfig, state: dict) -> Portfolio:
    """Restore a Portfolio from persisted state."""
    portfolio = Portfolio(config)
    portfolio.cash = float(state.get("cash", config.initial_capital))
    portfolio.quantity = int(state.get("quantity", 0))
    portfolio.initial_quantity = int(state.get("initial_quantity", 0))
    portfolio.entry_price = state.get("entry_price")
    entry_time = state.get("entry_time")
    portfolio.entry_time = None if entry_time is None else _parse_datetime(entry_time)
    portfolio.entry_probability = state.get("entry_probability")
    portfolio.peak_price = state.get("peak_price")
    portfolio.bars_held = int(state.get("bars_held", 0))
    portfolio._target_1_hit = bool(state.get("target_1_hit", False))
    portfolio._target_2_hit = bool(state.get("target_2_hit", False))
    portfolio._target_3_hit = bool(state.get("target_3_hit", False))
    portfolio._stop_at_breakeven = bool(state.get("stop_at_breakeven", False))
    return portfolio


def filter_to_state(filters: TradeFilterState) -> dict:
    """Export trade filter counters."""
    return {
        "bars_since_last_exit": filters.bars_since_last_exit,
        "cooldown_remaining": filters.cooldown_remaining,
        "trades_by_day": {key.isoformat(): value for key, value in filters.trades_by_day.items()},
        "non_buy_streak": filters.non_buy_streak,
        "last_stop_exit": filters.last_stop_exit,
    }


def filter_from_state(config: BacktestConfig, state: dict) -> TradeFilterState:
    """Restore trade filter counters."""
    filters = TradeFilterState(config)
    filters.bars_since_last_exit = int(state.get("bars_since_last_exit", 0))
    filters.cooldown_remaining = int(state.get("cooldown_remaining", 0))
    filters.non_buy_streak = int(state.get("non_buy_streak", 0))
    filters.last_stop_exit = bool(state.get("last_stop_exit", False))
    trades_by_day = state.get("trades_by_day", {})
    filters.trades_by_day = {
        _parse_date(key): int(value) for key, value in trades_by_day.items()
    }
    return filters


def open_position_from_portfolio(
    *,
    security_id: str,
    symbol: str,
    strategy_id: str,
    portfolio: Portfolio,
    mark_price: float | None,
) -> dict | None:
    """Build open position payload when portfolio is long."""
    if not portfolio.is_long or portfolio.entry_price is None or portfolio.entry_time is None:
        return None

    unrealized = None
    if mark_price is not None:
        commission = portfolio._config.commission_pct
        cost_basis = portfolio.quantity * portfolio.entry_price * (1 + commission)
        proceeds = portfolio.quantity * mark_price * (1 - commission)
        unrealized = proceeds - cost_basis

    return {
        "security_id": security_id,
        "symbol": symbol,
        "strategy_id": strategy_id,
        "entry_time": portfolio.entry_time,
        "entry_price": portfolio.entry_price,
        "quantity": portfolio.quantity,
        "mark_price": mark_price,
        "unrealized_pnl": unrealized,
        "bars_held": portfolio.bars_held,
    }


def trades_since(
    portfolio: Portfolio,
    previous_count: int,
) -> list[BacktestTrade]:
    """Return newly closed trade legs since the previous count."""
    return portfolio.trades[previous_count:]


def _parse_datetime(value: str | object):
    from datetime import datetime

    if isinstance(value, datetime):
        return value
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _parse_date(value: str | object):
    from datetime import date, datetime

    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return date.fromisoformat(str(value))
