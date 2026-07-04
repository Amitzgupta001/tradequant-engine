"""Tests for paper trading helpers."""

from datetime import datetime, timedelta, timezone

from app.backtest.portfolio import Portfolio
from app.backtest.trade_filters import TradeFilterState
from app.domain.backtest import BacktestConfig, BacktestTrade
from app.paper.market_hours import is_market_open
from app.paper.models import PaperInstrumentState, PaperSession
from app.paper.portfolio_state import (
    filter_from_state,
    filter_to_state,
    portfolio_from_state,
    portfolio_to_state,
    trades_since,
)
from app.paper.store import PaperSessionStore


IST = timezone(timedelta(hours=5, minutes=30))


def test_market_open_weekday_session() -> None:
    monday_open = datetime(2026, 7, 6, 10, 0, tzinfo=IST)
    assert is_market_open(monday_open) is True


def test_market_closed_weekend() -> None:
    saturday = datetime(2026, 7, 4, 10, 0, tzinfo=IST)
    assert is_market_open(saturday) is False


def test_portfolio_state_roundtrip() -> None:
    config = BacktestConfig(initial_capital=100_000.0)
    portfolio = Portfolio(config)
    assert portfolio.buy(100.0, datetime(2026, 7, 6, 10, 0), 0.8) is True

    restored = portfolio_from_state(config, portfolio_to_state(portfolio))
    assert restored.is_long
    assert restored.quantity == portfolio.quantity
    assert restored.entry_price == portfolio.entry_price


def test_filter_state_roundtrip() -> None:
    config = BacktestConfig(initial_capital=100_000.0, min_bars_between_entries=5)
    filters = TradeFilterState(config)
    filters.on_entry(datetime(2026, 7, 6, 10, 0))
    filters.on_exit(datetime(2026, 7, 6, 11, 0), "stop_loss")

    restored = filter_from_state(config, filter_to_state(filters))
    assert restored.cooldown_remaining == config.cooldown_bars_after_stop
    assert restored.trades_by_day


def test_trades_since() -> None:
    config = BacktestConfig(initial_capital=100_000.0)
    portfolio = Portfolio(config)
    portfolio.buy(100.0, datetime(2026, 7, 6, 10, 0), 0.8)
    portfolio.sell(101.0, datetime(2026, 7, 6, 10, 30), exit_reason="target_1")
    new_trades = trades_since(portfolio, 0)
    assert len(new_trades) == 1
    assert isinstance(new_trades[0], BacktestTrade)


def test_bar_slot_changed() -> None:
    from app.paper.bar_clock import bar_slot, bar_slot_changed

    first = datetime(2026, 7, 6, 10, 4, tzinfo=IST)
    second = datetime(2026, 7, 6, 10, 6, tzinfo=IST)
    assert bar_slot(first) == datetime(2026, 7, 6, 10, 0, tzinfo=IST)
    assert bar_slot_changed(first, second) is True
    assert bar_slot_changed(second, second.replace(minute=7)) is False


def test_paper_safety_flag() -> None:
    from app.paper.safety import PAPER_TRADING_ONLY, verify_paper_trading_mode

    assert PAPER_TRADING_ONLY is True
    verify_paper_trading_mode()


def test_paper_session_store(tmp_path) -> None:
    store = PaperSessionStore(tmp_path)
    session = store.create_session(
        universe_id="nifty50",
        timeframe="MIN_5",
        initial_capital=1_000_000.0,
        capital_per_symbol=20_000.0,
        instrument_ids=["25", "157"],
        selector_universe_id="nifty50",
        session_id="test_session",
    )
    assert session.session_id == "test_session"
    assert store.get_active_session_id() == "test_session"

    state = PaperInstrumentState(security_id="25", symbol="ADANIENT")
    store.save_instrument_state(session.session_id, state)
    loaded = store.load_instrument_state(session.session_id, "25")
    assert loaded is not None
    assert loaded.symbol == "ADANIENT"

    stopped = store.stop_session(session.session_id)
    assert stopped.status.value == "stopped"
