"""Tests for stock universe registry."""

from app.data.universe import get_universe, list_universes


def test_list_universes_includes_nifty50() -> None:
    universes = list_universes()
    assert "nifty50" in universes
    assert "nifty500" in universes


def test_get_nifty50_universe() -> None:
    universe = get_universe("nifty50")
    assert universe.id == "nifty50"
    assert len(universe.instruments) == 50
    assert all(instrument.security_id for instrument in universe.instruments)
    assert universe.instruments[0].exchange_segment.value == "NSE_EQ"


def test_get_nifty500_universe() -> None:
    universe = get_universe("nifty500")
    assert universe.id == "nifty500"
    assert len(universe.instruments) >= 450
    assert all(instrument.security_id for instrument in universe.instruments)


def test_get_unknown_universe_raises() -> None:
    try:
        get_universe("unknown")
    except ValueError as exc:
        assert "Unknown universe" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
