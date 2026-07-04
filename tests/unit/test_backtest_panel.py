"""Tests for panel backtest orchestration."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.data.universe.registry import Universe
from app.domain.backtest import (
    BacktestConfig,
    BacktestMetrics,
    BacktestResult,
    BacktestTrade,
    EquityPoint,
)
from app.domain.enums.market import ExchangeSegment, InstrumentType, Timeframe
from app.domain.instrument import Instrument
from app.services.backtest_service import BacktestService


@pytest.fixture
def universe() -> Universe:
    return Universe(
        universe_id="nifty50",
        name="Nifty 50",
        instruments=[
            Instrument(
                security_id="111",
                exchange_segment=ExchangeSegment.NSE_EQ,
                instrument_type=InstrumentType.EQUITY,
                symbol="AAA",
            ),
            Instrument(
                security_id="222",
                exchange_segment=ExchangeSegment.NSE_EQ,
                instrument_type=InstrumentType.EQUITY,
                symbol="BBB",
            ),
        ],
    )


def _result(security_id: str, return_pct: float) -> BacktestResult:
    return BacktestResult(
        instrument_security_id=security_id,
        exchange_segment="NSE_EQ",
        timeframe="MIN_5",
        config=BacktestConfig(),
        metrics=BacktestMetrics(
            initial_capital=100_000,
            final_equity=100_000 + return_pct * 1000,
            total_return_pct=return_pct,
            max_drawdown_pct=1.0,
            total_trades=2,
            winning_trades=1,
            losing_trades=1,
            win_rate_pct=50.0,
            profit_factor=1.2,
        ),
        trades=[
            BacktestTrade(
                entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
                entry_price=100.0,
                exit_price=101.0,
                quantity=10,
                pnl=10.0,
                return_pct=0.01,
            )
        ],
        equity_curve=[
            EquityPoint(
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                equity=100_000,
                cash=100_000,
                position_value=0.0,
            )
        ],
    )


def test_run_panel_aggregates_symbol_results(universe: Universe, tmp_path: Path) -> None:
    raw_file = tmp_path / "raw.csv"
    feature_file = tmp_path / "features.csv"
    raw_file.write_text("data", encoding="utf-8")
    feature_file.write_text("data", encoding="utf-8")

    historical = Mock()
    historical.build_path.return_value = raw_file

    features = Mock()
    features.build_path.return_value = feature_file

    report_store = Mock()
    report_store.save_panel.return_value = tmp_path / "panel_summary.json"
    service = BacktestService(
        historical_repository=historical,
        feature_repository=features,
        model_registry=Mock(),
        report_store=report_store,
        engine=Mock(),
    )

    service._build_panel_strategy = Mock(return_value=Mock())
    service._run_instrument = Mock(
        side_effect=[_result("111", 5.0), _result("222", -2.0)]
    )

    panel_result = service.run_panel(
        universe,
        Timeframe.MIN_5,
        config=BacktestConfig(),
    )

    assert panel_result.symbols_backtested == 2
    assert panel_result.total_trades == 2
    assert panel_result.mean_return_pct == 1.5
    assert panel_result.summary_path == str(tmp_path / "panel_summary.json")
