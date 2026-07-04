"""Backtesting API routes."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.v1.dependencies import get_backtest_service
from app.domain.backtest import BacktestConfig
from app.domain.enums.market import Timeframe
from app.domain.instrument import Instrument
from app.services.backtest_service import BacktestService

router = APIRouter(prefix="/backtest", tags=["backtest"])


class BacktestRequest(BaseModel):
    """Request to run a backtest."""

    instrument: Instrument
    timeframe: Timeframe = Timeframe.DAILY
    initial_capital: float = Field(default=100_000.0, gt=0)
    probability_threshold: float = Field(default=0.55, gt=0.0, lt=1.0)
    commission_pct: float = Field(default=0.0003, ge=0)
    stop_loss_pct: float = Field(default=0.01, ge=0.0, le=0.05)
    trailing_stop_pct: float = Field(default=0.006, ge=0.0, le=0.05)
    trailing_activation_pct: float = Field(default=0.008, ge=0.0, le=0.05)
    max_hold_bars: int | None = Field(default=20, ge=1)


class BacktestResponse(BaseModel):
    """Backtest summary response."""

    summary_path: str | None
    equity_path: str | None
    metrics: dict
    trade_count: int


@router.post("/run", response_model=BacktestResponse)
def run_backtest(
    request: BacktestRequest,
    service: BacktestService = Depends(get_backtest_service),
) -> BacktestResponse:
    """Run ML strategy backtest on stored historical data."""
    config = BacktestConfig(
        initial_capital=request.initial_capital,
        probability_threshold=request.probability_threshold,
        commission_pct=request.commission_pct,
        stop_loss_pct=request.stop_loss_pct,
        trailing_stop_pct=request.trailing_stop_pct,
        trailing_activation_pct=request.trailing_activation_pct,
        max_hold_bars=request.max_hold_bars,
    )
    result, summary_path, equity_path = service.run(
        request.instrument,
        request.timeframe,
        config=config,
    )
    return BacktestResponse(
        summary_path=str(summary_path) if summary_path else None,
        equity_path=str(equity_path) if equity_path else None,
        metrics=result.metrics.model_dump(),
        trade_count=len(result.trades),
    )
