"""Paper trading API routes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.v1.dependencies import get_paper_trading_service
from app.domain.enums.market import Timeframe
from app.paper.models import PaperDashboardSnapshot, PaperSession, PaperTradeRecord
from app.services.paper_trading_service import PaperTradingService

router = APIRouter(prefix="/paper", tags=["paper"])


class StartPaperSessionRequest(BaseModel):
    """Start a paper trading session."""

    universe_id: str = "nifty50"
    selector_universe_id: str | None = None
    timeframe: Timeframe = Timeframe.MIN_5
    initial_capital: float = Field(default=1_000_000.0, gt=0)
    security_ids: list[str] | None = None
    session_id: str | None = None


class TickRequest(BaseModel):
    """Manually trigger one paper trading tick."""

    session_id: str | None = None
    force: bool = False


@router.get("/dashboard", response_model=PaperDashboardSnapshot)
def get_dashboard(
    session_id: str | None = None,
    service: PaperTradingService = Depends(get_paper_trading_service),
) -> PaperDashboardSnapshot:
    """Return dashboard snapshot for the active or requested session."""
    try:
        return service.dashboard(session_id=session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/session", response_model=PaperSession)
def get_session(
    service: PaperTradingService = Depends(get_paper_trading_service),
) -> PaperSession:
    """Return the active paper session."""
    session = service.get_active_session()
    if session is None:
        raise HTTPException(status_code=404, detail="No active paper trading session")
    return session


@router.get("/trades", response_model=list[PaperTradeRecord])
def list_trades(
    session_id: str | None = None,
    limit: int | None = 100,
    service: PaperTradingService = Depends(get_paper_trading_service),
) -> list[PaperTradeRecord]:
    """List closed paper trades."""
    try:
        return service.list_trades(session_id=session_id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/start", response_model=PaperSession)
def start_session(
    request: StartPaperSessionRequest,
    service: PaperTradingService = Depends(get_paper_trading_service),
) -> PaperSession:
    """Create and activate a new paper trading session."""
    return service.start_session(
        universe_id=request.universe_id,
        timeframe=request.timeframe,
        initial_capital=request.initial_capital,
        selector_universe_id=request.selector_universe_id,
        security_ids=request.security_ids,
        session_id=request.session_id,
    )


@router.post("/tick", response_model=PaperDashboardSnapshot)
def run_tick(
    request: TickRequest,
    service: PaperTradingService = Depends(get_paper_trading_service),
) -> PaperDashboardSnapshot:
    """Run one paper trading tick."""
    try:
        return service.tick(session_id=request.session_id, force=request.force)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/stop", response_model=PaperSession)
def stop_session(
    session_id: str | None = None,
    service: PaperTradingService = Depends(get_paper_trading_service),
) -> PaperSession:
    """Stop the active paper session."""
    try:
        return service.stop_session(session_id=session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
