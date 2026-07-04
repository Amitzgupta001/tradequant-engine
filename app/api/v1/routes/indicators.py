"""Indicator API routes."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.v1.dependencies import get_indicator_service
from app.domain.enums.market import Timeframe
from app.domain.instrument import Instrument
from app.services.indicator_service import IndicatorService

router = APIRouter(prefix="/indicators", tags=["indicators"])


class IndicatorComputeRequest(BaseModel):
    """Request to compute indicators from stored raw data."""

    instrument: Instrument
    timeframe: Timeframe = Timeframe.DAILY
    overwrite: bool = True


class IndicatorComputeResponse(BaseModel):
    """Response after computing and storing indicators."""

    path: str
    row_count: int = Field(ge=0)
    latest: dict | None = None


@router.post("/compute", response_model=IndicatorComputeResponse)
def compute_indicators(
    request: IndicatorComputeRequest,
    service: IndicatorService = Depends(get_indicator_service),
) -> IndicatorComputeResponse:
    """Compute technical indicators from stored OHLCV data."""
    snapshots, path = service.compute_and_store(
        request.instrument,
        request.timeframe,
        overwrite=request.overwrite,
    )
    latest_snapshot = snapshots[-1].model_dump(mode="json") if snapshots else None
    return IndicatorComputeResponse(
        path=str(path),
        row_count=len(snapshots),
        latest=latest_snapshot,
    )
