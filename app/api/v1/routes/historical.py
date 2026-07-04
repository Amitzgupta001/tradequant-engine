"""Historical data routes."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.v1.dependencies import get_historical_data_service
from app.domain.historical import HistoricalRequest
from app.services.historical_data_service import HistoricalDataService

router = APIRouter(prefix="/historical", tags=["historical"])


class HistoricalDownloadResponse(BaseModel):
    """Response after downloading and storing historical data."""

    path: str
    candle_count: int = Field(ge=0)


@router.post("/download", response_model=HistoricalDownloadResponse)
def download_historical_data(
    request: HistoricalRequest,
    overwrite: bool = True,
    service: HistoricalDataService = Depends(get_historical_data_service),
) -> HistoricalDownloadResponse:
    """Download historical OHLCV data from Dhan and store as CSV."""
    response, path = service.download_and_store(request, overwrite=overwrite)
    return HistoricalDownloadResponse(
        path=str(path),
        candle_count=len(response.candles),
    )
