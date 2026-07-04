"""Feature engineering API routes."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.v1.dependencies import get_feature_service
from app.domain.enums.market import Timeframe
from app.domain.instrument import Instrument
from app.services.feature_service import FeatureService

router = APIRouter(prefix="/features", tags=["features"])


class FeatureBuildRequest(BaseModel):
    """Request to build ML features from stored raw data."""

    instrument: Instrument
    timeframe: Timeframe = Timeframe.DAILY
    overwrite: bool = True


class FeatureBuildResponse(BaseModel):
    """Response after building and storing features."""

    path: str
    row_count: int = Field(ge=0)
    feature_columns: list[str]
    latest: dict | None = None


@router.post("/build", response_model=FeatureBuildResponse)
def build_features(
    request: FeatureBuildRequest,
    service: FeatureService = Depends(get_feature_service),
) -> FeatureBuildResponse:
    """Build ML-ready features from stored OHLCV data."""
    features, path = service.build_and_store(
        request.instrument,
        request.timeframe,
        overwrite=request.overwrite,
    )
    latest = features[-1].model_dump(mode="json") if features else None
    columns = list(features[0].model_fields.keys()) if features else []
    return FeatureBuildResponse(
        path=str(path),
        row_count=len(features),
        feature_columns=columns,
        latest=latest,
    )
