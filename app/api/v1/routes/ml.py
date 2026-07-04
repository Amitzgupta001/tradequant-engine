"""ML training API routes."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.v1.dependencies import get_training_service
from app.domain.enums.market import Timeframe
from app.domain.instrument import Instrument
from app.domain.training import TrainingConfig, TrainingTask
from app.services.training_service import TrainingService

router = APIRouter(prefix="/ml", tags=["ml"])


class TrainRequest(BaseModel):
    """Request to train a LightGBM model."""

    instrument: Instrument
    timeframe: Timeframe = Timeframe.DAILY
    years: int = Field(default=3, ge=1, le=10)
    task: TrainingTask = TrainingTask.CLASSIFICATION
    test_size: float = Field(default=0.2, gt=0.0, lt=0.5)
    skip_download: bool = False


class TrainResponse(BaseModel):
    """Response after model training."""

    model_path: str
    metadata_path: str
    metrics: dict
    feature_columns: list[str]


@router.post("/train", response_model=TrainResponse)
def train_model(
    request: TrainRequest,
    service: TrainingService = Depends(get_training_service),
) -> TrainResponse:
    """Train LightGBM on historical features."""
    config = TrainingConfig(task=request.task, test_size=request.test_size)
    if request.skip_download:
        result = service.train_from_features(
            request.instrument,
            request.timeframe,
            config=config,
        )
    else:
        result = service.train_with_history(
            request.instrument,
            request.timeframe,
            years=request.years,
            config=config,
        )
    return TrainResponse(
        model_path=result.model_path,
        metadata_path=result.metadata_path,
        metrics=result.metrics.model_dump(),
        feature_columns=result.feature_columns,
    )
