"""Label configuration for strategy ML pipelines."""

from enum import Enum

from pydantic import BaseModel, Field


class SignalLabel(str, Enum):
    """Classification labels for directional signals."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class LabelType(str, Enum):
    """Supported ML label types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TRIPLE_BARRIER = "triple_barrier"


class LabelConfig(BaseModel):
    """Configuration for strategy label generation."""

    label_type: LabelType = LabelType.CLASSIFICATION
    forward_horizon_bars: int = Field(default=5, ge=1, le=100)
    regression_threshold: float = Field(
        default=0.002,
        ge=0.0,
        description="Min absolute return for BUY/SELL classification",
    )
    take_profit_pct: float = Field(default=0.01, ge=0.001, le=0.2)
    stop_loss_pct: float = Field(default=0.005, ge=0.001, le=0.2)
    time_barrier_bars: int = Field(default=10, ge=1, le=200)
    version: str = "v1"
