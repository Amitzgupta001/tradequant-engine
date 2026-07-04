"""Trading signal model."""

from pydantic import BaseModel, Field

from app.domain.backtest import SignalAction


class Signal(BaseModel):
    """Trading signal produced by a strategy."""

    action: SignalAction
    confidence: float = Field(ge=0.0, le=1.0)
