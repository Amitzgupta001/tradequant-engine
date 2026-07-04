"""Trade entity (Phase 7)."""

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.instrument import Instrument
from app.domain.order import OrderSide


class Trade(BaseModel):
    """Executed trade placeholder for future execution layer."""

    instrument: Instrument
    side: OrderSide
    quantity: int = Field(gt=0)
    price: float = Field(gt=0)
    executed_at: datetime
    order_id: str | None = None
