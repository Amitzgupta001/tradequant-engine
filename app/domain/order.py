"""Order entity (Phase 7)."""

from enum import Enum

from pydantic import BaseModel, Field

from app.domain.instrument import Instrument


class OrderSide(str, Enum):
    """Order transaction side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """Order lifecycle status."""

    PENDING = "PENDING"
    TRADED = "TRADED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Order(BaseModel):
    """Trade order placeholder for future execution layer."""

    instrument: Instrument
    side: OrderSide
    quantity: int = Field(gt=0)
    price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
