"""Event system for decoupled pipeline components."""

from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from typing import Any

from loguru import logger


class EventType(str, Enum):
    """Platform event types."""

    NEW_CANDLE = "new_candle"
    FEATURES_READY = "features_ready"
    PREDICTION_READY = "prediction_ready"
    RISK_APPROVED = "risk_approved"
    ORDER_PLACED = "order_placed"
    TRADE_EXECUTED = "trade_executed"


EventHandler = Callable[[Any], None]


class EventBus:
    """Simple in-process publish/subscribe event bus."""

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a handler for an event type."""
        self._subscribers[event_type].append(handler)

    def publish(self, event_type: EventType, payload: Any) -> None:
        """Publish an event to all registered handlers."""
        logger.debug("Publishing event {} with payload type {}", event_type.value, type(payload).__name__)
        for handler in self._subscribers[event_type]:
            handler(payload)

    def clear(self) -> None:
        """Remove all subscribers."""
        self._subscribers.clear()


event_bus = EventBus()
