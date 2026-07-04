"""Broker client protocol."""

from typing import Any, Protocol

from app.domain.historical import HistoricalRequest, HistoricalResponse
from app.domain.instrument import Instrument


class BrokerClient(Protocol):
    """Contract for broker integrations."""

    def get_historical_data(self, request: HistoricalRequest) -> HistoricalResponse:
        """Fetch historical OHLCV candles for an instrument."""
        ...

    def connect_market_feed(self, instruments: list[Instrument]) -> None:
        """Connect to live market feed. Implemented in Phase 7."""
        ...

    def place_order(self, order: Any) -> Any:
        """Place a trade order. Implemented in Phase 7."""
        ...

    def get_positions(self) -> Any:
        """Retrieve open positions. Implemented in Phase 7."""
        ...
