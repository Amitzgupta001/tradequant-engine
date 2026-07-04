"""Historical data repository protocol."""

from pathlib import Path
from typing import Protocol

from app.domain.enums.market import Timeframe
from app.domain.historical import HistoricalResponse
from app.domain.instrument import Instrument


class HistoricalRepository(Protocol):
    """Contract for historical OHLCV persistence."""

    def save(self, response: HistoricalResponse, overwrite: bool = True) -> Path:
        """Persist historical candles."""
        ...

    def load(self, instrument: Instrument, timeframe: Timeframe) -> HistoricalResponse:
        """Load historical candles."""
        ...
