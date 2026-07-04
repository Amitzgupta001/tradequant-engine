"""Strategy base contract."""

from typing import Protocol

from app.domain.candle import Candle
from app.domain.features import FeatureVector
from app.domain.instrument import Instrument
from app.domain.signal import Signal


class Strategy(Protocol):
    """Contract for trading strategies."""

    def generate_signal(self, features: FeatureVector) -> Signal:
        """Generate a trading signal from feature vector."""
        ...
