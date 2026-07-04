"""Stop-loss rules (Phase 6+)."""


class StopLossManager:
    """Manage stop-loss levels for open positions."""

    def calculate_stop(self, entry_price: float, atr: float, multiplier: float = 2.0) -> float:
        """Calculate stop-loss price."""
        raise NotImplementedError("StopLossManager is available in Phase 6.")
