"""Position sizing (Phase 6+)."""


class PositionSizer:
    """Calculate position size based on risk parameters."""

    def size_for(self, capital: float, risk_pct: float, stop_loss_pct: float) -> int:
        """Return quantity for a given risk budget."""
        raise NotImplementedError("PositionSizer is available in Phase 6.")
