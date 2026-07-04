"""Exposure limits (Phase 6+)."""


class ExposureManager:
    """Enforce portfolio exposure limits."""

    def can_open(self, current_exposure: float, max_exposure: float) -> bool:
        """Return True if a new position is within exposure limits."""
        raise NotImplementedError("ExposureManager is available in Phase 6.")
