"""Broker-related exceptions."""


class BrokerError(Exception):
    """Base exception for broker operations."""


class BrokerAuthenticationError(BrokerError):
    """Raised when broker authentication fails."""


class BrokerAPIError(BrokerError):
    """Raised when a broker API call fails."""

    def __init__(self, message: str, remarks: object | None = None) -> None:
        super().__init__(message)
        self.remarks = remarks


class DhanAPIError(BrokerAPIError):
    """Raised when the Dhan API returns an error response."""
