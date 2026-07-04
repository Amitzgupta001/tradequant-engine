"""Dhan live market feed wrapper (WebSocket, read-only)."""

from collections.abc import Callable
from threading import Thread

from dhanhq import DhanContext, MarketFeed
from loguru import logger

from app.domain.enums.market import ExchangeSegment
from app.domain.instrument import Instrument
from app.paper.safety import verify_paper_trading_mode


class DhanLiveMarketFeed:
    """Subscribe to Dhan ticker WebSocket for live LTP updates."""

    _EXCHANGE_MAP = {
        ExchangeSegment.NSE_EQ: MarketFeed.NSE,
        ExchangeSegment.BSE_EQ: MarketFeed.BSE,
    }

    def __init__(
        self,
        client_id: str,
        access_token: str,
        instruments: list[Instrument],
        on_tick: Callable[[str, float], None],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        verify_paper_trading_mode()
        self._on_tick = on_tick
        self._on_error = on_error
        self._subscriptions = self._build_subscriptions(instruments)
        self._feed = MarketFeed(
            DhanContext(client_id, access_token),
            self._subscriptions,
            version="v2",
            on_message=self._handle_message,
            on_error=self._handle_error,
        )
        self._thread: Thread | None = None

    @staticmethod
    def _build_subscriptions(instruments: list[Instrument]) -> list[tuple[int, int, int]]:
        subscriptions: list[tuple[int, int, int]] = []
        for instrument in instruments:
            exchange = DhanLiveMarketFeed._EXCHANGE_MAP.get(instrument.exchange_segment, MarketFeed.NSE)
            subscriptions.append((exchange, int(instrument.security_id), MarketFeed.Ticker))
        return subscriptions

    def start(self) -> None:
        """Start the WebSocket feed in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        logger.info("Starting Dhan live market feed for {} instruments", len(self._subscriptions))
        self._thread = self._feed.start()

    def stop(self) -> None:
        """Stop the WebSocket feed."""
        logger.info("Stopping Dhan live market feed")
        self._feed.close_connection()

    def _handle_message(self, _feed: MarketFeed, data: dict | str) -> None:
        if not isinstance(data, dict):
            return
        if data.get("type") != "Ticker Data":
            return
        security_id = str(data.get("security_id", ""))
        ltp_raw = data.get("LTP")
        if not security_id or ltp_raw is None:
            return
        try:
            ltp = float(ltp_raw)
        except (TypeError, ValueError):
            return
        self._on_tick(security_id, ltp)

    def _handle_error(self, _feed: MarketFeed, error: Exception) -> None:
        logger.error("Dhan live feed error: {}", error)
        if self._on_error is not None:
            self._on_error(error)
