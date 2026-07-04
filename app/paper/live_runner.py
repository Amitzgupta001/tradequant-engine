"""Live WebSocket runner for paper trading."""

import time
from datetime import datetime

from loguru import logger

from app.brokers.dhan.live_feed import DhanLiveMarketFeed
from app.core.config import Settings, get_settings
from app.data.universe.registry import get_universe
from app.domain.enums.market import Timeframe
from app.paper.bar_clock import bar_slot_changed
from app.paper.market_hours import is_market_open
from app.paper.safety import verify_paper_trading_mode
from app.services.paper_trading_service import PaperTradingService


class PaperLiveRunner:
    """Use Dhan WebSocket for live LTP and REST for 5m bar closes."""

    def __init__(
        self,
        service: PaperTradingService,
        settings: Settings | None = None,
    ) -> None:
        verify_paper_trading_mode()
        self._service = service
        self._settings = settings or get_settings()
        self._feed: DhanLiveMarketFeed | None = None
        self._last_bar_slot: datetime | None = None

    def run(
        self,
        *,
        session_id: str | None = None,
        poll_seconds: float = 1.0,
        force: bool = False,
    ) -> None:
        """Run until interrupted."""
        session = self._service.get_active_session()
        if session is None:
            msg = "No active paper session. Run paper-trade --start first."
            raise ValueError(msg)
        if session_id is None:
            session_id = session.session_id

        universe = get_universe(session.universe_id)
        instruments = [
            item
            for item in universe.instruments
            if item.security_id in set(session.instrument_ids)
        ]

        self._feed = DhanLiveMarketFeed(
            client_id=self._settings.dhan_client_id,
            access_token=self._settings.dhan_access_token,
            instruments=instruments,
            on_tick=lambda security_id, ltp: self._service.update_live_price(
                security_id,
                ltp,
                session_id=session_id,
            ),
        )
        self._feed.start()
        logger.info(
            "Paper live runner started session={} symbols={} mode=websocket+bar_close",
            session_id,
            len(instruments),
        )

        try:
            while True:
                now = datetime.now()
                if (is_market_open(now) or force) and bar_slot_changed(self._last_bar_slot, now):
                    snapshot = self._service.tick(
                        session_id=session_id,
                        force=force,
                        lookback_days=PaperTradingService.BAR_CLOSE_LOOKBACK_DAYS,
                    )
                    self._last_bar_slot = now
                    logger.info(
                        "Bar close processed trades={} pnl={:.2f} open={}",
                        snapshot.total_trades,
                        snapshot.total_realized_pnl,
                        len(snapshot.open_positions),
                    )
                time.sleep(poll_seconds)
        finally:
            if self._feed is not None:
                self._feed.stop()
