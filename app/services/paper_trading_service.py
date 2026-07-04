"""Paper trading orchestration for live forward testing."""

from datetime import date, datetime, timedelta
from pathlib import Path

from loguru import logger

from app.core.config import Settings, get_settings
from app.data.repositories.base import HistoricalRepository
from app.data.universe.registry import Universe, get_universe
from app.domain.backtest import BacktestConfig
from app.domain.enums.market import Timeframe
from app.domain.instrument import Instrument
from app.ml.datasets.strategy_builder import StrategyDatasetBuilder
from app.ml.registry.strategy_registry import StrategyModelRegistry
from app.paper.engine import PaperInstrumentEngine
from app.paper.market_hours import is_market_open, session_date
from app.paper.models import (
    OpenPosition,
    PaperDashboardSnapshot,
    PaperInstrumentState,
    PaperSession,
    PaperSessionStatus,
    PaperTradeRecord,
)
from app.paper.portfolio_state import open_position_from_portfolio, portfolio_from_state
from app.paper.store import PaperSessionStore
from app.services.strategy_selector_service import StrategySelectorService
from app.services.training_service import TrainingService
from app.paper.safety import verify_paper_trading_mode
from app.strategy.presets import BEST_5M_BACKTEST, BEST_5M_DAYS


class PaperTradingService:
    """Run and inspect paper trading sessions."""

    LOOKBACK_DAYS = 60
    BAR_CLOSE_LOOKBACK_DAYS = 5

    def __init__(
        self,
        repository: HistoricalRepository,
        training_service: TrainingService,
        selector_service: StrategySelectorService,
        settings: Settings | None = None,
    ) -> None:
        verify_paper_trading_mode()
        self._settings = settings or get_settings()
        storage_path = Path(self._settings.storage_path)
        self._repository = repository
        self._training_service = training_service
        self._selector_service = selector_service
        self._store = PaperSessionStore(storage_path / "paper")
        self._model_registry = StrategyModelRegistry(storage_path)
        self._dataset_builder = StrategyDatasetBuilder(repository, storage_path)
        self._engine = PaperInstrumentEngine(
            self._dataset_builder,
            self._model_registry,
            self._store,
        )

    @property
    def store(self) -> PaperSessionStore:
        return self._store

    def start_session(
        self,
        *,
        universe_id: str,
        timeframe: Timeframe = Timeframe.MIN_5,
        initial_capital: float = 100_000.0,
        selector_universe_id: str | None = None,
        security_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> PaperSession:
        """Create a new paper trading session for a universe watchlist."""
        universe = get_universe(universe_id)
        instruments = self._resolve_instruments(universe, security_ids)
        if not instruments:
            msg = f"No instruments resolved for universe '{universe_id}'"
            raise ValueError(msg)

        capital_per_symbol = initial_capital / len(instruments)
        session = self._store.create_session(
            universe_id=universe_id,
            timeframe=timeframe.value,
            initial_capital=initial_capital,
            capital_per_symbol=capital_per_symbol,
            instrument_ids=[item.security_id for item, _ in instruments],
            selector_universe_id=selector_universe_id or universe_id,
            session_id=session_id,
        )

        for instrument, symbol in instruments:
            state = PaperInstrumentState(
                security_id=instrument.security_id,
                symbol=symbol,
                portfolio_state={"cash": capital_per_symbol},
            )
            self._store.save_instrument_state(session.session_id, state)

        logger.info(
            "Started paper session {} universe={} symbols={}",
            session.session_id,
            universe_id,
            len(instruments),
        )
        return session

    def get_active_session(self) -> PaperSession | None:
        return self._store.load_active_session()

    def stop_session(self, session_id: str | None = None) -> PaperSession:
        active = session_id or self._store.get_active_session_id()
        if active is None:
            msg = "No active paper trading session"
            raise ValueError(msg)
        return self._store.stop_session(active)

    def tick(
        self,
        session_id: str | None = None,
        *,
        force: bool = False,
        lookback_days: int | None = None,
    ) -> PaperDashboardSnapshot:
        """Refresh market data, process new bars, and persist paper trades."""
        session = self._load_session(session_id)
        if session.status != PaperSessionStatus.RUNNING:
            msg = f"Paper session '{session.session_id}' is not running"
            raise ValueError(msg)

        market_open = is_market_open()
        if not market_open and not force:
            return self.dashboard(session.session_id)

        universe = get_universe(session.universe_id)
        timeframe = Timeframe(session.timeframe)
        config = BEST_5M_BACKTEST.model_copy(
            update={
                "initial_capital": session.capital_per_symbol,
                "require_strategy_signal": True,
            }
        )
        today = session_date()
        end = date.today()
        refresh_days = lookback_days if lookback_days is not None else self.LOOKBACK_DAYS
        start = end - timedelta(days=min(BEST_5M_DAYS, refresh_days))

        processed = 0
        last_error: str | None = None
        for instrument, symbol in self._resolve_instruments(
            universe,
            session.instrument_ids,
        ):
            try:
                self._refresh_market_data(instrument, timeframe, start, end)
                state = self._store.load_instrument_state(session.session_id, instrument.security_id)
                if state is None:
                    state = PaperInstrumentState(
                        security_id=instrument.security_id,
                        symbol=symbol,
                        portfolio_state={"cash": session.capital_per_symbol},
                    )

                strategy_id = self._resolve_strategy(
                    instrument,
                    timeframe,
                    state,
                    today,
                    session.selector_universe_id,
                )
                state, _ = self._engine.process(
                    session_id=session.session_id,
                    instrument=instrument,
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy_id=strategy_id,
                    config=config,
                    state=state,
                )
                self._store.save_instrument_state(session.session_id, state)
                processed += 1
            except Exception as exc:
                last_error = f"{symbol} ({instrument.security_id}): {exc}"
                logger.exception("Paper tick failed for {}", symbol)

        session.last_tick_at = datetime.now()
        session.last_error = last_error
        self._store.save_session(session)
        snapshot = self.dashboard(session.session_id)
        snapshot.symbols_processed = processed
        return snapshot

    def dashboard(self, session_id: str | None = None) -> PaperDashboardSnapshot:
        """Build dashboard snapshot for API/UI."""
        session = self._load_session(session_id)
        trades = self._store.load_trades(session.session_id)
        states = self._store.list_instrument_states(session.session_id)
        open_positions: list[OpenPosition] = []

        for state in states:
            if not state.portfolio_state or not state.strategy_id:
                continue
            config = BEST_5M_BACKTEST.model_copy(
                update={"initial_capital": session.capital_per_symbol}
            )
            portfolio = portfolio_from_state(config, state.portfolio_state)
            payload = open_position_from_portfolio(
                security_id=state.security_id,
                symbol=state.symbol,
                strategy_id=state.strategy_id,
                portfolio=portfolio,
                mark_price=state.last_mark_price,
            )
            if payload is not None:
                open_positions.append(OpenPosition.model_validate(payload))

        total_realized = sum(trade.pnl for trade in trades)
        return PaperDashboardSnapshot(
            session=session,
            market_open=is_market_open(),
            open_positions=open_positions,
            recent_trades=trades[-50:],
            total_realized_pnl=total_realized,
            total_trades=len(trades),
            symbols_processed=len(states),
            last_tick_at=session.last_tick_at,
        )

    def list_trades(
        self,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> list[PaperTradeRecord]:
        session = self._load_session(session_id)
        return self._store.load_trades(session.session_id, limit=limit)

    def update_live_price(
        self,
        security_id: str,
        mark_price: float,
        session_id: str | None = None,
    ) -> None:
        """Update mark price from WebSocket tick without placing broker orders."""
        session = self._load_session(session_id)
        state = self._store.load_instrument_state(session.session_id, security_id)
        if state is None:
            return
        state.last_mark_price = mark_price
        self._store.save_instrument_state(session.session_id, state)

    def _load_session(self, session_id: str | None) -> PaperSession:
        if session_id is not None:
            return self._store.load_session(session_id)
        session = self._store.load_active_session()
        if session is None:
            msg = "No active paper trading session. Run paper-trade --start first."
            raise ValueError(msg)
        return session

    def _refresh_market_data(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        start: date,
        end: date,
    ) -> None:
        self._training_service.prepare_data(instrument, timeframe, start, end)
        self._training_service.release_batch_memory()

    def _resolve_strategy(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        state: PaperInstrumentState,
        today: date,
        selector_universe_id: str | None,
    ) -> str:
        if state.strategy_id and state.strategy_date == today:
            return state.strategy_id

        recommendations = self._selector_service.recommend(
            instrument,
            timeframe,
            universe_id=selector_universe_id,
            top_n=3,
        )
        if not recommendations:
            msg = f"No strategy recommendation for security_id={instrument.security_id}"
            raise ValueError(msg)

        strategy_id = recommendations[0].strategy_id
        state.strategy_id = strategy_id
        state.strategy_date = today
        return strategy_id

    @staticmethod
    def _resolve_instruments(
        universe: Universe,
        security_ids: list[str] | None,
    ) -> list[tuple[Instrument, str]]:
        if security_ids:
            lookup = {item.security_id: item for item in universe.instruments}
            resolved: list[tuple[Instrument, str]] = []
            for security_id in security_ids:
                item = lookup.get(str(security_id))
                if item is None:
                    continue
                resolved.append((item, item.symbol or security_id))
            return resolved

        return [(item, item.symbol or item.security_id) for item in universe.instruments]
