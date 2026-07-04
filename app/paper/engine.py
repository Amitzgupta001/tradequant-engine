"""Incremental paper trading bar processor."""

from datetime import date, datetime

import pandas as pd
from loguru import logger

from app.backtest.position_manager import manage_open_position
from app.backtest.portfolio import Portfolio
from app.backtest.trade_filters import TradeFilterState
from app.domain.backtest import BacktestConfig, SignalAction
from app.domain.candle import Candle
from app.domain.enums.market import Timeframe
from app.domain.instrument import Instrument
from app.ml.datasets.strategy_builder import StrategyDatasetBuilder
from app.ml.labels.base import LabelConfig
from app.ml.registry.strategy_registry import StrategyModelRegistry
from app.paper.models import PaperInstrumentState, PaperTradeRecord
from app.paper.portfolio_state import (
    filter_from_state,
    filter_to_state,
    portfolio_from_state,
    portfolio_to_state,
    trades_since,
)
from app.paper.store import PaperSessionStore
from app.strategy.presets import REWARD_PCT_5M, RISK_PCT_5M
from app.strategy.strategy_model_bridge import StrategyModelBridge


class PaperInstrumentEngine:
    """Process new bars for one instrument using a selected strategy model."""

    def __init__(
        self,
        dataset_builder: StrategyDatasetBuilder,
        model_registry: StrategyModelRegistry,
        store: PaperSessionStore,
    ) -> None:
        self._dataset_builder = dataset_builder
        self._model_registry = model_registry
        self._store = store

    def process(
        self,
        *,
        session_id: str,
        instrument: Instrument,
        symbol: str,
        timeframe: Timeframe,
        strategy_id: str,
        config: BacktestConfig,
        state: PaperInstrumentState | None = None,
    ) -> tuple[PaperInstrumentState, list[PaperTradeRecord]]:
        """Process all unprocessed bars and return updated state plus new trades."""
        state = state or PaperInstrumentState(
            security_id=instrument.security_id,
            symbol=symbol,
            portfolio_state={},
            filter_state={},
        )
        state.strategy_id = strategy_id

        label_config = LabelConfig(
            forward_horizon_bars=20,
            regression_threshold=REWARD_PCT_5M,
            take_profit_pct=REWARD_PCT_5M,
            stop_loss_pct=RISK_PCT_5M,
        )
        frame, _ = self._dataset_builder.build(
            strategy_id,
            instrument,
            timeframe,
            label_config,
        )
        working = frame.dropna(subset=["open", "high", "low", "close"]).copy()
        if len(working) < 2:
            logger.warning("Not enough strategy rows for paper trading: {}", symbol)
            return state, []

        symbol_config = config.model_copy(update={"initial_capital": config.initial_capital})
        portfolio = (
            portfolio_from_state(symbol_config, state.portfolio_state)
            if state.portfolio_state
            else Portfolio(symbol_config)
        )
        filters = (
            filter_from_state(symbol_config, state.filter_state)
            if state.filter_state
            else TradeFilterState(symbol_config)
        )

        signal_lookup: dict[object, int] = {}
        for _, row in working.iterrows():
            timestamp = row["timestamp"]
            if isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()
            signal_lookup[timestamp] = int(row.get("strategy_signal", 0) or 0)

        bridge = StrategyModelBridge.load(
            self._model_registry,
            strategy_id,
            signal_lookup,
            config=symbol_config,
        )

        rows = working.reset_index(drop=True)
        start_index = self._first_unprocessed_index(rows, state.last_processed_bar)
        if start_index >= len(rows) - 1:
            last_row = rows.iloc[-1]
            state.last_mark_price = float(last_row["close"])
            state.portfolio_state = portfolio_to_state(portfolio)
            state.filter_state = filter_to_state(filters)
            return state, []

        previous_trade_count = len(portfolio.trades)
        new_records: list[PaperTradeRecord] = []

        for index in range(start_index, len(rows) - 1):
            current = self._to_candle(rows.iloc[index])
            nxt = self._to_candle(rows.iloc[index + 1])
            row = rows.iloc[index]

            if portfolio.is_long:
                atr_pct = row.get("atr_pct") or row.get("atr_14")
                if isinstance(atr_pct, (int, float)) and atr_pct > 1:
                    atr_pct = atr_pct / current.close

                def should_exit_on_signal() -> bool:
                    signal = bridge.generate_signal_from_row(row)
                    if signal.action == SignalAction.BUY:
                        filters.reset_non_buy_streak()
                        return False
                    return filters.register_non_buy()

                manage_open_position(
                    portfolio,
                    filters,
                    current,
                    nxt,
                    atr_pct,
                    should_exit_on_signal,
                )
            else:
                filters.on_bar(current.timestamp)
                if filters.can_enter(current.timestamp):
                    signal = bridge.generate_signal_from_row(row)
                    if signal.action == SignalAction.BUY:
                        if portfolio.buy(nxt.open, nxt.timestamp, signal.confidence):
                            filters.on_entry(nxt.timestamp)

            state.last_processed_bar = nxt.timestamp

        closed = trades_since(portfolio, previous_trade_count)
        for trade in closed:
            record = PaperTradeRecord(
                trade_id=self._store.new_trade_id(),
                session_id=session_id,
                security_id=instrument.security_id,
                symbol=symbol,
                strategy_id=strategy_id,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                quantity=trade.quantity,
                pnl=trade.pnl,
                return_pct=trade.return_pct,
                exit_reason=trade.exit_reason,
                probability_at_entry=trade.probability_at_entry,
            )
            new_records.append(record)
            state.realized_pnl += trade.pnl
            state.trade_count += 1
            self._store.append_trade(session_id, record)

        last_row = rows.iloc[-1]
        state.last_mark_price = float(last_row["close"])
        state.portfolio_state = portfolio_to_state(portfolio)
        state.filter_state = filter_to_state(filters)
        return state, new_records

    @staticmethod
    def _first_unprocessed_index(rows: pd.DataFrame, last_processed_bar: datetime | None) -> int:
        if last_processed_bar is None:
            return 0
        for index in range(len(rows)):
            timestamp = rows.iloc[index]["timestamp"]
            if isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()
            if timestamp > last_processed_bar:
                return max(0, index - 1)
        return len(rows) - 1

    @staticmethod
    def _to_candle(row: pd.Series) -> Candle:
        timestamp = row["timestamp"]
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()
        return Candle(
            timestamp=timestamp,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0) or 0),
        )
