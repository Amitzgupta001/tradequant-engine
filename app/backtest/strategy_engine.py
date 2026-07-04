"""Backtest engine for Phase 3 per-strategy models."""

from loguru import logger
import pandas as pd

from app.backtest.metrics import attach_drawdowns, summarize_backtest
from app.backtest.position_manager import manage_open_position
from app.backtest.portfolio import Portfolio
from app.backtest.trade_filters import TradeFilterState
from app.domain.backtest import BacktestConfig, BacktestResult, EquityPoint
from app.domain.candle import Candle
from app.domain.enums.market import Timeframe
from app.domain.instrument import Instrument
from app.domain.backtest import SignalAction
from app.strategy.strategy_model_bridge import StrategyModelBridge


class StrategyBacktestEngine:
    """Simulate long-only trading on a strategy dataset dataframe."""

    def run(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        frame: pd.DataFrame,
        bridge: StrategyModelBridge,
        config: BacktestConfig | None = None,
    ) -> BacktestResult:
        """Execute backtest bar-by-bar on aligned strategy rows."""
        config = config or BacktestConfig()
        working = frame.dropna(subset=["open", "high", "low", "close"]).copy()
        if len(working) < 2:
            msg = "Need at least 2 strategy rows for backtesting"
            raise ValueError(msg)

        portfolio = Portfolio(config)
        filters = TradeFilterState(config)
        equity_curve: list[EquityPoint] = []
        rows = working.reset_index(drop=True)

        for index in range(len(rows) - 1):
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

            equity_curve.append(
                EquityPoint(
                    timestamp=nxt.timestamp,
                    equity=portfolio.equity(nxt.close),
                    cash=portfolio.cash,
                    position_value=portfolio.quantity * nxt.close,
                )
            )

        last = self._to_candle(rows.iloc[-1])
        if portfolio.is_long:
            portfolio.sell(last.close, last.timestamp, exit_reason="end_of_data")

        equity_curve = attach_drawdowns(equity_curve)
        metrics = summarize_backtest(config, portfolio.trades, equity_curve)
        logger.info(
            "Strategy backtest complete trades={} return={:.2f}%",
            metrics.total_trades,
            metrics.total_return_pct,
        )
        return BacktestResult(
            instrument_security_id=instrument.security_id,
            exchange_segment=instrument.exchange_segment.value,
            timeframe=timeframe.value,
            config=config,
            metrics=metrics,
            trades=portfolio.trades,
            equity_curve=equity_curve,
        )

    @staticmethod
    def _to_candle(row: pd.Series) -> Candle:
        return Candle(
            timestamp=row["timestamp"].to_pydatetime()
            if hasattr(row["timestamp"], "to_pydatetime")
            else row["timestamp"],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row["volume"]),
        )
