"""Bar-by-bar backtesting engine."""

from typing import Protocol

from loguru import logger

from app.backtest.metrics import attach_drawdowns, summarize_backtest
from app.backtest.position_manager import manage_open_position
from app.backtest.portfolio import Portfolio
from app.backtest.trade_filters import TradeFilterState
from app.domain.backtest import BacktestConfig, BacktestResult, EquityPoint, SignalAction
from app.domain.candle import Candle
from app.domain.enums.market import Timeframe
from app.domain.features import FeatureVector
from app.domain.instrument import Instrument
from app.domain.signal import Signal
from app.strategy.ml_strategy import MLStrategy


class SignalStrategy(Protocol):
    """Contract for strategies used by the backtest engine."""

    def generate_signal(self, features: FeatureVector) -> Signal:
        """Generate a trading signal from feature vector."""


class BacktestEngine:
    """Simulate long-only trading on historical data."""

    def run(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        candles: list[Candle],
        features: list[FeatureVector],
        strategy: SignalStrategy,
        config: BacktestConfig | None = None,
        strategy_signals: dict | None = None,
    ) -> BacktestResult:
        """Execute backtest aligned on candles and feature vectors."""
        config = config or BacktestConfig()
        if len(candles) != len(features):
            msg = "candles and features must have equal length"
            raise ValueError(msg)
        if len(candles) < 2:
            msg = "need at least 2 candles for backtesting"
            raise ValueError(msg)

        portfolio = Portfolio(config)
        filters = TradeFilterState(config)
        equity_curve: list[EquityPoint] = []
        signal_lookup = strategy_signals or {}

        for index in range(len(candles) - 1):
            current = candles[index]
            nxt = candles[index + 1]
            feature = features[index]

            if portfolio.is_long:
                atr_pct = feature.atr_pct

                def should_exit_on_signal() -> bool:
                    signal = strategy.generate_signal(feature)
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
                if filters.can_enter(current.timestamp) and self._entry_allowed(
                    feature,
                    strategy,
                    config,
                    signal_lookup,
                ):
                    signal = strategy.generate_signal(feature)
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

        last_candle = candles[-1]
        if portfolio.is_long:
            portfolio.sell(last_candle.close, last_candle.timestamp, exit_reason="end_of_data")

        if not equity_curve or equity_curve[-1].timestamp != last_candle.timestamp:
            equity_curve.append(
                EquityPoint(
                    timestamp=last_candle.timestamp,
                    equity=portfolio.cash,
                    cash=portfolio.cash,
                    position_value=0.0,
                )
            )
        else:
            equity_curve[-1] = EquityPoint(
                timestamp=last_candle.timestamp,
                equity=portfolio.cash,
                cash=portfolio.cash,
                position_value=0.0,
            )

        equity_curve = attach_drawdowns(equity_curve)
        metrics = summarize_backtest(config, portfolio.trades, equity_curve)

        logger.info(
            "Backtest complete trades={} return={:.2f}% sharpe={}",
            metrics.total_trades,
            metrics.total_return_pct,
            metrics.sharpe_ratio,
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
    def _entry_allowed(
        feature: FeatureVector,
        strategy: SignalStrategy,
        config: BacktestConfig,
        signal_lookup: dict,
    ) -> bool:
        """Check strategy signal and expected-value filters before entry."""
        if config.require_strategy_signal:
            raw_signal = signal_lookup.get(feature.timestamp, 0)
            if raw_signal <= 0:
                return False

        signal = strategy.generate_signal(feature)
        if signal.action != SignalAction.BUY:
            return False

        if config.min_expected_value is not None:
            probability = signal.confidence
            expected_value = (
                probability * config.expected_win_pct
                - (1 - probability) * config.expected_loss_pct
            )
            if expected_value < config.min_expected_value:
                return False
        return True
