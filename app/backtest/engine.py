"""Bar-by-bar backtesting engine."""

from loguru import logger

from app.backtest.metrics import attach_drawdowns, summarize_backtest
from app.backtest.portfolio import Portfolio
from app.domain.backtest import BacktestConfig, BacktestResult, EquityPoint, SignalAction
from app.domain.candle import Candle
from app.domain.enums.market import Timeframe
from app.domain.features import FeatureVector
from app.domain.instrument import Instrument
from app.strategy.ml_strategy import MLStrategy


class BacktestEngine:
    """Simulate long-only trading on historical data."""

    def run(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        candles: list[Candle],
        features: list[FeatureVector],
        strategy: MLStrategy,
        config: BacktestConfig | None = None,
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
        equity_curve: list[EquityPoint] = []

        for index in range(len(candles) - 1):
            current = candles[index]
            nxt = candles[index + 1]

            if portfolio.is_long:
                portfolio.on_bar(current)
                atr_pct = features[index].atr_pct
                stop_price = portfolio.check_stop_hit(current, atr_pct=atr_pct)
                if stop_price is not None:
                    portfolio.sell(stop_price, current.timestamp, exit_reason="stop_loss")
                elif portfolio.should_time_exit():
                    portfolio.sell(nxt.open, nxt.timestamp, exit_reason="max_hold")
                else:
                    signal = strategy.generate_signal(features[index])
                    if signal.action != SignalAction.BUY:
                        portfolio.sell(nxt.open, nxt.timestamp, exit_reason="signal")
            else:
                signal = strategy.generate_signal(features[index])
                if signal.action == SignalAction.BUY:
                    portfolio.buy(nxt.open, nxt.timestamp, signal.confidence)

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
