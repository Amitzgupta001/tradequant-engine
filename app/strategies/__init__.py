"""Built-in trading strategies for per-strategy ML pipelines."""

from app.strategies.bollinger.mean_reversion import BollingerMeanReversionStrategy
from app.strategies.breakout.breakdown import BreakdownStrategy
from app.strategies.breakout.breakout import BreakoutStrategy
from app.strategies.breakout.price_action import PriceActionBreakoutStrategy
from app.strategies.cpr.strategy import CprBreakoutStrategy
from app.strategies.ema.crossover import EmaCrossoverStrategy
from app.strategies.ema.pullback import EmaPullbackStrategy
from app.strategies.macd.momentum import MacdMomentumStrategy
from app.strategies.orb.strategy import OpeningRangeBreakoutStrategy
from app.strategies.registry import get_strategy, list_strategies, list_strategy_ids, register_strategy
from app.strategies.rsi.reversal import RsiReversalStrategy
from app.strategies.supertrend.strategy import SuperTrendStrategy
from app.strategies.vwap.breakout import VwapBreakoutStrategy


def _register_builtin_strategies() -> None:
    for strategy in (
        EmaCrossoverStrategy(),
        EmaPullbackStrategy(),
        VwapBreakoutStrategy(),
        OpeningRangeBreakoutStrategy(),
        CprBreakoutStrategy(),
        SuperTrendStrategy(),
        MacdMomentumStrategy(),
        RsiReversalStrategy(),
        BollingerMeanReversionStrategy(),
        PriceActionBreakoutStrategy(),
        BreakoutStrategy(),
        BreakdownStrategy(),
    ):
        register_strategy(strategy)


_register_builtin_strategies()

__all__ = [
    "get_strategy",
    "list_strategies",
    "list_strategy_ids",
    "register_strategy",
]
