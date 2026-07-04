"""Parameter sweep utilities for backtest optimization."""

import itertools
import json
from pathlib import Path

from collections.abc import Callable

from loguru import logger
from pydantic import BaseModel, Field

from app.backtest.engine import BacktestEngine
from app.domain.backtest import BacktestConfig, BacktestResult
from app.domain.enums.market import Timeframe
from app.domain.instrument import Instrument
from app.strategy.ml_strategy import MLStrategy


class SweepResult(BaseModel):
    """Single backtest outcome from a parameter combination."""

    parameters: dict[str, float | int | None]
    total_return_pct: float
    sharpe_ratio: float | None
    profit_factor: float | None
    win_rate_pct: float | None
    total_trades: int
    max_drawdown_pct: float


class BacktestSweep:
    """Grid search over backtest configuration parameters."""

    def __init__(self, engine: BacktestEngine | None = None) -> None:
        self._engine = engine or BacktestEngine()

    def run(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        candles: list,
        features: list,
        strategy: MLStrategy,
        base_config: BacktestConfig,
        grid: dict[str, list[float | int | None]],
        strategy_factory: Callable[[BacktestConfig], MLStrategy] | None = None,
    ) -> list[SweepResult]:
        """Run backtests for all parameter combinations."""
        keys = list(grid.keys())
        results: list[SweepResult] = []
        combinations = list(itertools.product(*(grid[key] for key in keys)))
        logger.info("Running backtest sweep with {} combinations", len(combinations))

        for values in combinations:
            overrides = dict(zip(keys, values, strict=True))
            config = base_config.model_copy(update=overrides)
            active_strategy = strategy_factory(config) if strategy_factory else strategy
            result = self._engine.run(
                instrument,
                timeframe,
                candles,
                features,
                active_strategy,
                config=config,
            )
            results.append(self._to_sweep_result(overrides, result))

        results.sort(key=lambda item: item.total_return_pct, reverse=True)
        return results

    @staticmethod
    def _to_sweep_result(parameters: dict, result: BacktestResult) -> SweepResult:
        metrics = result.metrics
        return SweepResult(
            parameters=parameters,
            total_return_pct=metrics.total_return_pct,
            sharpe_ratio=metrics.sharpe_ratio,
            profit_factor=metrics.profit_factor,
            win_rate_pct=metrics.win_rate_pct,
            total_trades=metrics.total_trades,
            max_drawdown_pct=metrics.max_drawdown_pct,
        )

    @staticmethod
    def save(results: list[SweepResult], path: Path) -> Path:
        """Persist sweep results as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [result.model_dump() for result in results]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
