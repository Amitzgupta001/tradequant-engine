"""Backtest orchestration service."""

from pathlib import Path

from loguru import logger

from app.backtest.engine import BacktestEngine
from app.backtest.panel_summary import summarize_panel_backtest
from app.backtest.report import BacktestReportStore
from app.data.universe.registry import Universe
from app.data.repositories.base import HistoricalRepository
from app.domain.backtest import BacktestConfig, BacktestResult, PanelBacktestResult
from app.domain.enums.market import Timeframe
from app.domain.instrument import Instrument
from app.ml.feature_store.repository import FeatureRepository
from app.ml.inference.predictor import LightGBMPredictor
from app.ml.registry.model_registry import ModelRegistry
from app.strategy.ml_strategy import MLStrategy


class BacktestService:
    """Run ML strategy backtests on stored data."""

    def __init__(
        self,
        historical_repository: HistoricalRepository,
        feature_repository: FeatureRepository,
        model_registry: ModelRegistry,
        report_store: BacktestReportStore,
        engine: BacktestEngine | None = None,
    ) -> None:
        self._historical_repository = historical_repository
        self._feature_repository = feature_repository
        self._model_registry = model_registry
        self._report_store = report_store
        self._engine = engine or BacktestEngine()

    def run(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        config: BacktestConfig | None = None,
        save_report: bool = True,
    ) -> tuple[BacktestResult, Path | None, Path | None]:
        """Run backtest using a per-stock registered model."""
        config = config or BacktestConfig()
        strategy = self._build_strategy(
            instrument.exchange_segment.value,
            instrument.security_id,
            timeframe.value,
            config=config,
        )
        result = self._run_instrument(instrument, timeframe, strategy, config)

        summary_path = None
        equity_path = None
        if save_report:
            summary_path, equity_path = self._report_store.save(instrument, timeframe, result)

        return result, summary_path, equity_path

    def run_panel(
        self,
        universe: Universe,
        timeframe: Timeframe,
        config: BacktestConfig | None = None,
        save_report: bool = True,
        save_per_symbol: bool = False,
    ) -> PanelBacktestResult:
        """Run backtests for all universe symbols using a shared panel model."""
        config = config or BacktestConfig()
        exchange = universe.instruments[0].exchange_segment.value
        strategy = self._build_panel_strategy(universe.id, exchange, timeframe.value, config)

        results: list[BacktestResult] = []
        skipped = 0
        total = len(universe.instruments)

        logger.info(
            "Panel backtest for {} on {} symbols",
            universe.name,
            total,
        )

        for index, instrument in enumerate(universe.instruments, start=1):
            if not self._has_backtest_data(instrument, timeframe):
                skipped += 1
                continue

            label = instrument.symbol or instrument.security_id
            logger.info("[{}/{}] Backtesting {} ({})", index, total, label, instrument.security_id)
            result = self._run_instrument(instrument, timeframe, strategy, config)
            results.append(result)

            if save_per_symbol:
                self._report_store.save(instrument, timeframe, result)

        if not results:
            msg = f"No backtest data found for universe '{universe.id}'"
            raise FileNotFoundError(msg)

        panel_result = summarize_panel_backtest(
            universe.id,
            timeframe.value,
            config,
            results,
            symbols_total=total,
            symbols_skipped=skipped,
            instruments_by_id={
                instrument.security_id: instrument.symbol for instrument in universe.instruments
            },
        )

        if save_report:
            summary_path = self._report_store.save_panel(universe.id, timeframe, panel_result)
            panel_result = panel_result.model_copy(update={"summary_path": str(summary_path)})

        logger.info(
            "Panel backtest complete symbols={} trades={} mean_return={:.2f}%",
            panel_result.symbols_backtested,
            panel_result.total_trades,
            panel_result.mean_return_pct or 0.0,
        )
        return panel_result

    def _build_strategy(
        self,
        exchange_segment: str,
        security_id: str,
        timeframe: str,
        config: BacktestConfig,
    ) -> MLStrategy:
        predictor = LightGBMPredictor(self._model_registry)
        predictor.load(exchange_segment, security_id, timeframe)
        metadata = self._model_registry.load_metadata(exchange_segment, security_id, timeframe)
        return MLStrategy(
            predictor,
            metadata=metadata,
            probability_threshold=config.probability_threshold,
        )

    def _build_panel_strategy(
        self,
        universe_id: str,
        exchange_segment: str,
        timeframe: str,
        config: BacktestConfig,
    ) -> MLStrategy:
        predictor = LightGBMPredictor(self._model_registry)
        predictor.load_panel(universe_id, timeframe)
        metadata = self._model_registry.load_panel_metadata(universe_id, timeframe)
        return MLStrategy(
            predictor,
            metadata=metadata,
            probability_threshold=config.probability_threshold,
        )

    def _run_instrument(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        strategy: MLStrategy,
        config: BacktestConfig,
    ) -> BacktestResult:
        historical = self._historical_repository.load(instrument, timeframe)
        features = self._feature_repository.load(instrument, timeframe)
        return self._engine.run(
            instrument,
            timeframe,
            historical.candles,
            features,
            strategy,
            config=config,
        )

    def _has_backtest_data(self, instrument: Instrument, timeframe: Timeframe) -> bool:
        raw_path = self._historical_repository.build_path(
            instrument.exchange_segment,
            instrument.security_id,
            timeframe,
        )
        feature_path = self._feature_repository.build_path(instrument, timeframe)
        return raw_path.exists() and feature_path.exists()
