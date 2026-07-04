"""Orchestrate Phase 3 strategy model backtests."""

from pathlib import Path

import pandas as pd
from loguru import logger

from app.backtest.report import BacktestReportStore
from app.backtest.strategy_engine import StrategyBacktestEngine
from app.core.config import Settings, get_settings
from app.data.repositories.base import HistoricalRepository
from app.domain.backtest import BacktestConfig, BacktestResult
from app.domain.enums.market import Timeframe
from app.domain.instrument import Instrument
from app.ml.datasets.strategy_builder import StrategyDatasetBuilder
from app.ml.labels.base import LabelConfig
from app.ml.registry.strategy_registry import StrategyModelRegistry
from app.services.strategy_training_service import StrategyTrainingService
from app.strategy.presets import BEST_5M_BACKTEST, REWARD_PCT_5M, RISK_PCT_5M
from app.strategy.strategy_model_bridge import StrategyModelBridge


class StrategyBacktestService:
    """Build, train, and backtest per-strategy models."""

    def __init__(
        self,
        repository: HistoricalRepository,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        storage_path = Path(self._settings.storage_path)
        self._dataset_builder = StrategyDatasetBuilder(repository, storage_path)
        self._training_service = StrategyTrainingService(repository, self._settings)
        self._model_registry = StrategyModelRegistry(storage_path)
        self._engine = StrategyBacktestEngine()
        self._report_store = BacktestReportStore(storage_path / "backtests")

    def run(
        self,
        strategy_id: str,
        instrument: Instrument,
        timeframe: Timeframe,
        config: BacktestConfig | None = None,
        label_config: LabelConfig | None = None,
        retrain: bool = False,
        model_version: int | None = None,
    ) -> tuple[BacktestResult, Path | None]:
        """Backtest a trained strategy model on stored data."""
        config = config or BEST_5M_BACKTEST.model_copy(update={"require_strategy_signal": True})
        label_config = label_config or LabelConfig(
            forward_horizon_bars=20,
            regression_threshold=REWARD_PCT_5M,
            take_profit_pct=REWARD_PCT_5M,
            stop_loss_pct=RISK_PCT_5M,
        )

        frame, metadata = self._dataset_builder.build(
            strategy_id,
            instrument,
            timeframe,
            label_config,
        )
        if retrain or self._model_registry.latest_version(strategy_id) is None:
            logger.info("Training strategy model before backtest: {}", strategy_id)
            self._training_service.train_strategy(
                strategy_id,
                instrument,
                timeframe,
                label_config=label_config,
            )

        signal_lookup = {}
        for _, row in frame.iterrows():
            timestamp = row["timestamp"]
            if isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()
            signal_lookup[timestamp] = int(row.get("strategy_signal", 0) or 0)
        bridge = StrategyModelBridge.load(
            self._model_registry,
            strategy_id,
            signal_lookup,
            version=model_version,
            config=config,
        )
        result = self._engine.run(instrument, timeframe, frame, bridge, config=config)
        summary_path, _ = self._report_store.save(
            instrument,
            timeframe,
            result,
            strategy_id=strategy_id,
        )
        return result, summary_path
