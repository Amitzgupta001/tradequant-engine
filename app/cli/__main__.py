"""CLI entry point."""

import argparse
import json
import sys
import time
from datetime import date

from loguru import logger

from app.brokers.dhan.client import DhanClient
from app.cache.memory import InMemoryCache
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.data.providers.historical_data_provider import HistoricalDataProvider
from app.data.repositories.csv_historical_repository import CSVHistoricalRepository
from app.domain.enums.market import ExchangeSegment, InstrumentType, Timeframe
from app.domain.features import FeatureVector
from app.domain.historical import HistoricalRequest, HistoricalResponse
from app.domain.backtest import BacktestConfig
from app.domain.instrument import Instrument
from app.backtest.engine import BacktestEngine
from app.backtest.report import BacktestReportStore
from app.ml.datasets.builder import FeatureDatasetBuilder
from app.ml.feature_store.repository import CSVFeatureRepository
from app.ml.registry.model_registry import ModelRegistry
from app.ml.trainer.lightgbm_trainer import LightGBMTrainer
from app.services.backtest_service import BacktestService
from app.services.feature_service import FeatureService
from app.services.historical_data_service import HistoricalDataService
from app.services.indicator_service import IndicatorService
from app.data.universe import get_universe, list_universes
from app.services.batch_data_service import BatchDataService
from app.backtest.sweep import BacktestSweep
from app.services.strategy_backtest_service import StrategyBacktestService
from app.services.strategy_selector_service import StrategySelectorService
from app.services.paper_trading_service import PaperTradingService
from app.paper.live_runner import PaperLiveRunner
from app.ml.datasets.strategy_selector_builder import (
    SelectionObjective,
    StrategySelectorBuilderConfig,
)
from app.strategy.ml_strategy import MLStrategy
from app.services.training_service import TrainingService
from app.domain.training import SetupType, TrainingConfig, TrainingTask
from app.ml.datasets.preparation import training_config_for_timeframe
from app.strategy.presets import BEST_5M_BACKTEST, BEST_5M_DAYS, BEST_5M_TIMEFRAME, BEST_5M_TRAINING


def _build_repository() -> CSVHistoricalRepository:
    settings = get_settings()
    return CSVHistoricalRepository(base_path=settings.storage_path)


def _build_service() -> HistoricalDataService:
    settings = get_settings()
    broker = DhanClient(settings.dhan_client_id, settings.dhan_access_token)
    cache = InMemoryCache[HistoricalResponse](default_ttl_seconds=settings.cache_ttl_seconds)
    provider = HistoricalDataProvider(broker=broker, cache=cache)
    repository = _build_repository()
    return HistoricalDataService(provider=provider, repository=repository)


def _build_indicator_service() -> IndicatorService:
    settings = get_settings()
    repository = _build_repository()
    return IndicatorService(
        repository=repository,
        processed_path=settings.storage_path / "processed",
    )


def _build_feature_service() -> FeatureService:
    settings = get_settings()
    repository = _build_repository()
    feature_repository = CSVFeatureRepository(base_path=settings.storage_path / "features")
    builder = FeatureDatasetBuilder(repository=repository)
    return FeatureService(
        repository=repository,
        feature_repository=feature_repository,
        builder=builder,
    )


def _build_training_service() -> TrainingService:
    settings = get_settings()
    repository = _build_repository()
    feature_repository = CSVFeatureRepository(base_path=settings.storage_path / "features")
    registry = ModelRegistry(base_path=settings.storage_path / "models")
    trainer = LightGBMTrainer(registry=registry)
    return TrainingService(
        historical_service=_build_service(),
        feature_service=_build_feature_service(),
        feature_repository=feature_repository,
        trainer=trainer,
    )


def _build_backtest_service() -> BacktestService:
    settings = get_settings()
    repository = _build_repository()
    return BacktestService(
        historical_repository=repository,
        feature_repository=CSVFeatureRepository(base_path=settings.storage_path / "features"),
        model_registry=ModelRegistry(base_path=settings.storage_path / "models"),
        report_store=BacktestReportStore(base_path=settings.storage_path / "backtests"),
        engine=BacktestEngine(),
    )


def _build_paper_trading_service() -> PaperTradingService:
    repository = _build_repository()
    return PaperTradingService(
        repository=repository,
        training_service=_build_training_service(),
        selector_service=StrategySelectorService(repository),
    )


def cmd_download(args: argparse.Namespace) -> int:
    """Download and store historical OHLCV data."""
    request = HistoricalRequest(
        instrument=Instrument(
            security_id=args.security_id,
            exchange_segment=ExchangeSegment(args.exchange),
            instrument_type=InstrumentType(args.instrument_type),
            symbol=args.symbol,
        ),
        from_date=date.fromisoformat(args.from_date),
        to_date=date.fromisoformat(args.to_date),
        timeframe=Timeframe(args.timeframe),
        include_oi=args.include_oi,
    )
    service = _build_service()
    response, path = service.download_and_store(request, overwrite=not args.no_overwrite)
    result = {
        "path": str(path),
        "candle_count": len(response.candles),
        "source": response.source,
    }
    print(json.dumps(result, indent=2))
    return 0


def cmd_indicators(args: argparse.Namespace) -> int:
    """Compute technical indicators from stored raw OHLCV data."""
    instrument = Instrument(
        security_id=args.security_id,
        exchange_segment=ExchangeSegment(args.exchange),
        instrument_type=InstrumentType(args.instrument_type),
        symbol=args.symbol,
    )
    service = _build_indicator_service()
    snapshots, path = service.compute_and_store(
        instrument,
        Timeframe(args.timeframe),
        overwrite=not args.no_overwrite,
    )
    latest = snapshots[-1].model_dump(mode="json") if snapshots else None
    result = {
        "path": str(path),
        "row_count": len(snapshots),
        "latest": latest,
    }
    print(json.dumps(result, indent=2))
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    """Build ML-ready features from stored raw OHLCV data."""
    instrument = Instrument(
        security_id=args.security_id,
        exchange_segment=ExchangeSegment(args.exchange),
        instrument_type=InstrumentType(args.instrument_type),
        symbol=args.symbol,
    )
    service = _build_feature_service()
    features, path = service.build_and_store(
        instrument,
        Timeframe(args.timeframe),
        overwrite=not args.no_overwrite,
    )
    latest = features[-1].model_dump(mode="json") if features else None
    result = {
        "path": str(path),
        "row_count": len(features),
        "feature_columns": list(FeatureVector.model_fields.keys()),
        "latest": latest,
    }
    print(json.dumps(result, indent=2))
    return 0


def _batch_sleep_seconds(args: argparse.Namespace) -> float:
    """Resolve pause duration between batch symbols (CLI overrides env)."""
    if getattr(args, "sleep_seconds", None) is not None:
        return max(0.0, args.sleep_seconds)
    return max(0.0, get_settings().batch_sleep_seconds)


def _apply_train_preset(args: argparse.Namespace) -> None:
    """Apply best validated preset overrides to train args."""
    if args.preset != "best":
        return
    args.timeframe = BEST_5M_TIMEFRAME.value
    args.days = BEST_5M_DAYS
    args.setup_type = BEST_5M_TRAINING.setup_type.value
    args.forward_horizon = BEST_5M_TRAINING.forward_horizon_bars
    args.move_threshold = BEST_5M_TRAINING.move_threshold


def _apply_backtest_preset(args: argparse.Namespace) -> None:
    """Apply best validated preset overrides to backtest args."""
    if args.preset != "best":
        return
    args.timeframe = BEST_5M_TIMEFRAME.value
    preset = BEST_5M_BACKTEST
    args.stop_loss_pct = preset.stop_loss_pct
    args.trailing_stop_pct = preset.trailing_stop_pct
    args.trailing_activation_pct = preset.trailing_activation_pct
    args.max_hold_bars = preset.max_hold_bars
    args.atr_stop_multiplier = preset.atr_stop_multiplier
    args.probability_threshold = preset.probability_threshold
    args.min_bars_between_entries = preset.min_bars_between_entries
    args.max_trades_per_day = preset.max_trades_per_day
    args.cooldown_bars_after_stop = preset.cooldown_bars_after_stop
    args.exit_confirmation_bars = preset.exit_confirmation_bars
    args.min_expected_value = preset.min_expected_value
    args.expected_win_pct = preset.expected_win_pct
    args.expected_loss_pct = preset.expected_loss_pct
    args.use_scaled_targets = preset.use_scaled_targets
    args.target_1_pct = preset.target_1_pct
    args.target_2_pct = preset.target_2_pct
    args.target_3_pct = preset.target_3_pct
    args.target_1_qty_pct = preset.target_1_qty_pct
    args.target_2_qty_pct = preset.target_2_qty_pct
    args.move_stop_to_breakeven_after_t1 = preset.move_stop_to_breakeven_after_t1


def _build_backtest_config(args: argparse.Namespace) -> BacktestConfig:
    """Build BacktestConfig from CLI args."""
    return BacktestConfig(
        initial_capital=args.initial_capital,
        probability_threshold=args.probability_threshold,
        commission_pct=args.commission_pct,
        stop_loss_pct=args.stop_loss_pct,
        trailing_stop_pct=args.trailing_stop_pct,
        trailing_activation_pct=args.trailing_activation_pct,
        max_hold_bars=args.max_hold_bars,
        atr_stop_multiplier=args.atr_stop_multiplier,
        min_bars_between_entries=args.min_bars_between_entries,
        max_trades_per_day=args.max_trades_per_day,
        cooldown_bars_after_stop=args.cooldown_bars_after_stop,
        exit_confirmation_bars=args.exit_confirmation_bars,
        min_expected_value=args.min_expected_value,
        expected_win_pct=args.expected_win_pct,
        expected_loss_pct=args.expected_loss_pct,
        use_scaled_targets=args.use_scaled_targets,
        target_1_pct=args.target_1_pct,
        target_2_pct=args.target_2_pct,
        target_3_pct=args.target_3_pct,
        target_1_qty_pct=args.target_1_qty_pct,
        target_2_qty_pct=args.target_2_qty_pct,
        move_stop_to_breakeven_after_t1=args.move_stop_to_breakeven_after_t1,
    )


def _add_backtest_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_universe: bool = False,
    universe_help: str = "Backtest all symbols using a panel model",
) -> None:
    """Register shared backtest CLI arguments."""
    parser.add_argument("--security-id", default=None, help="Single instrument (omit with --universe)")
    if include_universe:
        parser.add_argument(
            "--universe",
            choices=list_universes(),
            default=None,
            help=universe_help,
        )
    parser.add_argument("--exchange", default="NSE_EQ")
    parser.add_argument("--instrument-type", default="EQUITY")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", default="DAILY")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--preset", choices=["best"], default=None, help="Use validated best strategy preset")
    parser.add_argument("--probability-threshold", type=float, default=None)
    parser.add_argument("--commission-pct", type=float, default=0.0003)
    parser.add_argument("--stop-loss-pct", type=float, default=0.01)
    parser.add_argument("--trailing-stop-pct", type=float, default=0.006)
    parser.add_argument("--trailing-activation-pct", type=float, default=0.008)
    parser.add_argument("--max-hold-bars", type=int, default=20)
    parser.add_argument("--atr-stop-multiplier", type=float, default=2.0)
    parser.add_argument("--min-bars-between-entries", type=int, default=0)
    parser.add_argument("--max-trades-per-day", type=int, default=None)
    parser.add_argument("--cooldown-bars-after-stop", type=int, default=0)
    parser.add_argument("--exit-confirmation-bars", type=int, default=1)
    parser.add_argument("--min-expected-value", type=float, default=None)
    parser.add_argument("--expected-win-pct", type=float, default=0.003)
    parser.add_argument("--expected-loss-pct", type=float, default=0.007)
    parser.add_argument("--use-scaled-targets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-1-pct", type=float, default=0.005)
    parser.add_argument("--target-2-pct", type=float, default=0.010)
    parser.add_argument("--target-3-pct", type=float, default=0.015)
    parser.add_argument("--target-1-qty-pct", type=float, default=0.33)
    parser.add_argument("--target-2-qty-pct", type=float, default=0.33)
    parser.add_argument(
        "--move-stop-to-breakeven-after-t1",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--per-symbol",
        action="store_true",
        help="Save individual backtest reports for each symbol in a panel run",
    )


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run ML strategy backtest on stored data."""
    _apply_backtest_preset(args)
    timeframe = Timeframe(args.timeframe)
    config = _build_backtest_config(args)
    service = _build_backtest_service()

    if args.universe:
        universe = get_universe(args.universe)
        panel_result = service.run_panel(
            universe,
            timeframe,
            config=config,
            save_per_symbol=args.per_symbol,
        )
        output = {
            "universe": args.universe,
            "summary_path": panel_result.summary_path,
            "symbols_backtested": panel_result.symbols_backtested,
            "symbols_skipped": panel_result.symbols_skipped,
            "total_trades": panel_result.total_trades,
            "metrics": {
                "mean_return_pct": panel_result.mean_return_pct,
                "median_return_pct": panel_result.median_return_pct,
                "pooled_win_rate_pct": panel_result.pooled_win_rate_pct,
                "pooled_profit_factor": panel_result.pooled_profit_factor,
                "symbols_positive_return": panel_result.symbols_positive_return,
            },
        }
        print(json.dumps(output, indent=2))
        return 0

    instrument = Instrument(
        security_id=args.security_id,
        exchange_segment=ExchangeSegment(args.exchange),
        instrument_type=InstrumentType(args.instrument_type),
        symbol=args.symbol,
    )
    result, summary_path, equity_path = service.run(
        instrument,
        timeframe,
        config=config,
    )
    output = {
        "summary_path": str(summary_path) if summary_path else None,
        "equity_path": str(equity_path) if equity_path else None,
        "metrics": result.metrics.model_dump(),
        "trade_count": len(result.trades),
    }
    print(json.dumps(output, indent=2))
    return 0


def cmd_backtest_sweep(args: argparse.Namespace) -> int:
    """Grid-search backtest parameters for a single instrument."""
    _apply_backtest_preset(args)
    timeframe = Timeframe(args.timeframe)
    base_config = _build_backtest_config(args)
    instrument = Instrument(
        security_id=args.security_id,
        exchange_segment=ExchangeSegment(args.exchange),
        instrument_type=InstrumentType(args.instrument_type),
        symbol=args.symbol,
    )
    service = _build_backtest_service()
    historical = service._historical_repository.load(instrument, timeframe)
    features = service._feature_repository.load(instrument, timeframe)

    def strategy_factory(config: BacktestConfig) -> MLStrategy:
        return service._build_strategy(
            instrument.exchange_segment.value,
            instrument.security_id,
            timeframe.value,
            config=config,
        )

    grid = {
        "probability_threshold": [0.45, 0.50, 0.55],
        "stop_loss_pct": [0.007, 0.010],
        "max_hold_bars": [15, 20],
    }
    results = BacktestSweep().run(
        instrument,
        timeframe,
        historical.candles,
        features,
        strategy_factory(base_config),
        base_config,
        grid,
        strategy_factory=strategy_factory,
    )
    settings = get_settings()
    output_path = settings.storage_path / "backtests" / "sweeps" / f"{args.security_id}_{timeframe.value.lower()}.json"
    BacktestSweep.save(results, output_path)
    print(json.dumps({"sweep_path": str(output_path), "top_5": [item.model_dump() for item in results[:5]]}, indent=2))
    return 0


def cmd_backtest_strategy(args: argparse.Namespace) -> int:
    """Run Phase 3 per-strategy model backtest."""
    _apply_backtest_preset(args)
    timeframe = Timeframe(args.timeframe)
    config = _build_backtest_config(args).model_copy(update={"require_strategy_signal": True})
    instrument = Instrument(
        security_id=args.security_id,
        exchange_segment=ExchangeSegment(args.exchange),
        instrument_type=InstrumentType(args.instrument_type),
        symbol=args.symbol,
    )
    service = StrategyBacktestService(_build_repository())
    result, summary_path = service.run(
        args.strategy_id,
        instrument,
        timeframe,
        config=config,
        retrain=args.retrain,
    )
    print(
        json.dumps(
            {
                "strategy_id": args.strategy_id,
                "summary_path": str(summary_path) if summary_path else None,
                "metrics": result.metrics.model_dump(),
                "trade_count": len(result.trades),
            },
            indent=2,
        )
    )
    return 0


def cmd_train_strategy_selector(args: argparse.Namespace) -> int:
    """Train meta-model on walk-forward backtests across all strategies."""
    _apply_backtest_preset(args)
    timeframe = Timeframe(args.timeframe)
    config = _build_backtest_config(args).model_copy(update={"require_strategy_signal": True})
    builder_config = StrategySelectorBuilderConfig(
        train_window=args.train_window,
        step_size=args.step_size,
        min_trades=args.min_trades,
        objective=SelectionObjective(args.objective),
    )
    service = StrategySelectorService(_build_repository())

    if args.universe:
        universe = get_universe(args.universe)
        metadata = service.train_panel(
            universe,
            timeframe,
            backtest_config=config,
            builder_config=builder_config,
            rebuild_dataset=args.rebuild_dataset,
            train_strategy_models=not args.skip_strategy_training,
        )
        meta = service.simulate_meta_backtest(
            universe=universe,
            timeframe=timeframe,
            selector_version=metadata.version,
        )
        output = {
            "universe": args.universe,
            "selector_version": metadata.version,
            "constituent_count": metadata.constituent_count,
            "strategy_ids": metadata.strategy_ids,
            "objective": metadata.objective.value,
            "metrics": metadata.metrics.model_dump(),
            "backtest_tuning": metadata.backtest_metrics.model_dump() if metadata.backtest_metrics else None,
            "min_confidence": metadata.min_confidence,
            "meta_backtest": {
                "windows": meta.windows,
                "selected_windows": meta.selected_windows,
                "cumulative_return_pct": meta.cumulative_return_pct,
                "selector_hit_rate": meta.selector_hit_rate,
                "top_strategy_counts": meta.top_strategy_counts,
            },
        }
    else:
        instrument = Instrument(
            security_id=args.security_id,
            exchange_segment=ExchangeSegment(args.exchange),
            instrument_type=InstrumentType(args.instrument_type),
            symbol=args.symbol,
        )
        metadata = service.train(
            instrument,
            timeframe,
            backtest_config=config,
            builder_config=builder_config,
            rebuild_dataset=args.rebuild_dataset,
            train_strategy_models=not args.skip_strategy_training,
        )
        meta = service.simulate_meta_backtest(
            instrument,
            timeframe,
            selector_version=metadata.version,
        )
        output = {
            "security_id": args.security_id,
            "selector_version": metadata.version,
            "strategy_ids": metadata.strategy_ids,
            "objective": metadata.objective.value,
            "metrics": metadata.metrics.model_dump(),
            "backtest_tuning": metadata.backtest_metrics.model_dump() if metadata.backtest_metrics else None,
            "min_confidence": metadata.min_confidence,
            "meta_backtest": {
                "windows": meta.windows,
                "selected_windows": meta.selected_windows,
                "cumulative_return_pct": meta.cumulative_return_pct,
                "selector_hit_rate": meta.selector_hit_rate,
                "top_strategy_counts": meta.top_strategy_counts,
            },
        }

    print(json.dumps(output, indent=2))
    return 0


def cmd_recommend_strategy(args: argparse.Namespace) -> int:
    """Recommend best strategy for current market conditions."""
    timeframe = Timeframe(args.timeframe)
    instrument = Instrument(
        security_id=args.security_id,
        exchange_segment=ExchangeSegment(args.exchange),
        instrument_type=InstrumentType(args.instrument_type),
        symbol=args.symbol,
    )
    service = StrategySelectorService(_build_repository())
    recommendations = service.recommend(
        instrument,
        timeframe,
        selector_version=args.selector_version,
        universe_id=args.universe,
        top_n=args.top_n,
    )
    print(
        json.dumps(
            {
                "security_id": args.security_id,
                "universe": args.universe,
                "timeframe": timeframe.value,
                "recommendations": [item.model_dump(mode="json") for item in recommendations],
            },
            indent=2,
        )
    )
    return 0


def cmd_backtest_auto(args: argparse.Namespace) -> int:
    """Backtest using the selector-recommended strategy."""
    _apply_backtest_preset(args)
    timeframe = Timeframe(args.timeframe)
    config = _build_backtest_config(args).model_copy(update={"require_strategy_signal": True})
    instrument = Instrument(
        security_id=args.security_id,
        exchange_segment=ExchangeSegment(args.exchange),
        instrument_type=InstrumentType(args.instrument_type),
        symbol=args.symbol,
    )
    service = StrategySelectorService(_build_repository())
    recommendations = service.recommend(
        instrument,
        timeframe,
        selector_version=args.selector_version,
        universe_id=args.universe,
        top_n=3,
    )

    if args.mode == "single":
        strategy_id, result, recommendations = service.backtest_recommended(
            instrument,
            timeframe,
            backtest_config=config,
            selector_version=args.selector_version,
            universe_id=args.universe,
            retrain=args.retrain,
        )
        meta = service.simulate_meta_backtest(
            instrument,
            timeframe,
            selector_version=args.selector_version,
            universe_id=args.universe,
        )
        print(
            json.dumps(
                {
                    "mode": "single",
                    "recommended_strategy_id": strategy_id,
                    "universe": args.universe,
                    "recommendations": [item.model_dump(mode="json") for item in recommendations],
                    "backtest_metrics": result.metrics.model_dump(),
                    "trade_count": len(result.trades),
                    "meta_backtest": {
                        "cumulative_return_pct": meta.cumulative_return_pct,
                        "selector_hit_rate": meta.selector_hit_rate,
                        "selected_windows": meta.selected_windows,
                        "top_strategy_counts": meta.top_strategy_counts,
                    },
                },
                indent=2,
            )
        )
        return 0

    rolling = service.backtest_rolling(
        instrument,
        timeframe,
        backtest_config=config,
        selector_version=args.selector_version,
        universe_id=args.universe,
    )
    meta = service.simulate_meta_backtest(
        instrument,
        timeframe,
        selector_version=args.selector_version,
        universe_id=args.universe,
    )
    print(
        json.dumps(
            {
                "mode": "rolling",
                "universe": args.universe,
                "recommendations": [item.model_dump(mode="json") for item in recommendations],
                "rolling_backtest": {
                    "windows": rolling.windows,
                    "traded_windows": rolling.traded_windows,
                    "skipped_low_confidence": rolling.skipped_low_confidence,
                    "compounded_return_pct": rolling.compounded_return_pct,
                    "total_trades": rolling.total_trades,
                    "strategy_counts": rolling.strategy_counts,
                    "initial_capital": rolling.initial_capital,
                    "final_equity": rolling.final_equity,
                },
                "meta_backtest_simulated": {
                    "cumulative_return_pct": meta.cumulative_return_pct,
                    "selector_hit_rate": meta.selector_hit_rate,
                    "selected_windows": meta.selected_windows,
                    "top_strategy_counts": meta.top_strategy_counts,
                },
            },
            indent=2,
        )
    )
    return 0


def cmd_paper_trade(args: argparse.Namespace) -> int:
    """Run paper trading session for live forward testing."""
    service = _build_paper_trading_service()
    timeframe = Timeframe(args.timeframe)
    security_ids = None
    if args.security_ids:
        security_ids = [item.strip() for item in args.security_ids.split(",") if item.strip()]

    if args.start:
        session = service.start_session(
            universe_id=args.universe,
            timeframe=timeframe,
            initial_capital=args.capital,
            selector_universe_id=args.selector_universe or args.universe,
            security_ids=security_ids,
            session_id=args.session_id,
        )
        print(json.dumps(session.model_dump(mode="json"), indent=2))
        return 0

    if args.stop:
        session = service.stop_session(args.session_id)
        print(json.dumps(session.model_dump(mode="json"), indent=2))
        return 0

    if args.tick or args.run:
        if args.run and service.get_active_session() is None:
            session = service.start_session(
                universe_id=args.universe,
                timeframe=timeframe,
                initial_capital=args.capital,
                selector_universe_id=args.selector_universe or args.universe,
                security_ids=security_ids,
                session_id=args.session_id,
            )
            logger.info("Started paper session {}", session.session_id)

        if args.run and args.mode == "live":
            runner = PaperLiveRunner(service)
            runner.run(
                session_id=args.session_id,
                poll_seconds=min(args.poll_seconds, 5.0),
                force=args.force,
            )
            return 0

        while True:
            snapshot = service.tick(session_id=args.session_id, force=args.force)
            print(
                json.dumps(
                    {
                        "market_open": snapshot.market_open,
                        "session_id": snapshot.session.session_id,
                        "total_realized_pnl": snapshot.total_realized_pnl,
                        "total_trades": snapshot.total_trades,
                        "open_positions": len(snapshot.open_positions),
                        "symbols_processed": snapshot.symbols_processed,
                        "last_error": snapshot.session.last_error,
                    },
                    indent=2,
                )
            )
            if not args.run:
                break
            time.sleep(args.poll_seconds)

        return 0

    snapshot = service.dashboard(session_id=args.session_id)
    print(json.dumps(snapshot.model_dump(mode="json"), indent=2))
    return 0


def cmd_batch_download(args: argparse.Namespace) -> int:
    """Download OHLCV and build features for all symbols in a universe."""
    _apply_train_preset(args)
    universe = get_universe(args.universe)
    timeframe = Timeframe(args.timeframe)
    service = BatchDataService(_build_training_service())
    results = service.download_universe(
        universe,
        timeframe,
        years=args.years,
        days=args.days,
        skip_existing=args.skip_existing,
        sleep_seconds=_batch_sleep_seconds(args),
    )
    ok = [result for result in results if result.status == "ok"]
    failed = [result for result in results if result.status != "ok"]
    output = {
        "universe": universe.id,
        "timeframe": timeframe.value,
        "symbols_total": len(results),
        "symbols_ok": len(ok),
        "symbols_failed": len(failed),
        "total_candles": sum(result.candle_count for result in ok),
        "total_features": sum(result.feature_count for result in ok),
        "results": [result.to_dict() for result in results],
    }
    print(json.dumps(output, indent=2))
    return 1 if failed and not ok else 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train LightGBM model on engineered features."""
    _apply_train_preset(args)
    timeframe = Timeframe(args.timeframe)
    config_kwargs: dict = {
        "task": TrainingTask(args.task),
        "setup_type": SetupType(args.setup_type),
        "test_size": args.test_size,
        "n_estimators": args.n_estimators,
    }
    if args.forward_horizon is not None:
        config_kwargs["forward_horizon_bars"] = args.forward_horizon
    if args.move_threshold is not None:
        config_kwargs["move_threshold"] = args.move_threshold
    if args.preset == "best":
        config = training_config_for_timeframe(timeframe, BEST_5M_TRAINING)
    else:
        config = training_config_for_timeframe(timeframe, TrainingConfig(**config_kwargs))
    service = _build_training_service()

    if args.universe:
        universe = get_universe(args.universe)
        if args.skip_download:
            result = service.train_panel_from_features(universe, timeframe, config=config)
        else:
            result = service.train_panel_with_history(
                universe,
                timeframe,
                years=args.years,
                days=args.days,
                config=config,
                skip_existing=args.skip_existing,
                sleep_seconds=_batch_sleep_seconds(args),
            )
    else:
        instrument = Instrument(
            security_id=args.security_id,
            exchange_segment=ExchangeSegment(args.exchange),
            instrument_type=InstrumentType(args.instrument_type),
            symbol=args.symbol,
        )
        if args.skip_download:
            result = service.train_from_features(
                instrument,
                timeframe,
                config=config,
            )
        else:
            result = service.train_with_history(
                instrument,
                timeframe,
                years=args.years,
                days=args.days,
                config=config,
            )

    output = {
        "model_path": result.model_path,
        "metadata_path": result.metadata_path,
        "metrics": result.metrics.model_dump(),
        "feature_columns": result.feature_columns,
    }
    if args.universe:
        output["universe"] = args.universe
    print(json.dumps(output, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(prog="tradequant", description="TradeQuant CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download historical OHLCV data")
    download.add_argument("--security-id", required=True)
    download.add_argument("--exchange", default="NSE_EQ")
    download.add_argument("--instrument-type", default="EQUITY")
    download.add_argument("--symbol", default=None)
    download.add_argument("--from-date", required=True)
    download.add_argument("--to-date", required=True)
    download.add_argument("--timeframe", default="DAILY")
    download.add_argument("--include-oi", action="store_true")
    download.add_argument("--no-overwrite", action="store_true")
    download.set_defaults(func=cmd_download)

    indicators = subparsers.add_parser("indicators", help="Compute technical indicators")
    indicators.add_argument("--security-id", required=True)
    indicators.add_argument("--exchange", default="NSE_EQ")
    indicators.add_argument("--instrument-type", default="EQUITY")
    indicators.add_argument("--symbol", default=None)
    indicators.add_argument("--timeframe", default="DAILY")
    indicators.add_argument("--no-overwrite", action="store_true")
    indicators.set_defaults(func=cmd_indicators)

    features_cmd = subparsers.add_parser("features", help="Build ML-ready features")
    features_cmd.add_argument("--security-id", required=True)
    features_cmd.add_argument("--exchange", default="NSE_EQ")
    features_cmd.add_argument("--instrument-type", default="EQUITY")
    features_cmd.add_argument("--symbol", default=None)
    features_cmd.add_argument("--timeframe", default="DAILY")
    features_cmd.add_argument("--no-overwrite", action="store_true")
    features_cmd.set_defaults(func=cmd_features)

    backtest = subparsers.add_parser("backtest", help="Run ML strategy backtest (Phase 5)")
    _add_backtest_arguments(backtest, include_universe=True)
    backtest.set_defaults(func=cmd_backtest)

    sweep = subparsers.add_parser("backtest-sweep", help="Grid-search backtest parameters")
    _add_backtest_arguments(sweep)
    sweep.set_defaults(func=cmd_backtest_sweep)

    strategy_bt = subparsers.add_parser("backtest-strategy", help="Backtest a Phase 3 strategy model")
    _add_backtest_arguments(strategy_bt)
    strategy_bt.add_argument("--strategy-id", required=True)
    strategy_bt.add_argument("--retrain", action="store_true")
    strategy_bt.set_defaults(func=cmd_backtest_strategy)

    selector_train = subparsers.add_parser(
        "train-strategy-selector",
        help="Train meta-model on walk-forward backtests across all strategies",
    )
    _add_backtest_arguments(
        selector_train,
        include_universe=True,
        universe_help="Train pooled selector on this universe (omit --security-id)",
    )
    selector_train.add_argument("--train-window", type=int, default=400)
    selector_train.add_argument("--step-size", type=int, default=50)
    selector_train.add_argument("--min-trades", type=int, default=2)
    selector_train.add_argument(
        "--objective",
        choices=[item.value for item in SelectionObjective],
        default=SelectionObjective.PROFIT_FACTOR.value,
    )
    selector_train.add_argument("--rebuild-dataset", action="store_true")
    selector_train.add_argument(
        "--skip-strategy-training",
        action="store_true",
        help="Do not auto-train missing per-strategy models",
    )
    selector_train.set_defaults(func=cmd_train_strategy_selector)

    recommend = subparsers.add_parser(
        "recommend-strategy",
        help="Recommend best strategy using backtest-tuned selector model",
    )
    recommend.add_argument("--security-id", required=True)
    recommend.add_argument("--exchange", default="NSE_EQ")
    recommend.add_argument("--instrument-type", default="EQUITY")
    recommend.add_argument("--symbol", default=None)
    recommend.add_argument("--timeframe", default="MIN_5")
    recommend.add_argument(
        "--universe",
        choices=list_universes(),
        default=None,
        help="Use pooled selector model trained on this universe",
    )
    recommend.add_argument("--selector-version", type=int, default=None)
    recommend.add_argument("--top-n", type=int, default=5)
    recommend.set_defaults(func=cmd_recommend_strategy)

    auto_bt = subparsers.add_parser(
        "backtest-auto",
        help="Run backtest with selector-recommended strategy",
    )
    _add_backtest_arguments(
        auto_bt,
        include_universe=True,
        universe_help="Use pooled selector model trained on this universe",
    )
    auto_bt.add_argument("--selector-version", type=int, default=None)
    auto_bt.add_argument(
        "--mode",
        choices=["rolling", "single"],
        default="rolling",
        help="rolling switches strategy per window (default); single uses latest pick on full history",
    )
    auto_bt.add_argument("--retrain", action="store_true")
    auto_bt.set_defaults(func=cmd_backtest_auto)

    paper = subparsers.add_parser(
        "paper-trade",
        help="Paper trade live market with strategy selector (forward testing)",
    )
    paper.add_argument("--universe", choices=list_universes(), default="nifty50")
    paper.add_argument(
        "--selector-universe",
        choices=list_universes(),
        default=None,
        help="Pooled selector universe (defaults to --universe)",
    )
    paper.add_argument("--timeframe", default="MIN_5")
    paper.add_argument("--capital", type=float, default=1_000_000.0)
    paper.add_argument(
        "--security-ids",
        default=None,
        help="Comma-separated security ids (default: full universe)",
    )
    paper.add_argument("--session-id", default=None)
    paper.add_argument("--start", action="store_true", help="Create a new paper session")
    paper.add_argument("--stop", action="store_true", help="Stop the active paper session")
    paper.add_argument("--tick", action="store_true", help="Run one paper trading tick")
    paper.add_argument(
        "--run",
        action="store_true",
        help="Poll market data and process paper trades during market hours",
    )
    paper.add_argument(
        "--mode",
        choices=["live", "poll"],
        default="live",
        help="live=WebSocket LTP + REST on 5m bar close (default); poll=REST only loop",
    )
    paper.add_argument("--poll-seconds", type=int, default=60)
    paper.add_argument(
        "--force",
        action="store_true",
        help="Run ticks even when market is closed",
    )
    paper.set_defaults(func=cmd_paper_trade)

    batch_dl = subparsers.add_parser(
        "batch-download",
        help="Download OHLCV and build features for a stock universe",
    )
    batch_dl.add_argument(
        "--universe",
        required=True,
        choices=list_universes(),
        help="Stock universe to download",
    )
    batch_dl.add_argument("--timeframe", default="DAILY")
    batch_dl.add_argument("--years", type=int, default=5)
    batch_dl.add_argument("--days", type=int, default=90, help="Lookback days for intraday timeframes")
    batch_dl.add_argument("--preset", choices=["best"], default=None, help="Use validated best strategy preset")
    batch_dl.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip symbols that already have feature files (resume batch runs)",
    )
    batch_dl.add_argument(
        "--sleep-seconds",
        type=float,
        default=None,
        help="Pause between symbols in seconds (default: BATCH_SLEEP_SECONDS env, 3s)",
    )
    batch_dl.set_defaults(func=cmd_batch_download)

    train = subparsers.add_parser("train", help="Train LightGBM model (Phase 4)")
    train.add_argument("--security-id", default=None, help="Single instrument (omit when using --universe)")
    train.add_argument(
        "--universe",
        choices=list_universes(),
        default=None,
        help="Train pooled model across all symbols in a universe",
    )
    train.add_argument("--exchange", default="NSE_EQ")
    train.add_argument("--instrument-type", default="EQUITY")
    train.add_argument("--symbol", default=None)
    train.add_argument("--timeframe", default="DAILY")
    train.add_argument("--years", type=int, default=5)
    train.add_argument("--days", type=int, default=90, help="Lookback days for intraday timeframes")
    train.add_argument("--setup-type", default="long", choices=["long", "short"])
    train.add_argument("--forward-horizon", type=int, default=None, help="Forward horizon in bars")
    train.add_argument("--move-threshold", type=float, default=None)
    train.add_argument("--preset", choices=["best"], default=None, help="Use validated best strategy preset")
    train.add_argument("--task", default="classification", choices=["classification", "regression"])
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--n-estimators", type=int, default=200)
    train.add_argument("--skip-download", action="store_true")
    train.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip symbols that already have feature files during batch download",
    )
    train.add_argument(
        "--sleep-seconds",
        type=float,
        default=None,
        help="Pause between symbols during batch download (default: BATCH_SLEEP_SECONDS env, 3s)",
    )
    train.set_defaults(func=cmd_train)

    return parser


def _validate_backtest_args(args: argparse.Namespace) -> None:
    """Ensure backtest command has security-id or universe."""
    if not args.universe and not args.security_id:
        msg = "backtest requires --security-id or --universe"
        raise SystemExit(msg)
    if args.universe and args.security_id:
        msg = "backtest accepts --security-id or --universe, not both"
        raise SystemExit(msg)


def _validate_train_args(args: argparse.Namespace) -> None:
    """Ensure train command has security-id or universe."""
    if not args.universe and not args.security_id:
        msg = "train requires --security-id or --universe"
        raise SystemExit(msg)
    if args.universe and args.security_id:
        msg = "train accepts --security-id or --universe, not both"
        raise SystemExit(msg)


def _validate_selector_train_args(args: argparse.Namespace) -> None:
    """Ensure selector train command has security-id or universe."""
    if not args.universe and not args.security_id:
        msg = "train-strategy-selector requires --security-id or --universe"
        raise SystemExit(msg)
    if args.universe and args.security_id:
        msg = "train-strategy-selector accepts --security-id or --universe, not both"
        raise SystemExit(msg)


def _validate_selector_infer_args(args: argparse.Namespace) -> None:
    """Ensure recommend/backtest-auto have security-id."""
    if not args.security_id:
        msg = f"{args.command} requires --security-id"
        raise SystemExit(msg)


def main(argv: list[str] | None = None) -> int:
    """CLI main entry point."""
    settings = get_settings()
    setup_logging(settings.log_level)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        _validate_train_args(args)
    if args.command == "backtest":
        _validate_backtest_args(args)
    if args.command in ("backtest-sweep", "backtest-strategy") and not args.security_id:
        raise SystemExit(f"{args.command} requires --security-id")
    if args.command == "train-strategy-selector":
        _validate_selector_train_args(args)
    if args.command in ("recommend-strategy", "backtest-auto"):
        _validate_selector_infer_args(args)
    try:
        return args.func(args)
    except NotImplementedError as exc:
        logger.error(str(exc))
        return 1
    except Exception as exc:
        logger.exception("CLI command failed: {}", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
