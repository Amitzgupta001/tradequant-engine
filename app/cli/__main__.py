"""CLI entry point."""

import argparse
import json
import sys
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
    args.probability_threshold = None


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run ML strategy backtest on stored data."""
    _apply_backtest_preset(args)
    timeframe = Timeframe(args.timeframe)
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        probability_threshold=args.probability_threshold,
        commission_pct=args.commission_pct,
        stop_loss_pct=args.stop_loss_pct,
        trailing_stop_pct=args.trailing_stop_pct,
        trailing_activation_pct=args.trailing_activation_pct,
        max_hold_bars=args.max_hold_bars,
    )
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
    backtest.add_argument("--security-id", default=None, help="Single instrument (omit with --universe)")
    backtest.add_argument(
        "--universe",
        choices=list_universes(),
        default=None,
        help="Backtest all symbols using a panel model",
    )
    backtest.add_argument("--exchange", default="NSE_EQ")
    backtest.add_argument("--instrument-type", default="EQUITY")
    backtest.add_argument("--symbol", default=None)
    backtest.add_argument("--timeframe", default="DAILY")
    backtest.add_argument("--initial-capital", type=float, default=100_000.0)
    backtest.add_argument("--preset", choices=["best"], default=None, help="Use validated best strategy preset")
    backtest.add_argument("--probability-threshold", type=float, default=None)
    backtest.add_argument("--commission-pct", type=float, default=0.0003)
    backtest.add_argument("--stop-loss-pct", type=float, default=0.01)
    backtest.add_argument("--trailing-stop-pct", type=float, default=0.006)
    backtest.add_argument("--trailing-activation-pct", type=float, default=0.008)
    backtest.add_argument("--max-hold-bars", type=int, default=20)
    backtest.add_argument(
        "--per-symbol",
        action="store_true",
        help="Save individual backtest reports for each symbol in a panel run",
    )
    backtest.set_defaults(func=cmd_backtest)

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
