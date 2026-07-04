"""FastAPI v1 dependencies."""

from functools import lru_cache

from app.brokers.dhan.client import DhanClient
from app.cache.memory import InMemoryCache
from app.core.config import Settings, get_settings
from app.data.providers.historical_data_provider import HistoricalDataProvider
from app.data.repositories.csv_historical_repository import CSVHistoricalRepository
from app.domain.historical import HistoricalResponse
from app.backtest.engine import BacktestEngine
from app.backtest.report import BacktestReportStore
from app.ml.feature_store.repository import CSVFeatureRepository
from app.ml.registry.model_registry import ModelRegistry
from app.ml.trainer.lightgbm_trainer import LightGBMTrainer
from app.services.backtest_service import BacktestService
from app.services.feature_service import FeatureService
from app.services.historical_data_service import HistoricalDataService
from app.services.indicator_service import IndicatorService
from app.services.paper_trading_service import PaperTradingService
from app.services.strategy_selector_service import StrategySelectorService
from app.services.training_service import TrainingService
from app.ml.datasets.builder import FeatureDatasetBuilder


@lru_cache
def get_dhan_client() -> DhanClient:
    """Return a cached Dhan client instance."""
    settings = get_settings()
    return DhanClient(
        client_id=settings.dhan_client_id,
        access_token=settings.dhan_access_token,
    )


@lru_cache
def get_historical_cache() -> InMemoryCache[HistoricalResponse]:
    """Return a cached in-memory historical data cache."""
    settings = get_settings()
    return InMemoryCache(default_ttl_seconds=settings.cache_ttl_seconds)


@lru_cache
def get_historical_repository() -> CSVHistoricalRepository:
    """Return a cached historical repository instance."""
    settings = get_settings()
    return CSVHistoricalRepository(base_path=settings.storage_path)


@lru_cache
def get_historical_data_service() -> HistoricalDataService:
    """Return a cached historical data service instance."""
    broker = get_dhan_client()
    cache = get_historical_cache()
    provider = HistoricalDataProvider(broker=broker, cache=cache)
    repository = get_historical_repository()
    return HistoricalDataService(provider=provider, repository=repository)


@lru_cache
def get_indicator_service() -> IndicatorService:
    """Return a cached indicator service instance."""
    settings = get_settings()
    repository = get_historical_repository()
    processed_path = settings.storage_path / "processed"
    return IndicatorService(repository=repository, processed_path=processed_path)


@lru_cache
def get_feature_repository() -> CSVFeatureRepository:
    """Return a cached feature repository instance."""
    settings = get_settings()
    return CSVFeatureRepository(base_path=settings.storage_path / "features")


@lru_cache
def get_feature_service() -> FeatureService:
    """Return a cached feature service instance."""
    repository = get_historical_repository()
    builder = FeatureDatasetBuilder(repository=repository)
    return FeatureService(
        repository=repository,
        feature_repository=get_feature_repository(),
        builder=builder,
    )


@lru_cache
def get_model_registry() -> ModelRegistry:
    """Return a cached model registry instance."""
    settings = get_settings()
    return ModelRegistry(base_path=settings.storage_path / "models")


@lru_cache
def get_training_service() -> TrainingService:
    """Return a cached training service instance."""
    registry = get_model_registry()
    trainer = LightGBMTrainer(registry=registry)
    return TrainingService(
        historical_service=get_historical_data_service(),
        feature_service=get_feature_service(),
        feature_repository=get_feature_repository(),
        trainer=trainer,
    )


@lru_cache
def get_backtest_report_store() -> BacktestReportStore:
    """Return a cached backtest report store."""
    settings = get_settings()
    return BacktestReportStore(base_path=settings.storage_path / "backtests")


@lru_cache
def get_backtest_service() -> BacktestService:
    """Return a cached backtest service instance."""
    return BacktestService(
        historical_repository=get_historical_repository(),
        feature_repository=get_feature_repository(),
        model_registry=get_model_registry(),
        report_store=get_backtest_report_store(),
        engine=BacktestEngine(),
    )


@lru_cache
def get_strategy_selector_service() -> StrategySelectorService:
    """Return a cached strategy selector service instance."""
    return StrategySelectorService(get_historical_repository())


@lru_cache
def get_paper_trading_service() -> PaperTradingService:
    """Return a cached paper trading service instance."""
    return PaperTradingService(
        repository=get_historical_repository(),
        training_service=get_training_service(),
        selector_service=get_strategy_selector_service(),
    )


def get_app_settings() -> Settings:
    """Return application settings for dependency injection."""
    return get_settings()
