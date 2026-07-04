"""Registry for discovering and retrieving trading strategies."""

from app.strategies.base import TradingStrategy

_REGISTRY: dict[str, TradingStrategy] = {}


def register_strategy(strategy: TradingStrategy) -> None:
    """Register a strategy instance by its strategy_id."""
    _REGISTRY[strategy.strategy_id] = strategy


def get_strategy(strategy_id: str) -> TradingStrategy:
    """Return a registered strategy by id."""
    if strategy_id not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        msg = f"Unknown strategy '{strategy_id}'. Available: {available}"
        raise KeyError(msg)
    return _REGISTRY[strategy_id]


def list_strategies() -> list[TradingStrategy]:
    """Return all registered strategies sorted by id."""
    return [_REGISTRY[key] for key in sorted(_REGISTRY)]


def list_strategy_ids() -> list[str]:
    """Return registered strategy ids."""
    return sorted(_REGISTRY)
