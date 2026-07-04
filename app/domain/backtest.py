"""Backtesting domain models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SignalAction(str, Enum):
    """Trading signal actions."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class BacktestConfig(BaseModel):
    """Backtest simulation parameters."""

    initial_capital: float = Field(default=100_000.0, gt=0)
    commission_pct: float = Field(default=0.0003, ge=0)
    probability_threshold: float | None = None
    position_size_pct: float = Field(default=0.95, gt=0.0, le=1.0)
    annualization_factor: int = Field(default=252, ge=1)
    stop_loss_pct: float = Field(default=0.01, ge=0.0, le=0.05)
    trailing_stop_pct: float = Field(default=0.006, ge=0.0, le=0.05)
    trailing_activation_pct: float = Field(default=0.008, ge=0.0, le=0.05)
    atr_stop_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)
    max_hold_bars: int | None = Field(default=20, ge=1)


class BacktestTrade(BaseModel):
    """Single round-trip trade recorded during backtest."""

    entry_time: datetime
    exit_time: datetime
    entry_price: float = Field(gt=0)
    exit_price: float = Field(gt=0)
    quantity: int = Field(gt=0)
    pnl: float
    return_pct: float
    probability_at_entry: float | None = None
    exit_reason: str | None = None


class BacktestMetrics(BaseModel):
    """Summary statistics from a backtest run."""

    initial_capital: float
    final_equity: float
    total_return_pct: float
    sharpe_ratio: float | None = None
    max_drawdown_pct: float
    win_rate_pct: float | None = None
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float | None = None
    avg_trade_return_pct: float | None = None


class EquityPoint(BaseModel):
    """Portfolio equity snapshot."""

    timestamp: datetime
    equity: float
    cash: float
    position_value: float
    drawdown_pct: float = 0.0


class BacktestResult(BaseModel):
    """Complete backtest output."""

    instrument_security_id: str
    exchange_segment: str
    timeframe: str
    config: BacktestConfig
    metrics: BacktestMetrics
    trades: list[BacktestTrade]
    equity_curve: list[EquityPoint]


class SymbolBacktestSummary(BaseModel):
    """Per-symbol metrics from a panel backtest run."""

    security_id: str
    symbol: str | None = None
    total_return_pct: float
    total_trades: int
    win_rate_pct: float | None = None
    profit_factor: float | None = None
    max_drawdown_pct: float | None = None


class PanelBacktestResult(BaseModel):
    """Aggregated backtest output across a stock universe."""

    universe_id: str
    timeframe: str
    config: BacktestConfig
    symbols_total: int
    symbols_backtested: int
    symbols_skipped: int
    total_trades: int
    pooled_win_rate_pct: float | None = None
    mean_return_pct: float | None = None
    median_return_pct: float | None = None
    symbols_positive_return: int = 0
    pooled_profit_factor: float | None = None
    per_symbol: list[SymbolBacktestSummary]
    summary_path: str | None = None
