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
    atr_stop_multiplier: float = Field(
        default=2.0,
        ge=0.0,
        le=5.0,
        description="Set to 0 to use fixed stop_loss_pct only (no ATR widening)",
    )
    max_hold_bars: int | None = Field(default=20, ge=1)
    min_bars_between_entries: int = Field(default=0, ge=0)
    max_trades_per_day: int | None = Field(default=None, ge=1)
    cooldown_bars_after_stop: int = Field(default=0, ge=0)
    exit_confirmation_bars: int = Field(
        default=1,
        ge=1,
        description="Consecutive non-BUY bars required before signal exit",
    )
    min_expected_value: float | None = Field(
        default=None,
        description="Minimum EV = p*avg_win - (1-p)*avg_loss to enter (disabled when None)",
    )
    expected_win_pct: float = Field(default=0.003, ge=0.0)
    expected_loss_pct: float = Field(default=0.007, ge=0.0)
    require_strategy_signal: bool = Field(
        default=False,
        description="Only enter when underlying strategy signal is active",
    )
    use_scaled_targets: bool = Field(
        default=False,
        description="Enable T1/T2/T3 partial profit booking",
    )
    target_1_pct: float = Field(default=0.005, ge=0.0, le=0.2)
    target_2_pct: float = Field(default=0.010, ge=0.0, le=0.2)
    target_3_pct: float = Field(default=0.015, ge=0.0, le=0.2)
    target_1_qty_pct: float = Field(default=0.33, gt=0.0, le=1.0)
    target_2_qty_pct: float = Field(default=0.33, gt=0.0, le=1.0)
    move_stop_to_breakeven_after_t1: bool = True


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
