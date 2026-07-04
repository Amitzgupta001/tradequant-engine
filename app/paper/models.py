"""Paper trading session and trade models."""

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


class PaperSessionStatus(str, Enum):
    """Lifecycle state for a paper trading session."""

    RUNNING = "running"
    STOPPED = "stopped"


class OpenPosition(BaseModel):
    """Currently open simulated long position."""

    security_id: str
    symbol: str
    strategy_id: str
    entry_time: datetime
    entry_price: float = Field(gt=0)
    quantity: int = Field(gt=0)
    mark_price: float | None = None
    unrealized_pnl: float | None = None
    bars_held: int = 0


class PaperTradeRecord(BaseModel):
    """Closed paper trade leg."""

    trade_id: str
    session_id: str
    security_id: str
    symbol: str
    strategy_id: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float = Field(gt=0)
    exit_price: float = Field(gt=0)
    quantity: int = Field(gt=0)
    pnl: float
    return_pct: float
    exit_reason: str | None = None
    probability_at_entry: float | None = None


class PaperInstrumentState(BaseModel):
    """Per-symbol paper trading runtime state."""

    security_id: str
    symbol: str
    strategy_id: str | None = None
    strategy_date: date | None = None
    last_processed_bar: datetime | None = None
    last_mark_price: float | None = None
    portfolio_state: dict = Field(default_factory=dict)
    filter_state: dict = Field(default_factory=dict)
    realized_pnl: float = 0.0
    trade_count: int = 0


class PaperSession(BaseModel):
    """Paper trading session metadata."""

    session_id: str
    universe_id: str
    timeframe: str
    initial_capital: float = Field(gt=0)
    capital_per_symbol: float = Field(gt=0)
    status: PaperSessionStatus = PaperSessionStatus.RUNNING
    created_at: datetime
    last_tick_at: datetime | None = None
    last_error: str | None = None
    selector_universe_id: str | None = None
    instrument_ids: list[str] = Field(default_factory=list)


class PaperDashboardSnapshot(BaseModel):
    """Aggregated view for the live dashboard."""

    session: PaperSession
    market_open: bool
    open_positions: list[OpenPosition]
    recent_trades: list[PaperTradeRecord]
    total_realized_pnl: float
    total_trades: int
    symbols_processed: int
    last_tick_at: datetime | None = None
