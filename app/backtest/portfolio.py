"""Simulated portfolio for backtesting."""

import math
from datetime import datetime

from app.domain.backtest import BacktestConfig, BacktestTrade
from app.domain.candle import Candle


class Portfolio:
    """Long-only cash equity portfolio with stop-loss management."""

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        self.cash = config.initial_capital
        self.quantity = 0
        self.entry_price: float | None = None
        self.entry_time: datetime | None = None
        self.entry_probability: float | None = None
        self.peak_price: float | None = None
        self.bars_held = 0
        self.trades: list[BacktestTrade] = []

    @property
    def is_long(self) -> bool:
        """Return True when holding a position."""
        return self.quantity > 0

    def equity(self, mark_price: float) -> float:
        """Mark-to-market portfolio value."""
        return self.cash + (self.quantity * mark_price)

    def on_bar(self, candle: Candle) -> None:
        """Update trailing peak and holding period on each bar while long."""
        if not self.is_long:
            return
        self.bars_held += 1
        self.peak_price = candle.high if self.peak_price is None else max(self.peak_price, candle.high)

    def buy(self, price: float, timestamp: datetime, probability: float | None = None) -> bool:
        """Enter long position at price."""
        if self.is_long or price <= 0:
            return False

        budget = self.cash * self._config.position_size_pct
        commission = self._config.commission_pct
        quantity = math.floor(budget / (price * (1 + commission)))
        if quantity <= 0:
            return False

        cost = quantity * price * (1 + commission)
        self.cash -= cost
        self.quantity = quantity
        self.entry_price = price
        self.entry_time = timestamp
        self.entry_probability = probability
        self.peak_price = price
        self.bars_held = 0
        return True

    def sell(
        self,
        price: float,
        timestamp: datetime,
        exit_reason: str | None = None,
    ) -> bool:
        """Exit long position at price."""
        if not self.is_long or self.entry_price is None or self.entry_time is None:
            return False

        commission = self._config.commission_pct
        proceeds = self.quantity * price * (1 - commission)
        pnl = proceeds - (self.quantity * self.entry_price * (1 + commission))
        return_pct = pnl / (self.quantity * self.entry_price * (1 + commission))

        self.trades.append(
            BacktestTrade(
                entry_time=self.entry_time,
                exit_time=timestamp,
                entry_price=self.entry_price,
                exit_price=price,
                quantity=self.quantity,
                pnl=pnl,
                return_pct=return_pct,
                probability_at_entry=self.entry_probability,
                exit_reason=exit_reason,
            )
        )

        self.cash += proceeds
        self.quantity = 0
        self.entry_price = None
        self.entry_time = None
        self.entry_probability = None
        self.peak_price = None
        self.bars_held = 0
        return True

    def stop_level(self, atr_pct: float | None = None) -> float | None:
        """Return the active stop price (fixed + trailing, tightest for long)."""
        if not self.is_long or self.entry_price is None:
            return None

        stops: list[float] = []
        fixed_pct = self._config.stop_loss_pct
        if atr_pct is not None:
            fixed_pct = max(fixed_pct, atr_pct * self._config.atr_stop_multiplier)
        if fixed_pct > 0:
            stops.append(self.entry_price * (1 - fixed_pct))

        if self.peak_price is not None and self._config.trailing_stop_pct > 0:
            gain_pct = (self.peak_price - self.entry_price) / self.entry_price
            if gain_pct >= self._config.trailing_activation_pct:
                trail_pct = self._config.trailing_stop_pct
                if atr_pct is not None:
                    trail_pct = max(trail_pct, atr_pct)
                stops.append(self.peak_price * (1 - trail_pct))

        return max(stops) if stops else None

    def check_stop_hit(self, candle: Candle, atr_pct: float | None = None) -> float | None:
        """Return fill price if intrabar stop is triggered."""
        stop = self.stop_level(atr_pct=atr_pct)
        if stop is None or candle.low > stop:
            return None
        return min(stop, candle.open) if candle.open > stop else stop

    def should_time_exit(self) -> bool:
        """Return True when max holding period is exceeded."""
        if self._config.max_hold_bars is None:
            return False
        return self.bars_held >= self._config.max_hold_bars
