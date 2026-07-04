"""Simulated portfolio for backtesting."""

import math
from datetime import datetime

from app.domain.backtest import BacktestConfig, BacktestTrade
from app.domain.candle import Candle


class Portfolio:
    """Long-only cash equity portfolio with stop-loss and scaled target management."""

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        self.cash = config.initial_capital
        self.quantity = 0
        self.initial_quantity = 0
        self.entry_price: float | None = None
        self.entry_time: datetime | None = None
        self.entry_probability: float | None = None
        self.peak_price: float | None = None
        self.bars_held = 0
        self.trades: list[BacktestTrade] = []
        self._target_1_hit = False
        self._target_2_hit = False
        self._target_3_hit = False
        self._stop_at_breakeven = False

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
        self.initial_quantity = quantity
        self.entry_price = price
        self.entry_time = timestamp
        self.entry_probability = probability
        self.peak_price = price
        self.bars_held = 0
        self._target_1_hit = False
        self._target_2_hit = False
        self._target_3_hit = False
        self._stop_at_breakeven = False
        return True

    def sell(
        self,
        price: float,
        timestamp: datetime,
        exit_reason: str | None = None,
    ) -> bool:
        """Exit the full remaining long position."""
        return self.sell_partial(self.quantity, price, timestamp, exit_reason or "exit")

    def sell_partial(
        self,
        quantity: int,
        price: float,
        timestamp: datetime,
        exit_reason: str,
    ) -> bool:
        """Exit part of the long position and record a trade leg."""
        if not self.is_long or self.entry_price is None or self.entry_time is None:
            return False

        quantity = min(quantity, self.quantity)
        if quantity <= 0:
            return False

        commission = self._config.commission_pct
        cost_basis = quantity * self.entry_price * (1 + commission)
        proceeds = quantity * price * (1 - commission)
        pnl = proceeds - cost_basis
        return_pct = pnl / cost_basis if cost_basis else 0.0

        self.trades.append(
            BacktestTrade(
                entry_time=self.entry_time,
                exit_time=timestamp,
                entry_price=self.entry_price,
                exit_price=price,
                quantity=quantity,
                pnl=pnl,
                return_pct=return_pct,
                probability_at_entry=self.entry_probability,
                exit_reason=exit_reason,
            )
        )

        self.cash += proceeds
        self.quantity -= quantity
        if self.quantity <= 0:
            self._reset_position()
        return True

    def process_profit_targets(self, candle: Candle) -> None:
        """Book T1, T2, and T3 partial exits when intrabar highs reach targets."""
        if not self._config.use_scaled_targets or not self.is_long or self.entry_price is None:
            return

        entry = self.entry_price
        targets = (
            (self._config.target_1_pct, self._config.target_1_qty_pct, "target_1", "_target_1_hit"),
            (self._config.target_2_pct, self._config.target_2_qty_pct, "target_2", "_target_2_hit"),
            (self._config.target_3_pct, None, "target_3", "_target_3_hit"),
        )

        for target_pct, qty_pct, reason, flag_name in targets:
            if getattr(self, flag_name) or not self.is_long:
                continue
            target_price = entry * (1 + target_pct)
            if candle.high < target_price:
                continue

            if qty_pct is None:
                quantity = self.quantity
            else:
                quantity = max(1, math.floor(self.initial_quantity * qty_pct))
                quantity = min(quantity, self.quantity)

            fill_price = max(target_price, candle.open) if candle.open < target_price else target_price
            self.sell_partial(quantity, fill_price, candle.timestamp, reason)
            setattr(self, flag_name, True)

            if reason == "target_1" and self._config.move_stop_to_breakeven_after_t1:
                self._stop_at_breakeven = True

    def stop_level(self, atr_pct: float | None = None) -> float | None:
        """Return the active stop price (fixed + trailing + breakeven, tightest for long)."""
        if not self.is_long or self.entry_price is None:
            return None

        stops: list[float] = []
        if self._stop_at_breakeven:
            stops.append(self.entry_price)

        fixed_pct = self._config.stop_loss_pct
        if atr_pct is not None and self._config.atr_stop_multiplier > 0:
            fixed_pct = max(fixed_pct, atr_pct * self._config.atr_stop_multiplier)
        if fixed_pct > 0:
            stops.append(self.entry_price * (1 - fixed_pct))

        if self.peak_price is not None and self._config.trailing_stop_pct > 0:
            gain_pct = (self.peak_price - self.entry_price) / self.entry_price
            if gain_pct >= self._config.trailing_activation_pct:
                trail_pct = self._config.trailing_stop_pct
                if atr_pct is not None and self._config.atr_stop_multiplier > 0:
                    trail_pct = max(trail_pct, atr_pct)
                stops.append(self.peak_price * (1 - trail_pct))

        return max(stops) if stops else None

    def check_stop_hit(self, candle: Candle, atr_pct: float | None = None) -> float | None:
        """Return fill price if intrabar stop is triggered on remaining quantity."""
        stop = self.stop_level(atr_pct=atr_pct)
        if stop is None or candle.low > stop:
            return None
        return min(stop, candle.open) if candle.open > stop else stop

    def should_time_exit(self) -> bool:
        """Return True when max holding period is exceeded."""
        if self._config.max_hold_bars is None:
            return False
        return self.bars_held >= self._config.max_hold_bars

    def _reset_position(self) -> None:
        """Clear open position state."""
        self.quantity = 0
        self.initial_quantity = 0
        self.entry_price = None
        self.entry_time = None
        self.entry_probability = None
        self.peak_price = None
        self.bars_held = 0
        self._target_1_hit = False
        self._target_2_hit = False
        self._target_3_hit = False
        self._stop_at_breakeven = False
