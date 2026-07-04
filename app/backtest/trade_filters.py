"""Entry and exit filter state for backtest simulations."""

from datetime import date, datetime

from app.domain.backtest import BacktestConfig


class TradeFilterState:
    """Track cooldowns and daily trade limits during a backtest."""

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        self.bars_since_last_exit = 0
        self.cooldown_remaining = 0
        self.trades_by_day: dict[date, int] = {}
        self.non_buy_streak = 0
        self.last_stop_exit = False

    def on_bar(self, timestamp: datetime) -> None:
        """Advance bar counters while flat."""
        if self.bars_since_last_exit < self._config.min_bars_between_entries:
            self.bars_since_last_exit += 1
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    def on_exit(self, timestamp: datetime, exit_reason: str | None) -> None:
        """Record exit and reset entry filters."""
        self.bars_since_last_exit = 0
        self.non_buy_streak = 0
        session = timestamp.date()
        self.trades_by_day[session] = self.trades_by_day.get(session, 0)
        if exit_reason == "stop_loss":
            self.last_stop_exit = True
            self.cooldown_remaining = self._config.cooldown_bars_after_stop
        else:
            self.last_stop_exit = False

    def on_entry(self, timestamp: datetime) -> None:
        """Record a new entry."""
        session = timestamp.date()
        self.trades_by_day[session] = self.trades_by_day.get(session, 0) + 1
        self.non_buy_streak = 0

    def can_enter(self, timestamp: datetime) -> bool:
        """Return True when entry filters allow a new position."""
        if self.bars_since_last_exit < self._config.min_bars_between_entries:
            return False
        if self.cooldown_remaining > 0:
            return False
        if self._config.max_trades_per_day is not None:
            session = timestamp.date()
            if self.trades_by_day.get(session, 0) >= self._config.max_trades_per_day:
                return False
        return True

    def register_non_buy(self) -> bool:
        """Increment non-buy streak and return True when exit is confirmed."""
        self.non_buy_streak += 1
        return self.non_buy_streak >= self._config.exit_confirmation_bars

    def reset_non_buy_streak(self) -> None:
        """Reset exit confirmation when BUY signal reappears."""
        self.non_buy_streak = 0
