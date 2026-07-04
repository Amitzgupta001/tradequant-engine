"""Aggregate metrics from panel backtests."""

import statistics

from app.backtest.metrics import calculate_profit_factor
from app.domain.backtest import (
    BacktestConfig,
    BacktestResult,
    PanelBacktestResult,
    SymbolBacktestSummary,
)


def summarize_panel_backtest(
    universe_id: str,
    timeframe: str,
    config: BacktestConfig,
    results: list[BacktestResult],
    symbols_total: int,
    symbols_skipped: int,
    instruments_by_id: dict[str, str | None] | None = None,
) -> PanelBacktestResult:
    """Combine per-symbol backtest results into panel summary."""
    instruments_by_id = instruments_by_id or {}
    per_symbol: list[SymbolBacktestSummary] = []
    all_trades = []

    for result in results:
        per_symbol.append(
            SymbolBacktestSummary(
                security_id=result.instrument_security_id,
                symbol=instruments_by_id.get(result.instrument_security_id),
                total_return_pct=result.metrics.total_return_pct,
                total_trades=result.metrics.total_trades,
                win_rate_pct=result.metrics.win_rate_pct,
                profit_factor=result.metrics.profit_factor,
                max_drawdown_pct=result.metrics.max_drawdown_pct,
            )
        )
        all_trades.extend(result.trades)

    returns = [summary.total_return_pct for summary in per_symbol]
    winning_trades = [trade for trade in all_trades if trade.pnl > 0]
    pooled_win_rate = (len(winning_trades) / len(all_trades) * 100) if all_trades else None

    return PanelBacktestResult(
        universe_id=universe_id,
        timeframe=timeframe,
        config=config,
        symbols_total=symbols_total,
        symbols_backtested=len(results),
        symbols_skipped=symbols_skipped,
        total_trades=len(all_trades),
        pooled_win_rate_pct=pooled_win_rate,
        mean_return_pct=(statistics.mean(returns) if returns else None),
        median_return_pct=(statistics.median(returns) if returns else None),
        symbols_positive_return=sum(1 for value in returns if value > 0),
        pooled_profit_factor=calculate_profit_factor(all_trades),
        per_symbol=per_symbol,
    )
