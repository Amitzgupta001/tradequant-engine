"""Backtest performance metrics."""

import math

from app.domain.backtest import BacktestConfig, BacktestMetrics, BacktestTrade, EquityPoint


def calculate_max_drawdown(equity_curve: list[EquityPoint]) -> float:
    """Return maximum drawdown as a positive percentage."""
    if not equity_curve:
        return 0.0

    peak = equity_curve[0].equity
    max_drawdown = 0.0
    for point in equity_curve:
        peak = max(peak, point.equity)
        if peak > 0:
            drawdown = (peak - point.equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown * 100


def calculate_sharpe_ratio(
    equity_curve: list[EquityPoint],
    annualization_factor: int,
) -> float | None:
    """Annualized Sharpe ratio from equity curve returns."""
    if len(equity_curve) < 3:
        return None

    returns: list[float] = []
    for index in range(1, len(equity_curve)):
        previous = equity_curve[index - 1].equity
        current = equity_curve[index].equity
        if previous > 0:
            returns.append((current - previous) / previous)

    if len(returns) < 2:
        return None

    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance)
    if std_dev == 0:
        return None

    return (mean_return / std_dev) * math.sqrt(annualization_factor)


def calculate_profit_factor(trades: list[BacktestTrade]) -> float | None:
    """Gross profit divided by gross loss."""
    gross_profit = sum(trade.pnl for trade in trades if trade.pnl > 0)
    gross_loss = abs(sum(trade.pnl for trade in trades if trade.pnl < 0))
    if gross_loss == 0:
        return None
    return gross_profit / gross_loss


def summarize_backtest(
    config: BacktestConfig,
    trades: list[BacktestTrade],
    equity_curve: list[EquityPoint],
) -> BacktestMetrics:
    """Compute summary metrics from trades and equity curve."""
    initial = config.initial_capital
    final_equity = equity_curve[-1].equity if equity_curve else initial
    total_return_pct = ((final_equity - initial) / initial) * 100

    winning = [trade for trade in trades if trade.pnl > 0]
    losing = [trade for trade in trades if trade.pnl <= 0]
    win_rate = (len(winning) / len(trades) * 100) if trades else None
    avg_trade_return = (
        sum(trade.return_pct for trade in trades) / len(trades) * 100 if trades else None
    )

    return BacktestMetrics(
        initial_capital=initial,
        final_equity=final_equity,
        total_return_pct=total_return_pct,
        sharpe_ratio=calculate_sharpe_ratio(equity_curve, config.annualization_factor),
        max_drawdown_pct=calculate_max_drawdown(equity_curve),
        win_rate_pct=win_rate,
        total_trades=len(trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        profit_factor=calculate_profit_factor(trades),
        avg_trade_return_pct=avg_trade_return,
    )


def attach_drawdowns(equity_curve: list[EquityPoint]) -> list[EquityPoint]:
    """Populate drawdown percentage on each equity point."""
    if not equity_curve:
        return []

    peak = equity_curve[0].equity
    enriched: list[EquityPoint] = []
    for point in equity_curve:
        peak = max(peak, point.equity)
        drawdown_pct = ((peak - point.equity) / peak * 100) if peak > 0 else 0.0
        enriched.append(point.model_copy(update={"drawdown_pct": drawdown_pct}))
    return enriched
