"""Performance dashboard JSON report generation."""

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


class PerformanceDashboardReport(BaseModel):
    """Trading and ML metrics for dashboard consumption."""

    win_rate: float | None = None
    profit_factor: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    max_drawdown: float | None = None
    average_trade: float | None = None
    expected_value: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    roc_auc: float | None = None
    calibration: float | None = None

    def save(self, path: Path) -> None:
        """Persist dashboard metrics JSON."""
        path.write_text(json.dumps(self.model_dump(), indent=2), encoding="utf-8")


def build_dashboard_report(
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float] | None = None,
    trade_returns: list[float] | None = None,
) -> PerformanceDashboardReport:
    """Build dashboard metrics from predictions and optional trade returns."""
    report = PerformanceDashboardReport()
    if y_true:
        report.precision = precision_score(y_true, y_pred, zero_division=0)
        report.recall = recall_score(y_true, y_pred, zero_division=0)
        report.f1_score = f1_score(y_true, y_pred, zero_division=0)
        report.win_rate = sum(1 for value in y_true if value == 1) / len(y_true)
        if y_prob and len(set(y_true)) > 1:
            report.roc_auc = roc_auc_score(y_true, y_prob)
            report.calibration = 1.0 - abs(np.mean(y_prob) - np.mean(y_true))

    if trade_returns:
        wins = [value for value in trade_returns if value > 0]
        losses = [value for value in trade_returns if value < 0]
        report.average_trade = float(np.mean(trade_returns))
        report.expected_value = report.average_trade
        if losses:
            report.profit_factor = abs(sum(wins) / sum(losses)) if wins else 0.0
        report.sharpe_ratio = _sharpe(trade_returns)
        report.sortino_ratio = _sortino(trade_returns)
        report.max_drawdown = _max_drawdown(trade_returns)
        report.win_rate = len(wins) / len(trade_returns)

    return report


def _sharpe(returns: list[float]) -> float | None:
    if len(returns) < 2:
        return None
    std = float(np.std(returns))
    if std == 0:
        return None
    return float(np.mean(returns) / std * np.sqrt(252))


def _sortino(returns: list[float]) -> float | None:
    downside = [value for value in returns if value < 0]
    if len(downside) < 2:
        return None
    downside_std = float(np.std(downside))
    if downside_std == 0:
        return None
    return float(np.mean(returns) / downside_std * np.sqrt(252))


def _max_drawdown(returns: list[float]) -> float | None:
    if not returns:
        return None
    equity = np.cumprod([1 + value for value in returns])
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(drawdown.min())
