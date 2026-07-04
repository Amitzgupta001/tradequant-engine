"""Model evaluation metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.domain.training import TrainingMetrics


def evaluate_classification(
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float],
) -> TrainingMetrics:
    """Compute classification metrics."""
    majority_baseline = max(y_true.count(1), y_true.count(0)) / len(y_true) if y_true else None
    metrics = TrainingMetrics(
        train_rows=0,
        test_rows=len(y_true),
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        majority_baseline=majority_baseline,
    )
    if len(set(y_true)) > 1:
        metrics.roc_auc = roc_auc_score(y_true, y_prob)
    return metrics


def evaluate_regression(y_true: list[float], y_pred: list[float]) -> TrainingMetrics:
    """Compute regression metrics."""
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return TrainingMetrics(
        train_rows=0,
        test_rows=len(y_true),
        rmse=rmse,
        mae=mean_absolute_error(y_true, y_pred),
    )


def finalize_metrics(metrics: TrainingMetrics, train_rows: int) -> TrainingMetrics:
    """Attach train row count to metrics."""
    return metrics.model_copy(update={"train_rows": train_rows})


def apply_threshold(y_prob: list[float], threshold: float) -> list[int]:
    """Convert probabilities to binary predictions."""
    return [1 if probability >= threshold else 0 for probability in y_prob]


def signal_hit_rate(y_true: list[int], y_prob: list[float], threshold: float) -> tuple[int, float | None]:
    """Compute hit rate and count for positive signals above threshold."""
    signals = [index for index, probability in enumerate(y_prob) if probability >= threshold]
    if not signals:
        return 0, None
    hits = sum(y_true[index] for index in signals)
    return len(signals), hits / len(signals)


def simulated_profit_factor(
    y_true: list[int],
    y_prob: list[float],
    threshold: float,
    win_pct: float,
    loss_pct: float,
) -> tuple[float | None, int]:
    """Estimate profit factor for long signals at a threshold."""
    wins = 0.0
    losses = 0.0
    count = 0
    for label, probability in zip(y_true, y_prob, strict=True):
        if probability < threshold:
            continue
        count += 1
        if label == 1:
            wins += win_pct
        else:
            losses += loss_pct
    if count == 0 or losses == 0:
        return None, count
    return wins / losses, count


def simulated_sharpe(
    y_true: list[int],
    y_prob: list[float],
    threshold: float,
    win_pct: float,
    loss_pct: float,
) -> tuple[float | None, int]:
    """Estimate Sharpe from simulated trade returns at a threshold."""
    returns: list[float] = []
    for label, probability in zip(y_true, y_prob, strict=True):
        if probability < threshold:
            continue
        returns.append(win_pct if label == 1 else -loss_pct)
    if len(returns) < 2:
        return None, len(returns)
    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / len(returns)
    if variance == 0:
        return None, len(returns)
    return mean_return / (variance**0.5), len(returns)


def find_best_threshold(
    y_true: list[int],
    y_prob: list[float],
    min_signals: int = 5,
    objective: str = "hit_rate",
    win_pct: float = 0.003,
    loss_pct: float = 0.007,
) -> tuple[float, float | None, int, float | None]:
    """Find threshold that maximizes the selected validation objective."""
    if not y_prob:
        return 0.5, None, 0, None

    best_threshold = 0.5
    best_score = -1.0
    best_signal_count = 0
    best_secondary: float | None = None

    for threshold in np.arange(0.15, 0.91, 0.01):
        count, hit_rate = signal_hit_rate(y_true, y_prob, float(threshold))
        if objective == "profit_factor":
            score, count = simulated_profit_factor(y_true, y_prob, float(threshold), win_pct, loss_pct)
            secondary = hit_rate
        elif objective == "sharpe":
            score, count = simulated_sharpe(y_true, y_prob, float(threshold), win_pct, loss_pct)
            secondary = hit_rate
        else:
            score = hit_rate
            secondary = hit_rate
        if score is None or count < min_signals:
            continue
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_signal_count = count
            best_secondary = secondary

    if best_signal_count >= min_signals:
        return best_threshold, best_secondary, best_signal_count, best_score

    for threshold in np.arange(0.15, 0.91, 0.01):
        count, hit_rate = signal_hit_rate(y_true, y_prob, float(threshold))
        if count < 3 or hit_rate is None:
            continue
        if hit_rate > best_score:
            best_score = hit_rate
            best_threshold = float(threshold)
            best_signal_count = count
            best_secondary = hit_rate

    if best_signal_count > 0:
        return best_threshold, best_secondary, best_signal_count, best_score

    percentile_threshold = float(np.percentile(y_prob, 80))
    count, hit_rate = signal_hit_rate(y_true, y_prob, percentile_threshold)
    if count > 0:
        return percentile_threshold, hit_rate, count, hit_rate

    positive_rate = sum(y_true) / len(y_true)
    return max(0.15, positive_rate), None, 0, None
