"""Training configuration and result models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class TrainingTask(str, Enum):
    """Supported ML training tasks."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ThresholdTuningObjective(str, Enum):
    """Objective used when tuning the classification threshold."""

    HIT_RATE = "hit_rate"
    PROFIT_FACTOR = "profit_factor"
    SHARPE = "sharpe"


class SetupType(str, Enum):
    """Technical setup filter for swing signal training."""

    LONG = "long"
    SHORT = "short"


class TrainingConfig(BaseModel):
    """LightGBM training hyperparameters."""

    task: TrainingTask = TrainingTask.CLASSIFICATION
    setup_type: SetupType | None = Field(
        default=SetupType.LONG,
        description="Train on technical setup days only (targets ~80%% signal accuracy)",
    )
    forward_horizon_bars: int = Field(default=5, ge=1, le=100)
    move_threshold: float = Field(
        default=0.02,
        ge=0.001,
        le=0.05,
        description="Min absolute forward return for a successful setup",
    )
    test_size: float = Field(default=0.2, gt=0.0, lt=0.5)
    validation_size: float = Field(default=0.15, gt=0.0, lt=0.4)
    num_leaves: int = Field(default=7, ge=2)
    max_depth: int = Field(default=3, ge=2)
    learning_rate: float = Field(default=0.02, gt=0.0, le=1.0)
    n_estimators: int = Field(default=500, ge=10)
    min_child_samples: int = Field(default=15, ge=1)
    subsample: float = Field(default=0.7, gt=0.0, le=1.0)
    colsample_bytree: float = Field(default=0.7, gt=0.0, le=1.0)
    reg_alpha: float = Field(default=2.0, ge=0.0)
    reg_lambda: float = Field(default=2.0, ge=0.0)
    early_stopping_rounds: int = Field(default=50, ge=5)
    label_threshold: float = Field(default=0.0, ge=0.0)
    tune_threshold: bool = True
    threshold_objective: ThresholdTuningObjective = ThresholdTuningObjective.PROFIT_FACTOR
    assumed_win_pct: float = Field(default=0.003, ge=0.0)
    assumed_loss_pct: float = Field(default=0.007, ge=0.0)
    random_state: int = 42
    min_train_rows: int = Field(default=50, ge=20)


class TrainingMetrics(BaseModel):
    """Evaluation metrics from model training."""

    train_rows: int
    validation_rows: int = 0
    test_rows: int
    setup_rows: int = 0
    accuracy: float | None = None
    balanced_accuracy: float | None = None
    validation_accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    roc_auc: float | None = None
    prediction_threshold: float | None = None
    signal_count: int = 0
    signal_hit_rate: float | None = None
    majority_baseline: float | None = None
    rmse: float | None = None
    mae: float | None = None


class TrainingResult(BaseModel):
    """Outcome of a completed training run."""

    model_path: str
    metadata_path: str
    instrument_security_id: str
    timeframe: str
    feature_columns: list[str]
    metrics: TrainingMetrics
    trained_at: datetime
