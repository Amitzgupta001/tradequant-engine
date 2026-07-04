"""Validated strategy presets from backtest experiments."""

from app.domain.backtest import BacktestConfig
from app.domain.enums.market import Timeframe
from app.domain.training import SetupType, TrainingConfig

BEST_5M_TIMEFRAME = Timeframe.MIN_5
BEST_5M_DAYS = 365

BEST_5M_TRAINING = TrainingConfig(
    setup_type=SetupType.LONG,
    forward_horizon_bars=20,
    move_threshold=0.003,
    num_leaves=15,
    max_depth=4,
    min_child_samples=30,
    learning_rate=0.02,
    n_estimators=500,
    reg_alpha=1.0,
    reg_lambda=1.0,
    tune_threshold=True,
)

BEST_5M_BACKTEST = BacktestConfig(
    stop_loss_pct=0.01,
    trailing_stop_pct=0.006,
    trailing_activation_pct=0.008,
    atr_stop_multiplier=2.0,
    max_hold_bars=20,
    probability_threshold=None,
)
