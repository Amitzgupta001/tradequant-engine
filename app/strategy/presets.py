"""Validated strategy presets from backtest experiments."""

from app.domain.backtest import BacktestConfig
from app.domain.enums.market import Timeframe
from app.domain.training import SetupType, ThresholdTuningObjective, TrainingConfig

BEST_5M_TIMEFRAME = Timeframe.MIN_5
BEST_5M_DAYS = 365

# 1:3 risk-reward — risk 0.5% stop, reward 1.5% target move
RISK_REWARD_RATIO = 3.0
RISK_PCT_5M = 0.005
REWARD_PCT_5M = RISK_PCT_5M * RISK_REWARD_RATIO
TARGET_1_QTY_PCT = 0.33
TARGET_2_QTY_PCT = 0.33

BEST_5M_TRAINING = TrainingConfig(
    setup_type=SetupType.LONG,
    forward_horizon_bars=20,
    move_threshold=REWARD_PCT_5M,
    num_leaves=15,
    max_depth=4,
    min_child_samples=30,
    learning_rate=0.02,
    n_estimators=500,
    reg_alpha=1.0,
    reg_lambda=1.0,
    tune_threshold=True,
    threshold_objective=ThresholdTuningObjective.PROFIT_FACTOR,
    assumed_win_pct=REWARD_PCT_5M,
    assumed_loss_pct=RISK_PCT_5M,
)

BEST_5M_BACKTEST = BacktestConfig(
    stop_loss_pct=RISK_PCT_5M,
    trailing_stop_pct=RISK_PCT_5M * 0.8,
    trailing_activation_pct=REWARD_PCT_5M * 0.8,
    atr_stop_multiplier=0.0,
    max_hold_bars=20,
    probability_threshold=None,
    min_bars_between_entries=5,
    max_trades_per_day=3,
    cooldown_bars_after_stop=10,
    exit_confirmation_bars=2,
    min_expected_value=None,
    expected_win_pct=REWARD_PCT_5M,
    expected_loss_pct=RISK_PCT_5M,
    use_scaled_targets=True,
    target_1_pct=RISK_PCT_5M,
    target_2_pct=RISK_PCT_5M * 2,
    target_3_pct=REWARD_PCT_5M,
    target_1_qty_pct=TARGET_1_QTY_PCT,
    target_2_qty_pct=TARGET_2_QTY_PCT,
    move_stop_to_breakeven_after_t1=True,
)

IMPROVED_5M_BACKTEST = BEST_5M_BACKTEST
