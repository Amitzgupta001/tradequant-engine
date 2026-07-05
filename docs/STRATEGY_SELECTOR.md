# Strategy Selector (Phase 3)

Multi-strategy framework: per-strategy ML models, walk-forward benchmarks, and a meta-model that picks the best strategy for current market conditions.

## Overview

```text
12 rule-based strategies  →  per-strategy LightGBM models
                         →  walk-forward mini-backtests (benchmark dataset)
                         →  multiclass selector (LightGBM)
                         →  hybrid pick: model shortlist + backtest score
                         →  rolling auto-backtest
```

| Layer | Package | Role |
|---|---|---|
| Strategies | `app/strategies/` | Rule-based signals + strategy features |
| Per-strategy ML | `app/ml/trainer/strategy_trainer.py` | One LightGBM per strategy |
| Benchmark builder | `app/ml/datasets/strategy_selector_builder.py` | Label each window with best backtest strategy |
| Selector trainer | `app/ml/trainer/strategy_selector_trainer.py` | Multiclass model + margin tuning |
| Hybrid picker | `app/ml/selector/picker.py` | Model top-3 + backtest return |
| Recommendation | `app/ml/inference/recommendation.py` | Live strategy recommendation |
| Orchestration | `app/services/strategy_selector_service.py` | Train, recommend, rolling backtest |

## Registered strategies

`ema_crossover`, `ema_pullback`, `vwap_breakout`, `orb`, `cpr_breakout`, `supertrend`, `macd_momentum`, `rsi_reversal`, `bollinger_mean_reversion`, `price_action_breakout`, `breakout`, `breakdown`

Each strategy has its own dataset and model under `storage/models/{strategy_id}/v{N}/`.

## Best preset (`--preset best`)

1:3 risk-reward on 5-minute bars (see `app/strategy/presets.py`):

| Setting | Value |
|---|---|
| Stop loss | 0.5% |
| Reward / move threshold | 1.5% |
| T1 / T2 / T3 | +0.5% / +1.0% / +1.5% (33% / 33% / remainder) |
| Max hold | 20 bars |
| Max trades/day | 3 |
| Exit confirmation | 2 bars |

## CLI workflow

### 1. Per-strategy backtest

Train/load one strategy model and run a full backtest:

```bash
PYTHONPATH=. python3 -m app.cli backtest-strategy \
  --strategy-id breakout \
  --security-id 25 \
  --timeframe MIN_5 \
  --preset best
```

**Output:** `storage/backtests/strategies/{strategy_id}/NSE_EQ/{id}/min_5/`

### 2. Train strategy selector (single stock)

Runs walk-forward backtests for all strategies, builds benchmark dataset, trains selector:

```bash
PYTHONPATH=. python3 -m app.cli train-strategy-selector \
  --security-id 25 \
  --timeframe MIN_5 \
  --preset best
```

Use `--rebuild-dataset` to force fresh benchmarks. Use `--skip-strategy-training` if per-strategy models already exist.

**Output:**

| Artifact | Path |
|---|---|
| Benchmark dataset | `storage/datasets/strategy_selector/NSE_EQ/{id}/min_5/` |
| Selector model | `storage/models/strategy_selector/NSE_EQ/{id}/min_5/v{N}/` |

### 3. Train pooled selector (universe)

One selector trained across all symbols in a universe:

```bash
PYTHONPATH=. python3 -m app.cli train-strategy-selector \
  --universe nifty50 \
  --timeframe MIN_5 \
  --preset best
```

**Output:**

| Artifact | Path |
|---|---|
| Pooled benchmark | `storage/datasets/strategy_selector/panels/nifty50/min_5/` |
| Pooled selector | `storage/models/strategy_selector/panels/nifty50/min_5/v{N}/` |

Use `--universe nifty50` with `recommend-strategy` and `backtest-auto` to apply the pooled model to any symbol in that universe.

### 4. Recommend strategy

```bash
# Per-stock selector
PYTHONPATH=. python3 -m app.cli recommend-strategy \
  --security-id 25 \
  --timeframe MIN_5

# Pooled Nifty 50 selector
PYTHONPATH=. python3 -m app.cli recommend-strategy \
  --security-id 25 \
  --universe nifty50 \
  --timeframe MIN_5
```

### 5. Auto backtest (rolling)

**Default mode** switches strategy each walk-forward window using the hybrid picker (model shortlist + window backtest scores):

```bash
PYTHONPATH=. python3 -m app.cli backtest-auto \
  --security-id 25 \
  --timeframe MIN_5 \
  --preset best
```

Use `--mode single` to backtest only the latest recommended strategy over full history (usually misleading for selector use).

Use `--universe nifty50` to load the pooled selector model.

## How selection works

### Walk-forward benchmark

For each rolling window (default: 400 train bars → 50 eval bars):

1. Run a real mini-backtest for **every** strategy on the eval slice.
2. Label the window with the best strategy (profit factor by default).
3. Store market features + per-strategy returns in the benchmark row.

### Selector model

Multiclass LightGBM predicts `best_strategy_id` from market + regime features. Threshold tuning uses **probability margin** (top − second), not absolute confidence — required because 12 classes spread probability mass (~8% each).

### Hybrid picker

1. Model shortlists top-3 strategies by probability.
2. Picks the one with the **best backtest return** in that window (`strategy_returns_json`).
3. For live recommend, uses **historical average window return** per strategy from the benchmark dataset.

This prevents blindly following a model favorite (e.g. `macd_momentum`) when `breakout` scored better in backtests.

### Regime fallback

If no selector model exists or the gate rejects the prediction, static regime rules apply (e.g. sideways → `breakout`, `bollinger_mean_reversion`, …).

## Other commands

| Command | Description |
|---|---|
| `backtest-sweep` | Grid-search backtest parameters for ML swing path |
| `backtest` | Original per-stock ML swing backtest (`app/strategy/ml_strategy.py`) |

## Storage layout

```text
storage/
  datasets/
    {strategy_id}/NSE_EQ/{id}/min_5/          # per-strategy ML datasets
    strategy_selector/
      NSE_EQ/{id}/min_5/                       # single-stock benchmarks
      panels/{universe}/min_5/                 # pooled benchmarks
  models/
    {strategy_id}/v{N}/                        # per-strategy models
    strategy_selector/
      NSE_EQ/{id}/min_5/v{N}/                  # single-stock selector
      panels/{universe}/min_5/v{N}/            # pooled selector
  backtests/
    strategies/{strategy_id}/NSE_EQ/{id}/min_5/
```

## Typical workflow (ADANIENT example)

```bash
# Download + features first (if not done)
PYTHONPATH=. python3 -m app.cli batch-download --universe nifty50 --preset best --skip-existing

# Train selector on one stock
PYTHONPATH=. python3 -m app.cli train-strategy-selector \
  --security-id 25 --timeframe MIN_5 --preset best --rebuild-dataset

# Compare single strategy vs rolling auto
PYTHONPATH=. python3 -m app.cli backtest-strategy \
  --strategy-id breakout --security-id 25 --timeframe MIN_5 --preset best

PYTHONPATH=. python3 -m app.cli backtest-auto \
  --security-id 25 --timeframe MIN_5 --preset best

# Live recommendation
PYTHONPATH=. python3 -m app.cli recommend-strategy \
  --security-id 25 --timeframe MIN_5
```

## Related docs

- [Backtesting](BACKTESTING.md) — bar loop, walk-forward windows, rolling auto-backtest flow
- [Live Trading](LIVE_TRADING.md) — paper forward test logic, WebSocket + bar close flow
- [Paper Trading](PAPER_TRADING.md) — live forward testing with selector + dashboard
- [CLI Reference](CLI.md) — all command flags
- [Architecture](ARCHITECTURE.md) — layers and storage
- [ML Training](ML_TRAINING.md) — swing ML path (`train` / `backtest`)
