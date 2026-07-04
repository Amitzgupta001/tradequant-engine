# CLI Reference

All commands are available via Poetry or directly with Python:

```bash
poetry run tradequant <command> [options]
# or
PYTHONPATH=. python3 -m app.cli <command> [options]
```

## Commands

| Command | Description |
|---|---|
| `download` | Download OHLCV for a single instrument |
| `indicators` | Compute technical indicators from stored OHLCV |
| `features` | Build ML feature vectors from stored OHLCV |
| `batch-download` | Download OHLCV + features for an entire universe |
| `train` | Train a LightGBM model (single stock or panel) |
| `backtest` | Run ML swing strategy backtest on stored data |
| `backtest-strategy` | Backtest a Phase 3 per-strategy model |
| `backtest-sweep` | Grid-search ML backtest parameters |
| `backtest-auto` | Rolling backtest with strategy selector (default) |
| `train-strategy-selector` | Train meta-model on walk-forward strategy benchmarks |
| `recommend-strategy` | Recommend best strategy for current conditions |
| `paper-trade` | Live paper trading (WebSocket + 5m bar close, no broker orders) |

---

## download

Fetch historical candles from Dhan and save to CSV.

```bash
PYTHONPATH=. python3 -m app.cli download \
  --security-id 1333 \
  --from-date 2024-01-01 \
  --to-date 2024-12-31 \
  --timeframe DAILY
```

| Option | Default | Description |
|---|---|---|
| `--security-id` | required | Dhan security ID |
| `--exchange` | `NSE_EQ` | Exchange segment |
| `--instrument-type` | `EQUITY` | Instrument type |
| `--symbol` | — | Optional display symbol |
| `--from-date` | required | Start date (`YYYY-MM-DD`) |
| `--to-date` | required | End date (`YYYY-MM-DD`) |
| `--timeframe` | `DAILY` | `DAILY`, `MIN_5`, `MIN_15`, etc. |
| `--include-oi` | off | Include open interest |
| `--no-overwrite` | off | Skip if file exists |

**Output:** `storage/raw/NSE_EQ/{security_id}/{timeframe}.csv`

---

## indicators

Compute EMA, RSI, MACD, ATR, VWAP, Bollinger Bands from raw OHLCV.

```bash
PYTHONPATH=. python3 -m app.cli indicators --security-id 1333 --timeframe DAILY
```

**Output:** `storage/processed/NSE_EQ/{security_id}/{timeframe}_indicators.csv`

---

## features

Build ML-ready feature vectors (returns, indicator gaps, forward-return labels).

```bash
PYTHONPATH=. python3 -m app.cli features --security-id 1333 --timeframe MIN_5
```

**Output:** `storage/features/NSE_EQ/{security_id}/{timeframe}_features.csv`

---

## batch-download

Download and feature-engineer every symbol in a stock universe. Intraday requests are automatically split into 90-day API chunks.

```bash
# Nifty 50 — ~50 symbols
PYTHONPATH=. python3 -m app.cli batch-download --universe nifty50 --preset best

# Nifty 500 — ~466 symbols (several hours; use sleep to avoid overload)
PYTHONPATH=. python3 -m app.cli batch-download --universe nifty500 --preset best --skip-existing --sleep-seconds 5
```

| Option | Default | Description |
|---|---|---|
| `--universe` | required | `nifty50` or `nifty500` |
| `--timeframe` | `DAILY` | Overridden by `--preset best` → `MIN_5` |
| `--years` | `5` | Lookback for daily timeframes |
| `--days` | `90` | Lookback for intraday timeframes |
| `--preset` | — | `best` applies validated 5-min preset |
| `--skip-existing` | off | Skip symbols that already have feature files |
| `--sleep-seconds` | env / `3` | Pause between symbols (see `BATCH_SLEEP_SECONDS`) |

See [UNIVERSES.md](UNIVERSES.md) for universe details.

---

## train

Train a LightGBM classifier on swing-setup filtered rows.

### Single stock

```bash
# Best validated preset (5-min, 365 days, long setups)
PYTHONPATH=. python3 -m app.cli train --security-id 1333 --preset best

# Train on existing features only
PYTHONPATH=. python3 -m app.cli train --security-id 1333 --preset best --skip-download
```

**Output:** `storage/models/NSE_EQ/{security_id}/min_5/`

### Panel (multi-stock)

Pools feature rows from all universe symbols into one shared model.

```bash
# Download + train (Nifty 500 — long-running)
PYTHONPATH=. python3 -m app.cli train --universe nifty500 --preset best --skip-existing

# Train on already-downloaded data
PYTHONPATH=. python3 -m app.cli train --universe nifty500 --preset best --skip-download
```

**Output:** `storage/models/panels/{universe_id}/min_5/`

| Option | Default | Description |
|---|---|---|
| `--security-id` | — | Single instrument (omit with `--universe`) |
| `--universe` | — | `nifty50` or `nifty500` |
| `--timeframe` | `DAILY` | Overridden by `--preset best` |
| `--years` | `5` | Daily lookback |
| `--days` | `90` | Intraday lookback (preset: 365) |
| `--setup-type` | `long` | `long` or `short` setup filter |
| `--forward-horizon` | — | Forward horizon in bars |
| `--move-threshold` | — | Min return for successful setup |
| `--preset` | — | `best` — see [ML_TRAINING.md](ML_TRAINING.md) |
| `--task` | `classification` | `classification` or `regression` |
| `--test-size` | `0.2` | Holdout fraction |
| `--n-estimators` | `200` | LightGBM trees |
| `--skip-download` | off | Use existing feature files |
| `--skip-existing` | off | Skip already-downloaded symbols in batch |
| `--sleep-seconds` | env / `3` | Pause between symbols during batch download |

See [ML_TRAINING.md](ML_TRAINING.md) for training strategy details.

---

## backtest

Run the ML strategy backtest engine on stored data and a trained model.

### Single stock

```bash
PYTHONPATH=. python3 -m app.cli backtest --security-id 1333 --preset best
```

**Output:** `storage/backtests/NSE_EQ/{security_id}/{timeframe}/`

### Panel (universe)

Backtest every symbol that has stored data using the shared panel model:

```bash
PYTHONPATH=. python3 -m app.cli backtest --universe nifty500 --preset best
```

**Output:** `storage/backtests/panels/nifty500/min_5/summary.json`

Add `--per-symbol` to also save per-stock reports under `storage/backtests/NSE_EQ/{security_id}/`.

With custom parameters:

```bash
PYTHONPATH=. python3 -m app.cli backtest \
  --security-id 1333 \
  --initial-capital 100000 \
  --probability-threshold 0.55 \
  --stop-loss-pct 0.01 \
  --trailing-stop-pct 0.006 \
  --max-hold-bars 20
```

**Output:**

```text
storage/backtests/NSE_EQ/{security_id}/{timeframe}/
  summary.json
  equity_curve.csv
```

| Option | Default | Description |
|---|---|---|
| `--security-id` | — | Single instrument (omit with `--universe`) |
| `--universe` | — | `nifty50` or `nifty500` — uses panel model |
| `--preset` | — | `best` applies validated backtest config |
| `--probability-threshold` | — | Signal threshold (preset uses model's tuned value) |
| `--stop-loss-pct` | `0.01` | Fixed stop loss |
| `--trailing-stop-pct` | `0.006` | Trailing stop distance |
| `--trailing-activation-pct` | `0.008` | Profit level to activate trailing stop |
| `--max-hold-bars` | `20` | Maximum bars to hold a position |
| `--per-symbol` | off | Save per-stock reports during panel backtest |
| `--commission-pct` | `0.0003` | Commission per trade |

---

## backtest-strategy

Backtest one Phase 3 strategy (rule signal + per-strategy LightGBM filter).

```bash
PYTHONPATH=. python3 -m app.cli backtest-strategy \
  --strategy-id breakout \
  --security-id 25 \
  --timeframe MIN_5 \
  --preset best
```

| Option | Description |
|---|---|
| `--strategy-id` | required — e.g. `breakout`, `ema_pullback`, `macd_momentum` |
| `--retrain` | Retrain strategy model before backtest |
| `--security-id` | required |
| `--preset` | `best` — 1:3 R:R, T1/T2/T3 partial exits |

**Output:** `storage/backtests/strategies/{strategy_id}/NSE_EQ/{id}/min_5/`

See [STRATEGY_SELECTOR.md](STRATEGY_SELECTOR.md) for all strategy IDs.

---

## backtest-sweep

Grid-search probability threshold, stop loss, and max hold for the ML swing path:

```bash
PYTHONPATH=. python3 -m app.cli backtest-sweep \
  --security-id 25 \
  --timeframe MIN_5 \
  --preset best
```

**Output:** `storage/backtests/sweeps/{security_id}_{timeframe}.json`

---

## train-strategy-selector

Build walk-forward benchmarks across all strategies and train a multiclass selector model.

### Single stock

```bash
PYTHONPATH=. python3 -m app.cli train-strategy-selector \
  --security-id 25 \
  --timeframe MIN_5 \
  --preset best \
  --rebuild-dataset
```

### Pooled universe

```bash
PYTHONPATH=. python3 -m app.cli train-strategy-selector \
  --universe nifty50 \
  --timeframe MIN_5 \
  --preset best
```

| Option | Default | Description |
|---|---|---|
| `--security-id` | — | Single stock (omit with `--universe`) |
| `--universe` | — | `nifty50` or `nifty500` — pooled selector |
| `--train-window` | `400` | Rolling train bars per benchmark window |
| `--step-size` | `50` | Eval bars per window |
| `--min-trades` | `2` | Min trades to score a strategy in a window |
| `--objective` | `profit_factor` | `profit_factor`, `total_return`, or `sharpe` |
| `--rebuild-dataset` | off | Force fresh benchmark build |
| `--skip-strategy-training` | off | Skip auto-training missing per-strategy models |
| `--preset` | — | `best` backtest config for mini-backtests |

**Output:** `storage/models/strategy_selector/...` — see [STRATEGY_SELECTOR.md](STRATEGY_SELECTOR.md)

---

## recommend-strategy

Recommend strategies using the trained selector (hybrid: model + backtest priors).

```bash
PYTHONPATH=. python3 -m app.cli recommend-strategy \
  --security-id 25 \
  --timeframe MIN_5

# Use pooled Nifty 50 model on any symbol
PYTHONPATH=. python3 -m app.cli recommend-strategy \
  --security-id 25 \
  --universe nifty50 \
  --timeframe MIN_5
```

| Option | Default | Description |
|---|---|---|
| `--security-id` | required | Instrument to analyze |
| `--universe` | — | Load pooled selector for this universe |
| `--selector-version` | latest | Specific selector model version |
| `--top-n` | `5` | Number of ranked recommendations |

---

## backtest-auto

Rolling backtest that switches strategy each walk-forward window (default mode).

```bash
PYTHONPATH=. python3 -m app.cli backtest-auto \
  --security-id 25 \
  --timeframe MIN_5 \
  --preset best
```

| Option | Default | Description |
|---|---|---|
| `--mode` | `rolling` | `rolling` (per-window switch) or `single` (latest pick, full history) |
| `--universe` | — | Use pooled selector model |
| `--selector-version` | latest | Specific selector version |
| `--retrain` | off | Retrain recommended strategy model (`single` mode only) |
| `--preset` | — | `best` backtest config |

Falls back to regime-based rules when no selector model is available.

---

## Typical Workflows

### Phase 3 strategy selector (ADANIENT)

```bash
# Train selector (long — all strategies × walk-forward windows)
PYTHONPATH=. python3 -m app.cli train-strategy-selector \
  --security-id 25 --timeframe MIN_5 --preset best --rebuild-dataset

# Best single strategy (baseline)
PYTHONPATH=. python3 -m app.cli backtest-strategy \
  --strategy-id breakout --security-id 25 --timeframe MIN_5 --preset best

# Rolling auto-backtest
PYTHONPATH=. python3 -m app.cli backtest-auto \
  --security-id 25 --timeframe MIN_5 --preset best

# Live recommendation
PYTHONPATH=. python3 -m app.cli recommend-strategy \
  --security-id 25 --timeframe MIN_5
```

See [STRATEGY_SELECTOR.md](STRATEGY_SELECTOR.md) for architecture and pooled universe training.

---

## paper-trade

Live forward testing with **simulated** fills. Dhan is used for **market data only** — `place_order` is blocked.

Full guide with sequence diagrams: [PAPER_TRADING.md](PAPER_TRADING.md)

```bash
# Script wrapper (recommended)
./scripts/run_paper_nifty50.sh serve   # dashboard :4000
./scripts/run_paper_nifty50.sh start
./scripts/run_paper_nifty50.sh run    # --mode live (WebSocket + 5m bar close)

# CLI
PYTHONPATH=. python3 -m app.cli paper-trade \
  --universe nifty50 \
  --selector-universe nifty50 \
  --capital 1000000 \
  --start

PYTHONPATH=. python3 -m app.cli paper-trade \
  --universe nifty50 \
  --run \
  --mode live

# Poll fallback
PYTHONPATH=. python3 -m app.cli paper-trade --run --mode poll --poll-seconds 60
```

| Option | Default | Description |
|---|---|---|
| `--universe` | `nifty50` | Watchlist |
| `--selector-universe` | same | Pooled selector model |
| `--timeframe` | `MIN_5` | Bar interval |
| `--capital` | `1000000` | Virtual capital (split per symbol) |
| `--mode` | `live` | `live` = WebSocket LTP + REST on bar close; `poll` = REST loop |
| `--poll-seconds` | `60` | Bar-clock check (live) or REST interval (poll) |
| `--security-ids` | all | Comma-separated subset |
| `--start` | — | Create session |
| `--run` | — | Continuous loop |
| `--tick` | — | Single cycle |
| `--stop` | — | Stop session |
| `--force` | off | Run when market closed (testing) |

**Output:** `storage/paper/sessions/{session_id}/` · Dashboard: `GET /dashboard`

---

## Example workflows

### Quick single-stock experiment (HDFC Bank)

```bash
PYTHONPATH=. python3 -m app.cli train --security-id 1333 --preset best
PYTHONPATH=. python3 -m app.cli backtest --security-id 1333 --preset best
```

### Nifty 50 panel model

```bash
PYTHONPATH=. python3 -m app.cli batch-download --universe nifty50 --preset best
PYTHONPATH=. python3 -m app.cli train --universe nifty50 --preset best --skip-download
```

### Nifty 500 panel model (resume-friendly)

```bash
# Step 1: batch download (run in background; takes hours)
PYTHONPATH=. python3 -m app.cli batch-download --universe nifty500 --preset best --skip-existing

# Step 2: train when enough symbols are downloaded
PYTHONPATH=. python3 -m app.cli train --universe nifty500 --preset best --skip-download

# Or both in one command (recommended: add sleep for Nifty 500)
PYTHONPATH=. python3 -m app.cli train --universe nifty500 --preset best --skip-existing --sleep-seconds 5
```

Monitor a long-running job:

```bash
tail -f storage/logs/nifty500_train.log
ls storage/features/NSE_EQ | wc -l   # count downloaded symbols
```
