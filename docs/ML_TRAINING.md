# ML Training

## Overview

TradeQuant trains LightGBM models on engineered feature vectors with optional **swing setup filtering** — only rows matching oversold/overbought technical setups are used for training. This targets actionable intraday signals rather than predicting every bar.

Two training modes:

| Mode | CLI | Model path |
|---|---|---|
| Single stock | `train --security-id 1333` | `storage/models/NSE_EQ/{id}/min_5/` |
| Panel (multi-stock) | `train --universe nifty500` | `storage/models/panels/{universe}/min_5/` |

## Best preset (`--preset best`)

Validated 5-minute intraday configuration from backtest experiments on HDFC Bank:

| Setting | Value |
|---|---|
| Timeframe | `MIN_5` |
| Lookback | 365 days |
| Setup | Long (RSI/BB oversold) |
| Forward horizon | 20 bars (~100 minutes) |
| Move threshold | 0.3% |
| Model threshold | Auto-tuned on validation set |
| Stop loss (backtest) | 1% (2× ATR floor) |
| Trailing stop | 0.6% after +0.8% profit |
| Max hold | 20 bars |

Defined in `app/strategy/presets.py` as `BEST_5M_TRAINING` and `BEST_5M_BACKTEST`.

```bash
# Single stock
PYTHONPATH=. python3 -m app.cli train --security-id 1333 --preset best

# Panel
PYTHONPATH=. python3 -m app.cli train --universe nifty500 --preset best --skip-existing --sleep-seconds 5
```

## Feature columns

23 engineered features per candle (see `app/ml/datasets/preparation.py`):

- Returns: `return_1d`, `return_3d`, `return_5d`, lags, volatility
- Price action: `high_low_range_pct`, `body_pct`
- Indicators: `rsi_14`, `macd_histogram`, `atr_pct`, `bb_position`, etc.
- Labels: `forward_return_1d`, `forward_return_5d`, `forward_return_20b`

## Setup filter

Long setups (default) keep rows where any of:

- RSI &lt; 40
- Bollinger position &lt; 0.15
- RSI &lt; 45 and MACD histogram &gt; 0

Labels use `swing_setup_label`: success if forward return exceeds `move_threshold` (0.3% for 5-min preset).

## Threshold tuning

The trainer searches validation probabilities for the best signal threshold (0.15–0.90). If no validation signals fire, it falls back to the 75th percentile of predicted probabilities. `scale_pos_weight` balances class imbalance.

## Panel training

Panel mode:

1. Downloads/features each universe symbol (or skips existing with `--skip-existing`)
2. Loads all feature CSVs and concatenates rows
3. Trains one LightGBM model on the pooled dataset
4. Saves to `storage/models/panels/{universe_id}/min_5/`

More training rows → better generalization across stocks, but per-stock backtest still loads per-`security_id` models today. Panel inference/backtest wiring is a future step.

## Intraday download chunking

Dhan limits intraday history to **90 days per API request**. The provider automatically splits longer ranges into chunks and deduplicates candles. A 365-day 5-min download makes ~5 API calls per symbol.

## Honest expectations

- Daily direction accuracy ~50% is normal for liquid large-caps
- Setup-filtered intraday models target **signal hit rate**, not overall accuracy
- Majority-class baselines can look like 80%+ accuracy — always check `signal_hit_rate` and backtest P&amp;L

## Related

- [CLI Reference](CLI.md)
- [Stock Universes](UNIVERSES.md)
- [Architecture](ARCHITECTURE.md)
