# TradeQuant Engine

Production-quality AI quantitative trading platform for Indian markets using [Dhan](https://dhan.co).

## Documentation

| Doc | Description |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Layers, data flow, storage layout |
| [CLI Reference](docs/CLI.md) | All commands and options |
| [ML Training](docs/ML_TRAINING.md) | Presets, setup filters, panel training |
| [Stock Universes](docs/UNIVERSES.md) | Nifty 50 / Nifty 500 batch download |

## Architecture

```text
tradequant-engine/
├── app/
│   ├── api/v1/          # Versioned HTTP API
│   ├── brokers/         # Dhan (+ future Zerodha, Upstox)
│   ├── cache/           # In-memory cache (Redis later)
│   ├── cli/             # CLI: download, batch-download, train, backtest
│   ├── core/            # Config, Loguru logging, events
│   ├── data/
│   │   ├── providers/   # Historical data (90-day intraday chunks)
│   │   ├── repositories/
│   │   └── universe/    # Nifty 50 / Nifty 500 symbol lists
│   ├── domain/          # Candle, Instrument, Order, Trade
│   ├── indicators/      # EMA, RSI, MACD, ATR, VWAP, Bollinger
│   ├── ml/              # Features, LightGBM trainer, model registry
│   ├── risk/            # Position sizing, stops, exposure
│   ├── services/        # Orchestration (batch download, training)
│   ├── strategy/        # Presets (best 5-min validated config)
│   └── utils/
├── storage/
│   ├── raw/             # OHLCV CSV from Dhan
│   ├── processed/       # Indicator snapshots
│   ├── features/        # ML feature CSVs
│   ├── models/          # Per-stock and panel models
│   └── backtests/       # Backtest reports
├── tests/
└── docs/
```

## Requirements

- Python 3.12+ (Poetry) or Python 3.10+ with `PYTHONPATH=.`
- Dhan account with API access token

## Authentication

1. [web.dhan.co](https://web.dhan.co) → My Profile → Access DhanHQ APIs
2. Copy Client ID and Access Token into `.env`

```bash
cp .env.example .env
```

Key settings:

```bash
DHAN_CLIENT_ID=...
DHAN_ACCESS_TOKEN=...
BATCH_SLEEP_SECONDS=3   # pause between symbols in batch runs
```

## Setup

```bash
poetry install
# or: pip install -e .
```

## Quick start (single stock)

HDFC Bank (`security_id=1333`) with the validated **best** 5-minute preset:

```bash
PYTHONPATH=. python3 -m app.cli train --security-id 1333 --preset best
PYTHONPATH=. python3 -m app.cli backtest --security-id 1333 --preset best
```

## Multi-stock training (Nifty 50 / Nifty 500)

Batch download OHLCV and features for an entire index, then train a **panel model** pooled across all symbols.

```bash
# Nifty 50 (~50 symbols)
PYTHONPATH=. python3 -m app.cli batch-download --universe nifty50 --preset best
PYTHONPATH=. python3 -m app.cli train --universe nifty50 --preset best --skip-download

# Nifty 500 (~466 symbols — long-running; resume with --skip-existing)
PYTHONPATH=. python3 -m app.cli train --universe nifty500 --preset best --skip-existing --sleep-seconds 5
```

**Panel model output:** `storage/models/panels/nifty500/min_5/`

**Rate limiting:** Batch runs pause between symbols (`BATCH_SLEEP_SECONDS=3` by default, or `--sleep-seconds 5`) to avoid overloading the Dhan API or your machine.

Monitor progress:

```bash
tail -f storage/logs/nifty500_train.log
ls storage/features/NSE_EQ | wc -l
```

See [docs/UNIVERSES.md](docs/UNIVERSES.md) and [docs/ML_TRAINING.md](docs/ML_TRAINING.md) for details.

## CLI commands

| Command | Description |
|---|---|
| `download` | Download OHLCV for one instrument |
| `indicators` | Compute technical indicators |
| `features` | Build ML feature vectors |
| `batch-download` | Download + features for Nifty 50/500 |
| `train` | Train LightGBM (single stock or panel) |
| `backtest` | Run ML strategy backtest |

Full reference: [docs/CLI.md](docs/CLI.md)

## Best preset (`--preset best`)

Validated 5-minute intraday swing setup strategy:

| Setting | Value |
|---|---|
| Timeframe | 5-min, 365 days lookback |
| Setup | Long (RSI/BB oversold) |
| Forward horizon | 20 bars |
| Move threshold | 0.3% |
| Stop / trailing | 1% stop, 0.6% trail after +0.8% |

## Run API

```bash
poetry run serve
```

Health: `GET /health`

## Storage paths

| Data | Path |
|---|---|
| Raw OHLCV | `storage/raw/NSE_EQ/{security_id}/{timeframe}.csv` |
| Features | `storage/features/NSE_EQ/{security_id}/{timeframe}_features.csv` |
| Single-stock model | `storage/models/NSE_EQ/{security_id}/min_5/` |
| Panel model | `storage/models/panels/{universe}/min_5/` |
| Backtest | `storage/backtests/NSE_EQ/{security_id}/{timeframe}/` |

## Tests

```bash
PYTHONPATH=. python3 -m pytest
# or: poetry run pytest
```

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Foundation, Dhan data, CSV repository, CLI | Done |
| 2 | Technical indicators | Done |
| 3 | Feature engineering | Done |
| 4 | LightGBM training + registry | Done |
| 5 | Backtesting engine | Done |
| 5b | Multi-stock universes + panel training | Done |
| 6 | Paper trading | Planned |
| 7 | Live trading | Planned |
| 8 | LLM trade analysis | Planned |

## License

Apache License 2.0
