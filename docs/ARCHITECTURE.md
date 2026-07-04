# Architecture

## Layers

| Layer | Package | Responsibility |
|---|---|---|
| API | `app/api/v1/` | HTTP endpoints, versioned |
| CLI | `app/cli/` | Batch jobs without HTTP |
| Services | `app/services/` | Orchestration |
| Domain | `app/domain/` | Entities, value objects, enums |
| Data | `app/data/` | Providers, repositories, universes |
| Brokers | `app/brokers/` | External broker integrations |
| Core | `app/core/` | Config, logging, events |
| Cache | `app/cache/` | In-memory (Phase 1), Redis later |
| Strategy | `app/strategy/` | Trading strategies and presets |
| Risk | `app/risk/` | Position sizing, stops, exposure |
| ML | `app/ml/` | Full ML lifecycle |
| Indicators | `app/indicators/` | Technical indicators (Phase 2) |

## Sequence diagrams

### Single-stock training

```mermaid
sequenceDiagram
    actor User
    participant CLI as app/cli
    participant TS as TrainingService
    participant HDS as HistoricalDataService
    participant Dhan as DhanClient
    participant Raw as CSVHistoricalRepository
    participant FS as FeatureService
    participant Trainer as LightGBMTrainer
    participant Registry as ModelRegistry

    User->>CLI: train --security-id 1333 --preset best
    CLI->>TS: train_with_history(instrument, MIN_5, config)
    TS->>HDS: download_range_and_store(365 days)
    loop 90-day API chunks
        HDS->>Dhan: get_historical_data(MIN_5)
        Dhan-->>HDS: candles
    end
    HDS->>Raw: save → storage/raw/NSE_EQ/1333/min_5.csv
    TS->>FS: build_and_store(instrument, MIN_5)
    FS->>Raw: load candles
    FS->>FS: indicators + FeatureEngine
    FS->>FS: save → storage/features/.../min_5_features.csv
    TS->>Trainer: train(instrument, features, config)
    Trainer->>Trainer: setup filter + LightGBM fit
    Trainer->>Registry: save model + metadata
    Registry-->>CLI: storage/models/NSE_EQ/1333/min_5/
    CLI-->>User: metrics JSON
```

### Batch download (Nifty 500)

```mermaid
sequenceDiagram
    actor User
    participant CLI as app/cli
    participant Batch as BatchDataService
    participant TS as TrainingService
    participant HDS as HistoricalDataService
    participant Dhan as DhanClient
    participant Raw as CSVHistoricalRepository
    participant FS as FeatureService

    User->>CLI: batch-download --universe nifty500 --preset best
    CLI->>Batch: download_universe(nifty500, MIN_5, 365d)
    loop each symbol (466)
        alt --skip-existing and features exist
            Batch->>Batch: skip symbol
        else download
            Batch->>TS: prepare_data(instrument, start, end)
            TS->>HDS: download_range_and_store
            HDS->>Dhan: intraday chunks (90d each)
            HDS->>Raw: save raw CSV
            TS->>FS: build_and_store features
            TS->>TS: clear_cache + gc
            Batch->>Batch: sleep BATCH_SLEEP_SECONDS
        end
    end
    Batch-->>CLI: per-symbol results JSON
    CLI-->>User: symbols_ok / failed counts
```

### Panel training (Nifty 500)

```mermaid
sequenceDiagram
    actor User
    participant CLI as app/cli
    participant TS as TrainingService
    participant Batch as BatchDataService
    participant Panel as panel.build_panel_training_frame
    participant Trainer as LightGBMTrainer
    participant Registry as ModelRegistry

    User->>CLI: train --universe nifty500 --preset best --skip-download
    CLI->>TS: train_panel_from_features(nifty500, MIN_5)
    loop each symbol with feature CSV
        Panel->>Panel: read CSV → setup filter only
        Panel->>Panel: append rows (low memory)
    end
    Panel-->>TS: merged dataframe (~2.5M setup rows)
    TS->>Trainer: train_panel(universe_id, frame, config)
    Trainer->>Trainer: time-split + LightGBM fit + threshold tune
    Trainer->>Registry: save panel model
    Registry-->>CLI: storage/models/panels/nifty500/min_5/
    CLI-->>User: metrics JSON
```

### Panel backtest (Nifty 500)

```mermaid
sequenceDiagram
    actor User
    participant CLI as app/cli
    participant BS as BacktestService
    participant Pred as LightGBMPredictor
    participant Registry as ModelRegistry
    participant Engine as BacktestEngine
    participant Strategy as MLStrategy
    participant Report as BacktestReportStore

    User->>CLI: backtest --universe nifty500 --preset best
    CLI->>BS: run_panel(nifty500, MIN_5, config)
    BS->>Pred: load_panel(nifty500, MIN_5)
    Pred->>Registry: load panels/nifty500/min_5/model.txt
    Registry-->>Pred: threshold = 0.31
    loop each symbol with raw + features
        BS->>BS: load candles + features
        BS->>Strategy: MLStrategy(predictor, threshold)
        loop each bar
            Strategy->>Pred: predict(features[i])
            Pred-->>Strategy: probability
            Strategy->>Engine: BUY / HOLD signal
            Engine->>Engine: stops, trailing, max_hold
        end
        Engine-->>BS: trades + equity curve
    end
    BS->>BS: aggregate mean return, win rate, profit factor
    BS->>Report: save panels/nifty500/min_5/summary.json
    Report-->>CLI: panel summary path
    CLI-->>User: aggregated metrics JSON
```

### Single-stock backtest

```mermaid
sequenceDiagram
    actor User
    participant CLI as app/cli
    participant BS as BacktestService
    participant Pred as LightGBMPredictor
    participant Engine as BacktestEngine
    participant Strategy as MLStrategy

    User->>CLI: backtest --security-id 1333 --preset best
    CLI->>BS: run(instrument, MIN_5, config)
    BS->>Pred: load(NSE_EQ, 1333, MIN_5)
    BS->>BS: load raw candles + feature CSV
    BS->>Strategy: MLStrategy(predictor, model threshold)
    BS->>Engine: run(candles, features, strategy)
    loop bar-by-bar
        Engine->>Strategy: generate_signal(feature)
        Strategy->>Pred: predict()
        alt probability >= threshold
            Engine->>Engine: enter at next open
        end
        Engine->>Engine: stop loss / trailing / max hold
    end
    Engine-->>BS: BacktestResult
    BS-->>CLI: storage/backtests/NSE_EQ/1333/min_5/
    CLI-->>User: P&L metrics JSON
```

## Data flow

### Single instrument

```text
Dhan API → DhanClient → HistoricalDataProvider → HistoricalDataService
                                                         ↓
                                              CSVHistoricalRepository
                                                         ↓
                                              storage/raw/{segment}/{id}/
                                                         ↓
FeatureDatasetBuilder → FeatureEngine → CSVFeatureRepository
                                                         ↓
                                              storage/features/...
                                                         ↓
LightGBMTrainer → ModelRegistry → storage/models/{segment}/{id}/
```

### Panel (multi-stock)

```text
Universe (nifty50 / nifty500)
       ↓
BatchDataService ──loop──► prepare_data() per symbol (with sleep between symbols)
       ↓
load_panel_features() ──concat──► LightGBMTrainer.train_panel()
       ↓
storage/models/panels/{universe_id}/{timeframe}/
```

## Key components

| Component | Path | Role |
|---|---|---|
| Universe registry | `app/data/universe/registry.py` | Load Nifty 50/500 instrument lists |
| Batch download | `app/services/batch_data_service.py` | Loop download + features with rate pause |
| Panel loader | `app/ml/datasets/panel.py` | Merge feature CSVs across symbols |
| Intraday chunking | `app/data/providers/historical_data_provider.py` | 90-day API chunks |
| Strategy | `app/strategies/` | 12 rule-based strategies + features |
| Strategy ML | `app/ml/trainer/strategy_trainer.py` | Per-strategy LightGBM |
| Selector | `app/ml/trainer/strategy_selector_trainer.py` | Multiclass meta-model |
| Hybrid picker | `app/ml/selector/picker.py` | Model shortlist + backtest score |
| Paper trading | `app/services/paper_trading_service.py` | Live forward test orchestration |
| Live feed | `app/brokers/dhan/live_feed.py` | Dhan WebSocket ticker (read-only) |
| Paper safety | `app/paper/safety.py` | Blocks broker orders in paper mode |
| Best preset | `app/strategy/presets.py` | 1:3 R:R, T1/T2/T3 backtest config |
| Model registry | `app/ml/registry/model_registry.py` | Per-stock and panel model paths |

## Event pipeline (future)

```text
Raw OHLCV → Indicators → Features → ML Train → Inference → Strategy → Risk → Execution
```

Events are published via `app/core/events.py` (`NEW_CANDLE`, `FEATURES_READY`, etc.).

## Storage layout

```text
storage/
  raw/         OHLCV from brokers
  processed/   Indicator snapshots
  features/    Engineered feature CSVs
  models/
    NSE_EQ/{security_id}/{timeframe}/   # per-stock swing models
    panels/{universe_id}/{timeframe}/   # panel models
    {strategy_id}/v{N}/                 # per-strategy Phase 3 models
    strategy_selector/                  # meta-model (per-stock or panels/)
  datasets/
    {strategy_id}/...                   # strategy ML datasets
    strategy_selector/...               # walk-forward benchmark datasets
  backtests/   Backtest results (+ strategies/ subdir)
  paper/       Paper trading sessions, trades, state
  logs/        Long-running job logs (optional)
```

## Repository pattern

Swap storage without changing business logic:

- `HistoricalRepository` (protocol) → `CSVHistoricalRepository`
- `FeatureRepository` (protocol) → `CSVFeatureRepository`
- Future: Parquet, Postgres

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `DHAN_CLIENT_ID` | — | Dhan API client ID |
| `DHAN_ACCESS_TOKEN` | — | Dhan access token |
| `STORAGE_PATH` | `storage` | Local data directory |
| `BATCH_SLEEP_SECONDS` | `3` | Pause between symbols in batch runs |
| `CACHE_TTL_SECONDS` | `300` | In-memory cache TTL |

## Paper trading (live forward test)

See [PAPER_TRADING.md](PAPER_TRADING.md) for the full guide. Summary:

```mermaid
sequenceDiagram
    participant Runner as PaperLiveRunner
    participant WS as Dhan WebSocket
    participant PTS as PaperTradingService
    participant Dhan as Dhan REST
    participant Engine as PaperInstrumentEngine
    participant Store as storage/paper/

    Runner->>WS: subscribe tickers (LTP only)
    loop each tick
        WS-->>Runner: LTP
        Runner->>PTS: update_live_price (dashboard mark)
    end
    loop each 5m bar close (IST)
        Runner->>PTS: tick()
        PTS->>Dhan: intraday_minute_data (read only)
        PTS->>Engine: new bars → simulated buy/sell
        Engine->>Store: trades.jsonl
    end
    Note over Engine,Dhan: place_order never called
```

## Documentation

- [CLI Reference](CLI.md)
- [Paper Trading](PAPER_TRADING.md)
- [Strategy Selector](STRATEGY_SELECTOR.md)
- [ML Training](ML_TRAINING.md)
- [Stock Universes](UNIVERSES.md)
