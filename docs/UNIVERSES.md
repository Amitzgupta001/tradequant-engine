# Stock Universes

TradeQuant supports batch download and panel training across predefined index universes. Each universe is a JSON file under `app/data/universe/` with Dhan `security_id`s resolved from the [Dhan scrip master](https://images.dhan.co/api-data/api-scrip-master.csv).

## Available universes

| ID | Name | Symbols | File |
|---|---|---|---|
| `nifty50` | Nifty 50 | 50 | `app/data/universe/nifty50.json` |
| `nifty500` | Nifty 500 | ~466 | `app/data/universe/nifty500.json` |

The Nifty 500 list is sourced from the NSE index CSV (via a public mirror) and matched to Dhan IDs. Delisted or merged names (e.g. DHFL, JETAIRWAYS) are omitted; renamed symbols are mapped via aliases (e.g. TATAMOTORS → TMPV, ZOMATO → ETERNAL, HDFC → HDFCBANK).

## Usage

```bash
# Download all symbols
PYTHONPATH=. python3 -m app.cli batch-download --universe nifty50 --preset best

# Resume a large run (skip symbols that already have features)
PYTHONPATH=. python3 -m app.cli batch-download --universe nifty500 --preset best --skip-existing

# Train a pooled panel model
PYTHONPATH=. python3 -m app.cli train --universe nifty500 --preset best --skip-existing
```

## Storage layout

Per-symbol data (same as single-stock flow):

```text
storage/raw/NSE_EQ/{security_id}/min_5.csv
storage/features/NSE_EQ/{security_id}/min_5_features.csv
```

Panel model (shared across universe):

```text
storage/models/panels/{universe_id}/min_5/
  model.txt
  metadata.json
```

Panel metadata includes `universe_id`, `constituent_count`, and the tuned `prediction_threshold`.

## Rate limiting and memory

Batch runs pause between symbols to avoid overloading the Dhan API or your machine:

- **Environment:** `BATCH_SLEEP_SECONDS=3` (default 3 seconds)
- **CLI override:** `--sleep-seconds 5`

After each symbol, cached candles are cleared and memory is released. Skipped symbols (`--skip-existing`) do not sleep or download.

**Recommended for Nifty 500:** run download and training as separate steps (training loads all setup rows into RAM):

```bash
# Step 1: download only (resume-friendly)
PYTHONPATH=. python3 -m app.cli batch-download --universe nifty500 --preset best --skip-existing --sleep-seconds 5

# Step 2: train when download is far enough along
PYTHONPATH=. python3 -m app.cli train --universe nifty500 --preset best --skip-download
```

Panel training reads feature CSVs incrementally and keeps only setup-filtered rows (~500 MB for full Nifty 500 vs 10+ GB with the old approach).

## Adding a universe

1. Create `app/data/universe/my_universe.json`:

```json
{
  "id": "my_universe",
  "name": "My Universe",
  "instruments": [
    {"symbol": "HDFCBANK", "trading_symbol": "HDFCBANK", "security_id": "1333"}
  ]
}
```

2. Register it in `app/data/universe/registry.py`:

```python
_KNOWN_UNIVERSES = {
    "nifty50": "nifty50.json",
    "nifty500": "nifty500.json",
    "my_universe": "my_universe.json",
}
```

3. Use `--universe my_universe` in CLI commands.
