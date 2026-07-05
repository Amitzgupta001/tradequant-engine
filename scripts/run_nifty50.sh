#!/usr/bin/env bash
# Nifty 50 strategy-selector pipeline: download → train → backtest.
#
# Usage:
#   ./scripts/run_nifty50.sh download [--skip-existing]
#   ./scripts/run_nifty50.sh train [--rebuild-dataset] [--skip-strategy-training]
#   ./scripts/run_nifty50.sh backtest [--force]
#   ./scripts/run_nifty50.sh recommend [--security-id ID]
#   ./scripts/run_nifty50.sh all [--skip-existing] [--rebuild-dataset]
#
# Logs: storage/logs/nifty50_<step>.log

set -euo pipefail

UNIVERSE="nifty50"
TIMEFRAME="MIN_5"
PRESET="best"
SLEEP_SECONDS="${SLEEP_SECONDS:-5}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=.

LOG_DIR="$ROOT/storage/logs"
BACKTEST_DIR="$ROOT/storage/backtests/auto/nifty50"
UNIVERSE_JSON="$ROOT/app/data/universe/nifty50.json"

mkdir -p "$LOG_DIR" "$BACKTEST_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_cli() {
  log ">> python3 -m app.cli $*"
  python3 -m app.cli "$@"
}

cmd_download() {
  local extra=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --skip-existing) extra+=(--skip-existing) ;;
      *) echo "Unknown download flag: $1" >&2; exit 1 ;;
    esac
    shift
  done

  log "Step 1/3: batch-download ($UNIVERSE)"
  run_cli batch-download \
    --universe "$UNIVERSE" \
    --preset "$PRESET" \
    --sleep-seconds "$SLEEP_SECONDS" \
    "${extra[@]}" \
    2>&1 | tee "$LOG_DIR/nifty50_download.log"
}

cmd_train() {
  local extra=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --rebuild-dataset) extra+=(--rebuild-dataset) ;;
      --skip-strategy-training) extra+=(--skip-strategy-training) ;;
      *) echo "Unknown train flag: $1" >&2; exit 1 ;;
    esac
    shift
  done

  log "Step 2/3: train-strategy-selector ($UNIVERSE)"
  run_cli train-strategy-selector \
    --universe "$UNIVERSE" \
    --timeframe "$TIMEFRAME" \
    --preset "$PRESET" \
    "${extra[@]}" \
    2>&1 | tee "$LOG_DIR/nifty50_train_selector.log"
}

cmd_backtest() {
  local force=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --force) force=1 ;;
      *) echo "Unknown backtest flag: $1" >&2; exit 1 ;;
    esac
    shift
  done

  log "Step 3/3: backtest-auto for all $UNIVERSE symbols"
  python3 - "$UNIVERSE_JSON" "$BACKTEST_DIR" "$UNIVERSE" "$TIMEFRAME" "$PRESET" "$force" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

universe_json = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
universe_id = sys.argv[3]
timeframe = sys.argv[4]
preset = sys.argv[5]
force = sys.argv[6] == "1"

data = json.loads(universe_json.read_text())
summary_path = out_dir / "summary.jsonl"
instruments = data["instruments"]

ok = 0
failed = 0
skipped = 0

for item in instruments:
    sid = str(item["security_id"])
    sym = item["symbol"]
    out = out_dir / f"{sid}_{sym}.json"

    if out.exists() and not force:
        print(f"skip {sym} ({sid})", flush=True)
        skipped += 1
        continue

    print(f"backtest {sym} ({sid})...", flush=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.cli",
            "backtest-auto",
            "--security-id",
            sid,
            "--universe",
            universe_id,
            "--timeframe",
            timeframe,
            "--preset",
            preset,
        ],
        env={**dict(__import__("os").environ), "PYTHONPATH": "."},
        capture_output=True,
        text=True,
    )

    payload = result.stdout if result.returncode == 0 else (result.stderr or result.stdout)
    out.write_text(payload or "unknown error")

    row = {
        "security_id": sid,
        "symbol": sym,
        "ok": result.returncode == 0,
        "output": str(out),
    }
    with summary_path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")

    if result.returncode == 0:
        ok += 1
    else:
        failed += 1
        print(result.stderr, file=sys.stderr)

print(f"\nDone: ok={ok} failed={failed} skipped={skipped}", flush=True)
print(f"Results: {out_dir}", flush=True)
print(f"Summary: {summary_path}", flush=True)
PY
}

cmd_recommend() {
  local security_id=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --security-id)
        security_id="$2"
        shift 2
        ;;
      *) echo "Unknown recommend flag: $1" >&2; exit 1 ;;
    esac
  done

  if [[ -z "$security_id" ]]; then
    log "Recommendations for all $UNIVERSE symbols"
    python3 - "$UNIVERSE_JSON" "$UNIVERSE" "$TIMEFRAME" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

for item in json.loads(Path(sys.argv[1]).read_text())["instruments"]:
    sid = item["security_id"]
    sym = item["symbol"]
    print(f"\n=== {sym} ({sid}) ===", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "app.cli",
            "recommend-strategy",
            "--security-id",
            str(sid),
            "--universe",
            sys.argv[2],
            "--timeframe",
            sys.argv[3],
        ],
        env={**dict(__import__("os").environ), "PYTHONPATH": "."},
    )
PY
    return
  fi

  run_cli recommend-strategy \
    --security-id "$security_id" \
    --universe "$UNIVERSE" \
    --timeframe "$TIMEFRAME"
}

cmd_all() {
  local download_flags=()
  local train_flags=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --skip-existing) download_flags+=(--skip-existing) ;;
      --rebuild-dataset) train_flags+=(--rebuild-dataset) ;;
      --skip-strategy-training) train_flags+=(--skip-strategy-training) ;;
      *) echo "Unknown all flag: $1" >&2; exit 1 ;;
    esac
    shift
  done

  cmd_download "${download_flags[@]}"
  cmd_train "${train_flags[@]}"
  cmd_backtest
}

usage() {
  sed -n '3,10p' "$0" | tr -d '#'
  echo
  echo "Environment:"
  echo "  SLEEP_SECONDS  Pause between batch-download symbols (default: 5)"
}

main() {
  local command="${1:-}"
  shift || true

  case "$command" in
    download) cmd_download "$@" ;;
    train) cmd_train "$@" ;;
    backtest) cmd_backtest "$@" ;;
    recommend) cmd_recommend "$@" ;;
    all) cmd_all "$@" ;;
    -h|--help|help|"") usage ;;
    *)
      echo "Unknown command: $command" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
