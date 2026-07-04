#!/usr/bin/env bash
# Start paper trading for Nifty 50 with dashboard.
#
# Terminal 1 — API + dashboard:
#   ./scripts/run_paper_nifty50.sh serve
#
# Terminal 2 — paper trading loop (Monday market hours, WebSocket + 5m bar close):
#   ./scripts/run_paper_nifty50.sh run
#
# One-shot:
#   ./scripts/run_paper_nifty50.sh start
#   ./scripts/run_paper_nifty50.sh tick

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=.

UNIVERSE="${UNIVERSE:-nifty50}"
CAPITAL="${CAPITAL:-1000000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PORT="${PORT:-4000}"

cmd="${1:-help}"
shift || true

case "$cmd" in
  serve)
    echo "Dashboard: http://127.0.0.1:${PORT}/dashboard"
    python3 -m uvicorn app.api.main:create_app --factory --reload --host 0.0.0.0 --port "$PORT"
    ;;
  start)
    python3 -m app.cli paper-trade \
      --universe "$UNIVERSE" \
      --selector-universe "$UNIVERSE" \
      --timeframe MIN_5 \
      --capital "$CAPITAL" \
      --start
    ;;
  tick)
    python3 -m app.cli paper-trade \
      --universe "$UNIVERSE" \
      --selector-universe "$UNIVERSE" \
      --timeframe MIN_5 \
      --tick \
      "$@"
    ;;
  run)
    python3 -m app.cli paper-trade \
      --universe "$UNIVERSE" \
      --selector-universe "$UNIVERSE" \
      --timeframe MIN_5 \
      --capital "$CAPITAL" \
      --mode live \
      --run \
      --poll-seconds "$POLL_SECONDS" \
      "$@"
    ;;
  status)
    python3 -m app.cli paper-trade --universe "$UNIVERSE"
    ;;
  stop)
    python3 -m app.cli paper-trade --stop
    ;;
  help|*)
    sed -n '3,12p' "$0" | tr -d '#'
    echo
    echo "Commands: serve | start | tick | run | status | stop"
    ;;
esac
