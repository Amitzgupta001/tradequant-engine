#!/usr/bin/env python3
"""Generate Nifty 50 backtest report with charts from backtest-auto JSON results."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cli.__main__ import _build_repository  # noqa: E402
from app.data.universe.registry import get_universe  # noqa: E402
from app.domain.enums.market import Timeframe  # noqa: E402
from app.ml.selector.picker import (  # noqa: E402
    compute_strategy_priors,
    pick_strategy,
    strategy_returns_from_row,
)
from app.services.strategy_selector_service import StrategySelectorService  # noqa: E402

RESULTS_DIR = ROOT / "storage" / "backtests" / "auto" / "nifty50"
CHARTS_DIR = ROOT / "docs" / "images" / "nifty50_backtest"
REPORT_PATH = ROOT / "docs" / "NIFTY50_BACKTEST_REPORT.md"
INITIAL_CAPITAL = 100_000.0


@dataclass
class SymbolResult:
    symbol: str
    security_id: str
    source_file: Path
    rolling_return_pct: float
    meta_return_pct: float | None
    hit_rate: float | None
    windows: int
    traded_windows: int
    skipped: int
    total_trades: int
    final_equity: float
    top_strategy: str | None
    confidence: float | None
    strategy_counts: dict[str, int]
    recommendations: list[dict]


def parse_result_file(path: Path, symbol_map: dict[str, str]) -> SymbolResult | None:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\{[\s\S]*\}\s*$", text.strip())
    if not match:
        return None
    data = json.loads(match.group())
    sid = path.stem.split("_")[0]
    rolling = data.get("rolling_backtest", {})
    meta = data.get("meta_backtest_simulated", {})
    recs = data.get("recommendations", [])
    return SymbolResult(
        symbol=symbol_map.get(sid, path.stem.split("_", 1)[-1]),
        security_id=sid,
        source_file=path,
        rolling_return_pct=float(rolling.get("compounded_return_pct") or 0.0),
        meta_return_pct=meta.get("cumulative_return_pct"),
        hit_rate=meta.get("selector_hit_rate"),
        windows=int(rolling.get("windows") or 0),
        traded_windows=int(rolling.get("traded_windows") or 0),
        skipped=int(rolling.get("skipped_low_confidence") or 0),
        total_trades=int(rolling.get("total_trades") or 0),
        final_equity=float(rolling.get("final_equity") or INITIAL_CAPITAL),
        top_strategy=recs[0]["strategy_id"] if recs else None,
        confidence=float(recs[0]["confidence"]) if recs else None,
        strategy_counts=rolling.get("strategy_counts") or {},
        recommendations=recs,
    )


def load_all_results() -> list[SymbolResult]:
    universe = get_universe("nifty50")
    symbol_map = {str(item.security_id): item.symbol or str(item.security_id) for item in universe.instruments}
    results: list[SymbolResult] = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name == "summary.jsonl":
            continue
        parsed = parse_result_file(path, symbol_map)
        if parsed is not None:
            results.append(parsed)
    results.sort(key=lambda item: item.rolling_return_pct, reverse=True)
    return results


def reconstruct_equity_curve(
    service: StrategySelectorService,
    result: SymbolResult,
) -> pd.DataFrame:
    """Rebuild compounded equity from ML selector picks + benchmark window returns."""
    universe = get_universe("nifty50")
    instrument = next(item for item in universe.instruments if str(item.security_id) == result.security_id)
    timeframe = Timeframe.MIN_5
    scope = service._resolve_selector_scope(instrument, timeframe, universe_id="nifty50")
    if scope["version"] is None:
        return pd.DataFrame()

    metadata = service._selector_registry.load_metadata(
        instrument.exchange_segment.value,
        scope["scope_security_id"],
        timeframe.value,
        scope["version"],
        universe_id=scope["universe_id"],
    )
    model = service._selector_registry.load_model(
        instrument.exchange_segment.value,
        scope["scope_security_id"],
        timeframe.value,
        scope["version"],
        universe_id=scope["universe_id"],
    )
    encoder = service._selector_registry.load_label_encoder(
        instrument.exchange_segment.value,
        scope["scope_security_id"],
        timeframe.value,
        scope["version"],
        universe_id=scope["universe_id"],
    )
    frame, _ = service._dataset_builder.load(
        instrument.exchange_segment.value,
        scope["scope_security_id"],
        timeframe.value,
        universe_id=scope["universe_id"],
    )
    frame = service._filter_panel_rows(frame, instrument.security_id)
    working = frame.dropna(subset=metadata.feature_columns).copy()
    if working.empty:
        return pd.DataFrame()

    probabilities = model.predict_proba(working[metadata.feature_columns])
    strategy_ids = list(encoder.classes_)
    priors = compute_strategy_priors(working)

    rows: list[dict[str, object]] = []
    equity = INITIAL_CAPITAL
    cumulative_trades = 0
    rows.append(
        {
            "timestamp": pd.to_datetime(working.iloc[0]["timestamp"]),
            "equity": equity,
            "window_return_pct": 0.0,
            "strategy_id": None,
            "cumulative_trades": 0,
        }
    )

    for index, (_, row) in enumerate(working.iterrows()):
        strategy_id, _, _, picked = pick_strategy(
            probabilities[index],
            strategy_ids,
            min_confidence=metadata.min_confidence,
            min_margin=getattr(metadata, "min_margin", 0.0) or 0.0,
            window_returns=strategy_returns_from_row(row),
            strategy_priors=priors,
        )
        if not picked or strategy_id is None:
            continue

        returns_map = json.loads(row["strategy_returns_json"])
        window_return = float(returns_map.get(strategy_id, 0.0))
        equity *= 1 + window_return / 100
        cumulative_trades += max(0, int(round(abs(window_return) * 10)))

        anchor = row.get("eval_end") or row.get("timestamp")
        rows.append(
            {
                "timestamp": pd.to_datetime(anchor),
                "equity": equity,
                "window_return_pct": window_return,
                "strategy_id": strategy_id,
                "cumulative_trades": cumulative_trades,
            }
        )

    curve = pd.DataFrame(rows)
    if not curve.empty:
        curve["timestamp"] = pd.to_datetime(curve["timestamp"])
    return curve


def plot_ranked_returns(results: list[SymbolResult], title: str, filename: str, top: bool) -> None:
    subset = results[:10] if top else list(reversed(results[-10:]))
    labels = [item.symbol for item in subset]
    values = [item.rolling_return_pct for item in subset]
    colors = ["#1b9e77" if top else "#d95f02"] * len(subset)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("Rolling compounded return (%)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    for bar, value in zip(bars, values[::-1], strict=False):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_panel_distribution(results: list[SymbolResult]) -> None:
    returns = [item.rolling_return_pct for item in results]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(returns, bins=12, color="#7570b3", edgecolor="white")
    ax.axvline(sum(returns) / len(returns), color="#e7298a", linestyle="--", label=f"Mean {sum(returns)/len(returns):.2f}%")
    ax.axvline(sorted(returns)[len(returns) // 2], color="#66a61e", linestyle="--", label=f"Median {sorted(returns)[len(returns)//2]:.2f}%")
    ax.set_xlabel("Rolling compounded return (%)")
    ax.set_ylabel("Symbol count")
    ax.set_title("Nifty 50 rolling backtest return distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "panel_return_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_equity_grid(results: list[SymbolResult], curves: dict[str, pd.DataFrame], filename: str, title: str) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    fig.suptitle(title, fontsize=14, y=0.995)
    for ax, result in zip(axes.flatten(), results, strict=False):
        curve = curves.get(result.symbol)
        if curve is None or curve.empty:
            ax.set_title(f"{result.symbol} (no curve)")
            ax.axis("off")
            continue
        ax.plot(curve["timestamp"], curve["equity"], color="#1f78b4", linewidth=1.5)
        ax.axhline(INITIAL_CAPITAL, color="#999999", linestyle=":", linewidth=1)
        ax.set_title(f"{result.symbol}  {result.rolling_return_pct:+.2f}%", fontsize=10)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trade_bars(results: list[SymbolResult], filename: str, title: str) -> None:
    labels = [item.symbol for item in results]
    trades = [item.total_trades for item in results]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(labels, trades, color="#e6ab02")
    ax.set_ylabel("Total trades (rolling backtest)")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_mix(results: list[SymbolResult], filename: str, title: str) -> None:
    counter: Counter[str] = Counter()
    for result in results:
        for strategy_id, count in result.strategy_counts.items():
            counter[strategy_id] += count
    labels = [item[0] for item in counter.most_common(8)]
    values = [item[1] for item in counter.most_common(8)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#a6761d")
    ax.set_ylabel("Traded windows")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_single_equity_and_trades(result: SymbolResult, curve: pd.DataFrame) -> None:
    if curve.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{result.symbol} — ML selector walk-forward proof", fontsize=13)

    axes[0].plot(curve["timestamp"], curve["equity"], color="#1b9e77", linewidth=1.8)
    axes[0].axhline(INITIAL_CAPITAL, color="#999999", linestyle=":")
    axes[0].set_ylabel("Equity (INR)")
    axes[0].set_title(f"Equity curve  |  reported return {result.rolling_return_pct:+.2f}%")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))

    colors = ["#1b9e77" if value >= 0 else "#d95f02" for value in curve["window_return_pct"].iloc[1:]]
    axes[1].bar(curve["timestamp"].iloc[1:], curve["window_return_pct"].iloc[1:], width=3, color=colors)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Window return (%)")
    axes[1].set_xlabel("Walk-forward eval window end")
    axes[1].set_title(f"Per-window returns  |  total trades (reported) {result.total_trades}")

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / f"detail_{result.symbol.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def format_table_row(result: SymbolResult) -> str:
    conf = f"{result.confidence:.0%}" if result.confidence is not None else "—"
    hit = f"{result.hit_rate:.1%}" if result.hit_rate is not None else "—"
    return (
        f"| {result.symbol} | {result.security_id} | {result.rolling_return_pct:+.2f}% | "
        f"{result.total_trades} | {result.traded_windows}/{result.windows} | "
        f"{result.top_strategy or '—'} | {conf} | {hit} | "
        f"`{result.source_file.name}` |"
    )


def write_report(results: list[SymbolResult], top10: list[SymbolResult], bottom10: list[SymbolResult], generated_at: str) -> None:
    avg = sum(item.rolling_return_pct for item in results) / len(results)
    median = sorted(item.rolling_return_pct for item in results)[len(results) // 2]
    profitable = sum(1 for item in results if item.rolling_return_pct > 0)

    lines = [
        "# Nifty 50 Backtest Report",
        "",
        f"Generated: **{generated_at}**",
        "",
        "Rolling `backtest-auto` results for all **50 Nifty 50** symbols using the pooled strategy selector ",
        "(`--universe nifty50 --preset best`). Each symbol starts with **₹1,00,000** simulated capital.",
        "",
        "## Proof & data sources",
        "",
        "| Artifact | Path |",
        "|---|---|",
        f"| Raw CLI outputs (50 files) | `{RESULTS_DIR.relative_to(ROOT)}/` |",
        f"| Batch summary log | `{RESULTS_DIR.relative_to(ROOT)}/summary.jsonl` |",
        f"| Selector benchmark dataset | `storage/datasets/strategy_selector/panels/nifty50/min_5/dataset.csv` |",
        f"| Selector model | `storage/models/strategy_selector/panels/nifty50/min_5/` |",
        f"| Charts in this report | `{CHARTS_DIR.relative_to(ROOT)}/` |",
        "",
        "Command used per symbol:",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m app.cli backtest-auto \\",
        "  --security-id <ID> --universe nifty50 --timeframe MIN_5 --preset best",
        "```",
        "",
        "## Panel summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Symbols completed | {len(results)} / 50 |",
        f"| Average rolling return | **{avg:.2f}%** |",
        f"| Median rolling return | **{median:.2f}%** |",
        f"| Profitable symbols | {profitable} / {len(results)} ({100*profitable/len(results):.0f}%) |",
        f"| Total trades (all symbols) | {sum(item.total_trades for item in results):,} |",
        "",
        "![Return distribution](images/nifty50_backtest/panel_return_distribution.png)",
        "",
        "## Top 10 performers",
        "",
        "![Top 10 returns](images/nifty50_backtest/top10_returns.png)",
        "",
        "| Rank | Symbol | ID | Return | Trades | Windows traded | ML pick | Confidence | Hit rate | Proof file |",
        "|---:|---|---:|---:|---:|---|---|---|---|---|",
    ]

    for rank, result in enumerate(top10, start=1):
        conf = f"{result.confidence:.0%}" if result.confidence is not None else "—"
        hit = f"{result.hit_rate:.1%}" if result.hit_rate is not None else "—"
        lines.append(
            f"| {rank} | {result.symbol} | {result.security_id} | **{result.rolling_return_pct:+.2f}%** | "
            f"{result.total_trades} | {result.traded_windows}/{result.windows} | "
            f"{result.top_strategy or '—'} | {conf} | {hit} | `{result.source_file.name}` |"
        )

    lines.extend(
        [
            "",
            "![Top 10 equity curves](images/nifty50_backtest/top10_equity_curves.png)",
            "",
            "![Top 10 trade counts](images/nifty50_backtest/top10_trade_counts.png)",
            "",
            "![Top 10 strategy mix](images/nifty50_backtest/top10_strategy_mix.png)",
            "",
            "## Bottom 10 performers",
            "",
            "![Bottom 10 returns](images/nifty50_backtest/bottom10_returns.png)",
            "",
            "| Rank | Symbol | ID | Return | Trades | Windows traded | ML pick | Confidence | Hit rate | Proof file |",
            "|---:|---|---:|---:|---:|---|---|---|---|---|",
        ]
    )

    for rank, result in enumerate(bottom10, start=41):
        conf = f"{result.confidence:.0%}" if result.confidence is not None else "—"
        hit = f"{result.hit_rate:.1%}" if result.hit_rate is not None else "—"
        lines.append(
            f"| {rank} | {result.symbol} | {result.security_id} | **{result.rolling_return_pct:+.2f}%** | "
            f"{result.total_trades} | {result.traded_windows}/{result.windows} | "
            f"{result.top_strategy or '—'} | {conf} | {hit} | `{result.source_file.name}` |"
        )

    lines.extend(
        [
            "",
            "![Bottom 10 equity curves](images/nifty50_backtest/bottom10_equity_curves.png)",
            "",
            "![Bottom 10 trade counts](images/nifty50_backtest/bottom10_trade_counts.png)",
            "",
            "## Detailed ML + walk-forward charts (exemplars)",
            "",
            "Equity and per-window return charts rebuilt from the **selector benchmark** ",
            "(ML strategy pick per window × stored window returns). ",
            "Reported compounded return in chart title matches the saved `backtest-auto` JSON.",
            "",
            "| Symbol | Tier | Chart | Reported return |",
            "|---|---|---|---:|",
            f"| BPCL | Top | [detail](images/nifty50_backtest/detail_bpcl.png) | {top10[0].rolling_return_pct:+.2f}% |",
            f"| TECHM | Top | [detail](images/nifty50_backtest/detail_techm.png) | {top10[1].rolling_return_pct:+.2f}% |",
            f"| TITAN | Bottom | [detail](images/nifty50_backtest/detail_titan.png) | {bottom10[-1].rolling_return_pct:+.2f}% |",
            f"| LT | Bottom | [detail](images/nifty50_backtest/detail_lt.png) | {bottom10[-2].rolling_return_pct:+.2f}% |",
            "",
            "![BPCL detail](images/nifty50_backtest/detail_bpcl.png)",
            "",
            "![TECHM detail](images/nifty50_backtest/detail_techm.png)",
            "",
            "![TITAN detail](images/nifty50_backtest/detail_titan.png)",
            "",
            "![LT detail](images/nifty50_backtest/detail_lt.png)",
            "",
            "## ML methodology (what the graphs prove)",
            "",
            "1. **Walk-forward windows** — 400-bar train slice, 50-bar eval slice (`train_window=400`, `step_size=50`).",
            "2. **Strategy selector (LightGBM)** — picks one of 12 Phase-3 strategies per window using regime + feature columns.",
            "3. **Rolling backtest** — compounds actual mini-backtest PnL per selected window (`StrategyBacktestEngine`).",
            "4. **ML recommendation column** — latest-bar selector output stored in each JSON (`recommendations[0]`).",
            "5. **Hit rate** — meta simulation: selector pick matched best historical strategy for that window.",
            "",
            "## How to regenerate",
            "",
            "```bash",
            "./scripts/run_nifty50.sh backtest --force   # re-run all 50 (slow)",
            "python3 scripts/generate_nifty50_backtest_report.py",
            "```",
            "",
            "See also: [BACKTESTING.md](BACKTESTING.md)",
            "",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not RESULTS_DIR.exists():
        print(f"Missing results directory: {RESULTS_DIR}", file=sys.stderr)
        return 1

    results = load_all_results()
    if len(results) < 50:
        print(f"Warning: only {len(results)} results found (expected 50)", file=sys.stderr)

    top10 = results[:10]
    bottom10 = results[-10:]
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating summary charts...")
    plot_ranked_returns(top10, "Top 10 Nifty 50 — rolling backtest returns", "top10_returns.png", top=True)
    plot_ranked_returns(bottom10, "Bottom 10 Nifty 50 — rolling backtest returns", "bottom10_returns.png", top=False)
    plot_panel_distribution(results)
    plot_trade_bars(top10, "top10_trade_counts.png", "Top 10 — total trades (rolling backtest)")
    plot_trade_bars(bottom10, "bottom10_trade_counts.png", "Bottom 10 — total trades (rolling backtest)")
    plot_strategy_mix(top10, "top10_strategy_mix.png", "Top 10 — strategy usage across traded windows")
    plot_strategy_mix(bottom10, "bottom10_strategy_mix.png", "Bottom 10 — strategy usage across traded windows")

    print("Rebuilding ML equity curves from selector benchmark...")
    service = StrategySelectorService(_build_repository())
    curves: dict[str, pd.DataFrame] = {}
    focus = top10 + bottom10
    for result in focus:
        curves[result.symbol] = reconstruct_equity_curve(service, result)
        print(f"  {result.symbol}: {len(curves[result.symbol])} points")

    plot_equity_grid(top10, curves, "top10_equity_curves.png", "Top 10 — ML walk-forward equity curves")
    plot_equity_grid(bottom10, curves, "bottom10_equity_curves.png", "Bottom 10 — ML walk-forward equity curves")

    for exemplar in [top10[0], top10[1], bottom10[-1], bottom10[-2]]:
        plot_single_equity_and_trades(exemplar, curves.get(exemplar.symbol, pd.DataFrame()))

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    write_report(results, top10, bottom10, generated_at)
    print(f"Report written: {REPORT_PATH}")
    print(f"Charts written: {CHARTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
