"""Backtest report persistence."""

import csv
import json
from pathlib import Path

from loguru import logger

from app.domain.backtest import BacktestResult, PanelBacktestResult
from app.domain.enums.market import Timeframe
from app.domain.instrument import Instrument


class BacktestReportStore:
    """Save backtest results to storage/backtests/."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def report_dir(self, instrument: Instrument, timeframe: Timeframe) -> Path:
        """Directory for backtest artifacts."""
        return (
            self._base_path
            / instrument.exchange_segment.value
            / instrument.security_id
            / timeframe.value.lower()
        )

    def save(self, instrument: Instrument, timeframe: Timeframe, result: BacktestResult) -> tuple[Path, Path]:
        """Persist JSON summary and equity curve CSV."""
        directory = self.report_dir(instrument, timeframe)
        directory.mkdir(parents=True, exist_ok=True)

        summary_path = directory / "summary.json"
        equity_path = directory / "equity_curve.csv"

        summary_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

        with equity_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["timestamp", "equity", "cash", "position_value", "drawdown_pct"],
            )
            writer.writeheader()
            for point in result.equity_curve:
                writer.writerow(point.model_dump(mode="json"))

        logger.info("Saved backtest report to {}", directory)
        return summary_path, equity_path

    def panel_report_dir(self, universe_id: str, timeframe: Timeframe) -> Path:
        """Directory for panel backtest artifacts."""
        return self._base_path / "panels" / universe_id / timeframe.value.lower()

    def save_panel(
        self,
        universe_id: str,
        timeframe: Timeframe,
        result: PanelBacktestResult,
    ) -> Path:
        """Persist aggregated panel backtest summary."""
        directory = self.panel_report_dir(universe_id, timeframe)
        directory.mkdir(parents=True, exist_ok=True)
        summary_path = directory / "summary.json"
        summary_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        logger.info("Saved panel backtest report to {}", directory)
        return summary_path
