"""CSV implementation of historical data repository."""

import csv
from datetime import datetime
from pathlib import Path

from loguru import logger

from app.domain.candle import Candle
from app.domain.enums.market import ExchangeSegment, Timeframe
from app.domain.historical import HistoricalResponse
from app.domain.instrument import Instrument

CSV_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_interest",
]


class CSVHistoricalRepository:
    """Persist and load historical market data as CSV files under storage/raw/."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def build_path(
        self,
        exchange_segment: ExchangeSegment,
        security_id: str,
        timeframe: Timeframe,
    ) -> Path:
        """Build the CSV file path for an instrument and timeframe."""
        return (
            self._base_path
            / "raw"
            / exchange_segment.value
            / security_id
            / f"{timeframe.value.lower()}.csv"
        )

    def save(self, response: HistoricalResponse, overwrite: bool = True) -> Path:
        """Write historical candles to CSV."""
        path = self.build_path(
            response.instrument.exchange_segment,
            response.instrument.security_id,
            response.timeframe,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite:
            logger.info("Skipping existing CSV file at {}", path)
            return path

        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for candle in response.candles:
                writer.writerow(
                    {
                        "timestamp": candle.timestamp.isoformat(),
                        "open": candle.open,
                        "high": candle.high,
                        "low": candle.low,
                        "close": candle.close,
                        "volume": candle.volume,
                        "open_interest": candle.open_interest,
                    }
                )

        logger.info("Saved {} candles to {}", len(response.candles), path)
        return path

    def load(self, instrument: Instrument, timeframe: Timeframe) -> HistoricalResponse:
        """Load historical candles from CSV."""
        path = self.build_path(
            instrument.exchange_segment,
            instrument.security_id,
            timeframe,
        )
        if not path.exists():
            msg = f"CSV file not found: {path}"
            raise FileNotFoundError(msg)

        candles: list[Candle] = []
        with path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                open_interest = row.get("open_interest")
                candles.append(
                    Candle(
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=int(row["volume"]),
                        open_interest=int(open_interest) if open_interest else None,
                    )
                )

        return HistoricalResponse(
            instrument=instrument,
            timeframe=timeframe,
            candles=candles,
            source="csv",
        )
