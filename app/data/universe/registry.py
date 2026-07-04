"""Universe registry — load predefined stock lists for batch training."""

import json
from pathlib import Path

from app.domain.enums.market import ExchangeSegment, InstrumentType
from app.domain.instrument import Instrument

_UNIVERSE_DIR = Path(__file__).parent
_KNOWN_UNIVERSES = {
    "nifty50": "nifty50.json",
    "nifty500": "nifty500.json",
}


class Universe:
    """Named collection of tradable instruments."""

    def __init__(self, universe_id: str, name: str, instruments: list[Instrument]) -> None:
        self.id = universe_id
        self.name = name
        self.instruments = instruments


def get_universe(universe_id: str) -> Universe:
    """Load a universe by id (e.g. 'nifty50')."""
    filename = _KNOWN_UNIVERSES.get(universe_id.lower())
    if filename is None:
        known = ", ".join(sorted(_KNOWN_UNIVERSES))
        msg = f"Unknown universe '{universe_id}'. Known: {known}"
        raise ValueError(msg)

    path = _UNIVERSE_DIR / filename
    if not path.exists():
        msg = f"Universe file not found: {path}"
        raise FileNotFoundError(msg)

    payload = json.loads(path.read_text(encoding="utf-8"))
    instruments = [
        Instrument(
            security_id=entry["security_id"],
            exchange_segment=ExchangeSegment.NSE_EQ,
            instrument_type=InstrumentType.EQUITY,
            symbol=entry.get("symbol"),
        )
        for entry in payload["instruments"]
    ]
    return Universe(
        universe_id=payload.get("id", universe_id),
        name=payload.get("name", universe_id),
        instruments=instruments,
    )


def list_universes() -> list[str]:
    """Return available universe ids."""
    return sorted(_KNOWN_UNIVERSES.keys())
