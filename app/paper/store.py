"""Persist paper trading sessions, instrument state, and trade logs."""

import json
import uuid
from datetime import datetime
from pathlib import Path

from app.paper.models import (
    PaperInstrumentState,
    PaperSession,
    PaperSessionStatus,
    PaperTradeRecord,
)


class PaperSessionStore:
    """Filesystem store for paper trading state."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path
        self._base_path.mkdir(parents=True, exist_ok=True)

    @property
    def active_pointer(self) -> Path:
        return self._base_path / "active_session.json"

    def session_dir(self, session_id: str) -> Path:
        return self._base_path / "sessions" / session_id

    def set_active_session(self, session_id: str) -> None:
        """Point the active session marker at session_id."""
        self.active_pointer.write_text(json.dumps({"session_id": session_id}, indent=2))

    def get_active_session_id(self) -> str | None:
        """Return the active session id if configured."""
        if not self.active_pointer.exists():
            return None
        payload = json.loads(self.active_pointer.read_text())
        return payload.get("session_id")

    def save_session(self, session: PaperSession) -> Path:
        """Persist session metadata."""
        directory = self.session_dir(session.session_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "session.json"
        path.write_text(session.model_dump_json(indent=2))
        return path

    def load_session(self, session_id: str) -> PaperSession:
        """Load session metadata."""
        path = self.session_dir(session_id) / "session.json"
        return PaperSession.model_validate_json(path.read_text())

    def load_active_session(self) -> PaperSession | None:
        """Load the active session if present."""
        session_id = self.get_active_session_id()
        if session_id is None:
            return None
        return self.load_session(session_id)

    def save_instrument_state(self, session_id: str, state: PaperInstrumentState) -> None:
        """Persist per-symbol runtime state."""
        directory = self.session_dir(session_id) / "instruments"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{state.security_id}.json"
        path.write_text(state.model_dump_json(indent=2))

    def load_instrument_state(self, session_id: str, security_id: str) -> PaperInstrumentState | None:
        """Load per-symbol runtime state."""
        path = self.session_dir(session_id) / "instruments" / f"{security_id}.json"
        if not path.exists():
            return None
        return PaperInstrumentState.model_validate_json(path.read_text())

    def list_instrument_states(self, session_id: str) -> list[PaperInstrumentState]:
        """Load all instrument states for a session."""
        directory = self.session_dir(session_id) / "instruments"
        if not directory.exists():
            return []
        states: list[PaperInstrumentState] = []
        for path in sorted(directory.glob("*.json")):
            states.append(PaperInstrumentState.model_validate_json(path.read_text()))
        return states

    def append_trade(self, session_id: str, trade: PaperTradeRecord) -> None:
        """Append a closed trade to the session log."""
        directory = self.session_dir(session_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "trades.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(trade.model_dump_json())
            handle.write("\n")

    def load_trades(self, session_id: str, limit: int | None = None) -> list[PaperTradeRecord]:
        """Load closed trades for a session."""
        path = self.session_dir(session_id) / "trades.jsonl"
        if not path.exists():
            return []
        trades: list[PaperTradeRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            trades.append(PaperTradeRecord.model_validate_json(line))
        if limit is not None:
            return trades[-limit:]
        return trades

    def create_session(
        self,
        *,
        universe_id: str,
        timeframe: str,
        initial_capital: float,
        capital_per_symbol: float,
        instrument_ids: list[str],
        selector_universe_id: str | None = None,
        session_id: str | None = None,
    ) -> PaperSession:
        """Create and activate a new paper session."""
        resolved_id = session_id or datetime.now().strftime("paper_%Y%m%d_%H%M%S")
        session = PaperSession(
            session_id=resolved_id,
            universe_id=universe_id,
            timeframe=timeframe,
            initial_capital=initial_capital,
            capital_per_symbol=capital_per_symbol,
            status=PaperSessionStatus.RUNNING,
            created_at=datetime.now(),
            selector_universe_id=selector_universe_id or universe_id,
            instrument_ids=instrument_ids,
        )
        self.save_session(session)
        self.set_active_session(resolved_id)
        return session

    def new_trade_id(self) -> str:
        """Generate a unique trade id."""
        return uuid.uuid4().hex[:12]

    def stop_session(self, session_id: str) -> PaperSession:
        """Mark a session as stopped."""
        session = self.load_session(session_id)
        session.status = PaperSessionStatus.STOPPED
        self.save_session(session)
        return session
