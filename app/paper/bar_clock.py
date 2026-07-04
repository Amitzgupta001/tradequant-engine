"""5-minute bar boundary helpers for live paper trading."""

from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))


def to_ist(value: datetime) -> datetime:
    """Normalize a datetime to IST."""
    if value.tzinfo is None:
        return value.replace(tzinfo=IST)
    return value.astimezone(IST)


def bar_slot(value: datetime) -> datetime:
    """Return the start timestamp of the current 5-minute slot in IST."""
    current = to_ist(value).replace(second=0, microsecond=0)
    aligned_minute = (current.minute // 5) * 5
    return current.replace(minute=aligned_minute)


def bar_slot_changed(previous: datetime | None, current: datetime) -> bool:
    """Return True when the 5-minute slot has advanced."""
    if previous is None:
        return True
    return bar_slot(previous) != bar_slot(current)
