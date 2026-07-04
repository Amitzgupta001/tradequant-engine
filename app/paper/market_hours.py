"""Indian equity market hours helpers."""

from datetime import date, datetime, time, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


def now_ist() -> datetime:
    """Return current time in IST."""
    return datetime.now(tz=IST)


def is_market_open(now: datetime | None = None) -> bool:
    """Return True during NSE cash session on weekdays."""
    current = now or now_ist()
    if current.tzinfo is None:
        current = current.replace(tzinfo=IST)
    else:
        current = current.astimezone(IST)

    if current.weekday() >= 5:
        return False

    session_time = current.time()
    return MARKET_OPEN <= session_time <= MARKET_CLOSE


def session_date(now: datetime | None = None) -> date:
    """Return the trading session date in IST."""
    current = now or now_ist()
    if current.tzinfo is None:
        current = current.replace(tzinfo=IST)
    else:
        current = current.astimezone(IST)
    return current.date()
