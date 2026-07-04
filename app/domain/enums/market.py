"""Market-related enumerations."""

from enum import Enum


class ExchangeSegment(str, Enum):
    """Supported Dhan exchange segments."""

    NSE_EQ = "NSE_EQ"
    NSE_FNO = "NSE_FNO"
    BSE_EQ = "BSE_EQ"
    BSE_FNO = "BSE_FNO"
    MCX_COMM = "MCX_COMM"
    IDX_I = "IDX_I"


class InstrumentType(str, Enum):
    """Supported Dhan instrument types."""

    EQUITY = "EQUITY"
    INDEX = "INDEX"
    FUTIDX = "FUTIDX"
    OPTIDX = "OPTIDX"
    FUTSTK = "FUTSTK"
    OPTSTK = "OPTSTK"
    FUTCOM = "FUTCOM"
    OPTFUT = "OPTFUT"
    FUTCUR = "FUTCUR"
    OPTCUR = "OPTCUR"


class Timeframe(str, Enum):
    """Supported historical candle timeframes."""

    DAILY = "DAILY"
    MIN_1 = "MIN_1"
    MIN_5 = "MIN_5"
    MIN_15 = "MIN_15"
    MIN_30 = "MIN_30"
    MIN_60 = "MIN_60"

    @property
    def interval_minutes(self) -> int | None:
        """Return minute interval for intraday timeframes."""
        mapping = {
            Timeframe.MIN_1: 1,
            Timeframe.MIN_5: 5,
            Timeframe.MIN_15: 15,
            Timeframe.MIN_30: 30,
            Timeframe.MIN_60: 60,
        }
        return mapping.get(self)

    @property
    def is_daily(self) -> bool:
        """Return True when timeframe is daily."""
        return self == Timeframe.DAILY
