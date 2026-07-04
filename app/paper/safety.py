"""Safety guards — paper trading must never place broker orders."""

from loguru import logger

PAPER_TRADING_ONLY = True
PAPER_TRADING_MESSAGE = (
    "Paper trading mode: Dhan is used for market DATA only. "
    "No orders, positions, or funds APIs are called."
)


def verify_paper_trading_mode() -> None:
    """Log and assert that paper mode is active."""
    if not PAPER_TRADING_ONLY:
        msg = "Paper trading safety flag is disabled"
        raise RuntimeError(msg)
    logger.info(PAPER_TRADING_MESSAGE)


def block_broker_order(operation: str) -> None:
    """Raise if live order placement is attempted from paper paths."""
    msg = f"Blocked broker order attempt during paper trading: {operation}"
    raise RuntimeError(msg)
