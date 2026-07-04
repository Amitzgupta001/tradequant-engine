"""Loguru logging configuration."""

import sys

from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    """Configure Loguru with a consistent structured format."""
    logger.remove()
    logger.add(
        sys.stdout,
        level=level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
