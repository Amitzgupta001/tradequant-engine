"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.v1.router import v1_router
from app.brokers.exceptions import BrokerError
from app.core.config import get_settings
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize application resources on startup."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Starting {}", settings.app_name)
    yield
    logger.info("Shutting down {}", settings.app_name)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def root_health() -> dict[str, str]:
        return {"status": "ok", "app": settings.app_name}

    app.include_router(v1_router)

    @app.exception_handler(BrokerError)
    async def broker_error_handler(_: Request, exc: BrokerError) -> JSONResponse:
        logger.error("Broker error: {}", exc)
        return JSONResponse(
            status_code=502,
            content={"detail": str(exc)},
        )

    return app
