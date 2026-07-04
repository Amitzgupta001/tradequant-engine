"""Health check routes."""

from fastapi import APIRouter, Depends

from app.api.v1.dependencies import get_app_settings
from app.core.config import Settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(settings: Settings = Depends(get_app_settings)) -> dict[str, str]:
    """Return application health status."""
    return {"status": "ok", "app": settings.app_name}
