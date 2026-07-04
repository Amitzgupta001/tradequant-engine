"""API v1 router aggregation."""

from fastapi import APIRouter

from app.api.v1.routes import backtest, features, health, historical, indicators, ml, paper

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(health.router)
v1_router.include_router(historical.router)
v1_router.include_router(indicators.router)
v1_router.include_router(features.router)
v1_router.include_router(ml.router)
v1_router.include_router(backtest.router)
v1_router.include_router(paper.router)
