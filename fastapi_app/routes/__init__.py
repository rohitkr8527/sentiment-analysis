"""
API and Web routes package.
"""
from fastapi_app.routes.health import health_router
from fastapi_app.routes.api import api_router
from fastapi_app.routes.web import web_router
from fastapi_app.routes.metrics import metrics_router

__all__ = ["health_router", "api_router", "web_router", "metrics_router"]
