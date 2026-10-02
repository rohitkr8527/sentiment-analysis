from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from fastapi_app.core.config import settings
from fastapi_app.schemas.sentiment import HealthResponse
from fastapi_app.services.predictor import get_predictor

health_router = APIRouter(tags=["Health"])


@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns service availability and model readiness status."
)
async def health_check():
    """Health probe endpoint used by cloud orchestrators (e.g. Azure App Service / Container Apps)."""
    predictor = get_predictor()
    is_ready = predictor.is_ready
    status_str = "healthy" if is_ready else "degraded"

    content = {
        "status": status_str,
        "model_loaded": is_ready,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }

    status_code = status.HTTP_200_OK if is_ready else status.HTTP_200_OK
    return JSONResponse(status_code=status_code, content=content)


@health_router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Service readiness check",
    description="Readiness probe returning 200 OK only if model is fully loaded and ready."
)
async def readiness_check():
    """Readiness probe endpoint."""
    predictor = get_predictor()
    if not predictor.is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "degraded",
                "model_loaded": False,
                "version": settings.APP_VERSION,
                "environment": settings.ENVIRONMENT
            }
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "healthy",
            "model_loaded": True,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT
        }
    )
