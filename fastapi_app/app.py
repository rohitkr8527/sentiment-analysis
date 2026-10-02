import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from fastapi_app.core.config import settings
from fastapi_app.services.predictor import get_predictor
from fastapi_app.routes.health import health_router
from fastapi_app.routes.api import api_router
from fastapi_app.routes.web import web_router
from fastapi_app.routes.metrics import metrics_router

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
)
logger = logging.getLogger("sentiment_analysis_app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager for startup and shutdown tasks.
    Pre-warms the model predictor and initializes NLTK resources.
    """
    logger.info("Starting up %s (version: %s)...", settings.APP_NAME, settings.APP_VERSION)
    predictor = get_predictor()
    success = predictor.initialize()
    if success:
        logger.info("Predictor successfully initialized (Source: %s).", predictor.model_source)
    else:
        logger.warning("Predictor started in DEGRADED mode. Awaiting valid model artifacts.")

    yield

    logger.info("Shutting down %s...", settings.APP_NAME)


def create_app() -> FastAPI:
    """FastAPI application factory."""
    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=settings.APP_DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS Middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Routers
    application.include_router(health_router)
    application.include_router(api_router)
    application.include_router(web_router)
    application.include_router(metrics_router)

    return application


app = create_app()

if __name__ == "__main__":
    logger.info("Running local server on %s:%d", settings.HOST, settings.PORT)
    uvicorn.run("fastapi_app.app:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
