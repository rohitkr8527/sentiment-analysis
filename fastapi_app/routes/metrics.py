from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CollectorRegistry,
    CONTENT_TYPE_LATEST,
)

metrics_router = APIRouter(tags=["Monitoring"])

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total count of HTTP requests",
    ["method", "endpoint"],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    registry=registry
)

PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Count of model predictions by sentiment class",
    ["prediction"],
    registry=registry
)


@metrics_router.get("/metrics", summary="Prometheus application metrics")
async def get_metrics():
    """Expose Prometheus metrics for scrapers and cloud monitors."""
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
