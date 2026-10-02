import time
from fastapi import APIRouter, HTTPException, status
from fastapi_app.schemas.sentiment import SentimentRequest, SentimentResponse
from fastapi_app.services.predictor import get_predictor
from fastapi_app.routes.metrics import REQUEST_COUNT, REQUEST_LATENCY, PREDICTION_COUNT

api_router = APIRouter(prefix="/api/v1", tags=["REST API"])


@api_router.post(
    "/predict",
    response_model=SentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict sentiment (JSON REST API)",
    description="Accepts a text payload and returns sentiment classification ('Positive' or 'Negative') with confidence score."
)
async def predict_sentiment(payload: SentimentRequest):
    REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/predict").inc()
    start_time = time.time()

    predictor = get_predictor()
    if not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready to serve predictions. Ensure model artifacts are available."
        )

    try:
        result = predictor.predict(payload.text)
        PREDICTION_COUNT.labels(prediction=str(result["label"])).inc()
        REQUEST_LATENCY.labels(endpoint="/api/v1/predict").observe(time.time() - start_time)
        return SentimentResponse(**result)

    except Exception as e:
        REQUEST_LATENCY.labels(endpoint="/api/v1/predict").observe(time.time() - start_time)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )
