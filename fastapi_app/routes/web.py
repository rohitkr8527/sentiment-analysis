import os
import time
from typing import Optional
from fastapi import APIRouter, Request, Form, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from fastapi_app.services.predictor import get_predictor
from fastapi_app.routes.metrics import REQUEST_COUNT, REQUEST_LATENCY, PREDICTION_COUNT

web_router = APIRouter(tags=["Web Interface"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@web_router.get("/", response_class=HTMLResponse, summary="Home web interface")
async def home_page(request: Request):
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": None,
            "sentiment_label": None,
            "confidence": None,
            "latency_ms": None,
            "text_input": "",
            "error": None
        }
    )
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response


@web_router.post("/predict", summary="Sentiment prediction (Form or JSON)")
async def predict_endpoint(request: Request, text: Optional[str] = Form(None)):
    """
    Dual-mode prediction endpoint:
    - If Content-Type is application/json: returns structured JSON response.
    - If Content-Type is form data: returns rendered HTML web page.
    """
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()
    content_type = request.headers.get("content-type", "")

    input_text = text

    # Handle direct JSON request sent to /predict
    if "application/json" in content_type:
        try:
            body = await request.json()
            input_text = body.get("text", "")
        except Exception:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Invalid JSON payload"}
            )

    if not input_text or not input_text.strip():
        if "application/json" in content_type:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"detail": "Input 'text' field is required."}
            )
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "result": None,
                "sentiment_label": None,
                "confidence": None,
                "latency_ms": None,
                "text_input": "",
                "error": "Please enter some text to analyze."
            }
        )

    predictor = get_predictor()
    if not predictor.is_ready:
        err_msg = "Model service is currently unavailable. Please ensure model artifacts are present."
        if "application/json" in content_type:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": err_msg}
            )
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "result": None,
                "sentiment_label": None,
                "confidence": None,
                "latency_ms": None,
                "text_input": input_text,
                "error": err_msg
            }
        )

    try:
        prediction_data = predictor.predict(input_text)
        label = prediction_data["label"]
        sentiment = prediction_data["sentiment"]
        confidence = prediction_data["confidence"]
        latency_ms = prediction_data["latency_ms"]

        PREDICTION_COUNT.labels(prediction=str(label)).inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

        if "application/json" in content_type:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=prediction_data
            )

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "result": label,
                "sentiment_label": sentiment,
                "confidence": f"{confidence * 100:.1f}%",
                "latency_ms": latency_ms,
                "text_input": input_text,
                "error": None
            }
        )

    except Exception as e:
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        if "application/json" in content_type:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": str(e)}
            )
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "result": None,
                "sentiment_label": None,
                "confidence": None,
                "latency_ms": None,
                "text_input": input_text,
                "error": f"Error running sentiment analysis: {str(e)}"
            }
        )
