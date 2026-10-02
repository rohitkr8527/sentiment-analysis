from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    """Input payload for sentiment prediction."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="The text content to analyze for sentiment",
        examples=["I really love this product! The quality is amazing."]
    )


class SentimentResponse(BaseModel):
    """Output prediction response."""
    text: str = Field(..., description="Original input text")
    sentiment: str = Field(..., description="'Positive' or 'Negative'")
    label: int = Field(..., description="1 for Positive, 0 for Negative")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class HealthResponse(BaseModel):
    """Health check status response."""
    status: str = Field(..., description="'healthy' or 'degraded'")
    model_loaded: bool = Field(..., description="Whether the model and vectorizer are ready for inference")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Deployment environment")
