"""
Services package for sentiment analysis.
"""
from fastapi_app.services.predictor import SentimentPredictor, get_predictor

__all__ = ["SentimentPredictor", "get_predictor"]
