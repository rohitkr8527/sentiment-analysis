import os
import time
import pickle
import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

from fastapi_app.core.config import settings
from fastapi_app.services.text_preprocessor import normalize_text, ensure_nltk_resources

logger = logging.getLogger(__name__)


class SentimentPredictor:
    """
    Service responsible for model and vectorizer loading, caching,
    and performing real-time sentiment predictions.
    """

    def __init__(self):
        self.model: Optional[Any] = None
        self.vectorizer: Optional[Any] = None
        self.is_ready: bool = False
        self.model_source: str = "none"
        self.model_version: Optional[str] = None

    def initialize(self) -> bool:
        """
        Loads model and vectorizer artifacts.
        Tries local artifacts first, then optional MLflow remote registry.
        Does not crash if artifacts are missing; gracefully marks service as not ready.
        """
        logger.info("Initializing SentimentPredictor service...")
        ensure_nltk_resources()

        # 1. Try local artifacts first
        if self._load_local_artifacts():
            self.is_ready = True
            self.model_source = "local"
            return True

        # 2. Try MLflow if configured
        if settings.USE_MLFLOW_MODEL and self._load_mlflow_model():
            self.is_ready = True
            self.model_source = "mlflow"
            return True

        logger.warning(
            "SentimentPredictor initialized in UNREADY state. Neither local artifacts "
            "nor MLflow models were successfully loaded."
        )
        self.is_ready = False
        return False

    def _load_local_artifacts(self) -> bool:
        """Loads serialized model and vectorizer from filesystem."""
        model_path = settings.resolve_path(settings.MODEL_PATH)
        vectorizer_path = settings.resolve_path(settings.VECTORIZER_PATH)

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            logger.info(
                "Local model or vectorizer not found at: model=%s, vectorizer=%s",
                model_path, vectorizer_path
            )
            return False

        try:
            logger.info("Loading local vectorizer from: %s", vectorizer_path)
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)

            logger.info("Loading local model from: %s", model_path)
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

            # Ensure backward and forward compatibility across scikit-learn versions
            if hasattr(self.model, "__dict__") and not hasattr(self.model, "multi_class"):
                self.model.multi_class = "auto"

            logger.info("Successfully loaded local model and vectorizer artifacts.")
            return True
        except Exception as e:
            logger.error("Failed loading local artifacts: %s", e, exc_info=True)
            return False

    def _load_mlflow_model(self) -> bool:
        """Loads model from remote MLflow Model Registry."""
        token = settings.DAGSHUB_TOKEN
        tracking_uri = settings.MLFLOW_TRACKING_URI

        if not tracking_uri and settings.DAGSHUB_REPO_OWNER:
            tracking_uri = f"https://dagshub.com/{settings.DAGSHUB_REPO_OWNER}/{settings.DAGSHUB_REPO_NAME}.mlflow"

        if not tracking_uri or not token:
            logger.info("MLflow tracking credentials not fully configured; skipping MLflow loading.")
            return False

        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            os.environ["MLFLOW_TRACKING_USERNAME"] = token
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token
            mlflow.set_tracking_uri(tracking_uri)

            client = MlflowClient()
            model_name = settings.MLFLOW_MODEL_NAME
            stage = settings.MLFLOW_MODEL_STAGE

            versions = client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                versions = client.get_latest_versions(model_name, stages=["None"])

            if not versions:
                logger.warning("No versions found for model '%s' in MLflow.", model_name)
                return False

            latest_version = versions[0].version
            self.model_version = latest_version
            model_uri = f"models:/{model_name}/{latest_version}"
            logger.info("Loading model from MLflow URI: %s", model_uri)
            self.model = mlflow.pyfunc.load_model(model_uri)

            # Vectorizer is still needed
            vec_path = settings.resolve_path(settings.VECTORIZER_PATH)
            if os.path.exists(vec_path):
                with open(vec_path, "rb") as f:
                    self.vectorizer = pickle.load(f)
                return True
            else:
                logger.warning("MLflow model loaded, but local vectorizer not found at %s", vec_path)
                return False

        except Exception as e:
            logger.error("Failed loading model from MLflow: %s", e)
            return False

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Runs full inference on raw input text:
        1. Preprocessing & Normalization
        2. Vectorization
        3. Classification & Confidence calculation
        """
        if not self.is_ready or self.model is None or self.vectorizer is None:
            raise RuntimeError(
                "Model is not ready. Ensure model.pkl and tfidf_vectorizer.pkl exist or are configured."
            )

        start_time = time.perf_counter()

        cleaned_text = normalize_text(text)
        # Handle cases where all text was stripped (e.g. only punctuation or stopwords)
        inference_text = cleaned_text if cleaned_text.strip() else text.lower().strip()

        features = self.vectorizer.transform([inference_text])

        # Support both standard scikit-learn models and MLflow PyFunc wrappers
        if hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba(features)[0]
            except AttributeError as ae:
                if "multi_class" in str(ae):
                    self.model.multi_class = "auto"
                    probs = self.model.predict_proba(features)[0]
                else:
                    raise
            label = int(np.argmax(probs))
            confidence = float(probs[label])
        else:
            features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])
            preds = self.model.predict(features_df)
            label = int(preds[0])
            confidence = 1.0

        sentiment = "Positive" if label == 1 else "Negative"
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "text": text,
            "sentiment": sentiment,
            "label": label,
            "confidence": round(confidence, 4),
            "latency_ms": latency_ms,
        }


# Singleton service instance
_predictor_instance: Optional[SentimentPredictor] = None


def get_predictor() -> SentimentPredictor:
    """Returns the singleton predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = SentimentPredictor()
    return _predictor_instance
