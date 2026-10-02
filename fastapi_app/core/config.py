import os
from typing import List
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application runtime configuration settings."""

    # Server settings (supports Azure App Service WEBSITES_PORT / PORT)
    PORT: int = int(os.getenv("PORT") or os.getenv("WEBSITES_PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

    # API Metadata
    APP_NAME: str = "Sentiment Analysis API"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = "Production-ready sentiment analysis service with MLOps tracking, REST API, and web interface."

    # CORS settings
    CORS_ORIGINS: List[str] = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]

    # Model & Artifact resolution
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/model.pkl")
    VECTORIZER_PATH: str = os.getenv("VECTORIZER_PATH", "models/tfidf_vectorizer.pkl")

    # Optional MLflow / DagsHub remote configuration
    USE_MLFLOW_MODEL: bool = os.getenv("USE_MLFLOW_MODEL", "false").lower() in ("true", "1", "yes")
    DAGSHUB_TOKEN: str = os.getenv("DAGSHUB_TOKEN") or os.getenv("sentiment_analysis", "")
    DAGSHUB_REPO_OWNER: str = os.getenv("DAGSHUB_REPO_OWNER", "")
    DAGSHUB_REPO_NAME: str = os.getenv("DAGSHUB_REPO_NAME", "sentiment-analysis")
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "")
    MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "my_model")
    MLFLOW_MODEL_STAGE: str = os.getenv("MLFLOW_MODEL_STAGE", "Production")

    @classmethod
    def resolve_path(cls, relative_path: str) -> str:
        """
        Resolves file path checking multiple possible working directories
        (e.g., repository root, fastapi_app directory, or /app container directory).
        """
        if os.path.isabs(relative_path) and os.path.exists(relative_path):
            return relative_path

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.abspath(os.path.join(current_file_dir, ".."))
        repo_root = os.path.abspath(os.path.join(app_dir, ".."))

        candidates = [
            os.path.abspath(relative_path),
            os.path.join(repo_root, relative_path),
            os.path.join(app_dir, relative_path),
            os.path.join("/app", relative_path),
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        return os.path.abspath(relative_path)


settings = Settings()
