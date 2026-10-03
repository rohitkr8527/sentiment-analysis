import os
import json
import pickle
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from dotenv import load_dotenv

from src.logger import get_logger

logger = get_logger(__name__)
load_dotenv()


def setup_mlflow() -> Tuple[bool, str]:
    """
    Sets up MLflow tracking if credentials and repo details are configured in environment.
    Supports DAGSHUB_TOKEN (preferred) or legacy sentiment_analysis variable.
    Returns (is_configured, tracking_uri).
    """
    token = os.getenv("DAGSHUB_TOKEN") or os.getenv("sentiment_analysis")
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "rohitkr8527")
    repo_name = os.getenv("DAGSHUB_REPO_NAME", "sentiment-analysis")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"

    if token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            logger.info("MLflow tracking configured for DagsHub: %s", tracking_uri)
            return True, tracking_uri
        except Exception as e:
            logger.warning("Could not set up MLflow tracking URI: %s", e)

    logger.info("DAGSHUB_TOKEN / sentiment_analysis not configured; proceeding with local metrics generation.")
    return False, ""


def load_model(file_path: str = "models/model.pkl") -> Any:
    """Load serialized model artifact."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found at: {file_path}")
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    logger.info("Loaded trained model from %s", file_path)
    return model


def load_test_features(processed_dir: str = "data/processed") -> Tuple[Any, np.ndarray]:
    """Load test features and true labels from npz or CSV."""
    npz_path = os.path.join(processed_dir, "test_tfidf.npz")
    npy_path = os.path.join(processed_dir, "test_labels.npy")
    csv_path = os.path.join(processed_dir, "test_tfidf.csv")

    if os.path.exists(npz_path) and os.path.exists(npy_path):
        import scipy.sparse as sp
        logger.info("Loading test features from compressed sparse file %s...", npz_path)
        X_test = sp.load_npz(npz_path)
        y_test = np.load(npy_path)
        return X_test, y_test

    if os.path.exists(csv_path):
        logger.info("Loading test features from CSV %s...", csv_path)
        test_data = pd.read_csv(csv_path)
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values.astype(int)
        return X_test, y_test

    raise FileNotFoundError(f"Could not find test features in {processed_dir}. Run feature engineering first.")


def evaluate_model(clf: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Compute standard classification evaluation metrics."""
    logger.info("Evaluating model predictions on test set (%d samples)...", len(y_test))
    y_pred = clf.predict(X_test)

    # Some estimators may not implement predict_proba
    if hasattr(clf, "predict_proba"):
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_pred_proba))
    else:
        auc = 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc": auc
    }

    logger.info("Evaluation results: %s", metrics)
    return metrics


def save_metrics(metrics: Dict[str, Any], output_path: str = "reports/metrics.json") -> None:
    """Persist metrics to JSON file for DVC tracking."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Evaluation metrics saved to %s", output_path)


def save_experiment_info(
    run_id: str,
    model_path: str,
    tracking_uri: str,
    output_path: str = "reports/experiment_info.json"
) -> None:
    """Persist experiment metadata for downstream model registration."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    info = {
        "run_id": run_id,
        "model_path": model_path,
        "tracking_uri": tracking_uri
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
    logger.info("Experiment info saved to %s", output_path)


def main():
    try:
        clf = load_model("models/model.pkl")
        X_test, y_test = load_test_features("data/processed")
        metrics = evaluate_model(clf, X_test, y_test)
        save_metrics(metrics, "reports/metrics.json")

        mlflow_enabled, tracking_uri = setup_mlflow()
        run_id = "local-run"

        if mlflow_enabled:
            import mlflow
            import mlflow.sklearn
            experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "sentiment-analysis-pipeline")
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run() as run:
                run_id = run.info.run_id
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)

                if hasattr(clf, "get_params"):
                    mlflow.log_params(clf.get_params())

                mlflow.sklearn.log_model(clf, "model")
                mlflow.log_artifact("reports/metrics.json")
                if os.path.exists("models/tfidf_vectorizer.pkl"):
                    mlflow.log_artifact("models/tfidf_vectorizer.pkl")
                logger.info("Logged model and metrics to MLflow run: %s", run_id)

        save_experiment_info(
            run_id=run_id,
            model_path="model",
            tracking_uri=tracking_uri,
            output_path="reports/experiment_info.json"
        )
        logger.info("Model evaluation stage completed successfully.")

    except Exception as e:
        logger.error("Model evaluation stage failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
