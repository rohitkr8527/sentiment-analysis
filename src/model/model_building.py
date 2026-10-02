import os
import pickle
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
import yaml

from src.logger import get_logger

logger = get_logger(__name__)


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load model training parameters from YAML file."""
    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)
        return params or {}
    except Exception as e:
        logger.error("Failed to load parameters from %s: %s", params_path, e)
        raise


def load_training_data(
    processed_dir: str = "data/processed"
) -> Tuple[Any, np.ndarray]:
    """
    Loads training features and labels.
    Prefers compressed sparse npz representation if available; falls back to CSV.
    """
    npz_features_path = os.path.join(processed_dir, "train_tfidf.npz")
    npy_labels_path = os.path.join(processed_dir, "train_labels.npy")
    csv_path = os.path.join(processed_dir, "train_tfidf.csv")

    if os.path.exists(npz_features_path) and os.path.exists(npy_labels_path):
        logger.info("Loading compressed sparse features from %s...", npz_features_path)
        X_train = sp.load_npz(npz_features_path)
        y_train = np.load(npy_labels_path)
        logger.info("Loaded training data: %d samples, %d features.", X_train.shape[0], X_train.shape[1])
        return X_train, y_train

    if os.path.exists(csv_path):
        logger.info("Loading training features from CSV: %s...", csv_path)
        train_data = pd.read_csv(csv_path)
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values.astype(int)
        logger.info("Loaded training data from CSV: %d samples, %d features.", X_train.shape[0], X_train.shape[1])
        return X_train, y_train

    raise FileNotFoundError(
        f"Could not find training features in {processed_dir}. "
        "Expected train_tfidf.npz or train_tfidf.csv. Run feature engineering first."
    )


def train_classifier(X_train: Any, y_train: np.ndarray, params: Dict[str, Any]) -> LogisticRegression:
    """Instantiate and train LogisticRegression model with hyperparameters."""
    model_params = params.get("model_building", {})

    C = float(model_params.get("C", 1.0))
    solver = model_params.get("solver", "saga")
    penalty = model_params.get("penalty", "elasticnet")
    l1_ratio = float(model_params.get("l1_ratio", 0.5)) if penalty == "elasticnet" else None
    max_iter = int(model_params.get("max_iter", 500))
    random_state = int(model_params.get("random_state", 42))

    logger.info(
        "Training LogisticRegression (C=%s, solver=%s, penalty=%s, l1_ratio=%s, max_iter=%s, random_state=%s)...",
        C, solver, penalty, l1_ratio, max_iter, random_state
    )

    clf = LogisticRegression(
        C=C,
        solver=solver,
        penalty=penalty,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    logger.info("Model training completed successfully.")
    return clf


def save_model_artifact(model: Any, output_path: str = "models/model.pkl") -> None:
    """Serialize model artifact to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Serialized trained model saved to: %s", output_path)


def main():
    try:
        params = load_params("params.yaml")
        X_train, y_train = load_training_data("data/processed")
        clf = train_classifier(X_train, y_train, params)
        save_model_artifact(clf, "models/model.pkl")
        logger.info("Model building stage completed successfully.")
    except Exception as e:
        logger.error("Model building stage failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
