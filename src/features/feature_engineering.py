import os
import pickle
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml

from src.logger import get_logger

logger = get_logger(__name__)


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load configuration parameters from YAML file."""
    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)
        return params or {}
    except Exception as e:
        logger.error("Failed to load parameters from %s: %s", params_path, e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load and sanitize data from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_csv(file_path)
    df["text"] = df["text"].fillna("")
    return df


def apply_tfidf(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    params: Dict[str, Any],
    vectorizer_path: str = "models/tfidf_vectorizer.pkl"
) -> Tuple[sp.csr_matrix, np.ndarray, sp.csr_matrix, np.ndarray, TfidfVectorizer]:
    """
    Fits TF-IDF vectorizer on training text and transforms both train and test sets.
    Returns sparse matrices and label arrays for high performance and low memory consumption.
    """
    fe_params = params.get("feature_engineering", {})
    max_features = fe_params.get("max_features", 10000)
    max_df = float(fe_params.get("max_df", 0.9))
    min_df = int(fe_params.get("min_df", 5))
    ngram_range = tuple(fe_params.get("ngram_range", [1, 2]))

    logger.info(
        "Configuring TfidfVectorizer (max_features=%s, max_df=%s, min_df=%s, ngram_range=%s)",
        max_features, max_df, min_df, ngram_range
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        max_df=max_df,
        min_df=min_df,
        ngram_range=ngram_range,
        dtype=np.float32
    )

    X_train_raw = train_data["text"].values.astype(str)
    y_train = train_data["sentiment"].values.astype(int)

    X_test_raw = test_data["text"].values.astype(str)
    y_test = test_data["sentiment"].values.astype(int)

    logger.info("Fitting TF-IDF vectorizer on %d training records...", len(X_train_raw))
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    logger.info("Fitted vocabulary size: %d terms.", len(vectorizer.vocabulary_))

    logger.info("Transforming %d test records with fitted vectorizer...", len(X_test_raw))
    X_test_tfidf = vectorizer.transform(X_test_raw)

    # Serialize vectorizer
    os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info("Fitted TF-IDF vectorizer serialized to %s", vectorizer_path)

    return X_train_tfidf, y_train, X_test_tfidf, y_test, vectorizer


def save_processed_artifacts(
    X_train: sp.csr_matrix,
    y_train: np.ndarray,
    X_test: sp.csr_matrix,
    y_test: np.ndarray,
    output_dir: str = "data/processed",
    csv_sample_size: int = 2000
) -> None:
    """
    Saves sparse feature matrices and labels, and generates a structured
    test CSV for holdout validation and testing suites.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. High performance compressed sparse matrices & labels
    sp.save_npz(os.path.join(output_dir, "train_tfidf.npz"), X_train)
    np.save(os.path.join(output_dir, "train_labels.npy"), y_train)

    sp.save_npz(os.path.join(output_dir, "test_tfidf.npz"), X_test)
    np.save(os.path.join(output_dir, "test_labels.npy"), y_test)
    logger.info("Saved compressed sparse feature representations to %s", output_dir)

    # 2. Save holdout test CSV (for backward compatibility with pandas test readers)
    # Using a representative sample or full test set in float32
    sample_size = min(len(y_test), csv_sample_size)
    logger.info("Exporting %d test holdout rows to test_tfidf.csv...", sample_size)
    sample_sparse = X_test[:sample_size].toarray()
    test_df = pd.DataFrame(sample_sparse)
    test_df["label"] = y_test[:sample_size]
    test_csv_path = os.path.join(output_dir, "test_tfidf.csv")
    test_df.to_csv(test_csv_path, index=False)
    logger.info("Saved holdout validation dataset to %s (%d rows)", test_csv_path, len(test_df))


def main():
    try:
        params = load_params("params.yaml")
        interim_dir = os.path.join("data", "interim")
        processed_dir = os.path.join("data", "processed")

        train_path = os.path.join(interim_dir, "train_processed.csv")
        test_path = os.path.join(interim_dir, "test_processed.csv")

        train_data = load_data(train_path)
        test_data = load_data(test_path)

        X_train, y_train, X_test, y_test, _ = apply_tfidf(
            train_data,
            test_data,
            params,
            vectorizer_path="models/tfidf_vectorizer.pkl"
        )

        save_processed_artifacts(
            X_train, y_train, X_test, y_test,
            output_dir=processed_dir,
            csv_sample_size=2000
        )
        logger.info("Feature engineering stage completed successfully.")

    except Exception as e:
        logger.error("Feature engineering stage failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
