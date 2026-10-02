import os
from typing import Dict, Any, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from dotenv import load_dotenv

from src.logger import get_logger

logger = get_logger(__name__)
load_dotenv()


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load configuration parameters from a YAML file."""
    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)
        logger.info("Parameters loaded from %s", params_path)
        return params or {}
    except FileNotFoundError:
        logger.warning("Configuration file %s not found, using default parameters.", params_path)
        return {}
    except yaml.YAMLError as e:
        logger.error("Failed to parse YAML file %s: %s", params_path, e)
        raise


def load_data_from_blob(blob_name: str) -> Optional[pd.DataFrame]:
    """Fetch dataset from Azure Blob Storage if credentials are configured."""
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING") or os.getenv("AZURE_BLOB_CONNECTION_STRING")
    if not conn_str:
        logger.info("No Azure Storage connection string provided in environment; skipping blob download.")
        return None

    try:
        from src.connections.blob_connection import BlobOperations
        blob = BlobOperations()
        df = blob.fetch_file_from_blob(blob_name)
        if df is not None and not df.empty:
            logger.info("Successfully loaded data from Azure Blob Storage: %s", blob_name)
            return df
    except Exception as e:
        logger.warning("Could not load data from Azure Blob Storage (%s): %s", blob_name, e)

    return None


def load_local_data(fallback_path: str) -> pd.DataFrame:
    """Load dataset from local filesystem fallback."""
    candidate_paths = [
        fallback_path,
        "data/balanced_sentiment_dataset.csv",
        "data/raw/train.csv",
    ]

    for path in candidate_paths:
        if path and os.path.exists(path):
            logger.info("Loading local dataset from: %s", path)
            return pd.read_csv(path)

    raise FileNotFoundError(
        f"Unable to find dataset locally. Checked paths: {candidate_paths}. "
        "Please provide AZURE_STORAGE_CONNECTION_STRING or place dataset at one of these locations."
    )


def validate_and_format_data(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataset contains 'text' and 'sentiment' columns and remove null entries."""
    if "text" not in df.columns or "sentiment" not in df.columns:
        raise KeyError(
            f"Dataset must contain 'text' and 'sentiment' columns. Found columns: {list(df.columns)}"
        )

    clean_df = df[["text", "sentiment"]].dropna().copy()
    clean_df["sentiment"] = clean_df["sentiment"].astype(int)
    clean_df["text"] = clean_df["text"].astype(str)

    logger.info("Validated dataset with %d rows.", len(clean_df))
    return clean_df


def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = "data/raw") -> None:
    """Persist train and test datasets to disk."""
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Saved train data (%d rows) to %s", len(train_df), train_path)
    logger.info("Saved test data (%d rows) to %s", len(test_df), test_path)


def main():
    try:
        params = load_params("params.yaml")
        ingestion_params = params.get("data_ingestion", {})
        test_size = float(ingestion_params.get("test_size", 0.2))
        random_state = int(ingestion_params.get("random_state", 42))
        blob_name = ingestion_params.get("blob_name", "balanced_sentiment_dataset.csv")
        fallback_path = ingestion_params.get("local_fallback_path", "data/balanced_sentiment_dataset.csv")

        # 1. Try remote Azure Blob Storage first, then fallback to local
        df = load_data_from_blob(blob_name)
        if df is None:
            df = load_local_data(fallback_path)

        # 2. Validate columns and format
        formatted_df = validate_and_format_data(df)

        # 3. Stratified split to preserve balanced sentiment distribution
        train_data, test_data = train_test_split(
            formatted_df,
            test_size=test_size,
            random_state=random_state,
            stratify=formatted_df["sentiment"]
        )

        # 4. Save splits
        save_splits(train_data, test_data, output_dir="data/raw")
        logger.info("Data ingestion completed successfully.")

    except Exception as e:
        logger.error("Data ingestion pipeline failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
