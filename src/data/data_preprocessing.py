import os
import pandas as pd

from src.logger import get_logger
from src.utils.text_preprocessing import normalize_text, ensure_nltk_resources

logger = get_logger(__name__)


def preprocess_dataframe(df: pd.DataFrame, col: str = "text") -> pd.DataFrame:
    """
    Applies text normalization to the specified column and drops empty rows.
    """
    ensure_nltk_resources()
    logger.info("Applying text normalization across %d records...", len(df))

    cleaned_df = df.copy()
    cleaned_df[col] = cleaned_df[col].astype(str).apply(normalize_text)

    # Drop rows that became empty after normalization
    cleaned_df = cleaned_df[cleaned_df[col].str.strip() != ""]
    cleaned_df = cleaned_df.dropna(subset=[col, "sentiment"])
    logger.info("Preprocessing complete. Retained %d records.", len(cleaned_df))
    return cleaned_df


def main():
    try:
        raw_dir = os.path.join("data", "raw")
        interim_dir = os.path.join("data", "interim")
        os.makedirs(interim_dir, exist_ok=True)

        train_path = os.path.join(raw_dir, "train.csv")
        test_path = os.path.join(raw_dir, "test.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Raw data files not found in {raw_dir}. Run data ingestion stage first."
            )

        logger.info("Reading raw datasets...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        train_processed = preprocess_dataframe(train_df, col="text")
        test_processed = preprocess_dataframe(test_df, col="text")

        train_out_path = os.path.join(interim_dir, "train_processed.csv")
        test_out_path = os.path.join(interim_dir, "test_processed.csv")

        train_processed.to_csv(train_out_path, index=False)
        test_processed.to_csv(test_out_path, index=False)

        logger.info("Processed datasets successfully saved to %s", interim_dir)

    except Exception as e:
        logger.error("Data preprocessing stage failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()