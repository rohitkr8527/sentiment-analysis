import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
from dotenv import load_dotenv
from src.connections import blob_connection 

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise


def load_data_from_blob(blob_name: str) -> pd.DataFrame:
    """Fetch data from Azure Blob Storage."""
    try:
        blob = blob_connection.BlobOperations()
        df = blob.fetch_file_from_blob(blob_name)
        if df is None:
            raise ValueError(f"Failed to load data from blob: {blob_name}")
        return df
    except Exception as e:
        logging.error('Error loading data from Azure Blob Storage: %s', e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        logging.info("Preprocessing started...")

        # Ensure only 'text' and 'sentiment' columns are present
        if 'text' not in df.columns or 'sentiment' not in df.columns:
            raise KeyError("Expected columns 'text' and 'sentiment' not found in DataFrame")

        final_df = df[['text', 'sentiment']].copy()

        logging.info("Data preprocessing completed.")
        return final_df

    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise



def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets locally."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.info('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Error saving the data: %s', e)
        raise


def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params.get('data_ingestion', {}).get('test_size', 0.2)
        blob_name = params.get('data_ingestion', {}).get('blob_name', 'balanced_sentiment_dataset.csv')

        df = load_data_from_blob(blob_name)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data')

        logging.info("Data ingestion completed successfully.")

    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()


