from azure.storage.blob import BlobServiceClient
import pandas as pd
import logging
from io import StringIO
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

class BlobOperations:
    def __init__(self):
        """
        Initialize the BlobOperations class with Azure Blob Storage connection details from environment variables.
        """
        self.connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
        self.container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")

        if not self.connection_string or not self.container_name:
            raise ValueError("Missing Azure Blob Storage connection string or container name in environment variables.")

        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)

        logging.info("Initialized connection to Azure Blob Storage.")

    def fetch_file_from_blob(self, blob_name):
        """
        Fetches a CSV file from Azure Blob Storage and returns it as a Pandas DataFrame.

        :param blob_name: Name of the blob file (e.g., 'data/data.csv')
        :return: Pandas DataFrame or None if failed
        """
        try:
            logging.info(f" Fetching blob '{blob_name}' from container '{self.container_name}'...")
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_data = blob_client.download_blob().readall()
            df = pd.read_csv(StringIO(blob_data.decode('utf-8')))
            logging.info(f"Successfully loaded '{blob_name}' with {len(df)} records.")
            return df
        except Exception as e:
            logging.exception(f"Failed to fetch '{blob_name}' from Azure Blob Storage: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Replace this with your actual blob name
    BLOB_NAME = "balanced_sentiment_dataset.csv"

    blob_ops = BlobOperations()
    df = blob_ops.fetch_file_from_blob(BLOB_NAME)

    if df is not None:
        print(f"Data fetched successfully with {len(df)} records.")
        print(df.head())  # Show preview
    else:
        print("Failed to fetch the blob.")
