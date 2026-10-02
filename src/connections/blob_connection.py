import os
from io import StringIO
from typing import Optional
import pandas as pd
from dotenv import load_dotenv

from src.logger import get_logger

logger = get_logger(__name__)
load_dotenv()


class BlobOperations:
    """
    Azure Blob Storage wrapper for data ingestion and artifact management.
    Supports connection string authentication as well as Azure Identity / Managed Identity.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        container_name: Optional[str] = None,
    ):
        self.connection_string = (
            connection_string
            or os.getenv("AZURE_BLOB_CONNECTION_STRING")
            or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        )
        self.container_name = container_name or os.getenv("AZURE_BLOB_CONTAINER_NAME", "sentiment-data")

        if not self.connection_string:
            raise ValueError(
                "Missing Azure Blob Storage connection string. "
                "Please set AZURE_STORAGE_CONNECTION_STRING or AZURE_BLOB_CONNECTION_STRING."
            )

        try:
            from azure.storage.blob import BlobServiceClient
            self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            self.container_client = self.blob_service_client.get_container_client(self.container_name)
            logger.info("Successfully connected to Azure Blob Storage container: %s", self.container_name)
        except ImportError:
            logger.error("azure-storage-blob is not installed. Please install 'azure-storage-blob' to use BlobOperations.")
            raise
        except Exception as e:
            logger.error("Failed to initialize Azure Blob Storage client: %s", e)
            raise

    def fetch_file_from_blob(self, blob_name: str) -> Optional[pd.DataFrame]:
        """
        Fetches a CSV file from Azure Blob Storage and returns it as a Pandas DataFrame.

        :param blob_name: Name of the blob file (e.g. 'balanced_sentiment_dataset.csv')
        :return: Pandas DataFrame or None if fetching fails
        """
        try:
            logger.info("Fetching blob '%s' from container '%s'...", blob_name, self.container_name)
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_data = blob_client.download_blob().readall()
            df = pd.read_csv(StringIO(blob_data.decode("utf-8")))
            logger.info("Successfully loaded '%s' with %d records.", blob_name, len(df))
            return df
        except Exception as e:
            logger.exception("Failed to fetch '%s' from Azure Blob Storage: %s", blob_name, e)
            return None

    def upload_file_to_blob(self, local_file_path: str, blob_name: str, overwrite: bool = True) -> bool:
        """
        Uploads a local file to Azure Blob Storage.

        :param local_file_path: Local path to the file
        :param blob_name: Destination blob name
        :param overwrite: Whether to overwrite existing blob
        :return: True if successful, False otherwise
        """
        try:
            logger.info("Uploading '%s' to blob '%s'...", local_file_path, blob_name)
            blob_client = self.container_client.get_blob_client(blob_name)
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=overwrite)
            logger.info("Successfully uploaded '%s' to '%s'.", local_file_path, blob_name)
            return True
        except Exception as e:
            logger.exception("Failed to upload '%s' to Azure Blob Storage: %s", local_file_path, e)
            return False
