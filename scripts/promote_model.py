import os
import mlflow
from mlflow.exceptions import MlflowException

def promote_model():

    dagshub_token = os.getenv("sentiment_analysis")
    if not dagshub_token:
        raise EnvironmentError("sentiment_analysis environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "rohitkr8527"
    repo_name = "sentiment-analysis"

    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

    client = mlflow.MlflowClient()
    model_name = "my_model"

    # Get latest staging version
    try:
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not staging_versions:
            print("No model in Staging stage to promote.")
            return
        latest_staging_version = staging_versions[0].version
    except MlflowException as e:
        print(f"Error fetching staging model: {e}")
        return

    # Archive existing Production versions
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived",
            archive_existing_versions=False
        )

    # Promote the staging model to Production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_staging_version,
        stage="Production"
    )

    print(f"Model version {latest_staging_version} promoted to Production ✅")

if __name__ == "__main__":
    promote_model()
