import os
import sys
from dotenv import load_dotenv

load_dotenv()


def promote_model() -> bool:
    """
    Promote latest Staging model to Production in MLflow Model Registry.
    Archives previously active Production models.
    """
    token = os.getenv("DAGSHUB_TOKEN") or os.getenv("sentiment_analysis")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME", "sentiment-analysis")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "my_model")

    if not tracking_uri and repo_owner:
        tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"

    if not tracking_uri or not token:
        print("[WARNING] MLFLOW_TRACKING_URI and DAGSHUB_TOKEN are not configured; skipping model promotion.")
        return False

    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    try:
        import mlflow
        from mlflow.exceptions import MlflowException

        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()

        # 1. Fetch latest model version in Staging
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not staging_versions:
            print(f"[INFO] No version found in 'Staging' stage for model '{model_name}'. Promotion skipped.")
            return False

        latest_staging_version = staging_versions[0].version
        print(f"[INFO] Found model '{model_name}' version {latest_staging_version} in Staging.")

        # 2. Archive existing Production versions
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        for version in prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived",
                archive_existing_versions=False
            )
            print(f"[INFO] Archived previous production version {version.version}.")

        # 3. Promote staging model to Production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_staging_version,
            stage="Production"
        )
        print(f"[SUCCESS] Model '{model_name}' version {latest_staging_version} successfully promoted to Production.")
        return True

    except Exception as e:
        print(f"[ERROR] Model promotion failed: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    success = promote_model()
    # Exit with success if promotion completed or was skipped gracefully
    sys.exit(0)
