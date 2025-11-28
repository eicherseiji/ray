from ray.anyscale.lineage.common.constants import OL_PRODUCER

MLFLOW_OPENLINEAGE_PRODUCER = f"{OL_PRODUCER}/mlflow"

# MLflow plugin prefixes (generic)
ANYSCALE_MLFLOW_ARTIFACT_REPO_PREFIX = "anyscale-mlflow-artifact-repo"
ANYSCALE_MLFLOW_TRACKING_STORE_PREFIX = "anyscale-mlflow-tracking-store"

# MLflow tracking store specific prefixes
ANYSCALE_MLFLOW_TRACKING_STORE_FILE_PREFIX = "anyscale-mlflow-tracking-store-file"
ANYSCALE_MLFLOW_TRACKING_STORE_REST_PREFIX = "anyscale-mlflow-tracking-store-rest"

# MLflow artifact repo specific prefixes
ANYSCALE_MLFLOW_ARTIFACT_REPO_LOCAL_PREFIX = "anyscale-mlflow-artifact-repo-local"
ANYSCALE_MLFLOW_ARTIFACT_REPO_MODELS_PREFIX = "anyscale-mlflow-artifact-repo-models"
ANYSCALE_MLFLOW_ARTIFACT_REPO_S3_PREFIX = "anyscale-mlflow-artifact-repo-s3"
ANYSCALE_MLFLOW_ARTIFACT_REPO_RUNS_PREFIX = "anyscale-mlflow-artifact-repo-runs"
ANYSCALE_MLFLOW_ARTIFACT_REPO_HTTP_PREFIX = "anyscale-mlflow-artifact-repo-http"
ANYSCALE_MLFLOW_ARTIFACT_REPO_GCS_PREFIX = "anyscale-mlflow-artifact-repo-gcs"
