from ray.anyscale.lineage.mlflow_lineage.constants import (
    ANYSCALE_MLFLOW_ARTIFACT_REPO_PREFIX,
    ANYSCALE_MLFLOW_TRACKING_STORE_FILE_PREFIX,
    ANYSCALE_MLFLOW_TRACKING_STORE_REST_PREFIX,
)

# ============================================================================
# Anyscale environment constants
# ============================================================================
TEST_ORG_ID = "org-123"
TEST_CLOUD_ID = "cloud-456"
TEST_PROJECT_ID = "project-789"
TEST_USER_EMAIL = "test@example.com"
TEST_OWNER_EMAIL = "owner@example.com"
TEST_WORKLOAD_NAME = "test-workload"
TEST_WORKLOAD_TYPE = "job"
TEST_JOB_ID = "job-abc123"
TEST_SERVICE_ID = "service-xyz789"
TEST_WORKSPACE_ID = "workspace-def456"
TEST_RUN_ID = "run-ghi789"

# Version constants
TEST_RAY_VERSION = "2.7.0"
TEST_PYTHON_VERSION = "3.11"
TEST_OS_VERSION = "linux"

# Simple test values (for minimal test cases)
SIMPLE_ORG = "org"
SIMPLE_CLOUD = "cloud"
SIMPLE_PROJECT = "project"
SIMPLE_JOB = "job"
SIMPLE_IDENTIFIER = "identifier"

# OpenLineage structure constants (for testing OpenLineage objects)
TEST_OL_NAMESPACE = "test-namespace"
TEST_OL_JOB_NAME = "test-job"
TEST_OL_DATASET_NAME = "test-dataset"
TEST_OL_RUN_ID_STR = "test-run-id"
# Shorter variants for mock objects
TEST_OL_NAMESPACE_SHORT = "test-ns"
TEST_OL_NAME_GENERIC = "test-name"

# ============================================================================
# Ray Data-specific test constants
# ============================================================================
TEST_DATASET_ID = "dataset-123"
TEST_RUN_ID_SHORT = "run-id"

# Database/Collection names
TEST_DB_NAME = "test_db"
TEST_DB_COLLECTION = "test_collection"
PROD_DB_NAME = "prod_db"
PROD_DB_COLLECTION = "events"

# Databricks constants
TEST_DATABRICKS_HOST = "dbc-12345678-abcd.cloud.databricks.com"
TEST_WAREHOUSE_ID = "warehouse123"
TEST_CATALOG_NAME = "main"
TEST_SCHEMA_NAME = "default"
TEST_TABLE_NAME = "users"

# URI patterns for Ray Data testing
TEST_RAY_DATA_S3_URI = "s3://bucket/path"
TEST_RAY_DATA_FILE_URI = "file:///tmp/out"
TEST_MONGO_URI = "mongodb://localhost:27017"
TEST_MONGO_URI_AUTH = "mongodb://user:pass@cluster.mongodb.net:27017"
TEST_DATABRICKS_URL = f"https://{TEST_DATABRICKS_HOST}"

# Common namespace patterns for Ray Data
TEST_RAY_NAMESPACE_PATTERN = "org.cloud.project"
TEST_RAY_JOB_NAME_PATTERN = "job.job-id"

# UUID pattern for testing
TEST_UUID_PATTERN = "12345678-1234-5678-1234-567812345678"


# ============================================================================
# MLflow-specific test constants
# ============================================================================
TEST_MLFLOW_HOST_LOCAL = "mlflow.local"
TEST_MLFLOW_HOST_EXAMPLE = "mlflow.example"
TEST_MLFLOW_EXPERIMENT_ID = "exp-1"
TEST_MLFLOW_RUN_ID = "run-1"
TEST_MLFLOW_RUN_ID_ALT = "run-123"
TEST_MLFLOW_RUN_NAME = "test-run"
TEST_MLFLOW_MODEL_UUID = "uuid-123"
TEST_MLFLOW_MODEL_URI = "models:/uri"
TEST_MLFLOW_MODEL_NAME = "mlflow-model"
TEST_MLFLOW_VERSION = "2.7.0"
TEST_MLFLOW_VERSION_29 = "2.9.0"
TEST_MLFLOW_VERSION_30 = "3.0.0"

# MLflow artifact and storage URIs
TEST_MLFLOW_S3_URI = "s3://bucket"
TEST_MLFLOW_S3_URI_WITH_PATH = "s3://bucket/path"
TEST_MLFLOW_FILE_URI = "file:///tmp/artifacts"
TEST_LOCAL_MLRUNS_PATH = "/tmp/mlruns"
TEST_LOCAL_ARTIFACTS_PATH = "/tmp/artifacts"

# Artifact paths
TEST_ARTIFACT_PATH = "artifact/path"
TEST_MODEL_ARTIFACT_PATH = "model/file.pkl"

# Test UUIDs and run IDs for OpenLineage
TEST_GENERATED_RUN_ID = "12345678-1234-1234-1234-123456789abc"
TEST_WORKLOAD_OL_RUN_ID_ALT = "00000000-0000-0000-0000-00000000000a"

# Schema field names
TEST_SCHEMA_FIELD_COL1 = "col1"
TEST_SCHEMA_FIELD_COL2 = "col2"
TEST_SCHEMA_FIELD_COLA = "colA"
TEST_SCHEMA_FIELD_COLB = "colB"
TEST_SCHEMA_FIELD_F1 = "f1"

# Plugin prefixes
MLFLOW_TRACKING_STORE_FILE_PREFIX = f"{ANYSCALE_MLFLOW_TRACKING_STORE_FILE_PREFIX}:"
MLFLOW_TRACKING_STORE_REST_PREFIX = f"{ANYSCALE_MLFLOW_TRACKING_STORE_REST_PREFIX}:"
MLFLOW_ARTIFACT_REPO_PREFIX = f"{ANYSCALE_MLFLOW_ARTIFACT_REPO_PREFIX}:"
