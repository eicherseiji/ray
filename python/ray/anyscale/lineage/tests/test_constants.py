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

# Simple test values (for minimal test cases)
SIMPLE_ORG = "test-org-001"
SIMPLE_CLOUD = "test-cloud-aws"
SIMPLE_PROJECT = "ml-project"
SIMPLE_JOB = "test-job-001"
SIMPLE_IDENTIFIER = "test-id-123"

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
TEST_MOCK_RUN_ID_STR = "run-id"

# Database/Collection names
TEST_DB_NAME = "test_db"
TEST_DB_COLLECTION = "test_collection"
TEST_DB_NAME_ALT = "prod_db"
TEST_DB_COLLECTION_ALT = "events"

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

# Common namespace patterns for Ray Data (computed from SIMPLE_* constants)
TEST_RAY_NAMESPACE_PATTERN = f"{SIMPLE_ORG}.{SIMPLE_CLOUD}.{SIMPLE_PROJECT}"
TEST_RAY_JOB_NAME_PATTERN = f"job.{SIMPLE_JOB}"

# UUID pattern for testing (valid UUID v4 format)
TEST_UUID_PATTERN = "12345678-1234-4678-9234-567812345678"


# ============================================================================
# MLflow-specific test constants
# ============================================================================
TEST_MLFLOW_HOST_LOCAL = "mlflow.local"
TEST_MLFLOW_HOST_EXAMPLE = "mlflow.example"
TEST_MLFLOW_EXPERIMENT_ID = "exp-1"
TEST_MLFLOW_RUN_ID = "run-123"
TEST_MLFLOW_RUN_NAME = "test-run"
TEST_MLFLOW_MODEL_UUID = "12345678-abcd-4ef0-9012-3456789abcde"
TEST_MLFLOW_MODEL_URI = "models:/test-model/1"
TEST_MLFLOW_MODEL_NAME = "mlflow-model"

# MLflow artifact and storage URIs
TEST_MLFLOW_S3_URI = "s3://bucket"
TEST_LOCAL_MLRUNS_PATH = "/tmp/mlruns"
TEST_LOCAL_ARTIFACTS_PATH = "/tmp/artifacts"

# Artifact paths
TEST_ARTIFACT_PATH = "artifact/path"
TEST_MODEL_ARTIFACT_PATH = "model/file.pkl"

# Test UUIDs and run IDs for OpenLineage
TEST_WORKLOAD_OL_RUN_ID_ALT = "00000000-0000-0000-0000-00000000000a"
