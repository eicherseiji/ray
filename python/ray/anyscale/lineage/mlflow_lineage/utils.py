from enum import Enum, unique
from functools import wraps
from typing import Any, Callable

from ray.anyscale.lineage.common.exceptions import AnyscaleLineageMLflowError
from ray.anyscale.lineage.common.logging import get_logger
from ray.anyscale.lineage.mlflow_lineage.constants import (
    ANYSCALE_MLFLOW_ARTIFACT_REPO_GCS_PREFIX,
    ANYSCALE_MLFLOW_ARTIFACT_REPO_HTTP_PREFIX,
    ANYSCALE_MLFLOW_ARTIFACT_REPO_LOCAL_PREFIX,
    ANYSCALE_MLFLOW_ARTIFACT_REPO_MODELS_PREFIX,
    ANYSCALE_MLFLOW_ARTIFACT_REPO_PREFIX,
    ANYSCALE_MLFLOW_ARTIFACT_REPO_RUNS_PREFIX,
    ANYSCALE_MLFLOW_ARTIFACT_REPO_S3_PREFIX,
    ANYSCALE_MLFLOW_TRACKING_STORE_FILE_PREFIX,
    ANYSCALE_MLFLOW_TRACKING_STORE_PREFIX,
    ANYSCALE_MLFLOW_TRACKING_STORE_REST_PREFIX,
)

logger = get_logger(__name__)


STORE_URI_PREFIX_SEPARATOR = ":"


@unique
class StoreType(str, Enum):
    TRACKING_STORE = "tracking_store"
    ARTIFACT_REPO = "artifact_repo"


def extract_upstream_store_uri(store_uri: str, store_type: StoreType) -> str:
    """Extract the upstream store URI from the user provided store URI."""
    if store_type == StoreType.TRACKING_STORE:
        tracking_store_prefix_map = {
            ANYSCALE_MLFLOW_TRACKING_STORE_FILE_PREFIX: "file",
            ANYSCALE_MLFLOW_TRACKING_STORE_REST_PREFIX: "http",
        }

        for anyscale_prefix, upstream_prefix in tracking_store_prefix_map.items():
            if store_uri.startswith(anyscale_prefix):
                # Replace anyscale prefix with upstream prefix
                store_uri = store_uri.replace(anyscale_prefix, upstream_prefix, 1)
                break
        else:
            # Fallback to generic tracking store prefix stripping
            prefix = ANYSCALE_MLFLOW_TRACKING_STORE_PREFIX
            if store_uri.startswith(prefix):
                store_uri = store_uri.split(STORE_URI_PREFIX_SEPARATOR, 1)[1]

    elif store_type == StoreType.ARTIFACT_REPO:
        artifact_repo_prefix_map = {
            ANYSCALE_MLFLOW_ARTIFACT_REPO_LOCAL_PREFIX: "file",
            ANYSCALE_MLFLOW_ARTIFACT_REPO_MODELS_PREFIX: "models",
            ANYSCALE_MLFLOW_ARTIFACT_REPO_S3_PREFIX: "s3",
            ANYSCALE_MLFLOW_ARTIFACT_REPO_RUNS_PREFIX: "runs",
            ANYSCALE_MLFLOW_ARTIFACT_REPO_HTTP_PREFIX: "http",
            ANYSCALE_MLFLOW_ARTIFACT_REPO_GCS_PREFIX: "gs",
        }

        for anyscale_prefix, upstream_prefix in artifact_repo_prefix_map.items():
            if store_uri.startswith(anyscale_prefix):
                # Replace anyscale prefix with upstream prefix
                store_uri = store_uri.replace(anyscale_prefix, upstream_prefix, 1)
                break
        else:
            # Fallback to generic artifact repo prefix stripping
            prefix = ANYSCALE_MLFLOW_ARTIFACT_REPO_PREFIX
            if store_uri.startswith(prefix):
                store_uri = store_uri.split(STORE_URI_PREFIX_SEPARATOR, 1)[1]

    return store_uri


def catch_mlflow_store_exception(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrapper to catch MLflow store exceptions."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Error in MLflow store function '{func.__name__}' for args '{args!s}' "
                f"and kwargs '{kwargs!s}': {e!r}"
            )
            raise AnyscaleLineageMLflowError(e) from e

    return wrapper
