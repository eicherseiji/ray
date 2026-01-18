from functools import wraps
from typing import Any, Callable

from ray.anyscale.lineage.common.constants import IGNORE_ERRORS
from ray.anyscale.lineage.common.exceptions import AnyscaleLineageMLflowError
from ray.anyscale.lineage.common.logging import get_logger

logger = get_logger(__name__)


MLFLOW_ARTIFACTS_URI_SCHEME = "mlflow-artifacts:"


def catch_mlflow_store_exception(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrapper to catch MLflow store exceptions."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = (
                f"Error in MLflow store function '{func.__name__}' "
                f"for args '{args!s}' and kwargs '{kwargs!s}': {e!r}"
            )
            logger.error(error_msg)

            # If IGNORE_ERRORS=True, suppress error and allow workload to continue
            if not IGNORE_ERRORS:
                raise AnyscaleLineageMLflowError(error_msg) from e

    return wrapper


def resolve_http_uri_from_mlflow_artifacts_uri(artifact_uri: str) -> str:
    """Resolve an HTTP URI from a MLflow artifacts URI.

    Converts an mlflow-artifacts:/ URI to an HTTP URI using the tracking URI.
    Other URI schemes are returned unchanged.
    """
    if not artifact_uri.startswith(MLFLOW_ARTIFACTS_URI_SCHEME):
        return artifact_uri

    try:
        from mlflow.store.artifact.mlflow_artifacts_repo import (
            MlflowArtifactsRepository,
        )
        from mlflow.tracking import get_tracking_uri

        return MlflowArtifactsRepository.resolve_uri(artifact_uri, get_tracking_uri())
    except Exception as e:
        logger.debug(
            f"Failed to resolve HTTP URI from MLflow artifacts URI '{artifact_uri}': {e!r}"
        )
        return artifact_uri
