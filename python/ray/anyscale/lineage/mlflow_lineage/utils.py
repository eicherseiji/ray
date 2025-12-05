from functools import wraps
from typing import Any, Callable

from ray.anyscale.lineage.common.constants import IGNORE_ERRORS
from ray.anyscale.lineage.common.exceptions import AnyscaleLineageMLflowError
from ray.anyscale.lineage.common.logging import get_logger


logger = get_logger(__name__)


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
