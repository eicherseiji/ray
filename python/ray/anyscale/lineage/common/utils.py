import json
import os
import platform
import sys
import uuid
from datetime import datetime, timezone
from enum import Enum, unique
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, Type
from urllib.parse import urlsplit

import ray
from openlineage.client.event_v2 import (
    Dataset,
    InputDataset,
    Job,
    OutputDataset,
    Run,
)
from openlineage.client.facet_v2 import (
    DatasetFacet,
    InputDatasetFacet,
    JobFacet,
    OutputDatasetFacet,
    RunFacet,
)
from openlineage.client.uuid import generate_new_uuid

from ray.anyscale.lineage.common.constants import LINEAGE_EVENTS_LOG_FILENAME
from ray.anyscale.lineage.common.exceptions import AnyscaleLineageClientError
from ray.anyscale.lineage.common.logging import get_logger

logger = get_logger(__name__)


@unique
class AnyscaleEnvironmentVariables(Enum):
    ANYSCALE_CLOUD_ID = "ANYSCALE_CLOUD_ID"
    ANYSCALE_CLUSTER_ID = "ANYSCALE_CLUSTER_ID"
    ANYSCALE_JOB_ID = "ANYSCALE_JOB_ID"
    ANYSCALE_ORGANIZATION_ID = "ANYSCALE_ORGANIZATION_ID"
    ANYSCALE_PROJECT_ID = "ANYSCALE_PROJECT_ID"
    ANYSCALE_SERVICE_ID = "ANYSCALE_SERVICE_ID"
    ANYSCALE_USER_EMAIL = "ANYSCALE_USER_EMAIL"
    ANYSCALE_WORKLOAD_NAME = "ANYSCALE_WORKLOAD_NAME"
    ANYSCALE_WORKLOAD_TYPE = "ANYSCALE_WORKLOAD_TYPE"
    ANYSCALE_WORKSPACE_ID = "ANYSCALE_WORKSPACE_ID"


@unique
class AnyscaleWorkloadTypes(Enum):
    JOB = "JOB"
    SERVICE = "SERVICE"
    WORKSPACE = "WORKSPACE"


# Global variables for values that don't change for the lifetime of the process
_anyscale_workload_id: Optional[str] = None
_anyscale_workload_ol_job_name: Optional[str] = None
_anyscale_workload_ol_job_namespace: Optional[str] = None
_anyscale_workload_ol_run_id: Optional[str] = None
_anyscale_workload_os_version: Optional[str] = None
_anyscale_workload_python_version: Optional[str] = None
_anyscale_workload_ray_version: Optional[str] = None


def get_anyscale_workload_id() -> str:
    """Get the current Anyscale workload ID."""
    global _anyscale_workload_id
    if _anyscale_workload_id is not None:
        return _anyscale_workload_id

    anyscale_workload_type = os.environ.get(
        AnyscaleEnvironmentVariables.ANYSCALE_WORKLOAD_TYPE.value, ""
    )
    if anyscale_workload_type.lower() == AnyscaleWorkloadTypes.JOB.value.lower():
        _anyscale_workload_id = os.environ.get(
            AnyscaleEnvironmentVariables.ANYSCALE_JOB_ID.value, ""
        )
    elif anyscale_workload_type.lower() == AnyscaleWorkloadTypes.SERVICE.value.lower():
        _anyscale_workload_id = os.environ.get(
            AnyscaleEnvironmentVariables.ANYSCALE_SERVICE_ID.value, ""
        )
    elif (
        anyscale_workload_type.lower() == AnyscaleWorkloadTypes.WORKSPACE.value.lower()
    ):
        _anyscale_workload_id = os.environ.get(
            AnyscaleEnvironmentVariables.ANYSCALE_WORKSPACE_ID.value, ""
        )
    else:
        raise ValueError("Anyscale workload type is required.")
    return _anyscale_workload_id


def get_anyscale_workload_ol_job_namespace() -> str:
    """Get OpenLineage job namespace for the current Anyscale workload."""
    global _anyscale_workload_ol_job_namespace
    if _anyscale_workload_ol_job_namespace is not None:
        return _anyscale_workload_ol_job_namespace

    anyscale_organization_id = os.environ.get(
        AnyscaleEnvironmentVariables.ANYSCALE_ORGANIZATION_ID.value, ""
    )
    anyscale_cloud_id = os.environ.get(
        AnyscaleEnvironmentVariables.ANYSCALE_CLOUD_ID.value, ""
    )
    anyscale_project_id = os.environ.get(
        AnyscaleEnvironmentVariables.ANYSCALE_PROJECT_ID.value, ""
    )
    if not all((anyscale_organization_id, anyscale_cloud_id, anyscale_project_id)):
        raise ValueError(
            "Anyscale organization ID, cloud ID, and project ID are required."
        )

    _anyscale_workload_ol_job_namespace = ".".join(
        (
            anyscale_organization_id,
            anyscale_cloud_id,
            anyscale_project_id,
        )
    )
    return _anyscale_workload_ol_job_namespace


def get_anyscale_workload_ol_job_name() -> str:
    """Get OpenLineage job name for the current Anyscale workload."""
    global _anyscale_workload_ol_job_name
    if _anyscale_workload_ol_job_name is not None:
        return _anyscale_workload_ol_job_name

    anyscale_workload_type = os.environ.get(
        AnyscaleEnvironmentVariables.ANYSCALE_WORKLOAD_TYPE.value, ""
    )
    anyscale_workload_id = get_anyscale_workload_id()
    if not all((anyscale_workload_type, anyscale_workload_id)):
        raise ValueError("Anyscale workload type and ID are required.")

    _anyscale_workload_ol_job_name = ".".join(
        (
            str(anyscale_workload_type).lower(),
            str(anyscale_workload_id),
        )
    )
    return _anyscale_workload_ol_job_name


def get_anyscale_workload_ol_run_id() -> str:
    """Get OpenLineage run ID for the current Anyscale workload.

    Computes a deterministic UUID5 based on workload namespace and name.
    Same workload will always produce the same run ID.
    """
    global _anyscale_workload_ol_run_id
    if _anyscale_workload_ol_run_id is not None:
        return _anyscale_workload_ol_run_id

    # Get namespace and name for UUID5 computation
    namespace = get_anyscale_workload_ol_job_namespace()
    name = get_anyscale_workload_ol_job_name()

    # Compute deterministic UUID5
    combined = f"{namespace}.{name}"
    run_id_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, combined)

    _anyscale_workload_ol_run_id = str(run_id_uuid)
    return _anyscale_workload_ol_run_id


def get_python_version() -> str:
    """Get the current Python version."""
    global _anyscale_workload_python_version
    if _anyscale_workload_python_version is not None:
        return _anyscale_workload_python_version

    _anyscale_workload_python_version = sys.version
    return _anyscale_workload_python_version


def get_ray_version() -> str:
    """Get the current Ray version."""
    global _anyscale_workload_ray_version
    if _anyscale_workload_ray_version is not None:
        return _anyscale_workload_ray_version

    _anyscale_workload_ray_version = ray.__version__
    return _anyscale_workload_ray_version


def get_os_version() -> str:
    """Get the current OS version."""
    global _anyscale_workload_os_version
    if _anyscale_workload_os_version is not None:
        return _anyscale_workload_os_version

    _anyscale_workload_os_version = json.dumps(
        {
            "name": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        }
    )
    return _anyscale_workload_os_version


def get_now_utc_datetime() -> str:
    """Get the current UTC datetime."""
    return datetime.now(timezone.utc).isoformat()


def generate_openlineage_run_id() -> str:
    """Generate a new OpenLineage run ID."""
    return str(generate_new_uuid())


def create_openlineage_job_from_args(
    job_namespace: str,
    job_name: str,
    facets: Optional[Dict[str, JobFacet]] = None,
) -> Job:
    """Create an OpenLineage job from arguments."""
    return Job(namespace=job_namespace, name=job_name, facets=facets)


def create_openlineage_run_from_args(
    run_id: str,
    facets: Optional[Dict[str, RunFacet]] = None,
) -> Run:
    """Create an OpenLineage run from arguments."""
    return Run(runId=run_id, facets=facets)


def create_openlineage_dataset_from_args(
    dataset_namespace: str,
    dataset_name: str,
    facets: Optional[Dict[str, DatasetFacet]] = None,
) -> Dataset:
    """Create an OpenLineage dataset from arguments."""
    return Dataset(
        namespace=dataset_namespace,
        name=dataset_name,
        facets=facets,
    )


def create_openlineage_input_dataset_from_args(
    dataset_namespace: str,
    dataset_name: str,
    input_facets: Optional[Dict[str, InputDatasetFacet]] = None,
    facets: Optional[Dict[str, DatasetFacet]] = None,
) -> InputDataset:
    """Create an OpenLineage input dataset from arguments."""
    return InputDataset(
        namespace=dataset_namespace,
        name=dataset_name,
        inputFacets=input_facets,
        facets=facets,
    )


def create_openlineage_output_dataset_from_args(
    dataset_namespace: str,
    dataset_name: str,
    output_facets: Optional[Dict[str, OutputDatasetFacet]] = None,
    facets: Optional[Dict[str, DatasetFacet]] = None,
) -> OutputDataset:
    """Create an OpenLineage output dataset from arguments."""
    return OutputDataset(
        namespace=dataset_namespace,
        name=dataset_name,
        outputFacets=output_facets,
        facets=facets,
    )


def parse_uri(uri: str) -> dict[str, str]:
    """Parse a URI and return its components as a dictionary."""
    return urlsplit(uri)._asdict()


def catch_ol_client_exception(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrapper to catch OpenLineage client exceptions."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Error in OpenLineage client method '{func.__name__}' for args '{args!s}' "
                f"and kwargs '{kwargs!s}': {e!r}"
            )
            raise AnyscaleLineageClientError(e) from e

    return wrapper


def catch_class_method_exception(
    handler: Callable[..., Any], exceptions: Type[Exception] = Exception
) -> Callable[..., Any]:
    """Function decorator that wraps a class method to handle exceptions."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                return handler(e, func.__name__, args, kwargs)

        return wrapper

    return decorator


def wrap_class_methods(
    *,
    decorator: Callable[..., Any],
    exclude: Tuple[str, ...] = (),
    include_inherited: bool = False,
) -> Callable[..., Any]:
    """Class decorator that applies `decorator` to all callables/properties."""

    def class_decorator(cls: Type[Any]) -> Type[Any]:
        # Collect all methods to wrap
        methods_to_wrap = []

        if include_inherited:
            # Get all attributes including inherited ones
            for name in dir(cls):
                if name in exclude or (name.startswith("__") and name.endswith("__")):
                    continue
                try:
                    attr = getattr(cls, name)
                    if callable(attr) and not isinstance(attr, type):
                        methods_to_wrap.append(name)
                except AttributeError:
                    continue
        else:
            # Only get attributes defined directly in this class
            for name, _attr in cls.__dict__.items():
                if name in exclude or (name.startswith("__") and name.endswith("__")):
                    continue
                methods_to_wrap.append(name)

        # Wrap each method
        for name in methods_to_wrap:
            # Check if it's defined in the class's own __dict__ to handle special descriptors
            own_attr = cls.__dict__.get(name)

            if own_attr is not None:
                # Handle methods defined directly in this class
                if isinstance(own_attr, staticmethod):
                    setattr(cls, name, staticmethod(decorator(own_attr.__func__)))
                elif isinstance(own_attr, classmethod):
                    setattr(cls, name, classmethod(decorator(own_attr.__func__)))
                elif isinstance(own_attr, property):
                    fget = decorator(own_attr.fget) if own_attr.fget else None
                    fset = decorator(own_attr.fset) if own_attr.fset else None
                    fdel = decorator(own_attr.fdel) if own_attr.fdel else None
                    setattr(cls, name, property(fget, fset, fdel, own_attr.__doc__))
                elif callable(own_attr):
                    setattr(cls, name, decorator(own_attr))
            else:
                # Handle inherited methods - get the actual attribute and wrap it
                attr = getattr(cls, name)
                if callable(attr) and not isinstance(attr, type):
                    setattr(cls, name, decorator(attr))

        return cls

    return class_decorator


def get_os_env(key: str, default: str = "") -> str:
    """Get an environment variable and return a default value if it is not set."""
    return os.environ.get(key, default)


def evaluate_and_transform_uri(uri: str) -> Tuple[bool, str]:
    """Evaluate if a URI should be tracked, and conditionally transform it.

    For local paths, only track /mnt/cluster_storage/ and /mnt/shared_storage/ paths.
    These local paths are transformed to include file:// prefix which causes them
    to be treated as remote filesystem paths by OpenLineage. This is intended to include
    Anyscale cloud and workload identifiers in the OpenLineage dataset namespaces.

    Remote paths pass through unchanged and are always tracked.
    """
    # Handle None/empty input
    if not uri:
        return (False, uri)

    # Parse URI and extract scheme
    parsed = parse_uri(uri)
    scheme = parsed["scheme"]
    path = parsed["path"]

    # Only process file and local schemes
    # Non-local schemes (s3, http, etc.) pass through unchanged
    if scheme and scheme not in ("file", "local"):
        return (True, uri)

    # Check for remote filesystem (file://<host>/path)
    # Remote filesystem paths pass through unchanged
    if scheme == "file" and parsed["netloc"]:
        return (True, uri)

    # Don't track paths that don't start with /mnt/
    if not path.startswith("/mnt/"):
        return (False, uri)

    # Don't track /mnt/user_storage/ paths
    if path.startswith("/mnt/user_storage/"):
        return (False, uri)

    # Transform /mnt/cluster_storage/ paths with workload_id
    if path.startswith("/mnt/cluster_storage/"):
        try:
            workload_id = get_anyscale_workload_id()
            transformed_uri = f"file://{workload_id}{path}"
            return (True, transformed_uri)
        except ValueError as e:
            logger.error(
                f"Error getting Anyscale workload ID for path transformation: {e}. "
                f"Path will not be tracked: {uri}"
            )
            return (False, uri)

    # Transform /mnt/shared_storage/ paths with cloud_id
    if path.startswith("/mnt/shared_storage/"):
        cloud_id = get_os_env(AnyscaleEnvironmentVariables.ANYSCALE_CLOUD_ID.value)
        if cloud_id:
            transformed_uri = f"file://{cloud_id}{path}"
            return (True, transformed_uri)
        else:
            logger.error(
                f"Anyscale cloud ID not found for path transformation. "
                f"Path will not be tracked: {uri}"
            )
            return (False, uri)

    # Don't track any other paths
    return (False, uri)


def get_lineage_logs_dir() -> str:
    """Get the lineage logs directory path.

    Returns the path to <session_dir>/logs/lineage, creating it if necessary.
    """
    from ray._private.worker import _global_node

    logs_dir = _global_node.get_logs_dir_path()
    lineage_logs_dir = os.path.join(logs_dir, "lineage")
    os.makedirs(lineage_logs_dir, exist_ok=True)
    return lineage_logs_dir


def get_openlineage_events_log_path() -> str:
    """Get the full path for the OpenLineage events log file."""
    return os.path.join(get_lineage_logs_dir(), LINEAGE_EVENTS_LOG_FILENAME)


def get_anyscale_openlineage_config() -> dict:
    """Get the OpenLineage configuration with dynamic log file path."""
    return {
        "transport": {
            "type": "composite",
            "continue_on_success": False,  # stops emission if one of the transport succeeds
            "sort_transports": True,  # sorts the transports by priority
            "transports": {
                "first": {
                    "type": "http",
                    "url": "http://0.0.0.0:8691",  # Vector collector endpoint
                    "endpoint": "",
                },
                "second": {
                    "type": "file",
                    "log_file_path": get_openlineage_events_log_path(),
                    "append": True,
                    "priority": 1,
                },
            },
        }
    }
