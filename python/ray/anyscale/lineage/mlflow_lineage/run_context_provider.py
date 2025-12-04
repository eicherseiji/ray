import os
from enum import Enum, unique
from typing import Any, Dict, Optional, Type


def is_lineage_tracking_enabled() -> bool:
    """Check if lineage tracking is enabled."""
    # TODO (@sanketrai): check environment variable
    return True


@unique
class Tags(str, Enum):
    ANYSCALE_JOB_ID = "ANYSCALE_JOB_ID"
    ANYSCALE_PROJECT_ID = "ANYSCALE_PROJECT_ID"
    ANYSCALE_SERVICE_ID = "ANYSCALE_SERVICE_ID"
    ANYSCALE_WORKLOAD_NAME = "ANYSCALE_WORKLOAD_NAME"
    ANYSCALE_WORKLOAD_TYPE = "ANYSCALE_WORKLOAD_TYPE"
    ANYSCALE_WORKSPACE_ID = "ANYSCALE_WORKSPACE_ID"


_AnyscaleRunContextProviderClass: Optional[Type[Any]] = None


def _create_anyscale_run_context_provider_class() -> Type[Any]:
    """Create the AnyscaleRunContextProvider class."""
    # Import mlflow dependencies here to avoid circular imports during plugin registration.
    from mlflow.tracking.context.abstract_context import RunContextProvider

    class AnyscaleRunContextProvider(RunContextProvider):
        """Anyscale implementation of MLflow run context provider."""

        @staticmethod
        def in_context() -> bool:
            return is_lineage_tracking_enabled()

        @staticmethod
        def tags() -> Dict[str, str]:
            available_tags = {}
            for tag in Tags:
                if value := os.getenv(tag.value):
                    available_tags[tag.value] = value
            return available_tags

    return AnyscaleRunContextProvider


def AnyscaleRunContextProvider(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleRunContextProvider instances."""
    global _AnyscaleRunContextProviderClass
    if _AnyscaleRunContextProviderClass is None:
        _AnyscaleRunContextProviderClass = _create_anyscale_run_context_provider_class()
    return _AnyscaleRunContextProviderClass(*args, **kwargs)
