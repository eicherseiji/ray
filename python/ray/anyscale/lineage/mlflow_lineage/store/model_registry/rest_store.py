from functools import partial
from typing import Any, Optional, Type

from ray.anyscale.lineage.common.openlineage_client import (
    AnyscaleOpenLineageClient,
)

_AnyscaleRestStoreClass: Optional[Type[Any]] = None


def _create_anyscale_rest_store_class() -> Type[Any]:
    """Create the AnyscaleRestStore class."""
    # Import mlflow and mlflow_openlineage dependencies here to avoid
    # circular imports during plugin registration.
    from mlflow.store.model_registry.rest_store import RestStore
    from mlflow.utils.credentials import get_default_host_creds

    from ray.anyscale.lineage.common.logging import get_logger
    from ray.anyscale.lineage.mlflow_lineage.constants import (
        MLFLOW_OPENLINEAGE_PRODUCER,
    )
    from ray.anyscale.lineage.mlflow_lineage.store.model_registry.utils import (
        process_and_emit_ol_events_for_model_registration,
    )

    logger = get_logger(__name__)

    class AnyscaleRestStore(RestStore):
        """Anyscale implementation of MLflow REST model registry store."""

        def __init__(self, store_uri: str, tracking_uri: Optional[str] = None):
            self.is_plugin = True

            get_host_creds_func = partial(get_default_host_creds, store_uri)

            super().__init__(get_host_creds_func)  # type: ignore[no-untyped-call]

            self.ol_client: Optional[AnyscaleOpenLineageClient]
            # Ignore plugin errors to avoid affecting upstream MLflow functionality
            try:
                self.ol_client = AnyscaleOpenLineageClient(
                    ol_producer=MLFLOW_OPENLINEAGE_PRODUCER
                )
            except Exception as e:
                logger.warning(f"Error initializing AnyscaleOpenLineageClient: {e!r}")
                self.ol_client = None

        def create_model_version(
            self,
            name: str,
            source: str,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            result = super().create_model_version(  # type: ignore[no-untyped-call]
                name, source, *args, **kwargs
            )

            if self.ol_client:
                # Ignore plugin errors to avoid affecting upstream MLflow functionality
                try:
                    process_and_emit_ol_events_for_model_registration(
                        ol_client=self.ol_client,
                        model_name=name,
                        model_uri=source,
                        model_version=str(result.version),
                    )
                except Exception as e:
                    logger.warning(
                        f"Error processing and emitting OpenLineage events: {e!r}"
                    )

            return result

    return AnyscaleRestStore


def AnyscaleRestStore(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleRestStore instances."""
    global _AnyscaleRestStoreClass
    if _AnyscaleRestStoreClass is None:
        _AnyscaleRestStoreClass = _create_anyscale_rest_store_class()
    return _AnyscaleRestStoreClass(*args, **kwargs)
