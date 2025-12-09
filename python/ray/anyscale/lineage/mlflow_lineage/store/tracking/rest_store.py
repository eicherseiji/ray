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
    from mlflow.store.tracking.rest_store import RestStore
    from mlflow.utils.credentials import get_default_host_creds

    from ray.anyscale.lineage.common.logging import get_logger
    from ray.anyscale.lineage.mlflow_lineage.constants import (
        MLFLOW_OPENLINEAGE_PRODUCER,
    )
    from ray.anyscale.lineage.mlflow_lineage.store.tracking.utils import (
        process_and_emit_ol_events_for_model_logging,
    )

    logger = get_logger(__name__)

    class AnyscaleRestStore(RestStore):
        """Anyscale implementation of MLflow REST tracking store."""

        def __init__(self, store_uri: str, artifact_uri: Optional[str] = None):
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

        def record_logged_model(self, run_id: str, mlflow_model: Any) -> None:
            super().record_logged_model(run_id, mlflow_model)  # type: ignore[no-untyped-call]

            if self.ol_client:
                # Ignore plugin errors to avoid affecting upstream MLflow functionality
                try:
                    process_and_emit_ol_events_for_model_logging(
                        ol_client=self.ol_client,
                        run=self.get_run(run_id),  # type: ignore[no-untyped-call]
                        mlflow_model=mlflow_model,
                    )
                except Exception as e:
                    logger.warning(
                        f"Error processing and emitting OpenLineage events: {e!r}"
                    )

    return AnyscaleRestStore


def AnyscaleRestStore(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleRestStore instances."""
    global _AnyscaleRestStoreClass
    if _AnyscaleRestStoreClass is None:
        _AnyscaleRestStoreClass = _create_anyscale_rest_store_class()
    return _AnyscaleRestStoreClass(*args, **kwargs)
