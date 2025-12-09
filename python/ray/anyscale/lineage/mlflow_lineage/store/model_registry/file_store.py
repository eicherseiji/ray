from typing import Any, Optional, Type

from ray.anyscale.lineage.common.openlineage_client import (
    AnyscaleOpenLineageClient,
)

_AnyscaleFileStoreClass: Optional[Type[Any]] = None


def _create_anyscale_file_store_class() -> Type[Any]:
    """Create the AnyscaleFileStore class."""
    # Import mlflow and mlflow_openlineage dependencies here to avoid
    # circular imports during plugin registration.
    from mlflow.store.model_registry.file_store import FileStore

    from ray.anyscale.lineage.common.logging import get_logger
    from ray.anyscale.lineage.mlflow_lineage.constants import (
        MLFLOW_OPENLINEAGE_PRODUCER,
    )
    from ray.anyscale.lineage.mlflow_lineage.store.model_registry.utils import (
        process_and_emit_ol_events_for_model_registration,
    )

    logger = get_logger(__name__)

    class AnyscaleFileStore(FileStore):
        """Anyscale implementation of MLflow File model registry store."""

        def __init__(self, store_uri: str, tracking_uri: Optional[str] = None):
            self.is_plugin = True

            super().__init__(root_directory=store_uri)  # type: ignore[no-untyped-call]

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
                name=name, source=source, *args, **kwargs
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

    return AnyscaleFileStore


def AnyscaleFileStore(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleFileStore instances."""
    global _AnyscaleFileStoreClass
    if _AnyscaleFileStoreClass is None:
        _AnyscaleFileStoreClass = _create_anyscale_file_store_class()
    return _AnyscaleFileStoreClass(*args, **kwargs)
