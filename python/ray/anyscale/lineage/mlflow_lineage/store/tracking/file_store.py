from typing import Any, Optional, Type

from ray.anyscale.lineage.common.openlineage_client import (
    AnyscaleOpenLineageClient,
)
from ray.anyscale.lineage.mlflow_lineage.utils import catch_mlflow_store_exception

_AnyscaleFileStoreClass: Optional[Type[Any]] = None


def _create_anyscale_file_store_class() -> Type[Any]:
    """Create the AnyscaleFileStore class."""
    # Import mlflow and mlflow_openlineage dependencies here to avoid
    # circular imports during plugin registration.
    from mlflow.store.tracking.file_store import FileStore

    from ray.anyscale.lineage.mlflow_lineage.constants import (
        MLFLOW_OPENLINEAGE_PRODUCER,
    )
    from ray.anyscale.lineage.mlflow_lineage.store.tracking.utils import (
        process_and_emit_ol_events_for_model_logging,
    )

    class AnyscaleFileStore(FileStore):
        """Anyscale implementation of MLflow File tracking store."""

        def __init__(self, store_uri: str, artifact_uri: Optional[str] = None):
            self.is_plugin = True

            super().__init__(
                root_directory=store_uri, artifact_root_uri=artifact_uri
            )  # type: ignore[no-untyped-call]

            self.ol_client = AnyscaleOpenLineageClient(
                ol_producer=MLFLOW_OPENLINEAGE_PRODUCER
            )

            self.host = store_uri

        @catch_mlflow_store_exception
        def record_logged_model(self, run_id: str, mlflow_model: Any) -> None:
            super().record_logged_model(run_id, mlflow_model)  # type: ignore[no-untyped-call]

            process_and_emit_ol_events_for_model_logging(
                ol_client=self.ol_client,
                mlflow_host=self.host,
                run=self.get_run(run_id),  # type: ignore[no-untyped-call]
                mlflow_model=mlflow_model,
            )

    return AnyscaleFileStore


def AnyscaleFileStore(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleFileStore instances."""
    global _AnyscaleFileStoreClass
    if _AnyscaleFileStoreClass is None:
        _AnyscaleFileStoreClass = _create_anyscale_file_store_class()
    return _AnyscaleFileStoreClass(*args, **kwargs)
