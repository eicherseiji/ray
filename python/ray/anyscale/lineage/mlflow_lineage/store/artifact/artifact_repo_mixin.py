from typing import Any, Optional

from ray.anyscale.lineage.common.logging import get_logger
from ray.anyscale.lineage.common.openlineage_client import AnyscaleOpenLineageClient
from ray.anyscale.lineage.mlflow_lineage.constants import (
    MLFLOW_OPENLINEAGE_PRODUCER,
)
from ray.anyscale.lineage.mlflow_lineage.store.artifact.utils import (
    ArtifactRepoOperations,
    process_and_emit_ol_events_for_artifact_repo_operation,
)


logger = get_logger(__name__)


class AnyscaleArtifactRepositoryMixin:
    def __init__(self, artifact_uri: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(artifact_uri, *args, **kwargs)  # type: ignore[call-arg]

        # This flag tracks if we are inside a `log_artifacts` call
        # Since `log_artifacts` calls `log_artifact` for each file, skip emitting
        # OpenLineage events from `log_artifact` when inside a `log_artifacts` call
        self._inside_log_artifacts = False

        self.ol_client: Optional[AnyscaleOpenLineageClient]
        # Ignore plugin errors to avoid affecting upstream MLflow functionality
        try:
            self.ol_client = AnyscaleOpenLineageClient(
                ol_producer=MLFLOW_OPENLINEAGE_PRODUCER
            )
        except Exception as e:
            logger.warning(f"Error initializing AnyscaleOpenLineageClient: {e!r}")
            self.ol_client = None

    def download_artifacts(
        self,
        artifact_path: str,
        dst_path: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        result: str = super().download_artifacts(  # type: ignore[misc]
            artifact_path, dst_path, *args, **kwargs
        )

        if self.ol_client:
            # Ignore plugin errors to avoid affecting upstream MLflow functionality
            try:
                process_and_emit_ol_events_for_artifact_repo_operation(
                    operation=ArtifactRepoOperations.DOWNLOAD,
                    ol_client=self.ol_client,
                    artifact_uri=self.artifact_uri,  # type: ignore[attr-defined]
                    artifact_path=artifact_path,
                )
            except Exception as e:
                logger.warning(
                    f"Error processing and emitting OpenLineage events: {e!r}"
                )

        return result

    def log_artifact(
        self,
        local_file: str,
        artifact_path: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result: Any = super().log_artifact(local_file, artifact_path, *args, **kwargs)  # type: ignore[misc]

        if self._inside_log_artifacts:
            return result

        if artifact_path is None:
            artifact_path = local_file

        if self.ol_client:
            # Ignore plugin errors to avoid affecting upstream MLflow functionality
            try:
                process_and_emit_ol_events_for_artifact_repo_operation(
                    operation=ArtifactRepoOperations.LOG,
                    ol_client=self.ol_client,
                    artifact_uri=self.artifact_uri,  # type: ignore[attr-defined]
                    artifact_path=artifact_path,
                )
            except Exception as e:
                logger.warning(
                    f"Error processing and emitting OpenLineage events: {e!r}"
                )

        return result

    def log_artifacts(
        self,
        local_dir: str,
        artifact_path: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self._inside_log_artifacts = True

        try:
            result: Any = super().log_artifacts(local_dir, artifact_path, *args, **kwargs)  # type: ignore[misc]
        finally:
            self._inside_log_artifacts = False

        if artifact_path is None:
            artifact_path = local_dir

        if self.ol_client:
            # Ignore plugin errors to avoid affecting upstream MLflow functionality
            try:
                process_and_emit_ol_events_for_artifact_repo_operation(
                    operation=ArtifactRepoOperations.LOG,
                    ol_client=self.ol_client,
                    artifact_uri=self.artifact_uri,  # type: ignore[attr-defined]
                    artifact_path=artifact_path,
                )
            except Exception as e:
                logger.warning(
                    f"Error processing and emitting OpenLineage events: {e!r}"
                )

        return result
