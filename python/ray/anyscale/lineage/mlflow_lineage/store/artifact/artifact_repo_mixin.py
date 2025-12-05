import os
from typing import Any, Optional

from ray.anyscale.lineage.common.openlineage_client import AnyscaleOpenLineageClient
from ray.anyscale.lineage.mlflow_lineage.constants import (
    MLFLOW_OPENLINEAGE_PRODUCER,
)
from ray.anyscale.lineage.mlflow_lineage.utils import catch_mlflow_store_exception
from ray.anyscale.lineage.mlflow_lineage.store.artifact.utils import (
    ArtifactRepoOperations,
    process_and_emit_ol_events_for_artifact_repo_operation,
    should_emit_openlineage_event_for_artifact,
)


class AnyscaleArtifactRepositoryMixin:
    def __init__(self, artifact_uri: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(artifact_uri, *args, **kwargs)  # type: ignore[call-arg]
        self.artifact_uri = artifact_uri
        self.ol_client = AnyscaleOpenLineageClient(
            ol_producer=MLFLOW_OPENLINEAGE_PRODUCER
        )

    @catch_mlflow_store_exception
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

        # Skip OpenLineage events for runs:/ URIs
        full_artifact_path = os.path.join(self.artifact_uri, artifact_path)
        if not should_emit_openlineage_event_for_artifact(full_artifact_path):
            return result

        process_and_emit_ol_events_for_artifact_repo_operation(
            operation=ArtifactRepoOperations.DOWNLOAD,
            ol_client=self.ol_client,
            artifact_uri=self.artifact_uri,
            artifact_path=artifact_path,
        )

        return result

    @catch_mlflow_store_exception
    def log_artifact(
        self,
        local_file: str,
        artifact_path: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result: Any = super().log_artifact(local_file, artifact_path, *args, **kwargs)  # type: ignore[misc]

        if artifact_path is None:
            artifact_path = local_file

        # Skip OpenLineage events for runs:/ URIs
        full_artifact_path = os.path.join(self.artifact_uri, artifact_path)
        if not should_emit_openlineage_event_for_artifact(full_artifact_path):
            return result

        process_and_emit_ol_events_for_artifact_repo_operation(
            operation=ArtifactRepoOperations.LOG,
            ol_client=self.ol_client,
            artifact_uri=self.artifact_uri,
            artifact_path=artifact_path,
        )

        return result

    @catch_mlflow_store_exception
    def log_artifacts(
        self,
        local_dir: str,
        artifact_path: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result: Any = super().log_artifacts(local_dir, artifact_path, *args, **kwargs)  # type: ignore[misc]

        if artifact_path is None:
            artifact_path = local_dir

        # Skip OpenLineage events for runs:/ URIs
        full_artifact_path = os.path.join(self.artifact_uri, artifact_path)
        if not should_emit_openlineage_event_for_artifact(full_artifact_path):
            return result

        process_and_emit_ol_events_for_artifact_repo_operation(
            operation=ArtifactRepoOperations.LOG,
            ol_client=self.ol_client,
            artifact_uri=self.artifact_uri,
            artifact_path=artifact_path,
        )

        return result
