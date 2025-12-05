import os
from enum import Enum, unique
from typing import Dict, Optional

import mlflow
from openlineage.client.event_v2 import RunState
from openlineage.client.facet_v2 import DatasetFacet, JobFacet, RunFacet

from ray.anyscale.lineage.common.facets.dataset import DatasetType, FreeFormFileFormat
from ray.anyscale.lineage.common.openlineage_client import AnyscaleOpenLineageClient
from ray.anyscale.lineage.common.utils import (
    create_openlineage_input_dataset_from_args,
    create_openlineage_job_from_args,
    create_openlineage_output_dataset_from_args,
    create_openlineage_run_from_args,
    evaluate_and_transform_uri,
    get_anyscale_workload_ol_job_name,
    get_anyscale_workload_ol_job_namespace,
    get_anyscale_workload_ol_run_id,
)
from ray.anyscale.lineage.mlflow_lineage.facet_constructor import (
    MLflowFacetConstructor,
)
from ray.anyscale.lineage.version import __version__

ARTIFACT_PATH_FILE_FORMAT_SEPARATOR = "."


def should_emit_openlineage_event_for_artifact(artifact_uri: str) -> bool:
    """Determine if OpenLineage events should be emitted for an artifact URI.

    Args:
        artifact_uri: The artifact URI to check

    Returns:
        False if events should be skipped (e.g., for runs:/ URIs or /tmp paths), True otherwise
    """
    # Skip runs:/ URIs as they reference artifacts from other runs
    # Skip /tmp paths as they are temporary local files, not meaningful artifacts
    # Emitting events for these would create circular/duplicate lineage
    return not (artifact_uri.startswith("runs:/") or artifact_uri.startswith("/tmp"))


@unique
class ArtifactRepoOperations(Enum):
    """Artifact repository operations."""

    DOWNLOAD = "download"
    LOG = "log"


def _construct_ol_job_facets_for_artifact_repo_operation() -> Dict[str, JobFacet]:
    """Construct OpenLineage job facets for a artifact repository operation."""
    job_facets: Dict[str, JobFacet] = {}
    job_facets.update(
        MLflowFacetConstructor.construct_anyscale_workload_details_job_facet()
    )
    return job_facets


def _construct_ol_run_facets_for_artifact_repo_operation() -> Dict[str, RunFacet]:
    """Construct OpenLineage run facets for a artifact repository operation."""
    run_facets: Dict[str, RunFacet] = {}
    run_facets.update(
        MLflowFacetConstructor.construct_processing_engine_run_facet(
            engine_name="MLflow",
            engine_version=mlflow.__version__,
            openlineage_adapter_version=__version__,
        )
    )
    return run_facets


def _construct_ol_dataset_facets_for_artifact_repo_operation(
    full_artifact_path: str,
    file_format: Optional[str] = None,
) -> Dict[str, DatasetFacet]:
    """Construct OpenLineage dataset facets for a artifact repository operation."""
    dataset_facets: Dict[str, DatasetFacet] = {}
    dataset_facets.update(
        MLflowFacetConstructor.construct_dataset_type_dataset_facet(
            dataset_type=DatasetType.MODEL
        )
    )
    dataset_facets.update(
        MLflowFacetConstructor.construct_datasource_dataset_facet(
            name=None,
            uri=full_artifact_path,
        )
    )
    if file_format:
        dataset_facets.update(
            MLflowFacetConstructor.construct_file_format_dataset_facet(
                format=FreeFormFileFormat(format=file_format),
            )
        )
    dataset_facets.update(MLflowFacetConstructor.construct_ownership_dataset_facet())
    return dataset_facets


def process_and_emit_ol_events_for_artifact_repo_operation(
    operation: ArtifactRepoOperations,
    ol_client: AnyscaleOpenLineageClient,
    artifact_uri: str,
    artifact_path: str,
) -> None:
    """Process and emit OpenLineage events for a artifact repository operation."""
    full_artifact_path = os.path.join(artifact_uri, artifact_path)

    # Transform Anyscale-specific /mnt paths
    _, transformed_artifact_uri = evaluate_and_transform_uri(artifact_uri)
    should_track, transformed_full_artifact_path = evaluate_and_transform_uri(
        full_artifact_path
    )
    if not should_track:
        return

    file_format = (
        artifact_path.split(ARTIFACT_PATH_FILE_FORMAT_SEPARATOR)[-1]
        if ARTIFACT_PATH_FILE_FORMAT_SEPARATOR in artifact_path
        else None
    )

    # OpenLineage job corresponds to an Anyscale WSJ (Workspace, Service, or Job)
    # OpenLineage run corresponds to the execution of the Anyscale WSJ
    # OpenLineage datasets are associated with the Anyscale WSJ
    ol_job_namespace = get_anyscale_workload_ol_job_namespace()
    ol_job_name = get_anyscale_workload_ol_job_name()
    ol_run_id = get_anyscale_workload_ol_run_id()

    # Dataset corresponds to the artifact that was downloaded
    ol_dataset_namespace = transformed_artifact_uri
    ol_dataset_name = artifact_path

    # construct facets
    job_facets = _construct_ol_job_facets_for_artifact_repo_operation()
    run_facets = _construct_ol_run_facets_for_artifact_repo_operation()
    dataset_facets = _construct_ol_dataset_facets_for_artifact_repo_operation(
        full_artifact_path=transformed_full_artifact_path,
        file_format=file_format,
    )

    # construct datasets
    input_datasets, output_datasets = [], []
    if operation == ArtifactRepoOperations.DOWNLOAD:
        input_datasets = [
            create_openlineage_input_dataset_from_args(
                dataset_namespace=ol_dataset_namespace,
                dataset_name=ol_dataset_name,
                facets=dataset_facets,
            )
        ]
    elif operation == ArtifactRepoOperations.LOG:
        output_datasets = [
            create_openlineage_output_dataset_from_args(
                dataset_namespace=ol_dataset_namespace,
                dataset_name=ol_dataset_name,
                facets=dataset_facets,
            )
        ]

    # complete the OpenLineage run
    ol_client.emit_run_event(
        run=create_openlineage_run_from_args(
            run_id=ol_run_id,
            facets=run_facets,
        ),
        job=create_openlineage_job_from_args(
            job_namespace=ol_job_namespace,
            job_name=ol_job_name,
            facets=job_facets,
        ),
        event_type=RunState.COMPLETE,
        inputs=input_datasets,
        outputs=output_datasets,
    )
