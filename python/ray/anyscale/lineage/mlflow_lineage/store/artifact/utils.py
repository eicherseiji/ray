import os
from enum import Enum, unique
from typing import Dict

import mlflow
from openlineage.client.event_v2 import RunState
from openlineage.client.facet_v2 import DatasetFacet, JobFacet, RunFacet

from ray.anyscale.lineage.common.facets.dataset import DatasetType
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
from ray.anyscale.lineage.mlflow_lineage.utils import (
    resolve_http_uri_from_mlflow_artifacts_uri,
)
from ray.anyscale.lineage.version import __version__


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
    model_name: str,
    model_uri: str,
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
            name=model_name,
            uri=model_uri,
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
    # Extract model information
    artifact_uri = resolve_http_uri_from_mlflow_artifacts_uri(artifact_uri)
    model_name = artifact_path

    # Transform Anyscale-specific /mnt paths
    should_track, transformed_artifact_uri = evaluate_and_transform_uri(artifact_uri)
    if not should_track:
        return

    model_uri = os.path.join(transformed_artifact_uri, model_name)

    # OpenLineage job corresponds to an Anyscale WSJ (Workspace, Service, or Job)
    # OpenLineage run corresponds to the execution of the Anyscale WSJ
    # OpenLineage datasets are associated with the Anyscale WSJ
    ol_job_namespace = get_anyscale_workload_ol_job_namespace()
    ol_job_name = get_anyscale_workload_ol_job_name()
    ol_run_id = get_anyscale_workload_ol_run_id()

    # Dataset corresponds to the artifact that was downloaded or logged
    ol_dataset_namespace = transformed_artifact_uri
    ol_dataset_name = model_name

    # construct facets
    job_facets = _construct_ol_job_facets_for_artifact_repo_operation()
    run_facets = _construct_ol_run_facets_for_artifact_repo_operation()
    dataset_facets = _construct_ol_dataset_facets_for_artifact_repo_operation(
        model_name=model_name,
        model_uri=model_uri,
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
