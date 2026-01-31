import os
from typing import Dict

import mlflow
from mlflow.entities.run import Run
from mlflow.models.model import Model
from openlineage.client.event_v2 import RunState
from openlineage.client.facet_v2 import DatasetFacet, JobFacet, RunFacet

from ray.anyscale.lineage.common.facets.dataset import DatasetType, FreeFormFileFormat
from ray.anyscale.lineage.common.logging import get_logger
from ray.anyscale.lineage.common.openlineage_client import AnyscaleOpenLineageClient
from ray.anyscale.lineage.common.utils import (
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

logger = get_logger(__name__)


def _construct_ol_job_facets_for_model_logging() -> Dict[str, JobFacet]:
    """Construct OpenLineage job facets for a model logging."""
    job_facets: Dict[str, JobFacet] = {}
    job_facets.update(
        MLflowFacetConstructor.construct_anyscale_workload_details_job_facet()
    )
    return job_facets


def _construct_ol_run_facets_for_model_logging() -> Dict[str, RunFacet]:
    """Construct OpenLineage run facets for a model logging."""
    run_facets: Dict[str, RunFacet] = {}
    run_facets.update(
        MLflowFacetConstructor.construct_processing_engine_run_facet(
            engine_name="MLflow",
            engine_version=mlflow.__version__,
            openlineage_adapter_version=__version__,
        )
    )
    return run_facets


def _construct_ol_dataset_facets_for_model_logging(
    model_name: str,
    model_uri: str,
    model_flavors: str,
) -> Dict[str, DatasetFacet]:
    """Construct OpenLineage dataset facets for a model logging."""
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
    dataset_facets.update(
        MLflowFacetConstructor.construct_file_format_dataset_facet(
            format=FreeFormFileFormat(format=model_flavors),
        )
    )
    dataset_facets.update(MLflowFacetConstructor.construct_ownership_dataset_facet())
    return dataset_facets


def process_and_emit_ol_events_for_model_logging(
    ol_client: AnyscaleOpenLineageClient,
    run: Run,
    mlflow_model: Model,
) -> None:
    """Process and emit OpenLineage events for a model logging."""
    # Extract model logging information
    artifact_uri = resolve_http_uri_from_mlflow_artifacts_uri(run.info.artifact_uri)
    model_name = mlflow_model.artifact_path
    model_flavors = ",".join(mlflow_model.flavors.keys())

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

    # Dataset corresponds to the model that was logged
    ol_dataset_namespace = transformed_artifact_uri
    ol_dataset_name = model_name

    # construct facets
    job_facets = _construct_ol_job_facets_for_model_logging()
    run_facets = _construct_ol_run_facets_for_model_logging()
    dataset_facets = _construct_ol_dataset_facets_for_model_logging(
        model_name=model_name,
        model_uri=model_uri,
        model_flavors=model_flavors,
    )

    # construct datasets
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
        outputs=output_datasets,
    )
