from typing import Dict

import mlflow
from openlineage.client.event_v2 import RunState
from openlineage.client.facet_v2 import DatasetFacet, JobFacet, RunFacet

from ray.anyscale.lineage.common.facets.dataset import DatasetType
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
from ray.anyscale.lineage.mlflow_lineage.facet_constructor import MLflowFacetConstructor
from ray.anyscale.lineage.mlflow_lineage.utils import (
    resolve_http_uri_from_mlflow_artifacts_uri,
)
from ray.anyscale.lineage.version import __version__

logger = get_logger(__name__)


def _construct_ol_job_facets_for_model_registration() -> Dict[str, JobFacet]:
    """Construct OpenLineage job facets for model registration."""
    job_facets: Dict[str, JobFacet] = {}
    job_facets.update(
        MLflowFacetConstructor.construct_anyscale_workload_details_job_facet()
    )
    return job_facets


def _construct_ol_run_facets_for_model_registration() -> Dict[str, RunFacet]:
    """Construct OpenLineage run facets for model registration."""
    run_facets: Dict[str, RunFacet] = {}
    run_facets.update(
        MLflowFacetConstructor.construct_processing_engine_run_facet(
            engine_name="MLflow",
            engine_version=mlflow.__version__,
            openlineage_adapter_version=__version__,
        )
    )
    return run_facets


def _construct_ol_dataset_facets_for_model_registration(
    model_name: str,
    model_uri: str,
    model_version: str,
) -> Dict[str, DatasetFacet]:
    """Construct OpenLineage dataset facets for model registration."""
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
        MLflowFacetConstructor.construct_version_dataset_facet(
            version=model_version,
        )
    )
    dataset_facets.update(MLflowFacetConstructor.construct_ownership_dataset_facet())
    return dataset_facets


def process_and_emit_ol_events_for_model_registration(
    ol_client: AnyscaleOpenLineageClient,
    model_name: str,
    model_uri: str,
    model_version: str,
) -> None:
    """Process and emit OpenLineage events for model registration."""
    # Transform the model URI
    model_uri = resolve_http_uri_from_mlflow_artifacts_uri(model_uri)

    # Transform Anyscale-specific /mnt paths
    should_track, transformed_model_uri = evaluate_and_transform_uri(model_uri)
    if not should_track:
        return

    # OpenLineage job corresponds to an Anyscale WSJ
    ol_job_namespace = get_anyscale_workload_ol_job_namespace()
    ol_job_name = get_anyscale_workload_ol_job_name()
    ol_run_id = get_anyscale_workload_ol_run_id()

    # Dataset corresponds to the registered model version
    ol_dataset_namespace = transformed_model_uri
    ol_dataset_name = model_name

    # Construct facets
    job_facets = _construct_ol_job_facets_for_model_registration()
    run_facets = _construct_ol_run_facets_for_model_registration()
    dataset_facets = _construct_ol_dataset_facets_for_model_registration(
        model_name=model_name,
        model_uri=transformed_model_uri,
        model_version=model_version,
    )

    # Construct output datasets (model registration produces a registered model)
    output_datasets = [
        create_openlineage_output_dataset_from_args(
            dataset_namespace=ol_dataset_namespace,
            dataset_name=ol_dataset_name,
            facets=dataset_facets,
        )
    ]

    # Emit the OpenLineage run event
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
