from typing import Any, Dict, Optional

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
    model_uuid: str,
    model_uri: str,
    model_flavors: str,
    model_info: Any,
    _input_schema: Optional[list[dict[str, Any]]] = None,
    _output_schema: Optional[list[dict[str, Any]]] = None,
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
            name=model_name or model_uuid,
            uri=model_uri,
        )
    )
    dataset_facets.update(
        MLflowFacetConstructor.construct_file_format_dataset_facet(
            format=FreeFormFileFormat(format=model_flavors),
        )
    )
    dataset_facets.update(MLflowFacetConstructor.construct_ownership_dataset_facet())

    # Input/Ouput schema not needed for now, add back when needed
    # if input_schema:
    #     dataset_facets.update(
    #         MLflowFacetConstructor.construct_input_schema_dataset_facet(
    #             fields=input_schema,
    #         )
    #     )
    # if output_schema:
    #     dataset_facets.update(
    #         MLflowFacetConstructor.construct_output_schema_dataset_facet(
    #             fields=output_schema,
    #         )
    #     )

    dataset_facets.update(
        MLflowFacetConstructor.construct_version_dataset_facet(
            version=getattr(model_info, "version", "1.0"),
        )
    )
    return dataset_facets


def process_and_emit_ol_events_for_model_logging(
    ol_client: AnyscaleOpenLineageClient,
    mlflow_host: str,
    run: Run,
    mlflow_model: Model,
) -> None:
    """Process and emit OpenLineage events for a model logging."""
    # extract model logging information
    _experiment_id = run.info.experiment_id
    model_info = mlflow_model.get_model_info()
    _run_id = mlflow_model.run_id
    model_uuid = str(model_info.model_uuid)
    model_uri = model_info.model_uri
    model_name = getattr(model_info, "name", "")
    model_flavors = ",".join(model_info.flavors.keys())

    # Skip OpenLineage events for runs:/ URIs to avoid duplicate/circular lineage
    if model_uri and model_uri.startswith("runs:/"):
        logger.info(
            f"Skipping OpenLineage event for runs:/ URI: {model_uri}. "
            "These URIs reference artifacts from other runs and would create duplicate lineage."
        )
        return

    # Transform Anyscale-specific /mnt paths
    should_track, transformed_model_uri = evaluate_and_transform_uri(model_uri)
    if not should_track:
        return

    # Input/Ouput schema not needed for now, add back when needed
    # extract input and output schema
    # input_schema = mlflow_model.get_input_schema()  # type: ignore[no-untyped-call]
    # if input_schema:
    #     input_schema = [
    #         {"name": field, "type": val}
    #         for field, val in input_schema.input_types_dict().items()
    #     ]
    # output_schema = mlflow_model.get_output_schema()  # type: ignore[no-untyped-call]
    # if output_schema:
    #     output_schema = [
    #         {"name": field, "type": val}
    #         for field, val in output_schema.output_types_dict().items()
    #     ]

    # OpenLineage job corresponds to an Anyscale WSJ (Workspace, Service, or Job)
    # OpenLineage run corresponds to the execution of the Anyscale WSJ
    # OpenLineage datasets are associated with the Anyscale WSJ
    ol_job_namespace = get_anyscale_workload_ol_job_namespace()
    ol_job_name = get_anyscale_workload_ol_job_name()
    ol_run_id = get_anyscale_workload_ol_run_id()

    ol_dataset_namespace = mlflow_host
    ol_dataset_name = model_name or model_uuid

    # construct facets
    job_facets = _construct_ol_job_facets_for_model_logging()
    run_facets = _construct_ol_run_facets_for_model_logging()
    dataset_facets = _construct_ol_dataset_facets_for_model_logging(
        model_name=model_name,
        model_uuid=model_uuid,
        model_uri=transformed_model_uri,
        model_flavors=model_flavors,
        model_info=model_info,
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
