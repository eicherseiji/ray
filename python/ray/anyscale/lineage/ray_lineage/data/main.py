from typing import Dict, List, Tuple

from openlineage.client.event_v2 import InputDataset, OutputDataset, RunState
from openlineage.client.facet_v2 import JobFacet, RunFacet
from ray.data._internal.execution.execution_callback import ExecutionCallback
from ray.data._internal.execution.streaming_executor import StreamingExecutor

from ray.anyscale.lineage.common.exceptions import AnyscaleLineageRayDataError
from ray.anyscale.lineage.common.logging import configure_logging
from ray.anyscale.lineage.common.openlineage_client import AnyscaleOpenLineageClient
from ray.anyscale.lineage.common.utils import (
    create_openlineage_job_from_args,
    create_openlineage_run_from_args,
    get_anyscale_workload_ol_job_name,
    get_anyscale_workload_ol_job_namespace,
    get_anyscale_workload_ol_run_id,
    get_ray_version,
)
from ray.anyscale.lineage.ray_lineage.data.constants import (
    RAY_DATA_OPENLINEAGE_PRODUCER,
)
from ray.anyscale.lineage.ray_lineage.data.dataset_constructor import (
    construct_input_output_datasets,
)
from ray.anyscale.lineage.ray_lineage.data.facet_constructor import (
    RayDataFacetConstructor,
)

# Configure logging once at module level
configure_logging()


class RayDataOpenLineageExecutionCallback(ExecutionCallback):
    """Execution callback for lineage tracking for Ray Data workloads.

    Emits OpenLineage events to configured backends for input and output datasets
    that are read or written using Ray Data Dataset read and write APIs.
    This callback is initialized for each `Dataset` in a Ray session and the hooks
    are executed once for every materialization of the dataset.
    """

    def __init__(self) -> None:
        super().__init__()

        self.ol_client = AnyscaleOpenLineageClient(
            ol_producer=RAY_DATA_OPENLINEAGE_PRODUCER
        )

        # OpenLineage job corresponds to an Anyscale WSJ (Workspace, Service, or Job)
        # OpenLineage run corresponds to the execution of the Anyscale WSJ
        # OpenLineage datasets are associated with the Anyscale WSJ
        self.ol_job_namespace = get_anyscale_workload_ol_job_namespace()
        self.ol_job_name = get_anyscale_workload_ol_job_name()
        self.ol_run_id = get_anyscale_workload_ol_run_id()

    def before_execution_starts(self, executor: StreamingExecutor) -> None:
        """Called before the Dataset execution starts."""
        ...

    def on_execution_step(self, executor: StreamingExecutor) -> None:
        """Called at each step of the Dataset execution loop."""
        ...

    def _after_execution_completes(
        self, executor: StreamingExecutor
    ) -> Tuple[
        Dict[str, JobFacet],
        Dict[str, RunFacet],
        List[InputDataset],
        List[OutputDataset],
    ]:
        """Common logic for after execution succeeds and fails."""
        # construct job facets
        job_facets: Dict[str, JobFacet] = {}
        job_facets.update(
            RayDataFacetConstructor.construct_anyscale_workload_details_job_facet()
        )

        # construct run facets
        run_facets: Dict[str, RunFacet] = {}
        run_facets.update(
            RayDataFacetConstructor.construct_processing_engine_run_facet(
                engine_name="Ray Data",
                engine_version=get_ray_version(),
                openlineage_adapter_version=get_ray_version(),
            )
        )

        # construct input and output datasets
        input_datasets, output_datasets = construct_input_output_datasets(executor)

        return job_facets, run_facets, input_datasets, output_datasets

    def after_execution_succeeds(self, executor: StreamingExecutor) -> None:
        """Called after the Dataset execution succeeds."""
        (
            job_facets,
            run_facets,
            input_datasets,
            output_datasets,
        ) = self._after_execution_completes(executor)

        # complete the OpenLineage run
        try:
            self.ol_client.emit_run_event(
                run=create_openlineage_run_from_args(
                    run_id=self.ol_run_id,
                    facets=run_facets,
                ),
                job=create_openlineage_job_from_args(
                    job_namespace=self.ol_job_namespace,
                    job_name=self.ol_job_name,
                    facets=job_facets,
                ),
                event_type=RunState.COMPLETE,
                inputs=input_datasets,
                outputs=output_datasets,
            )
        except Exception as e:
            raise AnyscaleLineageRayDataError(
                f"Error emitting OpenLineage COMPLETE run event for run: id '{self.ol_run_id}'"
            ) from e

    def after_execution_fails(
        self, executor: StreamingExecutor, error: Exception
    ) -> None:
        """Called after the Dataset execution fails."""
        (
            job_facets,
            run_facets,
            input_datasets,
            output_datasets,
        ) = self._after_execution_completes(executor)

        # complete the OpenLineage run
        try:
            self.ol_client.emit_run_event(
                run=create_openlineage_run_from_args(
                    run_id=self.ol_run_id,
                    facets=run_facets,
                ),
                job=create_openlineage_job_from_args(
                    job_namespace=self.ol_job_namespace,
                    job_name=self.ol_job_name,
                    facets=job_facets,
                ),
                event_type=RunState.FAIL,
                inputs=input_datasets,
                outputs=output_datasets,
            )
        except Exception as e:
            raise AnyscaleLineageRayDataError(
                f"Error emitting OpenLineage FAIL run event for run: id '{self.ol_run_id}'"
            ) from e
