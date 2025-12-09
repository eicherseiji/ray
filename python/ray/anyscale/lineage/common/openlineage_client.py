from typing import List, Optional

from openlineage.client import OpenLineageClient, set_producer
from openlineage.client.event_v2 import (
    InputDataset,
    Job,
    JobEvent,
    OutputDataset,
    Run,
    RunEvent,
    RunState,
)

from ray.anyscale.lineage.common.logging import get_logger
from ray.anyscale.lineage.common.utils import (
    catch_ol_client_exception,
    get_anyscale_openlineage_config,
    get_now_utc_datetime,
)

logger = get_logger(__name__)


class AnyscaleOpenLineageClient:
    """OpenLineage client for Anyscale lineage tracking."""

    def __init__(self, ol_producer: str):
        """Initialize the OpenLineage client."""
        self._client = OpenLineageClient(config=get_anyscale_openlineage_config())
        self._ol_producer = ol_producer
        set_producer(self._ol_producer)

    def _get_ol_producer(self) -> str:
        """Get the OpenLineage producer."""
        return self._ol_producer

    @catch_ol_client_exception
    def emit_job_event(
        self,
        job: Job,
        inputs: Optional[List[InputDataset]] = None,
        outputs: Optional[List[OutputDataset]] = None,
    ) -> None:
        """Emit a job event."""
        logger.info(
            f"Emitting job event for job: namespace '{job.namespace}', name '{job.name}'"
        )
        self._client.emit(
            JobEvent(
                eventTime=get_now_utc_datetime(),
                producer=self._get_ol_producer(),
                job=job,
                inputs=inputs,
                outputs=outputs,
            )
        )

    @catch_ol_client_exception
    def emit_run_event(
        self,
        run: Run,
        job: Job,
        event_type: Optional[RunState] = None,
        inputs: Optional[List[InputDataset]] = None,
        outputs: Optional[List[OutputDataset]] = None,
    ) -> None:
        """Emit a run event."""
        logger.info(
            f"Emitting run event for run: id '{run.runId}', event type: '{event_type}', "
            f"job namespace: '{job.namespace}', job name: '{job.name}'"
        )
        self._client.emit(
            RunEvent(
                eventTime=get_now_utc_datetime(),
                producer=self._get_ol_producer(),
                run=run,
                job=job,
                eventType=event_type,
                inputs=inputs,
                outputs=outputs,
            )
        )

    @catch_ol_client_exception
    def async_http_transport_clean_shutdown(self) -> None:
        """Clean shutdown of the async HTTP transport."""
        # wait until all events are processed
        self._client.transport.close(timeout=-1)
