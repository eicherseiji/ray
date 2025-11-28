from unittest import mock

import pytest

from ray.anyscale.lineage.common import openlineage_client
from ray.anyscale.lineage.common.exceptions import AnyscaleLineageClientError
from ray.anyscale.lineage.tests.test_constants import (
    TEST_OL_JOB_NAME,
    TEST_OL_NAMESPACE_SHORT,
    TEST_OL_RUN_ID_STR,
)

pytestmark = [
    pytest.mark.timeout(30),
]


class FakeOpenLineageTransport:
    """Fake for OpenLineage transport."""

    def __init__(self):
        self.close_called = False
        self.close_timeout = None

    def close(self, timeout=-1):
        self.close_called = True
        self.close_timeout = timeout


class FakeOpenLineageClient:
    """Fake for OpenLineageClient."""

    def __init__(self):
        self.transport = FakeOpenLineageTransport()
        self.emitted_events = []

    def emit(self, event):
        self.emitted_events.append(event)


class FakeJob:
    """Fake for OpenLineage Job object."""

    def __init__(self, namespace: str, name: str):
        self.namespace = namespace
        self.name = name


class FakeRun:
    """Fake for OpenLineage Run object."""

    def __init__(self, run_id: str):
        self.runId = run_id


class FakeDataset:
    """Fake for OpenLineage Dataset object."""

    def __init__(self, namespace: str = "test", name: str = "test-dataset"):
        self.namespace = namespace
        self.name = name


@pytest.fixture
def mock_ol_client():
    """Mock OpenLineage client."""
    with mock.patch(
        "ray.anyscale.lineage.common.openlineage_client.OpenLineageClient.from_environment"
    ) as from_env_mock:
        fake_client = FakeOpenLineageClient()
        from_env_mock.return_value = fake_client
        yield fake_client


@pytest.fixture
def anyscale_client(mock_ol_client):
    """Create AnyscaleOpenLineageClient."""
    with mock.patch("ray.anyscale.lineage.common.openlineage_client.set_producer"):
        client = openlineage_client.AnyscaleOpenLineageClient("test-producer")
        return client


class TestAnyscaleOpenLineageClient:
    """Test suite for AnyscaleOpenLineageClient."""

    @mock.patch("ray.anyscale.lineage.common.openlineage_client.JobEvent")
    def test_emit_job_event_success(
        self, mock_job_event, anyscale_client, mock_ol_client
    ):
        """Test successful job event emission."""
        job = FakeJob(namespace=TEST_OL_NAMESPACE_SHORT, name=TEST_OL_JOB_NAME)
        inputs = [FakeDataset(namespace="input", name="input-dataset")]
        outputs = [FakeDataset(namespace="output", name="output-dataset")]

        anyscale_client.emit_job_event(job, inputs=inputs, outputs=outputs)

        mock_job_event.assert_called_once()
        assert len(mock_ol_client.emitted_events) == 1

    @mock.patch("ray.anyscale.lineage.common.openlineage_client.RunEvent")
    def test_emit_run_event_success(
        self, mock_run_event, anyscale_client, mock_ol_client
    ):
        """Test successful run event emission."""
        run = FakeRun(run_id=TEST_OL_RUN_ID_STR)
        job = FakeJob(namespace=TEST_OL_NAMESPACE_SHORT, name=TEST_OL_JOB_NAME)

        anyscale_client.emit_run_event(run, job)

        mock_run_event.assert_called_once()
        assert len(mock_ol_client.emitted_events) == 1

    @mock.patch("ray.anyscale.lineage.common.openlineage_client.RunEvent")
    def test_emit_run_event_exception_handling(
        self, mock_run_event, anyscale_client, mock_ol_client
    ):
        """Test that exceptions are properly wrapped."""
        run = FakeRun(run_id=TEST_OL_RUN_ID_STR)
        job = FakeJob(namespace=TEST_OL_NAMESPACE_SHORT, name=TEST_OL_JOB_NAME)

        def raise_error(event):
            raise RuntimeError("Connection failed")

        mock_ol_client.emit = raise_error

        with pytest.raises(AnyscaleLineageClientError) as error:
            anyscale_client.emit_run_event(run, job)

        assert "Connection failed" in str(error.value)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", "-x", __file__]))
