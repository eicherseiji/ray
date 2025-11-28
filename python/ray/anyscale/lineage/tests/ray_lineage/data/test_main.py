import pytest

from ray.anyscale.lineage.common.exceptions import AnyscaleLineageRayDataError
from ray.anyscale.lineage.ray_lineage.data import main
from ray.anyscale.lineage.tests.test_constants import (
    TEST_DATASET_ID,
    TEST_RAY_JOB_NAME_PATTERN,
    TEST_RAY_NAMESPACE_PATTERN,
    TEST_RUN_ID_SHORT,
    TEST_UUID_PATTERN,
)


class DummyExecutor:
    def __init__(self):
        self._dataset_id = TEST_DATASET_ID
        self._topology = []


class DummyClient:
    def __init__(self):
        self.emitted_events = []

    def generate_run_id(self):
        return TEST_RUN_ID_SHORT

    def create_job_from_args(self, **kwargs):
        return {"job": kwargs}

    def emit_job_event(self, **kwargs):
        self.emitted_events.append(("job", kwargs))

    def create_run_from_args(self, **kwargs):
        return {"run": kwargs}

    def emit_run_event(self, **kwargs):
        self.emitted_events.append(("run", kwargs))


@pytest.fixture(autouse=True)
def patch_client(monkeypatch):
    dummy_client = DummyClient()
    # Simplified client patching
    monkeypatch.setattr(
        main, "AnyscaleOpenLineageClient", lambda *args, **kwargs: dummy_client
    )
    return dummy_client


def sample_env(monkeypatch):
    from ray.anyscale.lineage.tests.test_constants import (
        SIMPLE_CLOUD,
        SIMPLE_JOB,
        SIMPLE_ORG,
        SIMPLE_PROJECT,
    )

    monkeypatch.setenv("ANYSCALE_ORGANIZATION_ID", SIMPLE_ORG)
    monkeypatch.setenv("ANYSCALE_CLOUD_ID", SIMPLE_CLOUD)
    monkeypatch.setenv("ANYSCALE_PROJECT_ID", SIMPLE_PROJECT)
    monkeypatch.setenv("ANYSCALE_WORKLOAD_TYPE", SIMPLE_JOB)
    monkeypatch.setenv("ANYSCALE_JOB_ID", "job-id")
    monkeypatch.setenv("ANYSCALE_WORKLOAD_VERSION_ID", "v1")
    monkeypatch.setenv(
        "ANYSCALE_WORKLOAD_OPENLINEAGE_RUN_ID",
        TEST_UUID_PATTERN,
    )


def test_after_execution_succeeds_emits_complete_event(monkeypatch):
    sample_env(monkeypatch)

    dummy_input = []
    dummy_output = []

    def fake_after_execution_completes(self, executor):
        return {}, {}, dummy_input, dummy_output

    monkeypatch.setattr(
        main.RayDataOpenLineageExecutionCallback,
        "_after_execution_completes",
        fake_after_execution_completes,
    )

    callback = main.RayDataOpenLineageExecutionCallback()
    executor = DummyExecutor()
    callback.before_execution_starts(executor)
    callback.after_execution_succeeds(executor)

    # Verify run event was emitted with COMPLETE state
    assert callback.ol_client.emitted_events[-1][0] == "run"
    assert (
        callback.ol_client.emitted_events[-1][1]["event_type"] == main.RunState.COMPLETE
    )


def test_after_execution_fails_includes_error_facet(monkeypatch):
    sample_env(monkeypatch)

    def fake_after_execution_completes(self, executor):
        return {}, {}, [], []

    monkeypatch.setattr(
        main.RayDataOpenLineageExecutionCallback,
        "_after_execution_completes",
        fake_after_execution_completes,
    )

    callback = main.RayDataOpenLineageExecutionCallback()
    executor = DummyExecutor()
    callback.before_execution_starts(executor)
    callback.after_execution_fails(executor, error=RuntimeError("boom"))

    run_event = callback.ol_client.emitted_events[-1]
    assert run_event[0] == "run"
    assert run_event[1]["event_type"] == main.RunState.FAIL


def test_on_execution_step_does_nothing(monkeypatch):
    sample_env(monkeypatch)

    callback = main.RayDataOpenLineageExecutionCallback()
    executor = DummyExecutor()

    # Should not raise any exceptions and should be a no-op
    callback.on_execution_step(executor)


def test_callback_initialization_sets_correct_attributes(monkeypatch):
    sample_env(monkeypatch)

    callback = main.RayDataOpenLineageExecutionCallback()

    assert callback.ol_client is not None
    assert callback.ol_job_namespace == TEST_RAY_NAMESPACE_PATTERN
    assert callback.ol_job_name == TEST_RAY_JOB_NAME_PATTERN
    assert callback.ol_run_id is not None
    # Verify it's a valid UUID
    from uuid import UUID

    UUID(callback.ol_run_id)


def test_after_execution_completes_constructs_datasets_and_facets(monkeypatch):
    sample_env(monkeypatch)

    dummy_input = []
    dummy_output = []

    def fake_construct_input_output_datasets(executor):
        return dummy_input, dummy_output

    monkeypatch.setattr(
        main,
        "construct_input_output_datasets",
        fake_construct_input_output_datasets,
    )

    callback = main.RayDataOpenLineageExecutionCallback()
    executor = DummyExecutor()

    job_facets, run_facets, inputs, outputs = callback._after_execution_completes(
        executor
    )

    assert isinstance(job_facets, dict)
    assert isinstance(run_facets, dict)
    assert inputs == dummy_input
    assert outputs == dummy_output


def test_after_execution_succeeds_raises_on_emit_failure(monkeypatch):
    sample_env(monkeypatch)

    class ErrorClient(DummyClient):
        def emit_run_event(self, **kwargs):
            if kwargs.get("event_type") == main.RunState.COMPLETE:
                raise RuntimeError("emit failure")

    monkeypatch.setattr(
        main,
        "AnyscaleOpenLineageClient",
        lambda *args, **kwargs: ErrorClient(),
    )

    def fake_after_execution_completes(self, executor):
        return {}, {}, [], []

    monkeypatch.setattr(
        main.RayDataOpenLineageExecutionCallback,
        "_after_execution_completes",
        fake_after_execution_completes,
    )

    callback = main.RayDataOpenLineageExecutionCallback()
    executor = DummyExecutor()
    callback.before_execution_starts(executor)

    with pytest.raises(
        AnyscaleLineageRayDataError,
        match="Error emitting OpenLineage COMPLETE run event",
    ):
        callback.after_execution_succeeds(executor)


def test_after_execution_fails_raises_on_emit_failure(monkeypatch):
    sample_env(monkeypatch)

    class ErrorClient(DummyClient):
        def emit_run_event(self, **kwargs):
            if kwargs.get("event_type") == main.RunState.FAIL:
                raise RuntimeError("emit failure")

    monkeypatch.setattr(
        main,
        "AnyscaleOpenLineageClient",
        lambda *args, **kwargs: ErrorClient(),
    )

    def fake_after_execution_completes(self, executor):
        return {}, {}, [], []

    monkeypatch.setattr(
        main.RayDataOpenLineageExecutionCallback,
        "_after_execution_completes",
        fake_after_execution_completes,
    )

    callback = main.RayDataOpenLineageExecutionCallback()
    executor = DummyExecutor()
    callback.before_execution_starts(executor)

    with pytest.raises(
        AnyscaleLineageRayDataError, match="Error emitting OpenLineage FAIL run event"
    ):
        callback.after_execution_fails(executor, error=RuntimeError("test error"))
