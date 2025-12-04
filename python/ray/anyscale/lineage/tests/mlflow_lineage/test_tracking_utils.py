from __future__ import annotations

from unittest import mock

import pytest
from openlineage.client.event_v2 import RunState

from ray.anyscale.lineage.mlflow_lineage.store.tracking.utils import (
    process_and_emit_ol_events_for_model_logging,
)
from ray.anyscale.lineage.tests.test_constants import (
    TEST_MLFLOW_EXPERIMENT_ID,
    TEST_MLFLOW_HOST_LOCAL,
    TEST_MLFLOW_MODEL_URI,
    TEST_MLFLOW_MODEL_UUID,
    TEST_MLFLOW_RUN_ID,
    TEST_MLFLOW_RUN_NAME,
    TEST_WORKLOAD_OL_RUN_ID_ALT,
)


@pytest.fixture
def workload_env(sample_anyscale_env, monkeypatch):
    monkeypatch.setenv(
        "ANYSCALE_WORKLOAD_OPENLINEAGE_RUN_ID",
        TEST_WORKLOAD_OL_RUN_ID_ALT,
    )
    return sample_anyscale_env


@pytest.fixture
def mock_run():
    run = mock.Mock()
    run.info.run_name = TEST_MLFLOW_RUN_NAME
    run.info.experiment_id = TEST_MLFLOW_EXPERIMENT_ID
    return run


@pytest.fixture
def mock_model():
    model_info = mock.Mock()
    model_info.model_uuid = TEST_MLFLOW_MODEL_UUID
    model_info.model_uri = TEST_MLFLOW_MODEL_URI
    model_info.flavors = {"python_function": {}}

    model = mock.Mock()
    model.get_model_info.return_value = model_info
    model.run_id = TEST_MLFLOW_RUN_ID
    # Schema methods are no longer used as input/output schema extraction is commented out
    return model


@pytest.fixture
def mock_client():
    client = mock.Mock()
    return client


def test_process_and_emit_ol_events_for_model_logging_happy_path(
    workload_env, mock_client, mock_run, mock_model, monkeypatch
) -> None:
    process_and_emit_ol_events_for_model_logging(
        ol_client=mock_client,
        mlflow_host=TEST_MLFLOW_HOST_LOCAL,
        run=mock_run,
        mlflow_model=mock_model,
    )

    # Verify run event was emitted once with COMPLETE state
    assert mock_client.emit_run_event.call_count == 1

    # Check the event type is COMPLETE
    call_args_list = mock_client.emit_run_event.call_args_list
    assert call_args_list[0][1]["event_type"] == RunState.COMPLETE


def test_process_and_emit_ol_events_for_model_logging_skips_runs_uri(
    workload_env, mock_client, mock_run
) -> None:
    """Test that model logging with runs:/ URI is skipped."""
    # Create a mock model with a runs:/ URI
    model_info = mock.Mock()
    model_info.model_uuid = TEST_MLFLOW_MODEL_UUID
    model_info.model_uri = "runs:/b4e6a62eb8a54755b9991bb4b3fe7d96/clip-base"
    model_info.flavors = {"transformers": {}}

    model = mock.Mock()
    model.get_model_info.return_value = model_info
    model.run_id = TEST_MLFLOW_RUN_ID

    # Call the function
    process_and_emit_ol_events_for_model_logging(
        ol_client=mock_client,
        mlflow_host=TEST_MLFLOW_HOST_LOCAL,
        run=mock_run,
        mlflow_model=model,
    )

    # Verify NO events were emitted
    mock_client.emit_job_event.assert_not_called()
    mock_client.emit_run_event.assert_not_called()
