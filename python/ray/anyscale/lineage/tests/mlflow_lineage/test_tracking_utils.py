"""Tests for tracking store OpenLineage event processing."""

from __future__ import annotations

from unittest import mock

import pytest
from openlineage.client.event_v2 import RunState

from ray.anyscale.lineage.mlflow_lineage.store.tracking.utils import (
    process_and_emit_ol_events_for_model_logging,
)
from ray.anyscale.lineage.tests.test_constants import (
    TEST_MLFLOW_EXPERIMENT_ID,
    TEST_MLFLOW_RUN_ID,
    TEST_MLFLOW_RUN_NAME,
    TEST_MLFLOW_S3_URI,
    TEST_WORKLOAD_OL_RUN_ID_ALT,
)


@pytest.fixture
def workload_env(sample_anyscale_env, monkeypatch):
    monkeypatch.setenv(
        "ANYSCALE_WORKLOAD_OPENLINEAGE_RUN_ID", TEST_WORKLOAD_OL_RUN_ID_ALT
    )
    return sample_anyscale_env


@pytest.fixture
def mock_run():
    run = mock.Mock()
    run.info.run_name = TEST_MLFLOW_RUN_NAME
    run.info.experiment_id = TEST_MLFLOW_EXPERIMENT_ID
    run.info.artifact_uri = TEST_MLFLOW_S3_URI
    return run


@pytest.fixture
def mock_model():
    model = mock.Mock()
    model.artifact_path = "test-model"
    model.flavors = {"python_function": {}, "sklearn": {}}
    model.run_id = TEST_MLFLOW_RUN_ID
    return model


@pytest.fixture
def mock_client():
    return mock.Mock()


class TestModelLogging:
    """Tests for model logging OpenLineage event emission."""

    def test_emits_complete_event_with_output_dataset(
        self, workload_env, mock_client, mock_run, mock_model
    ):
        """Model logging emits COMPLETE event with output dataset."""
        process_and_emit_ol_events_for_model_logging(
            ol_client=mock_client,
            run=mock_run,
            mlflow_model=mock_model,
        )

        assert mock_client.emit_run_event.call_count == 1
        call_args = mock_client.emit_run_event.call_args[1]
        assert call_args["event_type"] == RunState.COMPLETE
        assert len(call_args["outputs"]) == 1

    def test_dataset_uses_artifact_uri_as_namespace(
        self, workload_env, mock_client, mock_run, mock_model
    ):
        """Dataset namespace is derived from run's artifact_uri."""
        process_and_emit_ol_events_for_model_logging(
            ol_client=mock_client,
            run=mock_run,
            mlflow_model=mock_model,
        )

        call_args = mock_client.emit_run_event.call_args[1]
        output = call_args["outputs"][0]
        assert output.namespace == TEST_MLFLOW_S3_URI
        assert output.name == "test-model"

    def test_skips_untracked_uris(self, workload_env, mock_client, mock_model):
        """Model logging skips untracked URIs like /tmp paths."""
        run = mock.Mock()
        run.info.artifact_uri = "/tmp/mlruns/artifacts"

        process_and_emit_ol_events_for_model_logging(
            ol_client=mock_client,
            run=run,
            mlflow_model=mock_model,
        )

        mock_client.emit_run_event.assert_not_called()
