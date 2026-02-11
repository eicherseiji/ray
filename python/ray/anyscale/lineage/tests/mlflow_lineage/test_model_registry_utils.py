"""Tests for model registry OpenLineage event processing."""

from __future__ import annotations

from unittest import mock

import pytest
from openlineage.client.event_v2 import RunState

from ray.anyscale.lineage.mlflow_lineage.store.model_registry.utils import (
    process_and_emit_ol_events_for_model_registration,
)
from ray.anyscale.lineage.tests.test_constants import (
    TEST_MLFLOW_MODEL_NAME,
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
def mock_client():
    return mock.Mock()


class TestModelRegistration:
    """Tests for model registration OpenLineage event emission."""

    def test_emits_complete_event_with_output_dataset(self, workload_env, mock_client):
        """Model registration emits COMPLETE event with output dataset."""
        process_and_emit_ol_events_for_model_registration(
            ol_client=mock_client,
            model_name=TEST_MLFLOW_MODEL_NAME,
            model_uri=TEST_MLFLOW_S3_URI,
            model_version="1",
        )

        assert mock_client.emit_run_event.call_count == 1
        call_args = mock_client.emit_run_event.call_args[1]
        assert call_args["event_type"] == RunState.COMPLETE
        assert len(call_args["outputs"]) == 1

    def test_dataset_uses_model_uri_as_namespace(self, workload_env, mock_client):
        """Dataset namespace is derived from model URI."""
        process_and_emit_ol_events_for_model_registration(
            ol_client=mock_client,
            model_name=TEST_MLFLOW_MODEL_NAME,
            model_uri=TEST_MLFLOW_S3_URI,
            model_version="1",
        )

        call_args = mock_client.emit_run_event.call_args[1]
        output = call_args["outputs"][0]
        assert output.namespace == TEST_MLFLOW_S3_URI
        assert output.name == TEST_MLFLOW_MODEL_NAME

    def test_skips_untracked_uris(self, workload_env, mock_client):
        """Model registration skips untracked URIs like /tmp paths."""
        process_and_emit_ol_events_for_model_registration(
            ol_client=mock_client,
            model_name=TEST_MLFLOW_MODEL_NAME,
            model_uri="/tmp/mlruns/artifacts",
            model_version="1",
        )

        mock_client.emit_run_event.assert_not_called()
