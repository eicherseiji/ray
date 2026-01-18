"""Tests for artifact repository OpenLineage event processing."""

from __future__ import annotations

from unittest import mock

import pytest
from openlineage.client.event_v2 import RunState

from ray.anyscale.lineage.mlflow_lineage.store.artifact.utils import (
    ArtifactRepoOperations,
    process_and_emit_ol_events_for_artifact_repo_operation,
)
from ray.anyscale.lineage.tests.test_constants import (
    TEST_CLOUD_ID,
    TEST_JOB_ID,
    TEST_MLFLOW_S3_URI as TEST_S3_URI,
    TEST_MODEL_ARTIFACT_PATH,
    TEST_WORKLOAD_OL_RUN_ID_ALT,
)


@pytest.fixture
def mock_client():
    return mock.Mock()


@pytest.fixture
def workload_env(sample_anyscale_env, monkeypatch):
    monkeypatch.setenv(
        "ANYSCALE_WORKLOAD_OPENLINEAGE_RUN_ID", TEST_WORKLOAD_OL_RUN_ID_ALT
    )
    return sample_anyscale_env


class TestArtifactRepoOperations:
    """Tests for artifact repository OpenLineage event emission."""

    def test_download_emits_complete_event_with_input_dataset(
        self, workload_env, mock_client
    ):
        """Download operation emits COMPLETE event with input dataset."""
        process_and_emit_ol_events_for_artifact_repo_operation(
            operation=ArtifactRepoOperations.DOWNLOAD,
            ol_client=mock_client,
            artifact_uri=TEST_S3_URI,
            artifact_path=TEST_MODEL_ARTIFACT_PATH,
        )

        assert mock_client.emit_run_event.call_count == 1
        call_args = mock_client.emit_run_event.call_args[1]
        assert call_args["event_type"] == RunState.COMPLETE
        assert len(call_args["inputs"]) == 1
        assert call_args["inputs"][0].namespace == TEST_S3_URI
        assert call_args["inputs"][0].name == TEST_MODEL_ARTIFACT_PATH

    def test_log_emits_complete_event_with_output_dataset(
        self, workload_env, mock_client
    ):
        """Log operation emits COMPLETE event with output dataset."""
        process_and_emit_ol_events_for_artifact_repo_operation(
            operation=ArtifactRepoOperations.LOG,
            ol_client=mock_client,
            artifact_uri=TEST_S3_URI,
            artifact_path=TEST_MODEL_ARTIFACT_PATH,
        )

        assert mock_client.emit_run_event.call_count == 1
        call_args = mock_client.emit_run_event.call_args[1]
        assert call_args["event_type"] == RunState.COMPLETE
        assert len(call_args["outputs"]) == 1
        assert call_args["outputs"][0].namespace == TEST_S3_URI
        assert call_args["outputs"][0].name == TEST_MODEL_ARTIFACT_PATH


class TestUriTransformation:
    """Tests for URI transformation behavior."""

    def test_user_storage_paths_not_tracked(self, workload_env, mock_client):
        """/mnt/user_storage paths are not tracked (no events emitted)."""
        process_and_emit_ol_events_for_artifact_repo_operation(
            operation=ArtifactRepoOperations.DOWNLOAD,
            ol_client=mock_client,
            artifact_uri="/mnt/user_storage/artifacts",
            artifact_path="model/file.pkl",
        )
        assert mock_client.emit_run_event.call_count == 0

    def test_cluster_storage_paths_transformed(self, workload_env, mock_client):
        """/mnt/cluster_storage paths are transformed with job ID."""
        process_and_emit_ol_events_for_artifact_repo_operation(
            operation=ArtifactRepoOperations.LOG,
            ol_client=mock_client,
            artifact_uri="/mnt/cluster_storage/models",
            artifact_path="my_model",
        )

        call_args = mock_client.emit_run_event.call_args[1]
        output = call_args["outputs"][0]
        assert output.namespace == f"file://{TEST_JOB_ID}/mnt/cluster_storage/models"

    def test_shared_storage_paths_transformed(self, workload_env, mock_client):
        """/mnt/shared_storage paths are transformed with cloud ID."""
        process_and_emit_ol_events_for_artifact_repo_operation(
            operation=ArtifactRepoOperations.DOWNLOAD,
            ol_client=mock_client,
            artifact_uri="/mnt/shared_storage/common",
            artifact_path="data.csv",
        )

        call_args = mock_client.emit_run_event.call_args[1]
        input_ds = call_args["inputs"][0]
        assert input_ds.namespace == f"file://{TEST_CLOUD_ID}/mnt/shared_storage/common"

    def test_s3_paths_not_transformed(self, workload_env, mock_client):
        """S3 URIs are not transformed."""
        process_and_emit_ol_events_for_artifact_repo_operation(
            operation=ArtifactRepoOperations.DOWNLOAD,
            ol_client=mock_client,
            artifact_uri=TEST_S3_URI,
            artifact_path=TEST_MODEL_ARTIFACT_PATH,
        )

        call_args = mock_client.emit_run_event.call_args[1]
        assert call_args["inputs"][0].namespace == TEST_S3_URI
