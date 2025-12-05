from __future__ import annotations

from unittest import mock

import pytest
from openlineage.client.event_v2 import RunState

from ray.anyscale.lineage.mlflow_lineage.store.artifact.utils import (
    ArtifactRepoOperations,
    process_and_emit_ol_events_for_artifact_repo_operation,
    should_emit_openlineage_event_for_artifact,
)
from ray.anyscale.lineage.tests.test_constants import (
    TEST_MODEL_ARTIFACT_PATH,
    TEST_MLFLOW_S3_URI as TEST_S3_URI,
    TEST_WORKLOAD_OL_RUN_ID_ALT,
)


@pytest.fixture
def mock_client():
    client = mock.Mock()
    client.generate_run_id.side_effect = ["job-id", "run-id"]
    client.create_job_from_args.return_value = mock.Mock()
    client.create_run_from_args.return_value = mock.Mock()
    client.create_input_dataset_from_args.return_value = mock.Mock()
    client.create_output_dataset_from_args.return_value = mock.Mock()
    return client


@pytest.fixture
def workload_env(sample_anyscale_env, monkeypatch):
    monkeypatch.setenv(
        "ANYSCALE_WORKLOAD_OPENLINEAGE_RUN_ID",
        TEST_WORKLOAD_OL_RUN_ID_ALT,
    )
    return sample_anyscale_env


def test_process_and_emit_events_download(
    workload_env, mock_client, monkeypatch
) -> None:
    process_and_emit_ol_events_for_artifact_repo_operation(
        operation=ArtifactRepoOperations.DOWNLOAD,
        ol_client=mock_client,
        artifact_uri=TEST_S3_URI,
        artifact_path=TEST_MODEL_ARTIFACT_PATH,
    )

    # Verify run event was emitted once with COMPLETE state
    assert mock_client.emit_run_event.call_count == 1

    # Check the event type is COMPLETE
    call_args_list = mock_client.emit_run_event.call_args_list
    assert call_args_list[0][1]["event_type"] == RunState.COMPLETE


def test_process_and_emit_events_log(workload_env, mock_client, monkeypatch) -> None:
    process_and_emit_ol_events_for_artifact_repo_operation(
        operation=ArtifactRepoOperations.LOG,
        ol_client=mock_client,
        artifact_uri=TEST_S3_URI,
        artifact_path=TEST_MODEL_ARTIFACT_PATH,
    )

    # Verify run event was emitted once with COMPLETE state
    assert mock_client.emit_run_event.call_count == 1


def test_artifact_download_user_storage_not_tracked(workload_env, mock_client) -> None:
    """Test that /mnt/user_storage artifact URIs are NOT tracked during download."""
    artifact_uri = "/mnt/user_storage/artifacts"
    artifact_path = "model/file.pkl"

    process_and_emit_ol_events_for_artifact_repo_operation(
        operation=ArtifactRepoOperations.DOWNLOAD,
        ol_client=mock_client,
        artifact_uri=artifact_uri,
        artifact_path=artifact_path,
    )

    # /mnt/user_storage/ paths should NOT be tracked - no events emitted
    assert mock_client.emit_run_event.call_count == 0


def test_artifact_log_transforms_mnt_cluster_storage_path(
    workload_env, mock_client
) -> None:
    """Test that /mnt/cluster_storage artifact URIs are transformed during log."""
    from ray.anyscale.lineage.tests.test_constants import TEST_JOB_ID

    artifact_uri = "/mnt/cluster_storage/models"
    artifact_path = "my_model/checkpoint.ckpt"

    process_and_emit_ol_events_for_artifact_repo_operation(
        operation=ArtifactRepoOperations.LOG,
        ol_client=mock_client,
        artifact_uri=artifact_uri,
        artifact_path=artifact_path,
    )

    # Verify run event was emitted
    assert mock_client.emit_run_event.call_count == 1

    # Check the outputs to verify the transformed path
    call_args = mock_client.emit_run_event.call_args
    outputs = call_args[1].get("outputs", [])

    assert len(outputs) == 1
    output_dataset = outputs[0]

    # Check that the namespace is the transformed URI (format: file://{id}/path)
    expected_namespace = f"file://{TEST_JOB_ID}/mnt/cluster_storage/models"
    assert output_dataset.namespace == expected_namespace
    assert output_dataset.name == artifact_path


def test_artifact_download_transforms_mnt_shared_storage_path(
    workload_env, mock_client
) -> None:
    """Test that /mnt/shared_storage artifact URIs are transformed during download."""
    from ray.anyscale.lineage.tests.test_constants import TEST_CLOUD_ID

    artifact_uri = "/mnt/shared_storage/common/artifacts"
    artifact_path = "data/dataset.csv"

    process_and_emit_ol_events_for_artifact_repo_operation(
        operation=ArtifactRepoOperations.DOWNLOAD,
        ol_client=mock_client,
        artifact_uri=artifact_uri,
        artifact_path=artifact_path,
    )

    # Verify run event was emitted
    assert mock_client.emit_run_event.call_count == 1

    # Check the inputs to verify the transformed path
    call_args = mock_client.emit_run_event.call_args
    inputs = call_args[1].get("inputs", [])

    assert len(inputs) == 1
    input_dataset = inputs[0]

    # Check that the namespace is the transformed URI (format: file://{id}/path)
    expected_namespace = f"file://{TEST_CLOUD_ID}/mnt/shared_storage/common/artifacts"
    assert input_dataset.namespace == expected_namespace
    assert input_dataset.name == artifact_path


def test_artifact_download_does_not_transform_s3_paths(
    workload_env, mock_client
) -> None:
    """Test that S3 artifact URIs are not transformed."""
    artifact_uri = TEST_S3_URI
    artifact_path = TEST_MODEL_ARTIFACT_PATH

    process_and_emit_ol_events_for_artifact_repo_operation(
        operation=ArtifactRepoOperations.DOWNLOAD,
        ol_client=mock_client,
        artifact_uri=artifact_uri,
        artifact_path=artifact_path,
    )

    # Verify run event was emitted
    assert mock_client.emit_run_event.call_count == 1

    # Check the inputs to verify no transformation occurred
    call_args = mock_client.emit_run_event.call_args
    inputs = call_args[1].get("inputs", [])

    assert len(inputs) == 1
    input_dataset = inputs[0]

    # S3 URIs should not be transformed
    assert input_dataset.namespace == TEST_S3_URI
    assert input_dataset.name == TEST_MODEL_ARTIFACT_PATH


class TestShouldEmitOpenLineageEventForArtifact:
    """Test the should_emit_openlineage_event_for_artifact utility function."""

    def test_should_emit_for_regular_path(self):
        """Test that regular paths return True."""
        assert should_emit_openlineage_event_for_artifact(
            "/mnt/user_storage/data/file.csv"
        )
        assert should_emit_openlineage_event_for_artifact("s3://bucket/path/to/file")
        assert should_emit_openlineage_event_for_artifact(
            "file:/mnt/cluster_storage/model"
        )
        assert should_emit_openlineage_event_for_artifact(
            "/home/user/artifacts/model.pkl"
        )

    def test_should_skip_runs_uri(self):
        """Test that runs:/ URIs return False."""
        assert not should_emit_openlineage_event_for_artifact("runs:/run_id/model")
        assert not should_emit_openlineage_event_for_artifact(
            "runs:/abc123/artifacts/data.csv"
        )

    def test_should_skip_tmp_path(self):
        """Test that /tmp paths return False."""
        assert not should_emit_openlineage_event_for_artifact("/tmp/tmpfile")
        assert not should_emit_openlineage_event_for_artifact("/tmp/tmpdktllv6e")
        assert not should_emit_openlineage_event_for_artifact("/tmp/model.pkl")
        assert not should_emit_openlineage_event_for_artifact(
            "/tmp/nested/path/file.csv"
        )

    def test_should_emit_for_tmp_like_paths(self):
        """Test that paths containing 'tmp' but not starting with /tmp return True."""
        assert should_emit_openlineage_event_for_artifact("/home/tmp/file.csv")
        assert should_emit_openlineage_event_for_artifact("/data/tmpdir/model.pkl")
        assert should_emit_openlineage_event_for_artifact("s3://bucket/tmp/file")
