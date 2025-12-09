"""Tests for AnyscaleArtifactRepositoryMixin."""

from unittest import mock

import pytest

from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin import (
    AnyscaleArtifactRepositoryMixin,
)
from ray.anyscale.lineage.tests.test_constants import (
    TEST_ARTIFACT_PATH,
    TEST_MLFLOW_S3_URI as TEST_S3_URI,
)


class DummyBaseArtifactRepo:
    def __init__(self, artifact_uri: str, *args, **kwargs) -> None:
        self.artifact_uri = artifact_uri

    def download_artifacts(self, artifact_path, dst_path=None, *args, **kwargs):
        return "downloaded/path"

    def log_artifact(self, local_file, artifact_path=None, *args, **kwargs):
        return "logged-artifact"

    def log_artifacts(self, local_dir, artifact_path=None, *args, **kwargs):
        for i in range(3):
            self.log_artifact(f"{local_dir}/file{i}", artifact_path)
        return "logged-artifacts"


class DummyArtifactRepo(AnyscaleArtifactRepositoryMixin, DummyBaseArtifactRepo):
    pass


@pytest.fixture
def mock_process():
    return mock.Mock()


@pytest.fixture
def repo(monkeypatch, mock_process):
    monkeypatch.setattr(
        "ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin.AnyscaleOpenLineageClient",
        mock.Mock(return_value=mock.Mock()),
    )
    monkeypatch.setattr(
        "ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin.process_and_emit_ol_events_for_artifact_repo_operation",
        mock_process,
    )
    return DummyArtifactRepo(TEST_S3_URI)


class TestArtifactRepositoryMixin:
    """Tests for artifact repository mixin behavior."""

    def test_download_triggers_lineage_event(self, repo, mock_process):
        """Download artifacts triggers OpenLineage event."""
        result = repo.download_artifacts(TEST_ARTIFACT_PATH)
        assert result == "downloaded/path"
        mock_process.assert_called_once()

    def test_log_artifact_triggers_lineage_event(self, repo, mock_process):
        """Log artifact triggers OpenLineage event."""
        result = repo.log_artifact("local.file", artifact_path="remote/file")
        assert result == "logged-artifact"
        mock_process.assert_called_once()

    def test_log_artifacts_emits_single_event(self, repo, mock_process):
        """Log artifacts emits only one event despite calling log_artifact internally."""
        result = repo.log_artifacts("/local/dir", artifact_path="remote/dir")
        assert result == "logged-artifacts"
        mock_process.assert_called_once()

    def test_inside_log_artifacts_flag_prevents_duplicate_events(
        self, repo, mock_process
    ):
        """The _inside_log_artifacts flag prevents duplicate events from nested calls."""
        repo._inside_log_artifacts = True
        repo.log_artifact("local.file", artifact_path="remote/file")
        mock_process.assert_not_called()


class TestErrorHandling:
    """Tests for graceful error handling."""

    def test_client_init_failure_sets_client_to_none(self, monkeypatch):
        """When client initialization fails, ol_client is set to None."""
        monkeypatch.setattr(
            "ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin.AnyscaleOpenLineageClient",
            mock.Mock(side_effect=Exception("init failed")),
        )

        repo = DummyArtifactRepo(TEST_S3_URI)

        assert repo.ol_client is None
        assert repo.download_artifacts(TEST_ARTIFACT_PATH) == "downloaded/path"

    def test_process_failure_does_not_affect_operation(self, monkeypatch):
        """When event processing fails, the artifact operation still succeeds."""
        monkeypatch.setattr(
            "ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin.AnyscaleOpenLineageClient",
            mock.Mock(return_value=mock.Mock()),
        )
        monkeypatch.setattr(
            "ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin.process_and_emit_ol_events_for_artifact_repo_operation",
            mock.Mock(side_effect=Exception("processing failed")),
        )

        repo = DummyArtifactRepo(TEST_S3_URI)

        assert repo.download_artifacts(TEST_ARTIFACT_PATH) == "downloaded/path"
        assert repo.log_artifact("file") == "logged-artifact"
        assert repo.log_artifacts("/dir") == "logged-artifacts"
