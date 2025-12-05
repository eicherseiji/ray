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
        self.initialized_with = (artifact_uri, args, kwargs)
        self.artifact_uri = artifact_uri

    def download_artifacts(self, artifact_path, dst_path=None, *args, **kwargs):
        self.download_args = (artifact_path, dst_path, args, kwargs)
        return "downloaded/path"

    def log_artifact(self, local_file, artifact_path=None, *args, **kwargs):
        self.log_artifact_args = (local_file, artifact_path, args, kwargs)
        return "logged-artifact"

    def log_artifacts(self, local_dir, artifact_path=None, *args, **kwargs):
        self.log_artifacts_args = (local_dir, artifact_path, args, kwargs)
        return "logged-artifacts"


class DummyArtifactRepo(AnyscaleArtifactRepositoryMixin, DummyBaseArtifactRepo):
    def __init__(self, artifact_uri: str) -> None:
        super().__init__(artifact_uri)


@pytest.fixture
def repo(monkeypatch):
    client_mock = mock.Mock()
    monkeypatch.setattr(
        "ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin.AnyscaleOpenLineageClient",
        mock.Mock(return_value=client_mock),
    )
    process_mock = mock.Mock()
    monkeypatch.setattr(
        "ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin.process_and_emit_ol_events_for_artifact_repo_operation",
        process_mock,
    )

    artifact_repo = DummyArtifactRepo(TEST_S3_URI)
    return artifact_repo, client_mock, process_mock


def test_download_artifacts_triggers_lineage(repo):
    artifact_repo, _, process_mock = repo

    result = artifact_repo.download_artifacts(TEST_ARTIFACT_PATH, dst_path="/tmp")

    assert result == "downloaded/path"
    process_mock.assert_called_once()


def test_log_artifact_triggers_lineage(repo):
    artifact_repo, _, process_mock = repo

    result = artifact_repo.log_artifact("local.file", artifact_path="remote/file")

    assert result == "logged-artifact"
    process_mock.assert_called()


def test_log_artifacts_defaults_to_local_dir(repo):
    artifact_repo, _, process_mock = repo

    result = artifact_repo.log_artifacts("/local/dir")

    assert result == "logged-artifacts"
    process_mock.assert_called()
