from unittest import mock

import pytest

from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo import (
    AnyscaleArtifactRepository,
)


@pytest.fixture
def mock_artifact_repo(monkeypatch):
    """Mock ArtifactRepository to avoid MLflow dependencies."""
    mock_init = mock.Mock()
    monkeypatch.setattr(
        "mlflow.store.artifact.artifact_repo.ArtifactRepository.__init__",
        mock_init,
    )

    client_mock = mock.Mock()
    monkeypatch.setattr(
        "ray.anyscale.lineage.common.openlineage_client.AnyscaleOpenLineageClient",
        mock.Mock(return_value=client_mock),
    )

    return mock_init, client_mock


def test_anyscale_artifact_repository_inherits_mixin_functionality(
    mock_artifact_repo, monkeypatch
):
    """Test that base artifact repository inherits mixin functionality for lineage tracking."""
    _mock_init, _client_mock = mock_artifact_repo

    process_mock = mock.Mock()
    monkeypatch.setattr(
        "ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin.process_and_emit_ol_events_for_artifact_repo_operation",
        process_mock,
    )

    mock_download = mock.Mock(return_value="downloaded/file")
    mock_log_artifact = mock.Mock(return_value="logged")
    mock_log_artifacts = mock.Mock(return_value="logged-multiple")

    repo = AnyscaleArtifactRepository("file:///tmp/artifacts")

    type(repo).__bases__[1].download_artifacts = mock_download
    type(repo).__bases__[1].log_artifact = mock_log_artifact
    type(repo).__bases__[1].log_artifacts = mock_log_artifacts

    result = repo.download_artifacts("test/path", "/local/dst")
    assert result == "downloaded/file"
    process_mock.assert_called()

    process_mock.reset_mock()

    result = repo.log_artifact("/local/file", "remote/path")
    assert result == "logged"
    process_mock.assert_called()

    process_mock.reset_mock()

    result = repo.log_artifacts("/local/dir", "remote/dir")
    assert result == "logged-multiple"
    process_mock.assert_called()
