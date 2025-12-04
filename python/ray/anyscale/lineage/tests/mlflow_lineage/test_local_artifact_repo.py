from unittest import mock

import pytest

from ray.anyscale.lineage.mlflow_lineage.store.artifact.local_artifact_repo import (
    AnyscaleLocalArtifactRepository,
)


@pytest.fixture
def mock_local_repo(monkeypatch):
    """Mock LocalArtifactRepository to avoid file system dependencies."""
    mock_init = mock.Mock()
    monkeypatch.setattr(
        "mlflow.store.artifact.local_artifact_repo.LocalArtifactRepository.__init__",
        mock_init,
    )

    client_mock = mock.Mock()
    monkeypatch.setattr(
        "ray.anyscale.lineage.common.openlineage_client.AnyscaleOpenLineageClient",
        mock.Mock(return_value=client_mock),
    )

    return mock_init, client_mock


def test_anyscale_local_artifact_repository_extracts_uri(mock_local_repo):
    """Test that AnyscaleLocalArtifactRepository extracts upstream URI."""
    mock_init, _client_mock = mock_local_repo

    repo = AnyscaleLocalArtifactRepository(
        "anyscale-mlflow-artifact-repo-local:/local/path"
    )

    mock_init.assert_called_once_with("file:/local/path")
    assert repo.artifact_uri == "file:/local/path"
