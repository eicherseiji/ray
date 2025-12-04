from unittest import mock

import pytest

from ray.anyscale.lineage.mlflow_lineage.store.artifact.s3_artifact_repo import (
    AnyscaleS3ArtifactRepository,
)


@pytest.fixture
def mock_s3_repo(monkeypatch):
    """Mock S3ArtifactRepository to avoid AWS dependencies."""
    mock_init = mock.Mock()
    monkeypatch.setattr(
        "mlflow.store.artifact.s3_artifact_repo.S3ArtifactRepository.__init__",
        mock_init,
    )

    client_mock = mock.Mock()
    monkeypatch.setattr(
        "ray.anyscale.lineage.common.openlineage_client.AnyscaleOpenLineageClient",
        mock.Mock(return_value=client_mock),
    )

    return mock_init, client_mock


def test_anyscale_s3_artifact_repository_extracts_uri(mock_s3_repo):
    """Test that AnyscaleS3ArtifactRepository extracts upstream URI."""
    mock_init, _client_mock = mock_s3_repo

    repo = AnyscaleS3ArtifactRepository(
        "anyscale-mlflow-artifact-repo-s3://bucket/path"
    )

    mock_init.assert_called_once_with("s3://bucket/path")
    assert repo.artifact_uri == "s3://bucket/path"
