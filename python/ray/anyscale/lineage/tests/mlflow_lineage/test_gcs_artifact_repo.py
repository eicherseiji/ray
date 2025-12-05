from unittest import mock

import pytest

from ray.anyscale.lineage.mlflow_lineage.store.artifact.gcs_artifact_repo import (
    AnyscaleGCSArtifactRepository,
)


@pytest.fixture
def mock_gcs_repo(monkeypatch):
    """Mock GCSArtifactRepository to avoid GCP dependencies."""
    mock_init = mock.Mock()
    monkeypatch.setattr(
        "mlflow.store.artifact.gcs_artifact_repo.GCSArtifactRepository.__init__",
        mock_init,
    )

    client_mock = mock.Mock()
    monkeypatch.setattr(
        "ray.anyscale.lineage.common.openlineage_client.AnyscaleOpenLineageClient",
        mock.Mock(return_value=client_mock),
    )

    return mock_init, client_mock


def test_anyscale_gcs_artifact_repository_init_with_client(mock_gcs_repo):
    """Test that AnyscaleGCSArtifactRepository initializes with custom client."""
    mock_init, _client_mock = mock_gcs_repo

    custom_client = mock.Mock()
    credential_refresh = mock.Mock()

    repo = AnyscaleGCSArtifactRepository(
        "gs://bucket/path",
        client=custom_client,
        credential_refresh_def=credential_refresh,
    )

    mock_init.assert_called_once_with(
        "gs://bucket/path", custom_client, credential_refresh
    )

    assert repo.artifact_uri == "gs://bucket/path"
    assert hasattr(repo, "ol_client")
    assert repo.ol_client is not None
