from unittest import mock

import pytest


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
