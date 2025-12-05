from unittest import mock

import pytest


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
