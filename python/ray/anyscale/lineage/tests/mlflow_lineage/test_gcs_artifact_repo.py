"""Tests for AnyscaleGCSArtifactRepository."""

from unittest import mock

import pytest

from ray.anyscale.lineage.mlflow_lineage.store.artifact.gcs_artifact_repo import (
    AnyscaleGCSArtifactRepository,
)


@pytest.fixture
def mock_gcs_repo(monkeypatch):
    monkeypatch.setattr(
        "mlflow.store.artifact.gcs_artifact_repo.GCSArtifactRepository.__init__",
        mock.Mock(),
    )
    monkeypatch.setattr(
        "ray.anyscale.lineage.common.openlineage_client.AnyscaleOpenLineageClient",
        mock.Mock(return_value=mock.Mock()),
    )


def test_init_passes_args_to_parent(mock_gcs_repo, monkeypatch):
    """AnyscaleGCSArtifactRepository passes arguments through to parent."""
    parent_init = mock.Mock()
    monkeypatch.setattr(
        "mlflow.store.artifact.gcs_artifact_repo.GCSArtifactRepository.__init__",
        parent_init,
    )

    custom_client = mock.Mock()

    AnyscaleGCSArtifactRepository(
        "gs://bucket/path",
        client=custom_client,
    )

    parent_init.assert_called_once_with(
        "gs://bucket/path",
        client=custom_client,
    )
