"""Tests for AnyscaleArtifactRepository."""

from unittest import mock

import pytest

from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo import (
    AnyscaleArtifactRepository,
)
from ray.anyscale.lineage.tests.test_constants import TEST_MLFLOW_S3_URI


@pytest.fixture
def mock_artifact_repo(monkeypatch):
    monkeypatch.setattr(
        "mlflow.store.artifact.artifact_repo.ArtifactRepository.__init__",
        mock.Mock(),
    )
    monkeypatch.setattr(
        "ray.anyscale.lineage.common.openlineage_client.AnyscaleOpenLineageClient",
        mock.Mock(return_value=mock.Mock()),
    )


def test_inherits_mixin_functionality(mock_artifact_repo, monkeypatch):
    """AnyscaleArtifactRepository inherits mixin functionality for lineage tracking."""
    process_mock = mock.Mock()
    monkeypatch.setattr(
        "ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin.process_and_emit_ol_events_for_artifact_repo_operation",
        process_mock,
    )

    mock_download = mock.Mock(return_value="downloaded/file")
    mock_log = mock.Mock(return_value="logged")

    repo = AnyscaleArtifactRepository(TEST_MLFLOW_S3_URI)
    repo.artifact_uri = TEST_MLFLOW_S3_URI

    type(repo).__bases__[1].download_artifacts = mock_download
    type(repo).__bases__[1].log_artifact = mock_log

    assert repo.download_artifacts("test/path") == "downloaded/file"
    process_mock.assert_called()

    process_mock.reset_mock()

    assert repo.log_artifact("local/file") == "logged"
    process_mock.assert_called()
