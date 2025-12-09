"""Tests for mlflow_lineage.utils module."""

from unittest import mock

import pytest

from ray.anyscale.lineage.common.exceptions import AnyscaleLineageMLflowError
from ray.anyscale.lineage.mlflow_lineage import utils as mlflow_utils
from ray.anyscale.lineage.mlflow_lineage.utils import (
    MLFLOW_ARTIFACTS_URI_SCHEME,
    catch_mlflow_store_exception,
    resolve_http_uri_from_mlflow_artifacts_uri,
)


class TestCatchMlflowStoreException:
    """Tests for catch_mlflow_store_exception decorator."""

    def test_decorator_catches_and_wraps_exception(self, monkeypatch):
        """Test that decorator catches exceptions and wraps them."""
        monkeypatch.setattr(mlflow_utils, "IGNORE_ERRORS", False)

        @catch_mlflow_store_exception
        def failing_function():
            raise ValueError("test error")

        with pytest.raises(AnyscaleLineageMLflowError) as exc_info:
            failing_function()

        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_decorator_suppresses_error_when_ignore_errors_true(self, monkeypatch):
        """With IGNORE_ERRORS=True, exceptions are suppressed."""
        monkeypatch.setattr(mlflow_utils, "IGNORE_ERRORS", True)

        @catch_mlflow_store_exception
        def failing_function():
            raise ValueError("test error")

        result = failing_function()
        assert result is None


class TestResolveHttpUriFromMlflowArtifactsUri:
    """Tests for resolve_http_uri_from_mlflow_artifacts_uri function."""

    def test_non_mlflow_artifacts_uri_returned_unchanged(self):
        """Non mlflow-artifacts: URIs are returned unchanged."""
        assert (
            resolve_http_uri_from_mlflow_artifacts_uri("s3://bucket/path")
            == "s3://bucket/path"
        )
        assert (
            resolve_http_uri_from_mlflow_artifacts_uri("gs://bucket/path")
            == "gs://bucket/path"
        )
        assert (
            resolve_http_uri_from_mlflow_artifacts_uri("/local/path") == "/local/path"
        )

    def test_mlflow_artifacts_uri_resolved(self, monkeypatch):
        """mlflow-artifacts: URIs are resolved using MLflow."""
        mock_resolve = mock.Mock(return_value="http://mlflow-server:5000/api/artifacts")
        mock_repo = mock.Mock()
        mock_repo.resolve_uri = mock_resolve

        monkeypatch.setattr(
            "mlflow.store.artifact.mlflow_artifacts_repo.MlflowArtifactsRepository",
            mock_repo,
        )
        monkeypatch.setattr(
            "mlflow.tracking.get_tracking_uri",
            lambda: "http://mlflow-server:5000",
        )

        result = resolve_http_uri_from_mlflow_artifacts_uri(
            f"{MLFLOW_ARTIFACTS_URI_SCHEME}/my-run/artifacts"
        )

        assert result == "http://mlflow-server:5000/api/artifacts"
        mock_resolve.assert_called_once()

    def test_resolution_failure_returns_original_uri(self, monkeypatch):
        """When resolution fails, original URI is returned."""
        monkeypatch.setattr(
            "mlflow.store.artifact.mlflow_artifacts_repo.MlflowArtifactsRepository",
            mock.Mock(
                resolve_uri=mock.Mock(side_effect=Exception("resolution failed"))
            ),
        )
        monkeypatch.setattr(
            "mlflow.tracking.get_tracking_uri",
            lambda: "http://mlflow-server:5000",
        )

        original_uri = f"{MLFLOW_ARTIFACTS_URI_SCHEME}/my-run/artifacts"
        result = resolve_http_uri_from_mlflow_artifacts_uri(original_uri)

        assert result == original_uri
