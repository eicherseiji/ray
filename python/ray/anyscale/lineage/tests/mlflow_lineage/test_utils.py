"""Tests for mlflow_openlineage.utils module."""

import pytest

from ray.anyscale.lineage.common.exceptions import AnyscaleLineageMLflowError
from ray.anyscale.lineage.mlflow_lineage.constants import (
    ANYSCALE_MLFLOW_ARTIFACT_REPO_GCS_PREFIX,
    ANYSCALE_MLFLOW_ARTIFACT_REPO_HTTP_PREFIX,
    ANYSCALE_MLFLOW_ARTIFACT_REPO_LOCAL_PREFIX,
    ANYSCALE_MLFLOW_ARTIFACT_REPO_MODELS_PREFIX,
    ANYSCALE_MLFLOW_ARTIFACT_REPO_PREFIX,
    ANYSCALE_MLFLOW_ARTIFACT_REPO_RUNS_PREFIX,
    ANYSCALE_MLFLOW_ARTIFACT_REPO_S3_PREFIX,
    ANYSCALE_MLFLOW_TRACKING_STORE_FILE_PREFIX,
    ANYSCALE_MLFLOW_TRACKING_STORE_PREFIX,
    ANYSCALE_MLFLOW_TRACKING_STORE_REST_PREFIX,
)
from ray.anyscale.lineage.mlflow_lineage.utils import (
    StoreType,
    catch_mlflow_store_exception,
    extract_upstream_store_uri,
)


class TestExtractUpstreamStoreUri:
    """Tests for extract_upstream_store_uri function."""

    def test_extract_tracking_store_uri_with_prefix(self):
        """Test extracting tracking store URI with anyscale prefix."""
        store_uri = f"{ANYSCALE_MLFLOW_TRACKING_STORE_PREFIX}:http://localhost:5000"
        result = extract_upstream_store_uri(store_uri, StoreType.TRACKING_STORE)
        assert result == "http://localhost:5000"

    def test_extract_tracking_store_file_uri_with_prefix(self):
        """Test extracting file-based tracking store URI with prefix."""
        store_uri = f"{ANYSCALE_MLFLOW_TRACKING_STORE_PREFIX}:/tmp/mlruns"
        result = extract_upstream_store_uri(store_uri, StoreType.TRACKING_STORE)
        assert result == "/tmp/mlruns"

    def test_extract_tracking_store_file_uri_with_specific_prefix(self):
        """Test extracting file tracking store URI with specific prefix."""
        store_uri = f"{ANYSCALE_MLFLOW_TRACKING_STORE_FILE_PREFIX}:/tmp/mlruns"
        result = extract_upstream_store_uri(store_uri, StoreType.TRACKING_STORE)
        assert result == "file:/tmp/mlruns"

    def test_extract_tracking_store_rest_uri_with_specific_prefix(self):
        """Test extracting rest tracking store URI with specific prefix."""
        store_uri = f"{ANYSCALE_MLFLOW_TRACKING_STORE_REST_PREFIX}://localhost:5000"
        result = extract_upstream_store_uri(store_uri, StoreType.TRACKING_STORE)
        assert result == "http://localhost:5000"

    def test_extract_tracking_store_rest_uri_with_https(self):
        """Test extracting rest tracking store URI with https."""
        store_uri = f"{ANYSCALE_MLFLOW_TRACKING_STORE_REST_PREFIX}s://secure.host:443"
        result = extract_upstream_store_uri(store_uri, StoreType.TRACKING_STORE)
        assert result == "https://secure.host:443"

    def test_extract_artifact_repo_uri_with_prefix(self):
        """Test extracting artifact repo URI with anyscale prefix."""
        store_uri = f"{ANYSCALE_MLFLOW_ARTIFACT_REPO_PREFIX}:s3://bucket/path"
        result = extract_upstream_store_uri(store_uri, StoreType.ARTIFACT_REPO)
        assert result == "s3://bucket/path"

    def test_extract_artifact_repo_local_uri_with_prefix(self):
        """Test extracting local artifact repo URI with prefix."""
        store_uri = f"{ANYSCALE_MLFLOW_ARTIFACT_REPO_PREFIX}:/tmp/artifacts"
        result = extract_upstream_store_uri(store_uri, StoreType.ARTIFACT_REPO)
        assert result == "/tmp/artifacts"

    def test_extract_with_multiple_colons(self):
        """Test extracting URI with multiple colons (e.g., http://)."""
        store_uri = f"{ANYSCALE_MLFLOW_TRACKING_STORE_PREFIX}:http://host:5000"
        result = extract_upstream_store_uri(store_uri, StoreType.TRACKING_STORE)
        assert result == "http://host:5000"

    def test_extract_preserves_empty_string(self):
        """Test that empty strings are preserved."""
        store_uri = ""
        result = extract_upstream_store_uri(store_uri, StoreType.TRACKING_STORE)
        assert result == ""

    def test_extract_different_prefix_not_removed(self):
        """Test that wrong prefix is not removed."""
        # Using artifact repo prefix with tracking store type
        store_uri = f"{ANYSCALE_MLFLOW_ARTIFACT_REPO_PREFIX}:s3://bucket"
        result = extract_upstream_store_uri(store_uri, StoreType.TRACKING_STORE)
        # Should not strip the prefix since it doesn't match
        assert result == store_uri

    def test_extract_artifact_repo_local_uri_with_specific_prefix(self):
        """Test extracting local artifact repo URI with specific prefix."""
        store_uri = f"{ANYSCALE_MLFLOW_ARTIFACT_REPO_LOCAL_PREFIX}:/tmp/artifacts"
        result = extract_upstream_store_uri(store_uri, StoreType.ARTIFACT_REPO)
        assert result == "file:/tmp/artifacts"

    def test_extract_artifact_repo_models_uri_with_specific_prefix(self):
        """Test extracting models artifact repo URI with specific prefix."""
        store_uri = f"{ANYSCALE_MLFLOW_ARTIFACT_REPO_MODELS_PREFIX}:/model-name/version"
        result = extract_upstream_store_uri(store_uri, StoreType.ARTIFACT_REPO)
        assert result == "models:/model-name/version"

    def test_extract_artifact_repo_s3_uri_with_specific_prefix(self):
        """Test extracting s3 artifact repo URI with specific prefix."""
        store_uri = f"{ANYSCALE_MLFLOW_ARTIFACT_REPO_S3_PREFIX}://bucket/path"
        result = extract_upstream_store_uri(store_uri, StoreType.ARTIFACT_REPO)
        assert result == "s3://bucket/path"

    def test_extract_artifact_repo_runs_uri_with_specific_prefix(self):
        """Test extracting runs artifact repo URI with specific prefix."""
        store_uri = f"{ANYSCALE_MLFLOW_ARTIFACT_REPO_RUNS_PREFIX}:/run-id/path"
        result = extract_upstream_store_uri(store_uri, StoreType.ARTIFACT_REPO)
        assert result == "runs:/run-id/path"

    def test_extract_artifact_repo_http_uri_with_specific_prefix(self):
        """Test extracting http artifact repo URI with specific prefix."""
        store_uri = f"{ANYSCALE_MLFLOW_ARTIFACT_REPO_HTTP_PREFIX}://host/path"
        result = extract_upstream_store_uri(store_uri, StoreType.ARTIFACT_REPO)
        assert result == "http://host/path"

    def test_extract_artifact_repo_https_uri_with_specific_prefix(self):
        """Test extracting https artifact repo URI with specific prefix."""
        store_uri = f"{ANYSCALE_MLFLOW_ARTIFACT_REPO_HTTP_PREFIX}://secure.host/path"
        result = extract_upstream_store_uri(store_uri, StoreType.ARTIFACT_REPO)
        assert result == "http://secure.host/path"

    def test_extract_artifact_repo_gcs_uri_with_specific_prefix(self):
        """Test extracting gcs artifact repo URI with specific prefix."""
        store_uri = f"{ANYSCALE_MLFLOW_ARTIFACT_REPO_GCS_PREFIX}://bucket/path"
        result = extract_upstream_store_uri(store_uri, StoreType.ARTIFACT_REPO)
        assert result == "gs://bucket/path"


class TestCatchMlflowStoreException:
    """Tests for catch_mlflow_store_exception decorator."""

    def test_decorator_catches_and_wraps_exception(self):
        """Test that decorator catches exceptions and wraps them."""

        @catch_mlflow_store_exception
        def failing_function():
            raise ValueError("test error")

        with pytest.raises(AnyscaleLineageMLflowError) as exc_info:
            failing_function()

        # Check that the original exception is wrapped
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert str(exc_info.value.__cause__) == "test error"

    def test_decorator_works_with_args_and_kwargs(self):
        """Test that decorator works with various argument types."""

        @catch_mlflow_store_exception
        def complex_function(a, b, *args, c=None, **kwargs):
            return (a, b, args, c, kwargs)

        result = complex_function(1, 2, 3, 4, c=5, d=6, e=7)
        assert result == (1, 2, (3, 4), 5, {"d": 6, "e": 7})

    def test_decorator_handles_different_exception_types(self):
        """Test that decorator handles different exception types."""

        @catch_mlflow_store_exception
        def raise_runtime_error():
            raise RuntimeError("runtime error")

        @catch_mlflow_store_exception
        def raise_type_error():
            raise TypeError("type error")

        with pytest.raises(AnyscaleLineageMLflowError) as exc_info:
            raise_runtime_error()
        assert isinstance(exc_info.value.__cause__, RuntimeError)

        with pytest.raises(AnyscaleLineageMLflowError) as exc_info:
            raise_type_error()
        assert isinstance(exc_info.value.__cause__, TypeError)

    def test_decorator_wraps_exception_with_context(self):
        """Test that decorator wraps exception and preserves context."""

        @catch_mlflow_store_exception
        def failing_function(arg1, kwarg1=None):
            raise ValueError("test error")

        with pytest.raises(AnyscaleLineageMLflowError) as exc_info:
            failing_function("test_arg", kwarg1="test_kwarg")

        # Verify the original exception is wrapped
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert str(exc_info.value.__cause__) == "test error"

    def test_decorator_exception_chain_preserved(self):
        """Test that exception chain is properly preserved."""

        @catch_mlflow_store_exception
        def nested_exception():
            try:
                raise ValueError("inner error")
            except ValueError as e:
                raise RuntimeError("outer error") from e

        with pytest.raises(AnyscaleLineageMLflowError) as exc_info:
            nested_exception()

        # Check the exception chain
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert isinstance(exc_info.value.__cause__.__cause__, ValueError)
