"""Tests for mlflow_openlineage.utils module."""

import pytest

from ray.anyscale.lineage.common.exceptions import AnyscaleLineageMLflowError
from ray.anyscale.lineage.mlflow_lineage import utils as mlflow_utils
from ray.anyscale.lineage.mlflow_lineage.utils import catch_mlflow_store_exception


class TestCatchMlflowStoreException:
    """Tests for catch_mlflow_store_exception decorator."""

    @pytest.mark.parametrize("exception_class", [ValueError, RuntimeError, TypeError])
    def test_decorator_catches_and_wraps_exception(self, monkeypatch, exception_class):
        """Test that decorator catches various exceptions and wraps them."""
        monkeypatch.setattr(mlflow_utils, "IGNORE_ERRORS", False)

        @catch_mlflow_store_exception
        def failing_function():
            raise exception_class("test error")

        with pytest.raises(AnyscaleLineageMLflowError) as exc_info:
            failing_function()

        assert isinstance(exc_info.value.__cause__, exception_class)

    def test_decorator_works_with_args_and_kwargs(self):
        """Test that decorator works with various argument types."""

        @catch_mlflow_store_exception
        def complex_function(a, b, *args, c=None, **kwargs):
            return (a, b, args, c, kwargs)

        result = complex_function(1, 2, 3, 4, c=5, d=6, e=7)
        assert result == (1, 2, (3, 4), 5, {"d": 6, "e": 7})

    def test_decorator_exception_chain_preserved(self, monkeypatch):
        """Test that exception chain is properly preserved."""
        monkeypatch.setattr(mlflow_utils, "IGNORE_ERRORS", False)

        @catch_mlflow_store_exception
        def nested_exception():
            try:
                raise ValueError("inner error")
            except ValueError as e:
                raise RuntimeError("outer error") from e

        with pytest.raises(AnyscaleLineageMLflowError) as exc_info:
            nested_exception()

        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert isinstance(exc_info.value.__cause__.__cause__, ValueError)

    def test_decorator_suppresses_error_when_ignore_errors_true(self, monkeypatch):
        """With IGNORE_ERRORS=True, exceptions are suppressed."""
        monkeypatch.setattr(mlflow_utils, "IGNORE_ERRORS", True)

        @catch_mlflow_store_exception
        def failing_function():
            raise ValueError("test error")

        result = failing_function()
        assert result is None
