import pytest

from ray.anyscale.lineage.common.exceptions import AnyscaleLineageRayDataError
from ray.anyscale.lineage.ray_lineage.data.facet_constructor import (
    RayDataFacetConstructor,
)
from ray.anyscale.lineage.ray_lineage.data.facets.run.logical_plan import (
    LOGICAL_PLAN_RUN_FACET_KEY,
    LogicalPlanRunFacet,
)


class TestRayDataFacetConstructor:
    """Test cases for RayDataFacetConstructor."""

    def test_construct_logical_plan_run_facet_creates_correct_structure(self):
        """Test that logical plan run facet is constructed correctly."""
        logical_plan = "SELECT * FROM table"

        result = RayDataFacetConstructor.construct_logical_plan_run_facet(logical_plan)

        assert LOGICAL_PLAN_RUN_FACET_KEY in result
        facet = result[LOGICAL_PLAN_RUN_FACET_KEY]
        assert isinstance(facet, LogicalPlanRunFacet)
        assert facet.logical_plan == logical_plan

    def test_construct_logical_plan_run_facet_with_empty_plan(self):
        """Test logical plan run facet with empty plan."""
        logical_plan = ""

        result = RayDataFacetConstructor.construct_logical_plan_run_facet(logical_plan)

        assert LOGICAL_PLAN_RUN_FACET_KEY in result
        facet = result[LOGICAL_PLAN_RUN_FACET_KEY]
        assert facet.logical_plan == ""


class TestRayDataFacetConstructorErrorHandling:
    """Test error handling for RayDataFacetConstructor methods."""

    def test_construct_ownership_job_facet_wraps_errors(self, monkeypatch):
        """Test that errors in construct_ownership_job_facet are wrapped."""

        # Mock get_os_env to raise an exception
        def mock_get_os_env(key, default=""):
            raise RuntimeError("Environment error")

        monkeypatch.setattr(
            "ray.anyscale.lineage.common.facet_constructor.job_facet_constructor.get_os_env",
            mock_get_os_env,
        )

        with pytest.raises(AnyscaleLineageRayDataError):
            RayDataFacetConstructor.construct_ownership_job_facet()

    def test_construct_ownership_dataset_facet_wraps_errors(self, monkeypatch):
        """Test that errors in construct_ownership_dataset_facet are wrapped."""

        # Mock get_os_env to raise an exception
        def mock_get_os_env(key, default=""):
            raise RuntimeError("Environment error")

        monkeypatch.setattr(
            "ray.anyscale.lineage.common.facet_constructor.dataset_facet_constructor.get_os_env",
            mock_get_os_env,
        )

        with pytest.raises(AnyscaleLineageRayDataError):
            RayDataFacetConstructor.construct_ownership_dataset_facet()

    def test_construct_schema_dataset_facet_wraps_errors(self):
        """Test that errors in construct_schema_dataset_facet are wrapped."""
        import unittest.mock as mock

        # Mock the internal create function to raise an error
        with mock.patch(
            "ray.anyscale.lineage.common.facets.dataset.create_schema_dataset_facet",
            side_effect=RuntimeError("Schema error"),
        ), pytest.raises(AnyscaleLineageRayDataError):
            RayDataFacetConstructor.construct_schema_dataset_facet(
                [{"name": "field1", "type": "string"}]
            )

    def test_construct_anyscale_workload_details_job_facet_wraps_errors(
        self, monkeypatch
    ):
        """Test that errors in construct_anyscale_workload_details_job_facet are wrapped."""
        # Set up environment to make the method work up to a point
        monkeypatch.setenv("ANYSCALE_WORKLOAD_TYPE", "JOB")

        # Mock get_anyscale_workload_id to raise an exception
        def mock_get_anyscale_workload_id():
            raise RuntimeError("Workload ID error")

        monkeypatch.setattr(
            "ray.anyscale.lineage.common.facet_constructor.job_facet_constructor.get_anyscale_workload_id",
            mock_get_anyscale_workload_id,
        )

        with pytest.raises(AnyscaleLineageRayDataError):
            RayDataFacetConstructor.construct_anyscale_workload_details_job_facet()

    def test_construct_logical_plan_run_facet_wraps_errors(self):
        """Test that errors in construct_logical_plan_run_facet are wrapped."""
        import unittest.mock as mock

        # Mock the facet creation to raise an error
        with mock.patch(
            "ray.anyscale.lineage.ray_lineage.data.facets.run.create_logical_plan_run_facet",
            side_effect=RuntimeError("Logical plan error"),
        ), pytest.raises(AnyscaleLineageRayDataError):
            RayDataFacetConstructor.construct_logical_plan_run_facet("test plan")

    def test_error_handler_logs_error_details(self, monkeypatch):
        """Test that the error handler logs appropriate error details."""
        import unittest.mock as mock

        from ray.anyscale.lineage.ray_lineage.data import facet_constructor

        # Mock the logger
        mock_logger = mock.Mock()
        monkeypatch.setattr(facet_constructor, "logger", mock_logger)

        # Mock get_os_env to raise an exception
        def mock_get_os_env(key, default=""):
            raise ValueError("Test error message")

        monkeypatch.setattr(
            "ray.anyscale.lineage.common.facet_constructor.dataset_facet_constructor.get_os_env",
            mock_get_os_env,
        )

        with pytest.raises(AnyscaleLineageRayDataError):
            RayDataFacetConstructor.construct_ownership_dataset_facet()

        # Verify that logger.error was called
        assert mock_logger.error.called
        error_call = mock_logger.error.call_args[0][0]
        assert "construct_ownership_dataset_facet" in error_call
        assert "Test error message" in error_call or "ValueError" in error_call
