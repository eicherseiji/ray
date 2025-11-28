from ray.anyscale.lineage.ray_lineage.data.facets.run.logical_plan import (
    LogicalPlanRunFacet,
)


class TestLogicalPlanRunFacet:
    """Test cases for LogicalPlanRunFacet."""

    def test_logical_plan_run_facet_schema_url(self):
        """Test that schema URL is correctly formatted."""
        facet = LogicalPlanRunFacet(logical_plan="test")
        schema_url = facet._get_schema()

        assert "lineage/ray_lineage/data/facets/run/logical_plan.py" in schema_url
        assert schema_url.startswith("https://")

    def test_logical_plan_run_facet_with_empty_plan(self):
        """Test LogicalPlanRunFacet with empty plan."""
        facet = LogicalPlanRunFacet(logical_plan="")

        assert facet.logical_plan == ""
