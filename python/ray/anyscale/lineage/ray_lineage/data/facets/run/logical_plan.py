from typing import ClassVar, List

import attr
from openlineage.client.facet_v2 import RunFacet

from ray.anyscale.lineage.common.constants import REPO_URL

LOGICAL_PLAN_RUN_FACET_KEY: str = "logicalPlan"


@attr.define
class LogicalPlanRunFacet(RunFacet):
    """Logical plan run facet."""

    logical_plan: str
    """Logical plan"""

    _additional_skip_redact: ClassVar[List[str]] = [
        "logical_plan",
    ]

    @staticmethod
    def _get_schema() -> str:
        return (
            f"{REPO_URL}/blob/main/lineage/ray_lineage/data/facets/run/logical_plan.py"
        )


def create_logical_plan_run_facet(logical_plan: str) -> LogicalPlanRunFacet:
    return LogicalPlanRunFacet(logical_plan=logical_plan)
