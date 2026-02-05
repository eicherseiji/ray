from typing import Any, Dict

import ray.anyscale.lineage.ray_lineage.data.facets.run as ray_data_run_facets
from ray.anyscale.lineage.common.exceptions import AnyscaleLineageRayDataError
from ray.anyscale.lineage.common.facet_constructor import (
    DatasetFacetConstructor,
    JobFacetConstructor,
    RunFacetConstructor,
)
from ray.anyscale.lineage.common.logging import get_logger
from ray.anyscale.lineage.common.utils import (
    catch_class_method_exception,
    wrap_class_methods,
)

logger = get_logger(__name__)


def ray_data_facet_error_handler(
    e: Exception, func_name: str, func_args: Any, func_kwargs: Any
) -> Any:
    """Error handler for Ray Data facet constructor methods."""
    logger.error(
        f"Error in Ray Data facet constructor method '{func_name}' "
        f"for args '{func_args!s}' and kwargs '{func_kwargs!s}': {e!r}"
    )
    raise AnyscaleLineageRayDataError(e) from e


@wrap_class_methods(
    decorator=catch_class_method_exception(ray_data_facet_error_handler),
    exclude=("__init__",),
    include_inherited=True,
)
class RayDataFacetConstructor(
    JobFacetConstructor, RunFacetConstructor, DatasetFacetConstructor
):
    """Combined facet constructor for Ray Data OpenLineage integration."""

    @staticmethod
    def construct_logical_plan_run_facet(
        logical_plan: str,
    ) -> Dict[str, ray_data_run_facets.LogicalPlanRunFacet]:
        return {
            ray_data_run_facets.LOGICAL_PLAN_RUN_FACET_KEY: ray_data_run_facets.create_logical_plan_run_facet(
                logical_plan=logical_plan,
            )
        }
