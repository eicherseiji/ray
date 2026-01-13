from typing import Any, Optional

import mlflow

import ray.anyscale.lineage.mlflow_lineage.facets.dataset as mlflow_dataset_facets
import ray.anyscale.lineage.mlflow_lineage.facets.run as mlflow_run_facets
from ray.anyscale.lineage.common.exceptions import AnyscaleLineageMLflowError
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


def mlflow_facet_error_handler(
    e: Exception, func_name: str, func_args: Any, func_kwargs: Any
) -> Any:
    """Error handler for MLflow facet constructor methods."""
    logger.error(
        f"Error in MLflow facet constructor method '{func_name}' "
        f"for args '{func_args!s}' and kwargs '{func_kwargs!s}': {e!r}"
    )
    raise AnyscaleLineageMLflowError(e) from e


@wrap_class_methods(
    decorator=catch_class_method_exception(mlflow_facet_error_handler),
    exclude=("__init__",),
    include_inherited=True,
)
class MLflowFacetConstructor(
    DatasetFacetConstructor, JobFacetConstructor, RunFacetConstructor
):
    """Combined facet constructor for MLflow OpenLineage integration."""

    @staticmethod
    def construct_mlflow_workload_details_run_facet(
        host: str,
        experiment_id: str,
        run_id: str,
        mlflow_version: Optional[str] = None,
    ) -> dict[str, Any]:
        return {
            mlflow_run_facets.MLFLOW_WORKLOAD_DETAILS_RUN_FACET_KEY: mlflow_run_facets.create_mlflow_workload_details_run_facet(
                host=host,
                experiment_id=experiment_id,
                run_id=run_id,
                mlflow_version=mlflow_version or mlflow.__version__,
            )
        }

    @staticmethod
    def construct_input_schema_dataset_facet(
        fields: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            mlflow_dataset_facets.INPUT_SCHEMA_DATASET_FACET_KEY: mlflow_dataset_facets.create_input_schema_dataset_facet(
                fields=fields,
            )
        }

    @staticmethod
    def construct_output_schema_dataset_facet(
        fields: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            mlflow_dataset_facets.OUTPUT_SCHEMA_DATASET_FACET_KEY: mlflow_dataset_facets.create_output_schema_dataset_facet(
                fields=fields,
            )
        }
