from typing import ClassVar

import attr
from openlineage.client.facet_v2 import RunFacet

from ray.anyscale.lineage.common.constants import REPO_URL

MLFLOW_WORKLOAD_DETAILS_RUN_FACET_KEY: str = "mlflowWorkloadDetails"


@attr.define
class MLflowWorkloadDetailsRunFacet(RunFacet):
    """Run facet containing MLflow workload details."""

    host: str
    """MLflow host"""

    experiment_id: str
    """MLflow experiment id"""

    run_id: str
    """MLflow run id"""

    mlflow_version: str
    """MLflow version"""

    _additional_skip_redact: ClassVar[list[str]] = [
        "host",
        "experiment_id",
        "run_id",
        "mlflow_version",
    ]

    @staticmethod
    def _get_schema() -> str:
        return f"{REPO_URL}/blob/main/lineage/mlflow/facets/run/mlflow_workload_details"


def create_mlflow_workload_details_run_facet(
    host: str,
    experiment_id: str,
    run_id: str,
    mlflow_version: str,
) -> MLflowWorkloadDetailsRunFacet:
    return MLflowWorkloadDetailsRunFacet(
        host=host,
        experiment_id=experiment_id,
        run_id=run_id,
        mlflow_version=mlflow_version,
    )
