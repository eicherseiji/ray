from typing import Any

import attr
from openlineage.client.generated.schema_dataset import SchemaDatasetFacet

from ray.anyscale.lineage.common.constants import REPO_URL
from ray.anyscale.lineage.common.facets.dataset import create_schema_dataset_facet

OUTPUT_SCHEMA_DATASET_FACET_KEY: str = "outputSchema"


@attr.define
class OutputSchemaDatasetFacet(SchemaDatasetFacet):
    """Output schema dataset facet."""

    @staticmethod
    def _get_schema() -> str:
        return f"{REPO_URL}/blob/main/lineage/mlflow/facets/dataset/output_schema"


def create_output_schema_dataset_facet(
    fields: list[dict[str, Any]],
) -> OutputSchemaDatasetFacet:
    return OutputSchemaDatasetFacet(
        fields=create_schema_dataset_facet(fields=fields).fields,
    )
