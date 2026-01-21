from typing import Any

import attr
from openlineage.client.generated.schema_dataset import SchemaDatasetFacet

from ray.anyscale.lineage.common.constants import REPO_URL
from ray.anyscale.lineage.common.facets.dataset import create_schema_dataset_facet

INPUT_SCHEMA_DATASET_FACET_KEY: str = "inputSchema"


@attr.define
class InputSchemaDatasetFacet(SchemaDatasetFacet):
    """Input schema dataset facet."""

    @staticmethod
    def _get_schema() -> str:
        return f"{REPO_URL}/blob/main/lineage/mlflow/facets/dataset/input_schema"


def create_input_schema_dataset_facet(
    fields: list[dict[str, Any]],
) -> InputSchemaDatasetFacet:
    return InputSchemaDatasetFacet(
        fields=create_schema_dataset_facet(fields=fields).fields,
    )
