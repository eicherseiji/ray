from typing import Any

from openlineage.client.facet_v2 import schema_dataset

SCHEMA_DATASET_FACET_KEY: str = "schema"


def _create_schema_dataset_facet_fields(
    fields: list[dict[str, Any]],
) -> list[schema_dataset.SchemaDatasetFacetFields]:
    schema_fields = []
    for field in fields:
        if not field:
            continue

        field_name = field.get("name")
        if not field_name:
            continue

        field_type = field.get("type")
        field_description = field.get("description")
        field_fields = _create_schema_dataset_facet_fields(field.get("fields", []))
        schema_fields.append(
            schema_dataset.SchemaDatasetFacetFields(
                name=field_name,
                type=field_type,
                description=field_description,
                fields=field_fields,
            )
        )
    return schema_fields


def create_schema_dataset_facet(
    fields: list[dict[str, Any]],
) -> schema_dataset.SchemaDatasetFacet:
    schema_fields = _create_schema_dataset_facet_fields(fields)
    return schema_dataset.SchemaDatasetFacet(fields=schema_fields)
