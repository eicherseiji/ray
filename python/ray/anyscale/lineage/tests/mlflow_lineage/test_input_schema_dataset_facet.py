from ray.anyscale.lineage.mlflow_lineage.facets.dataset.input_schema import (
    InputSchemaDatasetFacet,
    create_input_schema_dataset_facet,
)


def test_create_input_schema_dataset_facet_from_fields() -> None:
    facet = create_input_schema_dataset_facet(fields=[{"name": "f1", "type": "string"}])

    assert isinstance(facet, InputSchemaDatasetFacet)
    assert facet.fields[0].name == "f1"
    assert facet.fields[0].type == "string"
