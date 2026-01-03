from ray.anyscale.lineage.mlflow_lineage.facets.dataset.output_schema import (
    OutputSchemaDatasetFacet,
    create_output_schema_dataset_facet,
)


def test_create_output_schema_dataset_facet_from_fields() -> None:
    facet = create_output_schema_dataset_facet(fields=[{"name": "f1", "type": "int"}])

    assert isinstance(facet, OutputSchemaDatasetFacet)
    assert facet.fields[0].name == "f1"
    assert facet.fields[0].type == "int"
