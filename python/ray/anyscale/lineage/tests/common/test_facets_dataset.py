from ray.anyscale.lineage.common.facets import dataset as dataset_facets


def test_create_file_format_dataset_facet_with_free_form():
    facet = dataset_facets.create_file_format_dataset_facet(
        dataset_facets.FreeFormFileFormat(format="custom-format")
    )

    assert facet.format == "custom-format"


def test_free_form_file_format_accepts_any_string():
    format1 = dataset_facets.FreeFormFileFormat(format="pkl")
    assert format1.value == "pkl"

    format2 = dataset_facets.FreeFormFileFormat(format="python_function,sklearn")
    assert format2.value == "python_function,sklearn"

    format3 = dataset_facets.FreeFormFileFormat(format="")
    assert format3.value == ""


def test_create_schema_dataset_facet():
    facet = dataset_facets.create_schema_dataset_facet(
        [
            {
                "name": "field",
                "type": "string",
                "description": "desc",
                "fields": [
                    {
                        "name": "nested",
                        "type": "int",
                    }
                ],
            }
        ]
    )

    top_field = facet.fields[0]
    assert top_field.name == "field"
    assert top_field.fields[0].name == "nested"
