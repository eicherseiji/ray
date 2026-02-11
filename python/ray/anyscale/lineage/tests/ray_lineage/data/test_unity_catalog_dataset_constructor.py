"""Tests for unity_catalog dataset constructor module."""


from ray.anyscale.lineage.ray_lineage.data.dataset_constructor import (
    main,
    unity_catalog,
)
from ray.anyscale.lineage.ray_lineage.data.utils import Datasources
from ray.anyscale.lineage.tests.test_constants import (
    TEST_CATALOG_NAME,
    TEST_DATABRICKS_HOST,
    TEST_DATABRICKS_URL,
    TEST_SCHEMA_NAME,
    TEST_TABLE_NAME,
    TEST_WAREHOUSE_ID,
)


def test_process_databricks_uc_datasource_builds_dataset(
    patch_facet_constructors, monkeypatch
):
    captured_args = {}

    def fake_create_input_dataset_from_args(**kwargs):
        captured_args.update(kwargs)
        return "databricks-uc-dataset"

    monkeypatch.setattr(
        unity_catalog,
        "create_openlineage_input_dataset_from_args",
        fake_create_input_dataset_from_args,
    )

    result = unity_catalog.process_databricks_uc_datasource(
        host=TEST_DATABRICKS_HOST,
        warehouse_id=TEST_WAREHOUSE_ID,
        catalog=TEST_CATALOG_NAME,
        schema=TEST_SCHEMA_NAME,
    )

    assert result == "databricks-uc-dataset"
    assert captured_args == {
        "dataset_namespace": TEST_DATABRICKS_HOST,
        "dataset_name": f"{TEST_WAREHOUSE_ID}.{TEST_CATALOG_NAME}.{TEST_SCHEMA_NAME}",
        "facets": {
            "dataset_type": unity_catalog.DatasetType.FILE,
            "datasource": f"{TEST_DATABRICKS_URL}/api/2.0/sql/statements/{TEST_WAREHOUSE_ID}/{TEST_CATALOG_NAME}/{TEST_SCHEMA_NAME}",
        },
    }


def test_process_unity_catalog_datasource_builds_dataset(
    patch_facet_constructors, monkeypatch
):
    captured_args = {}

    def fake_create_input_dataset_from_args(**kwargs):
        captured_args.update(kwargs)
        return "unity-catalog-dataset"

    monkeypatch.setattr(
        unity_catalog,
        "create_openlineage_input_dataset_from_args",
        fake_create_input_dataset_from_args,
    )

    result = unity_catalog.process_unity_catalog_datasource(
        base_url=TEST_DATABRICKS_URL,
        table_name=f"{TEST_CATALOG_NAME}.{TEST_SCHEMA_NAME}.{TEST_TABLE_NAME}",
    )

    assert result == "unity-catalog-dataset"
    assert captured_args == {
        "dataset_namespace": TEST_DATABRICKS_URL,
        "dataset_name": f"{TEST_CATALOG_NAME}.{TEST_SCHEMA_NAME}.{TEST_TABLE_NAME}",
        "facets": {
            "dataset_type": unity_catalog.DatasetType.FILE,
            "datasource": f"{TEST_DATABRICKS_URL}/api/2.1/unity-catalog/tables/{TEST_CATALOG_NAME}.{TEST_SCHEMA_NAME}.{TEST_TABLE_NAME}",
        },
    }


def test_process_datasource_prevents_duplicate_unity_catalog_uris(monkeypatch):
    """Test that duplicate Unity Catalog datasources are not processed twice."""

    processed_uris = []

    # Create a stub that is recognized as a Unity Catalog datasource
    unity_catalog_datasource_class = Datasources.UNITY_CATALOG_DATASOURCE.value

    # Create an instance with the required attributes
    datasource = unity_catalog_datasource_class.__new__(unity_catalog_datasource_class)
    datasource.base_url = TEST_DATABRICKS_URL
    datasource.table_full_name = (
        f"{TEST_CATALOG_NAME}.{TEST_SCHEMA_NAME}.{TEST_TABLE_NAME}"
    )

    def fake_process_unity_catalog_datasource(base_url, table_name):
        ds_uri = f"{base_url}/{table_name}"
        processed_uris.append(ds_uri)
        return f"dataset:{ds_uri}"

    monkeypatch.setattr(
        main,
        "get_file_format_datasources",
        lambda: [],
    )
    monkeypatch.setattr(
        main,
        "process_unity_catalog_datasource",
        fake_process_unity_catalog_datasource,
    )

    # First call - should process
    datasets1, seen1 = main.process_datasource(datasource, set())
    assert len(datasets1) == 1
    assert len(processed_uris) == 1

    # Second call with seen URIs - should skip
    datasets2, _ = main.process_datasource(datasource, seen1)
    assert len(datasets2) == 0  # Should not add duplicate
    assert len(processed_uris) == 1  # Should not process again
