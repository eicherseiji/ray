"""Tests for mongo dataset constructor module."""


from ray.anyscale.lineage.ray_lineage.data.dataset_constructor import main, mongo
from ray.anyscale.lineage.ray_lineage.data.utils import Datasinks, Datasources
from ray.anyscale.lineage.tests.test_constants import (
    PROD_DB_COLLECTION,
    PROD_DB_NAME,
    TEST_DB_COLLECTION,
    TEST_DB_NAME,
    TEST_MONGO_URI,
    TEST_MONGO_URI_AUTH,
)


def test_process_mongo_datasource_builds_dataset(patch_facet_constructors, monkeypatch):
    captured_args = {}

    def fake_create_input_dataset_from_args(**kwargs):
        captured_args.update(kwargs)
        return "mongo-input-dataset"

    monkeypatch.setattr(
        mongo,
        "create_openlineage_input_dataset_from_args",
        fake_create_input_dataset_from_args,
    )

    result = mongo.process_mongo_datasource(
        uri=TEST_MONGO_URI,
        database=TEST_DB_NAME,
        collection=TEST_DB_COLLECTION,
    )

    assert result == "mongo-input-dataset"
    assert captured_args == {
        "dataset_namespace": TEST_MONGO_URI,
        "dataset_name": f"{TEST_DB_NAME}.{TEST_DB_COLLECTION}",
        "facets": {
            "dataset_type": mongo.DatasetType.FILE,
            "datasource": TEST_MONGO_URI,
        },
    }


def test_process_mongo_datasink_builds_dataset(patch_facet_constructors, monkeypatch):
    captured_args = {}

    def fake_create_output_dataset_from_args(**kwargs):
        captured_args.update(kwargs)
        return "mongo-output-dataset"

    monkeypatch.setattr(
        mongo,
        "create_openlineage_output_dataset_from_args",
        fake_create_output_dataset_from_args,
    )

    result = mongo.process_mongo_datasink(
        uri=TEST_MONGO_URI_AUTH,
        database=PROD_DB_NAME,
        collection=PROD_DB_COLLECTION,
    )

    assert result == "mongo-output-dataset"
    assert captured_args == {
        "dataset_namespace": TEST_MONGO_URI_AUTH,
        "dataset_name": f"{PROD_DB_NAME}.{PROD_DB_COLLECTION}",
        "facets": {
            "dataset_type": mongo.DatasetType.FILE,
            "datasource": TEST_MONGO_URI_AUTH,
        },
    }


def test_process_datasource_prevents_duplicate_mongo_uris(monkeypatch):
    """Test that duplicate MongoDB datasources are not processed twice."""

    processed_uris = []

    # Create a stub that is recognized as a MongoDB datasource
    mongo_datasource_class = Datasources.MONGO_DATASOURCE.value

    # Create an instance with the required attributes
    datasource = mongo_datasource_class.__new__(mongo_datasource_class)
    datasource._uri = TEST_MONGO_URI
    datasource._database = TEST_DB_NAME
    datasource._collection = TEST_DB_COLLECTION

    def fake_process_mongo_datasource(uri, database, collection):
        ds_uri = f"{uri}/{database}/{collection}"
        processed_uris.append(ds_uri)
        return f"dataset:{ds_uri}"

    monkeypatch.setattr(
        main,
        "get_file_format_datasources",
        lambda: [],
    )
    monkeypatch.setattr(
        main,
        "process_mongo_datasource",
        fake_process_mongo_datasource,
    )

    # First call - should process
    datasets1, seen1 = main.process_datasource(datasource, set())
    assert len(datasets1) == 1
    assert len(processed_uris) == 1

    # Second call with seen URIs - should skip
    datasets2, _ = main.process_datasource(datasource, seen1)
    assert len(datasets2) == 0  # Should not add duplicate
    assert len(processed_uris) == 1  # Should not process again


def test_process_datasink_prevents_duplicate_mongo_uris(monkeypatch):
    """Test that duplicate MongoDB datasinks are not processed twice."""

    processed_uris = []

    # Create a stub that is recognized as a MongoDB datasink
    mongo_datasink_class = Datasinks.MONGO_DATASINK.value

    # Create an instance with the required attributes
    datasink = mongo_datasink_class.__new__(mongo_datasink_class)
    datasink.uri = TEST_MONGO_URI
    datasink.database = TEST_DB_NAME
    datasink.collection = TEST_DB_COLLECTION

    def fake_process_mongo_datasink(uri, database, collection):
        ds_uri = f"{uri}/{database}/{collection}"
        processed_uris.append(ds_uri)
        return f"dataset:{ds_uri}"

    monkeypatch.setattr(
        main,
        "get_file_format_datasinks",
        lambda: [],
    )
    monkeypatch.setattr(
        main,
        "process_mongo_datasink",
        fake_process_mongo_datasink,
    )

    # First call - should process
    datasets1, seen1 = main.process_datasink(datasink, set())
    assert len(datasets1) == 1
    assert len(processed_uris) == 1

    # Second call with seen URIs - should skip
    datasets2, _ = main.process_datasink(datasink, seen1)
    assert len(datasets2) == 0  # Should not add duplicate
    assert len(processed_uris) == 1  # Should not process again
