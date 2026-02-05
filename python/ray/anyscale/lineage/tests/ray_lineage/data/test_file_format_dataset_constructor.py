"""Tests for file_format dataset constructor module."""


from ray.anyscale.lineage.common.facets.dataset import FileFormats
from ray.anyscale.lineage.ray_lineage.data.dataset_constructor import file_format
from ray.anyscale.lineage.tests.test_constants import (
    TEST_CLOUD_ID,
    TEST_JOB_ID,
    TEST_RAY_DATA_FILE_URI as TEST_FILE_URI,
    TEST_RAY_DATA_S3_URI as TEST_S3_URI,
)


def test_process_file_format_datasource_builds_dataset(
    patch_facet_constructors, monkeypatch
):
    captured_args = {}

    def fake_create_input_dataset_from_args(**kwargs):
        captured_args.update(kwargs)
        return "input-dataset"

    monkeypatch.setattr(
        file_format,
        "create_openlineage_input_dataset_from_args",
        fake_create_input_dataset_from_args,
    )

    result = file_format.process_file_format_datasource(
        TEST_S3_URI, FileFormats.PARQUET
    )

    assert result == "input-dataset"
    assert captured_args == {
        "dataset_namespace": "namespace",
        "dataset_name": TEST_S3_URI,
        "facets": {
            "dataset_type": file_format.DatasetType.FILE,
            "datasource": TEST_S3_URI,
            "file_format": FileFormats.PARQUET,
        },
    }


def test_process_file_format_datasink_builds_dataset(
    patch_facet_constructors, monkeypatch
):
    captured_args = {}

    def fake_create_output_dataset_from_args(**kwargs):
        captured_args.update(kwargs)
        return "output-dataset"

    monkeypatch.setattr(
        file_format,
        "create_openlineage_output_dataset_from_args",
        fake_create_output_dataset_from_args,
    )

    result = file_format.process_file_format_datasink(TEST_FILE_URI, FileFormats.CSV)

    assert result == "output-dataset"
    assert captured_args == {
        "dataset_namespace": "namespace",
        "dataset_name": TEST_FILE_URI,
        "facets": {
            "dataset_type": file_format.DatasetType.FILE,
            "datasource": TEST_FILE_URI,
            "file_format": FileFormats.CSV,
        },
    }


def test_file_format_datasource_user_storage_not_tracked(
    sample_anyscale_env,
    monkeypatch,
):
    """Test that /mnt/user_storage paths are NOT tracked in file format datasources."""
    from ray.anyscale.lineage.ray_lineage.data.dataset_constructor import main

    processed_uris = []

    def fake_process(path, file_format):
        processed_uris.append(path)
        return f"dataset:{path}"

    monkeypatch.setattr(
        main,
        "process_file_format_datasource",
        fake_process,
    )

    class FakeDatasource:
        _source_paths = ["/mnt/user_storage/data/file.csv"]

    monkeypatch.setattr(
        main,
        "get_file_format_datasources",
        lambda: [FakeDatasource],
    )
    monkeypatch.setattr(
        main, "FILE_FORMATS_REGISTRY", {"FakeDatasource": FileFormats.CSV}
    )

    datasets, seen = main.process_datasource(FakeDatasource(), set())

    # /mnt/user_storage/ paths should NOT be tracked
    assert datasets == []
    assert processed_uris == []


def test_file_format_datasource_cluster_storage_transformed(
    sample_anyscale_env,
    monkeypatch,
):
    """Test that /mnt/cluster_storage paths are transformed in file format datasources."""
    from ray.anyscale.lineage.ray_lineage.data.dataset_constructor import main

    processed_uris = []

    def fake_process(path, file_format):
        processed_uris.append(path)
        return f"dataset:{path}"

    monkeypatch.setattr(
        main,
        "process_file_format_datasource",
        fake_process,
    )

    class FakeDatasource:
        _source_paths = ["/mnt/cluster_storage/models/model.pkl"]

    monkeypatch.setattr(
        main,
        "get_file_format_datasources",
        lambda: [FakeDatasource],
    )
    monkeypatch.setattr(
        main, "FILE_FORMATS_REGISTRY", {"FakeDatasource": FileFormats.UNKNOWN}
    )

    datasets, seen = main.process_datasource(FakeDatasource(), set())

    # /mnt/cluster_storage/ paths should be transformed with workload_id
    expected_transformed = f"file://{TEST_JOB_ID}/mnt/cluster_storage/models/model.pkl"
    assert len(datasets) == 1
    assert processed_uris == [expected_transformed]


def test_file_format_datasource_shared_storage_transformed(
    sample_anyscale_env,
    monkeypatch,
):
    """Test that /mnt/shared_storage paths are transformed in file format datasources."""
    from ray.anyscale.lineage.ray_lineage.data.dataset_constructor import main

    processed_uris = []

    def fake_process(path, file_format):
        processed_uris.append(path)
        return f"dataset:{path}"

    monkeypatch.setattr(
        main,
        "process_file_format_datasource",
        fake_process,
    )

    class FakeDatasource:
        _source_paths = ["/mnt/shared_storage/datasets/train.parquet"]

    monkeypatch.setattr(
        main,
        "get_file_format_datasources",
        lambda: [FakeDatasource],
    )
    monkeypatch.setattr(
        main, "FILE_FORMATS_REGISTRY", {"FakeDatasource": FileFormats.PARQUET}
    )

    datasets, seen = main.process_datasource(FakeDatasource(), set())

    # /mnt/shared_storage/ paths should be transformed with cloud_id
    expected_transformed = (
        f"file://{TEST_CLOUD_ID}/mnt/shared_storage/datasets/train.parquet"
    )
    assert len(datasets) == 1
    assert processed_uris == [expected_transformed]


def test_file_format_datasink_user_storage_not_tracked(
    sample_anyscale_env,
    monkeypatch,
):
    """Test that /mnt/user_storage paths are NOT tracked in file format datasinks."""
    from ray.anyscale.lineage.ray_lineage.data.dataset_constructor import main

    processed_uris = []

    def fake_process(path, file_format):
        processed_uris.append(path)
        return f"dataset:{path}"

    monkeypatch.setattr(
        main,
        "process_file_format_datasink",
        fake_process,
    )

    class FakeDatasink:
        unresolved_path = "/mnt/user_storage/output/result.json"

    monkeypatch.setattr(
        main,
        "get_file_format_datasinks",
        lambda: [FakeDatasink],
    )
    monkeypatch.setattr(
        main, "FILE_FORMATS_REGISTRY", {"FakeDatasink": FileFormats.JSON}
    )

    datasets, seen = main.process_datasink(FakeDatasink(), set())

    # /mnt/user_storage/ paths should NOT be tracked
    assert datasets == []
    assert processed_uris == []


def test_file_format_datasource_does_not_transform_non_mnt_paths(
    patch_facet_constructors,
    sample_anyscale_env,
):
    """Test that non-/mnt paths are not transformed."""
    from ray.anyscale.lineage.ray_lineage.data.dataset_constructor.file_format import (
        process_file_format_datasource,
    )

    path = "s3://bucket/data/file.csv"
    dataset = process_file_format_datasource(path, FileFormats.CSV)

    # Check that S3 paths are not transformed
    # Note: namespace is "namespace" due to the patch_dataset_naming fixture
    assert dataset.namespace == "namespace"
    assert dataset.name == path  # S3 path stays unchanged
    assert dataset.facets["datasource"] == path
