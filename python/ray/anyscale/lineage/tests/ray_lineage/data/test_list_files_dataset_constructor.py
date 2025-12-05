"""Tests for list_files dataset constructor module."""

from types import SimpleNamespace


from ray.anyscale.lineage.common.facets.dataset import FileFormats
from ray.anyscale.lineage.ray_lineage.data.dataset_constructor import list_files, main
from ray.anyscale.lineage.tests.test_constants import (
    TEST_RAY_DATA_S3_URI as TEST_S3_URI,
    TEST_CLOUD_ID,
)


def test_get_list_files_common_facets_with_known_extension(
    patch_facet_constructors, monkeypatch
):
    """Test get_list_files_common_facets with a known file extension."""
    monkeypatch.setattr(
        list_files,
        "FILE_EXTENSIONS_REGISTRY",
        {"ParquetDatasource": [".parquet", ".parq"]},
    )
    monkeypatch.setattr(
        list_files,
        "FILE_FORMATS_REGISTRY",
        {"ParquetDatasource": FileFormats.PARQUET},
    )

    facets = list_files.get_list_files_common_facets(TEST_S3_URI, [".parquet"])

    assert facets == {
        "dataset_type": list_files.DatasetType.FILE,
        "datasource": TEST_S3_URI,
        "file_format": FileFormats.PARQUET,
    }


def test_get_list_files_common_facets_with_unknown_extension(
    patch_facet_constructors, monkeypatch
):
    """Test get_list_files_common_facets with an unknown file extension."""
    monkeypatch.setattr(
        list_files,
        "FILE_EXTENSIONS_REGISTRY",
        {"ParquetDatasource": [".parquet", ".parq"]},
    )
    monkeypatch.setattr(
        list_files,
        "FILE_FORMATS_REGISTRY",
        {"ParquetDatasource": FileFormats.PARQUET},
    )

    facets = list_files.get_list_files_common_facets(TEST_S3_URI, [".unknown"])

    # Should have dataset_type, datasource, and UNKNOWN file_format
    assert facets == {
        "dataset_type": list_files.DatasetType.FILE,
        "datasource": TEST_S3_URI,
        "file_format": FileFormats.UNKNOWN,
    }


def test_process_list_files_operator_path_builds_dataset(
    patch_facet_constructors, monkeypatch
):
    """Test process_list_files_operator_path builds a complete dataset."""
    captured_args = {}

    def fake_create_input_dataset_from_args(**kwargs):
        captured_args.update(kwargs)
        return "list-files-input-dataset"

    monkeypatch.setattr(
        list_files,
        "create_openlineage_input_dataset_from_args",
        fake_create_input_dataset_from_args,
    )
    monkeypatch.setattr(
        list_files,
        "FILE_EXTENSIONS_REGISTRY",
        {"ParquetDatasource": [".parquet"]},
    )
    monkeypatch.setattr(
        list_files,
        "FILE_FORMATS_REGISTRY",
        {"ParquetDatasource": FileFormats.PARQUET},
    )

    result = list_files.process_list_files_operator_path("/data/files", [".parquet"])

    assert result == "list-files-input-dataset"
    assert captured_args == {
        "dataset_namespace": "namespace",
        "dataset_name": "/data/files",
        "facets": {
            "dataset_type": list_files.DatasetType.FILE,
            "datasource": "/data/files",
            "file_format": FileFormats.PARQUET,
        },
    }


def test_process_list_files_operator_with_valid_paths(monkeypatch):
    """Test process_list_files_operator with valid remote paths."""
    processed_paths = []

    def fake_process_path(path, file_extensions):
        processed_paths.append((path, file_extensions))
        return f"dataset:{path}"

    monkeypatch.setattr(
        main,
        "process_list_files_operator_path",
        fake_process_path,
    )

    # Use remote paths (s3://) which are always tracked
    operator = SimpleNamespace(
        _source_paths=["s3://bucket/a", "s3://bucket/b"],
        file_extensions=[".parquet", ".parq"],
    )

    datasets, seen = main.process_list_files_operator(operator, set())

    assert datasets == ["dataset:s3://bucket/a", "dataset:s3://bucket/b"]
    assert processed_paths == [
        ("s3://bucket/a", [".parquet", ".parq"]),
        ("s3://bucket/b", [".parquet", ".parq"]),
    ]
    assert seen == {"s3://bucket/a", "s3://bucket/b"}


def test_process_list_files_operator_skips_duplicates(monkeypatch):
    """Test process_list_files_operator skips duplicate paths."""
    processed_paths = []

    def fake_process_path(path, file_extensions):
        processed_paths.append(path)
        return f"dataset:{path}"

    monkeypatch.setattr(
        main,
        "process_list_files_operator_path",
        fake_process_path,
    )

    # Use remote paths (s3://) which are always tracked
    operator = SimpleNamespace(
        _source_paths=["s3://bucket/a", "s3://bucket/a", "s3://bucket/b"],
        file_extensions=[".csv"],
    )

    datasets, seen = main.process_list_files_operator(operator, set())

    assert len(datasets) == 2
    assert processed_paths == ["s3://bucket/a", "s3://bucket/b"]
    assert seen == {"s3://bucket/a", "s3://bucket/b"}


def test_process_list_files_operator_with_no_paths(monkeypatch):
    """Test process_list_files_operator with no paths returns empty list."""
    operator = SimpleNamespace(
        _source_paths=[],
        file_extensions=[".parquet"],
    )

    datasets, seen = main.process_list_files_operator(operator, set())

    assert datasets == []
    assert seen == set()


def test_process_list_files_operator_handles_string_path(monkeypatch):
    """Test process_list_files_operator converts string path to list."""
    processed_paths = []

    def fake_process_path(path, file_extensions):
        processed_paths.append(path)
        return f"dataset:{path}"

    monkeypatch.setattr(
        main,
        "process_list_files_operator_path",
        fake_process_path,
    )

    # Create operator with _source_paths as a string instead of list (remote path)
    operator = SimpleNamespace(
        _source_paths="s3://bucket/single_path",
        file_extensions=[".json"],
    )

    datasets, seen = main.process_list_files_operator(operator, set())

    assert len(datasets) == 1
    assert datasets == ["dataset:s3://bucket/single_path"]
    assert processed_paths == ["s3://bucket/single_path"]
    assert seen == {"s3://bucket/single_path"}


def test_list_files_operator_user_storage_not_tracked(
    sample_anyscale_env,
    monkeypatch,
):
    """Test that /mnt/user_storage paths are NOT tracked in ListFiles operator."""
    processed_paths = []

    def fake_process_path(path, file_extensions):
        processed_paths.append(path)
        return f"dataset:{path}"

    monkeypatch.setattr(
        main,
        "process_list_files_operator_path",
        fake_process_path,
    )

    operator = SimpleNamespace(
        _source_paths=["/mnt/user_storage/data"],
        file_extensions=[".csv"],
    )

    datasets, seen = main.process_list_files_operator(operator, set())

    # /mnt/user_storage/ paths should NOT be tracked
    assert datasets == []
    assert processed_paths == []
    assert seen == {"/mnt/user_storage/data"}


def test_list_files_operator_transforms_mnt_shared_storage_path(
    sample_anyscale_env,
    monkeypatch,
):
    """Test that /mnt/shared_storage paths are transformed in ListFiles operator."""
    processed_paths = []

    def fake_process_path(path, file_extensions):
        processed_paths.append(path)
        return f"dataset:{path}"

    monkeypatch.setattr(
        main,
        "process_list_files_operator_path",
        fake_process_path,
    )

    operator = SimpleNamespace(
        _source_paths=["/mnt/shared_storage/datasets"],
        file_extensions=[".parquet"],
    )

    datasets, seen = main.process_list_files_operator(operator, set())

    # /mnt/shared_storage/ paths should be transformed with cloud_id
    expected_transformed = f"file://{TEST_CLOUD_ID}/mnt/shared_storage/datasets"
    assert len(datasets) == 1
    assert processed_paths == [expected_transformed]
    assert seen == {"/mnt/shared_storage/datasets"}
