"""Tests for list_files dataset constructor module."""

from types import SimpleNamespace

import pytest

from ray.anyscale.lineage.common.facets.dataset import FileFormats, FreeFormFileFormat
from ray.anyscale.lineage.ray_lineage.data.dataset_constructor import list_files, main
from ray.anyscale.lineage.tests.test_constants import (
    TEST_RAY_DATA_S3_URI as TEST_S3_URI,
    TEST_CLOUD_ID,
)


# Test data for parameterized tests based on actual Ray Data datasource extensions
# Format: (datasource_name, registry_extensions, input_extension, expected_format)
FILE_FORMAT_TEST_CASES = [
    # CSV formats
    (
        "CSVDatasource",
        ["csv", "csv.gz", "csv.br", "csv.zst", "csv.lz4"],
        ".csv",
        FileFormats.CSV,
    ),
    (
        "CSVDatasource",
        ["csv", "csv.gz", "csv.br", "csv.zst", "csv.lz4"],
        ".csv.gz",
        FileFormats.CSV,
    ),
    (
        "CSVDatasource",
        ["csv", "csv.gz", "csv.br", "csv.zst", "csv.lz4"],
        "csv",
        FileFormats.CSV,
    ),
    # Parquet formats
    ("ParquetDatasource", ["parquet"], ".parquet", FileFormats.PARQUET),
    ("ParquetDatasource", ["parquet"], "parquet", FileFormats.PARQUET),
    # JSON formats
    (
        "ArrowJSONDatasource",
        ["json", "jsonl", "json.gz", "jsonl.gz"],
        ".json",
        FileFormats.JSON,
    ),
    (
        "ArrowJSONDatasource",
        ["json", "jsonl", "json.gz", "jsonl.gz"],
        ".jsonl",
        FileFormats.JSON,
    ),
    (
        "ArrowJSONDatasource",
        ["json", "jsonl", "json.gz", "jsonl.gz"],
        ".json.gz",
        FileFormats.JSON,
    ),
    # Image formats
    (
        "ImageDatasource",
        ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif"],
        ".png",
        FileFormats.IMAGE,
    ),
    (
        "ImageDatasource",
        ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif"],
        ".jpg",
        FileFormats.IMAGE,
    ),
    (
        "ImageDatasource",
        ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif"],
        ".jpeg",
        FileFormats.IMAGE,
    ),
    # Avro format
    ("AvroDatasource", ["avro"], ".avro", FileFormats.AVRO),
    # Numpy format
    ("NumpyDatasource", ["npy"], ".npy", FileFormats.NUMPY),
    # TFRecord format
    ("TFRecordDatasource", ["tfrecords"], ".tfrecords", FileFormats.TFRECORD),
    # Video formats
    (
        "VideoDatasource",
        ["mp4", "mkv", "mov", "avi", "webm"],
        ".mp4",
        FileFormats.VIDEO,
    ),
    (
        "VideoDatasource",
        ["mp4", "mkv", "mov", "avi", "webm"],
        ".webm",
        FileFormats.VIDEO,
    ),
    # Audio formats
    (
        "AudioDatasource",
        ["mp3", "wav", "aac", "flac", "ogg"],
        ".mp3",
        FileFormats.AUDIO,
    ),
    (
        "AudioDatasource",
        ["mp3", "wav", "aac", "flac", "ogg"],
        ".wav",
        FileFormats.AUDIO,
    ),
    # WebDataset format
    ("WebDatasetDatasource", ["tar"], ".tar", FileFormats.WEB_DATASET),
]


def test_get_list_files_common_facets_with_known_extension(
    patch_facet_constructors, monkeypatch
):
    """Test get_list_files_common_facets with a known file extension."""
    monkeypatch.setattr(
        list_files,
        "FILE_EXTENSIONS_REGISTRY",
        {"ParquetDatasource": ["parquet", "parq"]},
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


@pytest.mark.parametrize(
    "datasource_name,registry_extensions,input_extension,expected_format",
    FILE_FORMAT_TEST_CASES,
    ids=[f"{case[0]}-{case[2]}" for case in FILE_FORMAT_TEST_CASES],
)
def test_get_list_files_common_facets_file_formats(
    patch_facet_constructors,
    monkeypatch,
    datasource_name,
    registry_extensions,
    input_extension,
    expected_format,
):
    """Test for file format detection across all supported formats."""
    monkeypatch.setattr(
        list_files,
        "FILE_EXTENSIONS_REGISTRY",
        {datasource_name: registry_extensions},
    )
    monkeypatch.setattr(
        list_files,
        "FILE_FORMATS_REGISTRY",
        {datasource_name: expected_format},
    )

    facets = list_files.get_list_files_common_facets(TEST_S3_URI, [input_extension])

    assert facets == {
        "dataset_type": list_files.DatasetType.FILE,
        "datasource": TEST_S3_URI,
        "file_format": expected_format,
    }


def test_get_list_files_common_facets_matches_second_extension(
    patch_facet_constructors, monkeypatch
):
    """Test that file format is inferred from second extension when first doesn't match."""
    monkeypatch.setattr(
        list_files,
        "FILE_EXTENSIONS_REGISTRY",
        {"CSVDatasource": ["csv", "csv.gz"]},
    )
    monkeypatch.setattr(
        list_files,
        "FILE_FORMATS_REGISTRY",
        {"CSVDatasource": FileFormats.CSV},
    )

    # First extension ".unknown" won't match, but ".csv" should
    facets = list_files.get_list_files_common_facets(
        TEST_S3_URI, [".unknown", ".csv", ".txt"]
    )

    assert facets == {
        "dataset_type": list_files.DatasetType.FILE,
        "datasource": TEST_S3_URI,
        "file_format": FileFormats.CSV,
    }


def test_get_list_files_common_facets_with_unrecognized_extension(
    patch_facet_constructors, monkeypatch
):
    """Test get_list_files_common_facets with an unrecognized file extension uses FreeFormFileFormat."""
    monkeypatch.setattr(
        list_files,
        "FILE_EXTENSIONS_REGISTRY",
        {"ParquetDatasource": ["parquet", "parq"]},
    )
    monkeypatch.setattr(
        list_files,
        "FILE_FORMATS_REGISTRY",
        {"ParquetDatasource": FileFormats.PARQUET},
    )

    facets = list_files.get_list_files_common_facets(TEST_S3_URI, [".xyz"])

    assert facets["dataset_type"] == list_files.DatasetType.FILE
    assert facets["datasource"] == TEST_S3_URI
    assert isinstance(facets["file_format"], FreeFormFileFormat)
    assert facets["file_format"].value == "XYZ"


def test_get_list_files_common_facets_with_no_extensions(
    patch_facet_constructors, monkeypatch
):
    """Test get_list_files_common_facets with no file extensions returns UNKNOWN."""
    monkeypatch.setattr(
        list_files,
        "FILE_EXTENSIONS_REGISTRY",
        {"ParquetDatasource": ["parquet", "parq"]},
    )
    monkeypatch.setattr(
        list_files,
        "FILE_FORMATS_REGISTRY",
        {"ParquetDatasource": FileFormats.PARQUET},
    )

    facets = list_files.get_list_files_common_facets(TEST_S3_URI, [])

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
        {"ParquetDatasource": ["parquet"]},
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
