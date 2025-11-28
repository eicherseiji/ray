"""Tests for file_format dataset constructor module."""


from ray.anyscale.lineage.common.facets.dataset import FileFormats
from ray.anyscale.lineage.ray_lineage.data.dataset_constructor import file_format
from ray.anyscale.lineage.tests.test_constants import (
    TEST_RAY_DATA_FILE_URI as TEST_FILE_URI,
    TEST_RAY_DATA_S3_URI as TEST_S3_URI,
    TEST_JOB_ID,
    TEST_CLOUD_ID,
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


def test_file_format_datasource_transforms_mnt_user_storage_path(
    patch_facet_constructors,
    sample_anyscale_env,
):
    """Test that /mnt/user_storage paths are transformed in file format datasources."""
    from ray.anyscale.lineage.ray_lineage.data.dataset_constructor.file_format import (
        process_file_format_datasource,
    )

    path = "/mnt/user_storage/data/file.csv"
    dataset = process_file_format_datasource(path, FileFormats.CSV)

    # Check that the dataset namespace is "namespace" (local file system constant)
    assert dataset.namespace == "namespace"
    # Check that the dataset name is the transformed path
    assert dataset.name == f"{TEST_JOB_ID}:/mnt/user_storage/data/file.csv"

    # Check that the datasource facet contains the transformed URI
    expected_uri = f"{TEST_JOB_ID}:/mnt/user_storage/data/file.csv"
    assert dataset.facets["datasource"] == expected_uri


def test_file_format_datasource_transforms_mnt_cluster_storage_path(
    patch_facet_constructors,
    sample_anyscale_env,
):
    """Test that /mnt/cluster_storage paths are transformed in file format datasources."""
    from ray.anyscale.lineage.ray_lineage.data.dataset_constructor.file_format import (
        process_file_format_datasource,
    )

    path = "/mnt/cluster_storage/models/model.pkl"
    dataset = process_file_format_datasource(path, FileFormats.UNKNOWN)

    # Check that the dataset namespace is "namespace" (local file system constant)
    assert dataset.namespace == "namespace"
    # Check that the dataset name is the transformed path
    assert dataset.name == f"{TEST_JOB_ID}:/mnt/cluster_storage/models/model.pkl"

    # Check that the datasource facet contains the transformed URI
    expected_uri = f"{TEST_JOB_ID}:/mnt/cluster_storage/models/model.pkl"
    assert dataset.facets["datasource"] == expected_uri


def test_file_format_datasource_transforms_mnt_shared_storage_path(
    patch_facet_constructors,
    sample_anyscale_env,
):
    """Test that /mnt/shared_storage paths are transformed in file format datasources."""
    from ray.anyscale.lineage.ray_lineage.data.dataset_constructor.file_format import (
        process_file_format_datasource,
    )

    path = "/mnt/shared_storage/datasets/train.parquet"
    dataset = process_file_format_datasource(path, FileFormats.PARQUET)

    # Check that the dataset namespace is "namespace" (local file system constant)
    assert dataset.namespace == "namespace"
    # Check that the dataset name is the transformed path
    assert dataset.name == f"{TEST_CLOUD_ID}:/mnt/shared_storage/datasets/train.parquet"

    # Check that the datasource facet contains the transformed URI
    expected_uri = f"{TEST_CLOUD_ID}:/mnt/shared_storage/datasets/train.parquet"
    assert dataset.facets["datasource"] == expected_uri


def test_file_format_datasink_transforms_mnt_user_storage_path(
    patch_facet_constructors,
    sample_anyscale_env,
):
    """Test that /mnt/user_storage paths are transformed in file format datasinks."""
    from ray.anyscale.lineage.ray_lineage.data.dataset_constructor.file_format import (
        process_file_format_datasink,
    )

    path = "/mnt/user_storage/output/result.json"
    dataset = process_file_format_datasink(path, FileFormats.JSON)

    # Check that the dataset namespace is "namespace" (local file system constant)
    assert dataset.namespace == "namespace"
    # Check that the dataset name is the transformed path
    assert dataset.name == f"{TEST_JOB_ID}:/mnt/user_storage/output/result.json"

    # Check that the datasource facet contains the transformed URI
    expected_uri = f"{TEST_JOB_ID}:/mnt/user_storage/output/result.json"
    assert dataset.facets["datasource"] == expected_uri


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
