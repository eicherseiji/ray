from unittest import mock

import pytest

from ray.anyscale.lineage.common import dataset_naming
from ray.anyscale.lineage.common.exceptions import AnyscaleLineageSDKError
from ray.anyscale.lineage.tests.test_constants import (
    TEST_OL_NAME_GENERIC,
    TEST_OL_NAMESPACE,
)


class TestDatasetNamingClass:
    """Test dataset naming class retrieval."""

    @pytest.mark.parametrize(
        "dataset_type,expected_class_name",
        [
            (dataset_naming.OpenLineageDatasetNamingTypes.S3, "S3"),
            (dataset_naming.OpenLineageDatasetNamingTypes.BIG_QUERY, "BigQuery"),
            (dataset_naming.OpenLineageDatasetNamingTypes.POSTGRES, "Postgres"),
        ],
    )
    def test_get_ol_dataset_naming_class(self, dataset_type, expected_class_name):
        with mock.patch(
            "ray.anyscale.lineage.common.dataset_naming.dataset_naming"
        ) as mock_module:
            mock_class = mock.Mock()
            setattr(mock_module, expected_class_name, mock_class)

            result = dataset_naming.get_ol_dataset_naming_class(dataset_type)

            assert result is mock_class


class TestResolveDatasetNamespaceAndName:
    """Test dataset namespace and name resolution."""

    def test_resolve_with_valid_dataset_class(self):
        mock_dataset_cls = mock.Mock()
        mock_dataset = mock_dataset_cls.return_value
        mock_dataset.get_namespace.return_value = TEST_OL_NAMESPACE
        mock_dataset.get_name.return_value = TEST_OL_NAME_GENERIC

        with mock.patch(
            "ray.anyscale.lineage.common.dataset_naming.get_ol_dataset_naming_class",
            return_value=mock_dataset_cls,
        ):
            namespace, name = dataset_naming.resolve_ol_dataset_namespace_and_name(
                dataset_naming.OpenLineageDatasetNamingTypes.S3,
                bucket_name="test-bucket",
                object_key="/test/path",
            )

        mock_dataset_cls.assert_called_once_with(
            bucket_name="test-bucket", object_key="/test/path"
        )
        assert namespace == TEST_OL_NAMESPACE
        assert name == TEST_OL_NAME_GENERIC

    def test_resolve_with_missing_dataset_class_raises_error(self):
        with mock.patch(
            "ray.anyscale.lineage.common.dataset_naming.get_ol_dataset_naming_class",
            return_value=None,
        ), pytest.raises(AnyscaleLineageSDKError) as exc_info:
            dataset_naming.resolve_ol_dataset_namespace_and_name(
                dataset_naming.OpenLineageDatasetNamingTypes.S3, path="/test/path"
            )

        assert "Error resolving OpenLineage dataset namespace and name" in str(
            exc_info.value
        )

    @pytest.mark.parametrize("exception_type", [TypeError, ValueError])
    def test_resolve_with_exception_wrapped(self, exception_type):
        mock_dataset_cls = mock.Mock()
        mock_dataset_cls.side_effect = exception_type("Invalid arguments")

        with mock.patch(
            "ray.anyscale.lineage.common.dataset_naming.get_ol_dataset_naming_class",
            return_value=mock_dataset_cls,
        ), pytest.raises(AnyscaleLineageSDKError) as exc_info:
            dataset_naming.resolve_ol_dataset_namespace_and_name(
                dataset_naming.OpenLineageDatasetNamingTypes.S3,
                invalid_arg="value",
            )

        assert "Error resolving OpenLineage dataset namespace and name" in str(
            exc_info.value
        )


class TestParseDatasetURI:
    """Integration tests for parse_dataset_uri function."""

    @pytest.mark.parametrize(
        "uri,dataset_type,expected_keys",
        [
            (
                "awsathena://athena.us-west-2.amazonaws.com/awsdatacatalog.mydb.mytable",
                dataset_naming.OpenLineageDatasetNamingTypes.ATHENA,
                ["region_name", "catalog", "database", "table"],
            ),
            (
                "s3://my-bucket/path/to/object.parquet",
                dataset_naming.OpenLineageDatasetNamingTypes.S3,
                ["bucket_name", "object_key"],
            ),
            (
                "postgres://postgres.example.com:5432/mydb.public.users",
                dataset_naming.OpenLineageDatasetNamingTypes.POSTGRES,
                ["host", "port", "database", "schema", "table"],
            ),
            (
                "bigquery/my-project.my_dataset.my_table",
                dataset_naming.OpenLineageDatasetNamingTypes.BIG_QUERY,
                ["project_id", "dataset_name", "table_name"],
            ),
        ],
    )
    def test_parse_dataset_uri_integration(self, uri, dataset_type, expected_keys):
        result = dataset_naming.parse_dataset_uri(uri, dataset_type)

        for key in expected_keys:
            assert key in result, f"Expected key '{key}' not found in result"
            assert result[key], f"Key '{key}' has empty value"


class TestResolveDatasetNamingTypeAndAttributes:
    """Test dataset naming type and attributes resolution."""

    @pytest.mark.parametrize(
        "uri,expected_type",
        [
            ("s3://bucket/file.csv", dataset_naming.OpenLineageDatasetNamingTypes.S3),
            ("gs://bucket/file.csv", dataset_naming.OpenLineageDatasetNamingTypes.GCS),
            (
                "postgres://host:5432/db.public.table",
                dataset_naming.OpenLineageDatasetNamingTypes.POSTGRES,
            ),
            (
                "bigquery/project.dataset.table",
                dataset_naming.OpenLineageDatasetNamingTypes.BIG_QUERY,
            ),
            (
                "arn:aws:glue:us-west-2:123456789012/table/db/tbl",
                dataset_naming.OpenLineageDatasetNamingTypes.AWS_GLUE,
            ),
        ],
    )
    def test_resolve_uri_type(self, uri, expected_type):
        (
            dataset_type,
            _attributes,
        ) = dataset_naming.resolve_dataset_naming_type_and_attributes(uri)
        assert dataset_type == expected_type

    def test_resolve_unknown_defaults_to_local(self):
        (
            dataset_type,
            attributes,
        ) = dataset_naming.resolve_dataset_naming_type_and_attributes("unknown://path")
        assert (
            dataset_type
            == dataset_naming.OpenLineageDatasetNamingTypes.LOCAL_FILE_SYSTEM
        )
        assert attributes["path"] == "unknown://path"
