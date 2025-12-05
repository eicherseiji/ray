import pytest

from ray.anyscale.lineage.common.exceptions import AnyscaleLineageRayDataError
from ray.anyscale.lineage.ray_lineage.data import utils


def test_get_database_datasinks_contains_sql_datasink():
    datasinks = utils.get_database_datasinks()

    assert utils.Datasinks.SQL_DATASINK.value in datasinks


def test_file_extensions_registry_has_entries():
    registry = utils.FILE_EXTENSIONS_REGISTRY

    assert registry
    for extensions in registry.values():
        assert extensions is None or isinstance(extensions, (list, tuple))

    assert registry.get("CSVDatasource") is not None


def test_file_formats_registry_has_entries():
    """Test that FILE_FORMATS_REGISTRY is populated with datasource/datasink mappings."""
    from ray.anyscale.lineage.common.facets.dataset import FileFormats

    registry = utils.FILE_FORMATS_REGISTRY

    assert registry
    # Check that all values are FileFormats enum members
    for file_format in registry.values():
        assert isinstance(file_format, FileFormats)

    # Check some expected mappings
    assert registry.get("CSVDatasource") == FileFormats.CSV
    assert registry.get("ParquetDatasource") == FileFormats.PARQUET
    assert registry.get("CSVDatasink") == FileFormats.CSV
    assert registry.get("ParquetDatasink") == FileFormats.PARQUET


def test_file_formats_registry_contains_all_file_formats():
    """Test that FILE_FORMATS_REGISTRY has entries for file format datasources and datasinks."""

    registry = utils.FILE_FORMATS_REGISTRY

    # Get all datasources and datasinks
    datasources = utils.get_file_format_datasources()
    datasinks = utils.get_file_format_datasinks()

    # Check that at least some datasources and datasinks are mapped
    datasource_count = sum(1 for ds in datasources if ds.__name__ in registry)
    datasink_count = sum(1 for ds in datasinks if ds.__name__ in registry)

    assert datasource_count > 0, "Should have at least one datasource mapped"
    assert datasink_count > 0, "Should have at least one datasink mapped"


def test_build_file_extensions_registry_function():
    """Test that build_file_extensions_registry creates a proper registry."""
    registry = utils.build_file_extensions_registry()

    assert isinstance(registry, dict)
    assert len(registry) > 0

    # All values should be lists of strings
    for ds_name, extensions in registry.items():
        assert isinstance(ds_name, str)
        assert isinstance(extensions, (list, tuple))


def test_build_file_formats_registry_function():
    """Test that build_file_formats_registry creates a proper registry."""
    from ray.anyscale.lineage.common.facets.dataset import FileFormats

    registry = utils.build_file_formats_registry()

    assert isinstance(registry, dict)
    assert len(registry) > 0

    # All keys should be strings and values should be FileFormats
    for ds_name, file_format in registry.items():
        assert isinstance(ds_name, str)
        assert isinstance(file_format, FileFormats)


class TestCatchLineageCallbackException:
    def test_decorator_suppresses_error_when_ignore_errors_true(self, monkeypatch):
        monkeypatch.setattr(utils, "IGNORE_ERRORS", True)

        @utils.catch_lineage_callback_exception
        def failing_function():
            raise ValueError("test error")

        result = failing_function()
        assert result is None

    def test_decorator_raises_wrapped_error_when_ignore_errors_false(self, monkeypatch):
        monkeypatch.setattr(utils, "IGNORE_ERRORS", False)

        @utils.catch_lineage_callback_exception
        def failing_function():
            raise ValueError("test error")

        with pytest.raises(AnyscaleLineageRayDataError) as exc_info:
            failing_function()

        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "test error" in str(exc_info.value.__cause__)
