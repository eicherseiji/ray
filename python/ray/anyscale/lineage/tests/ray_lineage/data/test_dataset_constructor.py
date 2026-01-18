"""Tests for main dataset constructor module."""

from types import SimpleNamespace

from ray.anyscale.lineage.common.facets.dataset import FileFormats
from ray.anyscale.lineage.ray_lineage.data.dataset_constructor import main


def test_process_datasource_skips_duplicate_paths(monkeypatch):
    """Test that duplicate remote paths are deduplicated."""
    processed_paths = []

    class StubDatasource:
        # Use remote paths (s3://) which are always tracked
        _source_paths = ["s3://bucket/a", "s3://bucket/a", "s3://bucket/b"]

    monkeypatch.setattr(
        main,
        "get_file_format_datasources",
        lambda: [StubDatasource],
    )

    monkeypatch.setattr(
        main,
        "FILE_FORMATS_REGISTRY",
        {"StubDatasource": FileFormats.JSON},
    )

    def fake_process_file_format_datasource(path, file_format):
        processed_paths.append((path, file_format))
        return f"dataset:{path}"

    monkeypatch.setattr(
        main,
        "process_file_format_datasource",
        fake_process_file_format_datasource,
    )

    datasets, seen = main.process_datasource(StubDatasource(), set())

    assert datasets == ["dataset:s3://bucket/a", "dataset:s3://bucket/b"]
    assert processed_paths == [
        ("s3://bucket/a", FileFormats.JSON),
        ("s3://bucket/b", FileFormats.JSON),
    ]
    assert seen == {"s3://bucket/a", "s3://bucket/b"}


def test_construct_input_output_datasets_merges_common_facets(
    patch_facet_constructors, monkeypatch
):
    # Simplified fake operators with minimal required attributes
    class FakeOperator:
        def __init__(self, schema_names, schema_types, is_read=True):
            if is_read:
                self._datasource = SimpleNamespace()
                self._datasource_or_legacy_reader = SimpleNamespace()
            else:
                self._datasink_or_legacy_datasource = SimpleNamespace()
            self.schema_names = schema_names
            self.schema_types = schema_types

        def infer_schema(self):
            return SimpleNamespace(names=self.schema_names, types=self.schema_types)

    # Create mock operator classes for isinstance checks
    class MockReadOperator:
        pass

    class MockWriteOperator:
        pass

    monkeypatch.setattr(main, "ReadOperator", MockReadOperator)
    monkeypatch.setattr(main, "WriteOperator", MockWriteOperator)

    # Simplified fake dataset with facets
    class FakeDataset:
        def __init__(self):
            self.facets = {"existing": "facet"}

    def fake_process_datasource(datasource, seen):
        return ([FakeDataset()], seen)

    def fake_process_datasink(datasink, seen):
        return ([FakeDataset()], seen)

    monkeypatch.setattr(
        main,
        "process_datasource",
        fake_process_datasource,
    )
    monkeypatch.setattr(
        main,
        "process_datasink",
        fake_process_datasink,
    )

    # Create test operators that inherit from the mock classes
    class ReadOperatorInstance(MockReadOperator, FakeOperator):
        def __init__(self):
            FakeOperator.__init__(self, ["col"], ["string"], is_read=True)

    class WriteOperatorInstance(MockWriteOperator, FakeOperator):
        def __init__(self):
            FakeOperator.__init__(self, ["value"], ["int"], is_read=False)

    read_op = ReadOperatorInstance()
    write_op = WriteOperatorInstance()

    # Create a hashable physical operator mock
    class HashablePhysicalOp:
        def __init__(self, logical_operators):
            self._logical_operators = logical_operators

        def __hash__(self):
            return id(self)

    physical_op = HashablePhysicalOp([read_op, write_op])
    executor = SimpleNamespace(_topology={physical_op: None})

    inputs, outputs = main.construct_input_output_datasets(executor)

    assert len(inputs) == 1
    assert inputs[0].facets["schema"] == [{"name": "col", "type": "string"}]
    assert inputs[0].facets["owner"] == []

    assert len(outputs) == 1
    assert outputs[0].facets["schema"] == [{"name": "value", "type": "int"}]
    assert outputs[0].facets["owner"] == []


def test_process_datasource_handles_empty_topology():
    """Test that construct_input_output_datasets handles empty topology gracefully."""
    executor = SimpleNamespace(_topology=None)

    inputs, outputs = main.construct_input_output_datasets(executor)

    assert inputs == []
    assert outputs == []


def test_process_datasource_handles_missing_attributes():
    """Test that datasource processing handles missing attributes gracefully."""

    class IncompleteOperator:
        pass  # Missing required attributes

    # Create a hashable physical operator mock
    class HashablePhysicalOp:
        def __init__(self, logical_operators):
            self._logical_operators = logical_operators

        def __hash__(self):
            return id(self)

    physical_op = HashablePhysicalOp([IncompleteOperator()])
    executor = SimpleNamespace(_topology={physical_op: None})

    # Should not raise exceptions
    inputs, outputs = main.construct_input_output_datasets(executor)

    assert inputs == []
    assert outputs == []


def test_process_datasource_multiple_datasource_types(monkeypatch):
    """Test processing a datasource that doesn't match any known type."""

    class UnknownDatasource:
        pass

    monkeypatch.setattr(
        main,
        "get_file_format_datasources",
        lambda: [],
    )

    datasets, seen = main.process_datasource(UnknownDatasource(), set())

    assert datasets == []
    assert seen == set()


def test_process_datasink_multiple_datasink_types(monkeypatch):
    """Test processing a datasink that doesn't match any known type."""

    class UnknownDatasink:
        pass

    monkeypatch.setattr(
        main,
        "get_file_format_datasinks",
        lambda: [],
    )

    datasets, seen = main.process_datasink(UnknownDatasink(), set())

    assert datasets == []
    assert seen == set()


def test_process_datasink_prevents_duplicate_file_paths(monkeypatch):
    """Test that duplicate file datasinks are not processed twice."""
    processed_paths = []

    class StubDatasink:
        # Use remote path (s3://) which is always tracked
        unresolved_path = "s3://bucket/output.parquet"

    monkeypatch.setattr(
        main,
        "get_file_format_datasinks",
        lambda: [StubDatasink],
    )

    monkeypatch.setattr(
        main,
        "FILE_FORMATS_REGISTRY",
        {"StubDatasink": FileFormats.PARQUET},
    )

    def fake_process_file_format_datasink(path, file_format):
        processed_paths.append(path)
        return f"dataset:{path}"

    monkeypatch.setattr(
        main,
        "process_file_format_datasink",
        fake_process_file_format_datasink,
    )

    # Process the same datasink twice
    datasink = StubDatasink()

    # First call - should process
    datasets1, seen1 = main.process_datasink(datasink, set())
    assert len(datasets1) == 1
    assert len(processed_paths) == 1

    # Second call with seen URIs - should skip
    datasets2, _ = main.process_datasink(datasink, seen1)
    assert len(datasets2) == 0  # Should not add duplicate
    assert len(processed_paths) == 1  # Should not process again


def test_construct_input_output_datasets_handles_list_files_operator(
    patch_facet_constructors, monkeypatch
):
    """Test construct_input_output_datasets properly handles ListFiles operator."""

    class FakeOperator:
        def __init__(self, is_list_files=False):
            self._name = (
                main.LIST_FILES_LOGICAL_OPERATOR_NAME if is_list_files else "Other"
            )
            if is_list_files:
                self._source_paths = ["/data/file1", "/data/file2"]
                self.file_extensions = [".parquet"]
            self.schema_names = ["col"]
            self.schema_types = ["string"]

        def infer_schema(self):
            return SimpleNamespace(names=self.schema_names, types=self.schema_types)

    class FakeDataset:
        def __init__(self):
            self.facets = {"existing": "facet"}

    def fake_process_list_files_operator(operator, seen):
        return ([FakeDataset()], seen)

    monkeypatch.setattr(
        main,
        "process_list_files_operator",
        fake_process_list_files_operator,
    )

    list_files_op = FakeOperator(is_list_files=True)

    # Create a hashable physical operator mock
    class HashablePhysicalOp:
        def __init__(self, logical_operators):
            self._logical_operators = logical_operators

        def __hash__(self):
            return id(self)

    physical_op = HashablePhysicalOp([list_files_op])
    executor = SimpleNamespace(_topology={physical_op: None})

    inputs, outputs = main.construct_input_output_datasets(executor)

    assert len(inputs) == 1
    assert inputs[0].facets["schema"] == [{"name": "col", "type": "string"}]
    assert inputs[0].facets["owner"] == []
    assert inputs[0].facets["existing"] == "facet"
    assert len(outputs) == 0
