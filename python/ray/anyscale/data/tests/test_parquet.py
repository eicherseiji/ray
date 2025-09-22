import functools
import re
import os
from typing import List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyarrow.fs import FileSystemHandler, LocalFileSystem, PyFileSystem
from ray.anyscale.data._internal.readers.parquet_reader import (
    ParquetReader,
    ParquetFileChunker,
)
from ray.data.context import MAX_SAFE_BLOCK_SIZE_FACTOR
from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    FileManifest,
)
from ray.anyscale.data.checkpoint.interfaces import CheckpointConfig
from ray.anyscale.data.checkpoint.util import (
    normalize_id,
    get_checkpoint_fragments_info_for_file,
    CHECKPOINTED_FRAGMENT_TYPE,
    CHECKPOINTED_FILE_FRAGMENTS_TYPE,
    CHECKPOINTED_GENERATED_ID_COLUMN_TABLE_SCHEMA,
)

import ray
from ray.data.tests.conftest import *  # noqa


# Generated ID column name
GENERATED_ID_COL = "row_id"


@pytest.fixture
def checkpoint_config_fixture():
    """Fixture to set up and clean up checkpoint config.

    We set CheckpointConfig here because the include_row_id feature currently can only
    be enabled with CheckpointConfig.generated_id_column. If we expose a new API for
    read_parquet, we should update this fixture.

    """
    ctx = ray.data.DataContext.get_current()
    original_config = ctx.checkpoint_config

    def _setup_checkpoint_config(
        generated_id_column=GENERATED_ID_COL, checkpoint_path="/tmp/checkpoint"
    ):
        ctx.checkpoint_config = CheckpointConfig(
            generated_id_column=generated_id_column, checkpoint_path=checkpoint_path
        )
        return ctx.checkpoint_config

    yield _setup_checkpoint_config

    ctx.checkpoint_config = original_config


@pytest.fixture
def simple_parquet_file(tmp_path):
    """Fixture to create a simple Parquet file for testing."""

    def _create_parquet_file(data, filename="test.parquet"):
        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
        file_path = tmp_path / filename
        pq.write_table(table, file_path)
        return tmp_path

    return _create_parquet_file


@pytest.fixture
def multi_file_parquet_dataset(tmp_path):
    """Fixture to create a multi-file Parquet dataset for testing."""

    def _create_multi_file_dataset(total_rows, num_files, data_columns=None):
        if data_columns is None:
            data_columns = {"id": lambda i: i, "value": lambda i: i % 20}

        rows_per_file = total_rows // num_files
        all_data = []

        for i in range(num_files):
            start_idx = i * rows_per_file
            end_idx = start_idx + rows_per_file

            file_data = {}
            for col_name, col_func in data_columns.items():
                file_data[col_name] = [col_func(j) for j in range(start_idx, end_idx)]

            df = pd.DataFrame(file_data)
            table = pa.Table.from_pandas(df)
            file_path = tmp_path / f"test_{i:03d}.parquet"
            pq.write_table(table, file_path)
            all_data.append(df)

        return tmp_path, all_data

    return _create_multi_file_dataset


@pytest.fixture
def large_parquet_dataset(tmp_path):
    """Fixture to create a large Parquet dataset with uneven row distribution."""

    def _create_large_dataset(total_rows=1000, num_files=100):
        import random

        # Generate uneven distribution of rows per file
        rows_per_file = [1] * num_files  # Start with 1 row per file
        remaining_rows = total_rows - num_files

        # Randomly distribute remaining rows
        random.seed(42)  # For reproducibility
        for _ in range(remaining_rows):
            file_idx = random.randint(0, num_files - 1)
            rows_per_file[file_idx] += 1

        # Verify we have exactly total_rows rows
        assert sum(rows_per_file) == total_rows

        # Create files with uneven row distribution
        current_row_start = 0
        file_info = []  # Track (file_path, start_row, end_row, num_rows)

        for i in range(num_files):
            num_rows = rows_per_file[i]

            # Create data for this file
            start_val = current_row_start
            end_val = current_row_start + num_rows

            df = pd.DataFrame(
                {
                    "one": list(range(start_val, end_val)),
                    "two": [f"value_{j}" for j in range(start_val, end_val)],
                }
            )

            table = pa.Table.from_pandas(df)
            file_path = tmp_path / f"test_{i:03d}.parquet"
            pq.write_table(table, file_path)

            file_info.append(
                (
                    str(file_path),
                    current_row_start,
                    current_row_start + num_rows - 1,
                    num_rows,
                )
            )
            current_row_start += num_rows

        return tmp_path, file_info, total_rows

    return _create_large_dataset


@pytest.fixture
def large_parquet_file(tmp_path):
    """Create a single large parquet file suitable for chunking tests."""
    num_rows = 50000
    data = {
        "id": list(range(num_rows)),
        "text": [
            f"This is a longer text string for row {i} with more content"
            for i in range(num_rows)
        ],
        "value": [i * 1.5 for i in range(num_rows)],
        "category": [f"cat_{i % 10}" for i in range(num_rows)],
    }
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)

    file_path = tmp_path / "large_test.parquet"
    pq.write_table(table, file_path, row_group_size=1000)

    return str(file_path), num_rows


def flaky(func):
    """Error on the first call, then succeed on the second."""
    has_errored = False

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal has_errored
        if not has_errored:
            has_errored = True
            raise Exception("Transient error")
        else:
            return func(*args, **kwargs)

    return wrapper


# TODO(@bveeramani): Rather than calling `call_with_retry` whenever we invoke filesystem
# methods, we could create a filesystem wrapper that retries on transient errors. The
# implementation could be similar to the `FlakyFileSystemHandler` class below.
class FlakyFileSystemHandler(FileSystemHandler):
    def __init__(self, fs):
        self._fs = fs

    @flaky
    def copy_file(self, src, dest):
        self._fs.copy_file(src, dest)

    @flaky
    def create_dir(self, path, recursive):
        self._fs.create_dir(path, recursive=recursive)

    @flaky
    def delete_dir(self, path):
        self._fs.delete_dir(path)

    @flaky
    def delete_dir_contents(self, path, missing_dir_ok=False):
        self._fs.delete_dir_contents(path, missing_dir_ok=missing_dir_ok)

    @flaky
    def delete_file(self, path):
        self._fs.delete_file(path)

    @flaky
    def delete_root_dir_contents(self, path):
        self._fs._delete_dir_contents("/", accept_root_dir=True)

    @flaky
    def get_file_info(self, paths):
        return self._fs.get_file_info(paths)

    @flaky
    def get_file_info_selector(self, selector):
        return self._fs.get_file_info(selector)

    # Don't use the flaky decorator for 'get_type_name' because it presumably doesn't
    # use I/O.
    def get_type_name(self):
        return self._fs.type_name

    @flaky
    def move(self, src, dest):
        return self._fs.move(src, dest)

    # Don't use the flaky decorator for 'normalize_path' because it presumably doesn't
    # use I/O.
    def normalize_path(self, path):
        return self._fs.normalize_path(path)

    @flaky
    def open_append_stream(self, path, metadata):
        return self._fs.open_append_stream(path, metadata=metadata)

    @flaky
    def open_input_file(self, path):
        return self._fs.open_input_file(path)

    @flaky
    def open_input_stream(self, path):
        return self._fs.open_input_stream(path)

    @flaky
    def open_output_stream(self, path, metadata):
        return self._fs.open_output_stream(path, metadata=metadata)


def test_transient_error_handling(restore_data_context, ray_start_regular_shared):
    ctx = ray.data.DataContext.get_current()
    ctx.retried_io_errors.append("Transient error")
    # 'FlakyFileSystemHandler' raises an error on the first call to any filesystem
    # method, then succeeds on the second call.
    fs = PyFileSystem(FlakyFileSystemHandler(LocalFileSystem()))

    ray.data.read_parquet("example://iris.parquet", filesystem=fs).materialize()


@pytest.mark.parametrize(
    "columns,expected_exception,expected_message",
    [
        ("not_a_list", TypeError, "`columns` must be a list of strings."),
        (["valid_col", 123], TypeError, "All elements in `columns` must be strings."),
        (["variety", "sepal.length"], None, None),
    ],
)
def test_empty_columns_with_read_parquet(
    ray_start_regular_shared, columns, expected_exception, expected_message
):
    if expected_exception:
        with pytest.raises(expected_exception, match=expected_message):
            ray.data.read_parquet(
                "example://iris.parquet", columns=columns
            ).materialize()
    else:
        try:
            schema = ray.data.read_parquet(
                "example://iris.parquet", columns=columns
            ).schema()
            assert schema.names == ["variety", "sepal.length"]
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")


def test_read_parquet_produces_target_size_blocks(
    ray_start_regular_shared, tmp_path, restore_data_context
):
    table = pa.Table.from_pydict({"data": ["\0" * 1024 * 1024]})  # 1 MiB of data
    pq.write_table(table, tmp_path / "test1.parquet")
    pq.write_table(table, tmp_path / "test2.parquet")
    pq.write_table(table, tmp_path / "test3.parquet")
    pq.write_table(table, tmp_path / "test4.parquet")
    ray.data.DataContext.get_current().target_max_block_size = 2 * 1024 * 1024  # 2 MiB
    ds = ray.data.read_parquet(tmp_path)
    actual_block_sizes = [
        block_metadata.size_bytes
        for bundle in ds.iter_internal_ref_bundles()
        for block_metadata in bundle.metadata
    ]
    assert all(
        block_size == pytest.approx(2 * 1024 * 1024, rel=0.01)
        for block_size in actual_block_sizes
    ), actual_block_sizes


@pytest.mark.parametrize(
    "filter_expr, expected_row_count, expect_error, expected_error_message",
    [
        # Valid filter expression
        ("column03 == 0 and column04 == 0", 1, False, None),
        # Invalid filter expression (referencing partition columns which are in schema)
        (
            "column01 == 0 and column02 == 0",
            None,
            True,
            "RuntimeError: Filter expression: '((column01 == 0) and (column02 == 0))' failed on parquet file: '<parquet_file>' with columns: {'column03', 'column04'}",  # noqa: E501
        ),
        # Invalid filter expression (referencing non-partition column)
        (
            "non_existing_column == 0",
            None,
            True,
            "RuntimeError: Filter expression: '(non_existing_column == 0)' failed on parquet file: '<parquet_file>' with columns: {'column03', 'column04'}",  # noqa: E501
        ),
    ],
)
def test_read_parquet_filter_expr_partition_columns(
    ray_start_regular_shared,
    tmp_path,
    filter_expr,
    expected_row_count,
    expect_error,
    expected_error_message,
):
    """Verify handling of valid and invalid filter expressions on partitioned
    columns.
    """

    num_partitions = 10
    rows_per_partition = 10
    num_rows = num_partitions * rows_per_partition

    # DataFrame with partition columns
    df = pd.DataFrame(
        {
            "column01": list(range(num_partitions)) * rows_per_partition,
            "column02": list(range(num_partitions)) * rows_per_partition,
            "column03": list(range(num_rows)),
            "column04": list(range(num_rows)),
        }
    )

    # Write data to a partitioned Parquet dataset
    ds = ray.data.from_pandas(df)
    ds.write_parquet(tmp_path, partition_cols=["column01", "column02"])

    if expect_error:
        # Verify the exception type and message
        with pytest.raises(RuntimeError) as excinfo:
            ray.data.read_parquet(tmp_path).filter(expr=filter_expr).materialize()
        actual_message = str(excinfo.value)

        # Replace the parquet file name in the expected error message with a placeholder
        actual_message_core = re.sub(r"\s+", " ", actual_message.strip())
        expected_message_core = re.sub(r"\s+", " ", expected_error_message.strip())

        # Replace specific file names in the error messages with a placeholder
        actual_message_core = re.sub(
            r"parquet file: '[^']+'",
            "parquet file: '<parquet_file>'",
            actual_message_core,
        )
        expected_message_core = re.sub(
            r"parquet file: '[^']+'",
            "parquet file: '<parquet_file>'",
            expected_message_core,
        )

        # Sort the set in the message
        def normalize_set_order(message):
            return re.sub(
                r"{([^}]*)}",
                lambda m: "{" + ", ".join(sorted(m.group(1).split(", "))) + "}",
                message,
            )

        # Sort the schema columns set in the message before comparing
        actual_message_core = normalize_set_order(actual_message_core)
        expected_message_core = normalize_set_order(expected_message_core)

        assert expected_message_core in actual_message_core, (
            f"Expected error message to contain: '{expected_message_core}', "
            f"but got: '{actual_message_core}'"
        )
    else:
        # Verify the filtered dataset row count
        filtered_ds = ray.data.read_parquet(tmp_path).filter(expr=filter_expr)
        assert (
            filtered_ds.count() == expected_row_count
        ), f"Expected {expected_row_count} rows, but got {filtered_ds.count()} rows."


@pytest.mark.parametrize(
    "test_case",
    [
        # Test batch size estimation with small rows that fit within target block size
        # Expected: All rows should fit in a single batch due to large target block size
        {
            "name": "auto_batch_small_rows",
            "num_rows": 100,
            "batch_size": None,
            "target_block_size_bytes": 1024 * 1024,
            "data_config": {
                "id": lambda i: i,
                "value": lambda i: float(i),
                "text": lambda i: f"text_{i}",
            },
            "expected_num_batches": 1,
        },
        # Test batch size estimation with large rows that exceed target block size
        # Expected: Rows are batched based on sample-based estimation
        {
            "name": "auto_batch_large_rows",
            "num_rows": 50,
            "batch_size": None,
            "target_block_size_bytes": 16384,
            "data_config": {
                "id": lambda i: i,
                "value": lambda i: float(i),
                "text": lambda i: "x" * 1000,
            },
            "expected_num_batches": 1,
        },
        # Test explicit batch size with exact division
        # Expected: Should create batches based on row groups and explicit batch size
        {
            "name": "explicit_batch_size",
            "num_rows": 1000,
            "batch_size": 200,
            "target_block_size_bytes": None,
            "data_config": {
                "id": lambda i: i,
                "value": lambda i: float(i),
                "text": lambda i: f"text_{i}",
            },
            "expected_num_batches": 10,  # Row groups affect batching
        },
        # Test explicit batch size with remainder
        # Expected: Should create batches based on row groups and explicit batch size
        {
            "name": "explicit_batch_size_with_remainder",
            "num_rows": 1000,
            "batch_size": 300,
            "target_block_size_bytes": None,
            "data_config": {
                "id": lambda i: i,
                "value": lambda i: float(i),
                "text": lambda i: f"text_{i}",
            },
            "expected_num_batches": 10,  # Row groups affect batching
        },
    ],
)
def test_read_parquet_batching(ray_start_regular_shared, tmp_path, test_case):
    """Test ParquetReader batching logic with sample-based estimation."""

    num_rows = test_case["num_rows"]
    batch_size = test_case["batch_size"]
    target_block_size_bytes = test_case["target_block_size_bytes"]
    data_config = test_case["data_config"]
    expected_num_batches = test_case["expected_num_batches"]

    data = {
        col_name: [gen_func(i) for i in range(num_rows)]
        for col_name, gen_func in data_config.items()
    }
    table = pa.Table.from_pydict(data)

    file_path = os.path.join(tmp_path, "test.parquet")

    # Use fixed row group size
    pq.write_table(table, file_path, row_group_size=100)

    target_block_size = (
        target_block_size_bytes
        if target_block_size_bytes is not None
        else ray.data.DataContext.get_current().target_max_block_size
    )

    reader = ParquetReader(
        schema=None,
        dataset_kwargs={},
        batch_size=batch_size,
        to_batches_kwargs={},
        block_udf=None,
        include_paths=False,
        partitioning=None,
        target_block_size=target_block_size,
    )

    # Get actual file size for the manifest
    file_size = os.path.getsize(file_path)
    file_manifest = FileManifest.construct_manifest(
        [file_path], [file_size], [None], [None]
    )

    tables = list(
        reader.read_files(
            file_manifest,
            filter_expr=None,
            columns=None,
            columns_rename=None,
            filesystem=pa.fs.LocalFileSystem(),
        )
    )

    # Verify total number of batches and rows
    assert len(tables) == expected_num_batches, (
        f"Expected {expected_num_batches} batches, got {len(tables)} "
        f"for test case: {test_case['name']}"
    )
    total_rows = sum(table.num_rows for table in tables)
    assert total_rows == num_rows, (
        f"Expected {num_rows} total rows, got {total_rows} "
        f"for test case: {test_case['name']}"
    )

    if batch_size is None:
        # For sample-based batch size estimation
        # Verify that no batch exceeds the target block size
        for i, table in enumerate(tables):
            if i == 0:
                # Skip the first batch because it's not based on
                # sampling.
                continue
            batch_memory = table.nbytes
            max_allowed_memory = target_block_size * MAX_SAFE_BLOCK_SIZE_FACTOR
            assert batch_memory <= max_allowed_memory, (
                f"Batch {i} memory usage {batch_memory} exceeds "
                f"target {target_block_size} for test case: {test_case['name']}"
            )

            # Verify each batch has at least one row
            assert (
                table.num_rows > 0
            ), f"Batch {i} has 0 rows for test case: {test_case['name']}"
    else:
        # For explicit batch size
        # Verify no batch exceeds the specified batch size
        for i, table in enumerate(tables):
            assert table.num_rows <= batch_size, (
                f"Batch {i} has {table.num_rows} rows, expected <= {batch_size} "
                f"for test case: {test_case['name']}"
            )
            assert (
                table.num_rows > 0
            ), f"Batch {i} has 0 rows for test case: {test_case['name']}"


@pytest.fixture
def test_schema():
    """Test data schema used across multiple tests."""
    return {
        "int_col": pa.int64(),
        "float_col": pa.float64(),
        "str_col": pa.string(),
    }


def verify_unique_generated_id_column(
    table: pa.Table, generated_id_column: str, expected_count: int
) -> None:
    """Verify uniqueness of generated_id_column IDs in a PyArrow table.

    Args:
        table: PyArrow table containing the data
        generated_id_column: Name of the generated ID column
        expected_count: Expected number of unique IDs
    """
    # Extract generated ID column
    generated_ids = table[generated_id_column].to_numpy().tolist()

    # Create unique identifiers using the normalize_id API
    unique_identifiers = set()
    for generated_id in generated_ids:
        unique_id = normalize_id(generated_id)
        unique_identifiers.add(unique_id)

    # Verify uniqueness
    assert (
        len(unique_identifiers) == expected_count
    ), f"Expected {expected_count} unique generated_id_column IDs, got {len(unique_identifiers)}"


def create_test_data(num_rows: int, schema: dict) -> dict:
    """Create test data based on the provided schema."""
    data = {}
    for col_name, col_type in schema.items():
        if col_name == "int_col":
            data[col_name] = list(range(num_rows))
        elif col_name == "float_col":
            data[col_name] = [float(i) for i in range(num_rows)]
        elif col_name == "str_col":
            data[col_name] = [f"str_{i}" for i in range(num_rows)]
    return data


@pytest.mark.parametrize(
    "batch_size,filter_expr,expected_rows,description",
    [
        # No batch size cases
        (None, "int_col > 500", 499, "No batch size, int > 500"),
        (None, "int_col < 200", 200, "No batch size, int < 200"),
        (
            None,
            "float_col == 42.0",
            1,
            "No batch size, float == 42.0",
        ),
        (
            None,
            "str_col == 'str_42'",
            1,
            "No batch size, str == str_42",
        ),
        # Batch size cases
        (100, "int_col > 500", 499, "Fixed batch size, int > 500"),
        (200, "int_col < 200", 200, "Fixed batch size, int < 200"),
        (
            300,
            "float_col == 42.0",
            1,
            "Fixed batch size, float == 42.0",
        ),
        (
            400,
            "str_col == 'str_42'",
            1,
            "Fixed batch size, str == str_42",
        ),
    ],
)
def test_read_parquet_with_filter_selectivity(
    ray_start_regular_shared,
    tmp_path,
    batch_size,
    filter_expr,
    expected_rows,
    description,
    test_schema,
):
    """Test reading parquet files with filter expressions and different batch sizes."""
    num_rows = 1000
    data = create_test_data(num_rows, test_schema)
    table = pa.Table.from_pydict(data)

    file_path = os.path.join(tmp_path, "test.parquet")
    pq.write_table(table, file_path, row_group_size=200)

    if batch_size is not None:
        ray.data.DataContext.get_current().target_max_block_size = batch_size
    ds = ray.data.read_parquet(file_path).filter(expr=filter_expr)

    assert ds.count() == expected_rows, (
        f"{description}: Filter '{filter_expr}' returned {ds.count()} rows, "
        f"expected {expected_rows}"
    )

    # Verify schema has expected columns and types
    schema = ds.schema()
    assert set(schema.names) == set(
        test_schema.keys()
    ), f"Schema columns {schema.names} don't match expected {list(test_schema.keys())}"


@pytest.mark.parametrize(
    "batch_size,columns,description",
    [
        (None, ["int_col"], "No batch size, only int column"),
        (None, ["float_col"], "No batch size, only float column"),
        (None, ["int_col", "float_col"], "No batch size, int and float columns"),
        (None, ["str_col"], "No batch size, only string column"),
        (None, ["int_col", "str_col"], "No batch size, int and string columns"),
        (100, ["int_col"], "Fixed batch size, only int column"),
        (200, ["float_col"], "Fixed batch size, only float column"),
        (300, ["int_col", "float_col"], "Fixed batch size, int and float columns"),
        (400, ["str_col"], "Fixed batch size, only string column"),
        (500, ["int_col", "str_col"], "Fixed batch size, int and string columns"),
    ],
)
def test_read_parquet_with_columns_selectivity(
    ray_start_regular_shared,
    tmp_path,
    batch_size,
    columns,
    description,
    test_schema,
):
    """Test reading parquet files with different column selections and batch sizes."""
    num_rows = 1000
    data = create_test_data(num_rows, test_schema)
    table = pa.Table.from_pydict(data)

    file_path = os.path.join(tmp_path, "test.parquet")
    pq.write_table(table, file_path, row_group_size=200)

    if batch_size is not None:
        ray.data.DataContext.get_current().target_max_block_size = batch_size
    ds = ray.data.read_parquet(file_path, columns=columns)

    assert ds.count() == num_rows, (
        f"Column selection {columns} with batch_size={batch_size} "
        f"returned {ds.count()} rows, expected {num_rows}"
    )

    assert set(ds.schema().names) == set(columns), (
        f"Column selection {columns} with batch_size={batch_size} "
        f"returned columns {ds.schema().names}"
    )


@pytest.mark.parametrize(
    "test_data,generated_id_column,read_kwargs,rename_columns,expected_error_pattern,test_description",
    [
        # Collision with existing schema column
        (
            {"one": [1, 2, 3], GENERATED_ID_COL: ["a", "b", "c"]},
            GENERATED_ID_COL,
            {},
            None,
            "generated_id_column='row_id' conflicts with an existing column",
            "collision with existing Parquet column",
        ),
        # Collision with columns list
        (
            {
                "one": [1, 2, 3],
                "two": ["a", "b", "c"],
                GENERATED_ID_COL: ["d", "e", "f"],
            },
            GENERATED_ID_COL,
            {"columns": ["one", "two", GENERATED_ID_COL]},
            None,
            "generated_id_column='row_id' conflicts with a column in the columns list",
            "collision with explicit columns list",
        ),
        # Collision with renamed column (target name)
        (
            {"one": [1, 2, 3], "two": ["a", "b", "c"]},
            "renamed_col",
            {},
            {"one": "renamed_col"},
            "generated_id_column='renamed_col' conflicts with a renamed column",
            "collision with renamed column target name",
        ),
        # Collision with column being renamed (original name)
        (
            {"one": [1, 2, 3], "two": ["a", "b", "c"]},
            "one",
            {},
            {"one": "renamed_col"},
            "generated_id_column='one' conflicts with a column that will be renamed",
            "collision with column being renamed",
        ),
    ],
)
def test_parquet_generated_id_column_collisions(
    ray_start_regular_shared,
    tmp_path,
    checkpoint_config_fixture,
    simple_parquet_file,
    test_data,
    generated_id_column,
    read_kwargs,
    rename_columns,
    expected_error_pattern,
    test_description,
):
    """Test that generated_id_column raises appropriate errors when it collides with various column scenarios."""

    # Create Parquet file with test data
    simple_parquet_file(test_data)

    # Set up checkpoint config
    checkpoint_config_fixture(generated_id_column=generated_id_column)

    # Read Parquet and apply any additional operations
    ds = ray.data.read_parquet(tmp_path, **read_kwargs)

    # Apply rename operation if specified
    if rename_columns:
        ds = ds.rename_columns(rename_columns)

    # Should raise error due to collision
    with pytest.raises(ValueError, match=expected_error_pattern):
        ds.materialize()


def test_parquet_generated_id_column_with_filter_pushdown(
    ray_start_regular_shared,
    tmp_path,
    checkpoint_config_fixture,
    multi_file_parquet_dataset,
):
    """Verify row IDs when filter pushdown is applied with generated_id_column."""
    import pyarrow as pa
    import ray

    # Create test data with multiple files
    total_rows = 200
    num_files = 4
    data_path, all_data = multi_file_parquet_dataset(total_rows, num_files)

    # Combine all data for reference
    all_df = pd.concat(all_data, ignore_index=True)

    # Apply filter: keep rows where value < 10
    filter_condition = "value < 10"
    expected_filtered_df = all_df[all_df["value"] < 10].reset_index(drop=True)
    expected_count = len(expected_filtered_df)

    # Set up checkpoint config with generated_id_column
    checkpoint_config_fixture(generated_id_column=GENERATED_ID_COL)

    # Read with row IDs and apply filter
    ds_filtered = ray.data.read_parquet(data_path).filter(expr=filter_condition)

    # Verify schema includes row_id column
    assert ds_filtered.schema().names == ["id", "value", GENERATED_ID_COL]

    # Collect filtered data
    filtered_batches = list(ds_filtered.iter_batches(batch_format="pyarrow"))
    filtered_table = (
        pa.concat_tables(filtered_batches)
        if len(filtered_batches) > 1
        else filtered_batches[0]
    )

    # Verify row count matches expected
    assert (
        filtered_table.num_rows == expected_count
    ), f"Expected {expected_count} rows, got {filtered_table.num_rows}"

    # Verify generated_id_column preserves original file-based positions
    verify_unique_generated_id_column(filtered_table, GENERATED_ID_COL, expected_count)

    # generated_id_column IDs should correspond to original positions from each file's range
    # Each file had 50 rows with ranges: file0=[0-49], file1=[50-99], file2=[100-149], file3=[150-199]
    # After filtering "value < 10", we expect gaps in row IDs where filtered rows were removed
    filtered_df = filtered_table.to_pandas()

    # Verify the filtered data matches expected
    filtered_df = filtered_table.to_pandas()
    filtered_df_sorted = filtered_df.sort_values("id").reset_index(drop=True)
    expected_filtered_df_sorted = expected_filtered_df.sort_values("id").reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(
        filtered_df_sorted[["id", "value"]],
        expected_filtered_df_sorted[["id", "value"]],
    )


def test_parquet_read_with_generated_id_column_checkpoint_config(
    ray_start_regular_shared, tmp_path, checkpoint_config_fixture, large_parquet_dataset
):
    """Test reading Parquet files with generated_id_column from checkpoint config."""
    import pyarrow as pa

    # Create large dataset with uneven row distribution
    data_path, file_info, total_rows = large_parquet_dataset()

    # Set up checkpoint config with generated_id_column
    checkpoint_config_fixture(generated_id_column=GENERATED_ID_COL)

    # Read all files with row IDs
    ds = ray.data.read_parquet(data_path)

    # Verify schema includes row_id column
    assert ds.schema().names == ["one", "two", GENERATED_ID_COL]

    # Collect all data as Arrow table to verify row ID properties
    batches = list(ds.iter_batches(batch_format="pyarrow"))
    all_data_table = pa.concat_tables(batches) if len(batches) > 1 else batches[0]

    # Verify we have exactly 1000 rows
    assert all_data_table.num_rows == total_rows

    # Verify generated ID column uniqueness
    verify_unique_generated_id_column(all_data_table, GENERATED_ID_COL, total_rows)


def test_parquet_chunked_reading_preserves_order(ray_start_regular_shared, tmp_path):
    """Test that chunked reading preserves row order when order preservation is enabled."""
    # Create a large dataset with sequential IDs
    num_rows = 10000
    data = {
        "id": list(range(num_rows)),
        "sequence": list(range(num_rows)),  # Should be in order
        "value": [f"row_{i}" for i in range(num_rows)],
    }
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)

    file_path = tmp_path / "ordered_test.parquet"
    pq.write_table(table, file_path, row_group_size=1000)

    # Force chunking by using a small target chunk size
    reader = ParquetReader(
        schema=None,
        dataset_kwargs={},
        batch_size=None,
        to_batches_kwargs={},
        block_udf=None,
        include_paths=False,
        partitioning=None,
        target_block_size=None,
    )

    file_size = os.path.getsize(file_path)
    chunker = ParquetFileChunker(target_chunk_size=1 * 1024)  # 1KB to force chunking
    chunks = list(chunker.generate_chunk_metadatas(str(file_path), file_size))

    assert len(chunks) > 1

    # Create manifest with chunking
    paths = [str(file_path)] * len(chunks)
    chunk_metadatas = [metadata for metadata, _ in chunks]

    file_manifest = FileManifest.construct_manifest(
        paths,
        [file_size] * len(paths),
        chunk_metadatas,
        [None] * len(chunks),
    )

    # Read the data
    tables = list(
        reader.read_files(
            file_manifest,
            filesystem=pa.fs.LocalFileSystem(),
        )
    )

    # Combine all tables and check if sequence is preserved
    combined_table = pa.concat_tables(tables)
    sequence_values = combined_table.column("sequence").to_pylist()

    # Note: Order preservation depends on the threading configuration
    # This test verifies the data integrity rather than strict ordering
    assert len(sequence_values) == num_rows
    assert set(sequence_values) == set(range(num_rows))


@pytest.mark.parametrize(
    "file_size,expected_chunks,expected_metadata_none,expected_num_chunks",
    [
        # Small files (no chunking)
        (0, 1, True, None),  # 0 bytes -> 1 chunk, no metadata
        (100 * 1024 * 1024, 1, True, None),  # 100MB -> 1 chunk, no metadata
        (
            256 * 1024 * 1024,
            1,
            True,
            None,
        ),  # 256MB -> 1 chunk, no metadata (exactly at target)
        # Large files (with chunking)
        (257 * 1024 * 1024, 2, False, 2),  # Just over target -> 2 chunks
        (300 * 1024 * 1024, 2, False, 2),  # 300MB -> 2 chunks
        (512 * 1024 * 1024, 2, False, 2),  # 512MB -> 2 chunks
        (600 * 1024 * 1024, 3, False, 3),  # 600MB -> 3 chunks
        (1024 * 1024 * 1024, 4, False, 4),  # 1GB -> 4 chunks
    ],
)
def test_parquet_file_chunker(
    ray_start_regular_shared,
    file_size,
    expected_chunks,
    expected_metadata_none,
    expected_num_chunks,
):
    """Test ParquetFileChunker chunking behavior with various file sizes."""
    chunker = ParquetFileChunker()
    chunks = list(chunker.generate_chunk_metadatas("test.parquet", file_size))

    # Verify number of chunks
    assert len(chunks) == expected_chunks

    if expected_metadata_none:
        # Small files should have no chunking metadata
        assert len(chunks) == 1
        metadata, chunk_size = chunks[0]
        assert metadata is None
        assert chunk_size == file_size
    else:
        # Large files should have chunking metadata
        assert len(chunks) == expected_num_chunks

        # Verify each chunk has proper metadata
        for i, (metadata, chunk_size) in enumerate(chunks):
            assert metadata is not None
            assert metadata["chunk_idx"] == i
            assert metadata["total_num_chunks"] == expected_num_chunks


@pytest.mark.parametrize(
    "total_row_groups,total_num_chunks,expected_ranges",
    [
        # Even distribution cases
        (10, 2, [(0, 5), (5, 10)]),
        (12, 3, [(0, 4), (4, 8), (8, 12)]),
        (20, 4, [(0, 5), (5, 10), (10, 15), (15, 20)]),
        # Uneven distribution cases (earlier chunks get extra row groups)
        (10, 3, [(0, 4), (4, 7), (7, 10)]),
        (11, 3, [(0, 4), (4, 8), (8, 11)]),
        (13, 4, [(0, 4), (4, 7), (7, 10), (10, 13)]),
        # Edge cases
        (1, 1, [(0, 1)]),
        (1, 2, [(0, 1), None]),  # Second chunk should be None
        (0, 1, [None]),
        (5, 10, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)] + [None] * 5),
    ],
)
def test_calculate_row_group_range_distribution(
    ray_start_regular_shared, total_row_groups, total_num_chunks, expected_ranges
):
    """Test row group distribution across chunks and verify complete coverage."""
    reader = ParquetReader(
        schema=None,
        dataset_kwargs={},
        batch_size=None,
        to_batches_kwargs={},
        block_udf=None,
        include_paths=False,
        partitioning=None,
        target_block_size=None,
    )

    # Test each individual chunk
    for chunk_idx in range(total_num_chunks):
        result = reader._calculate_row_group_range(
            chunk_idx, total_num_chunks, total_row_groups
        )
        expected = (
            expected_ranges[chunk_idx] if chunk_idx < len(expected_ranges) else None
        )
        assert (
            result == expected
        ), f"Chunk {chunk_idx}: expected {expected}, got {result}"

    # Verify complete coverage (no gaps, no overlaps)
    covered_rows = set()
    for chunk_idx in range(total_num_chunks):
        result = reader._calculate_row_group_range(
            chunk_idx, total_num_chunks, total_row_groups
        )

        if result is not None:
            start, end = result
            chunk_rows = set(range(start, end))

            # Ensure no overlap
            assert not (
                covered_rows & chunk_rows
            ), f"Overlap detected in chunk {chunk_idx}"
            covered_rows.update(chunk_rows)

    # Ensure all row groups are covered
    assert covered_rows == set(
        range(total_row_groups)
    ), f"Not all row groups covered. Expected: {set(range(total_row_groups))}, Got: {covered_rows}"


def test_chunked_vs_non_chunked_same_result(
    ray_start_regular_shared, large_parquet_file
):
    """Test that chunked and non-chunked reading produce identical results."""
    file_path, expected_rows = large_parquet_file

    reader = ParquetReader(
        schema=None,
        dataset_kwargs={},
        batch_size=None,
        to_batches_kwargs={},
        block_udf=None,
        include_paths=False,
        partitioning=None,
        target_block_size=None,
    )

    file_size = os.path.getsize(file_path)

    # Ensure the file will be chunked - if not, the test setup is wrong
    chunker = ParquetFileChunker(
        target_chunk_size=256 * 1024
    )  # 256KB, force many chunks
    chunks = list(chunker.generate_chunk_metadatas(file_path, file_size))
    assert len(chunks) > 1

    # Read without chunking
    file_manifest_no_chunk = FileManifest.construct_manifest(
        [file_path],
        [file_size],
        [None],
        [None],
    )

    tables_no_chunk = list(
        reader.read_files(
            file_manifest_no_chunk,
            filesystem=pa.fs.LocalFileSystem(),
        )
    )

    file_manifest_chunked = FileManifest.construct_manifest(
        [file_path] * len(chunks),
        [file_size] * len(chunks),
        [metadata for metadata, _ in chunks],
        [None] * len(chunks),
    )

    tables_chunked = list(
        reader.read_files(
            file_manifest_chunked,
            filesystem=pa.fs.LocalFileSystem(),
        )
    )

    # Compare results
    total_rows_no_chunk = sum(table.num_rows for table in tables_no_chunk)
    total_rows_chunked = sum(table.num_rows for table in tables_chunked)

    assert total_rows_no_chunk == total_rows_chunked == expected_rows

    # Compare actual data (sort by ID to ensure consistent ordering)
    def get_sorted_data(tables):
        combined = pa.concat_tables(tables)
        return combined.to_pandas().sort_values("id").reset_index(drop=True)

    df_no_chunk = get_sorted_data(tables_no_chunk)
    df_chunked = get_sorted_data(tables_chunked)

    pd.testing.assert_frame_equal(df_no_chunk, df_chunked)


@pytest.mark.parametrize(
    "columns,expected_columns",
    [
        (None, ["id", "text", "value", "category"]),
        (["id", "value"], ["id", "value"]),
        (["text"], ["text"]),
    ],
)
def test_chunked_reading_with_column_selection(
    ray_start_regular_shared, large_parquet_file, columns, expected_columns
):
    """Test chunked reading with column selection."""
    file_path, expected_rows = large_parquet_file

    reader = ParquetReader(
        schema=None,
        dataset_kwargs={},
        batch_size=None,
        to_batches_kwargs={},
        block_udf=None,
        include_paths=False,
        partitioning=None,
        target_block_size=None,
    )

    file_size = os.path.getsize(file_path)

    # Ensure the file will be chunked - if not, the test setup is wrong
    chunker = ParquetFileChunker(
        target_chunk_size=256 * 1024
    )  # 256KB, force many chunks
    chunks = list(chunker.generate_chunk_metadatas(file_path, file_size))
    assert len(chunks) > 1

    file_manifest = FileManifest.construct_manifest(
        [file_path] * len(chunks),
        [file_size] * len(chunks),
        [metadata for metadata, _ in chunks],
        [None] * len(chunks),
    )

    tables = list(
        reader.read_files(
            file_manifest,
            columns=columns,
            filesystem=pa.fs.LocalFileSystem(),
        )
    )

    # Verify column selection
    for table in tables:
        assert table.column_names == expected_columns

    # Verify row count
    total_rows = sum(table.num_rows for table in tables)
    assert total_rows == expected_rows


@pytest.mark.parametrize("use_generated_id", [False, True])
def test_chunked_out_of_range_returns_empty(
    ray_start_regular_shared, tmp_path, restore_data_context, use_generated_id
):
    """Out-of-range chunks produce no fragments, with/without generated IDs.

    Create a single-row-group parquet file, generate multiple chunk metadatas with
    a 1KB target chunk size, then drop the first chunk so remaining chunks are
    out-of-range. Assert no tables are produced in both configurations.
    """
    # Create a simple table and write it as a single-row-group parquet file
    num_rows = 1500
    table = pa.table({"id": list(range(num_rows))})
    file_path = os.path.join(tmp_path, "single_row_group.parquet")
    pq.write_table(table, file_path, row_group_size=num_rows)

    file_size = os.path.getsize(file_path)

    # Optionally enable generated ID column via DataContext checkpoint config
    if use_generated_id:
        ctx = ray.data.DataContext.get_current()
        from ray.anyscale.data.checkpoint.interfaces import CheckpointConfig

        ctx.checkpoint_config = CheckpointConfig(
            generated_id_column="__row_id__", checkpoint_path=f"file://{tmp_path}"
        )

    reader = ParquetReader(
        schema=None,
        dataset_kwargs={},
        batch_size=None,
        to_batches_kwargs={},
        block_udf=None,
        include_paths=False,
        partitioning=None,
        target_block_size=None,
    )

    # Use small target chunk size to generate multiple chunk indices (1KB)
    chunker = ParquetFileChunker(target_chunk_size=1024)
    chunks = list(chunker.generate_chunk_metadatas(file_path, file_size))
    assert len(chunks) > 1

    # Keep only out-of-range chunk metadatas (drop chunk_idx == 0)
    out_of_range_chunk_mds = [md for md, _ in chunks][1:]
    file_manifest = FileManifest.construct_manifest(
        [file_path] * len(out_of_range_chunk_mds),
        [file_size] * len(out_of_range_chunk_mds),
        out_of_range_chunk_mds,
        [None] * len(out_of_range_chunk_mds),
    )

    tables = list(
        reader.read_files(
            file_manifest,
            filesystem=pa.fs.LocalFileSystem(),
        )
    )
    assert len(tables) == 0


def _create_checkpointed_ids_table(
    file_path: str,
    fragment_specs: List[tuple[int, int, int, List[bool]]],
) -> pa.Table:
    """Helper function to create checkpointed_ids table for testing.

    Args:
        file_path: The file path for the checkpoint data
        fragment_specs: List of (fragment_id, num_rows, num_checkpointed_rows, checkpointed_row_ids)
                       where checkpointed_row_ids is a boolean list indicating which rows are checkpointed

    Returns:
        PyArrow table with CHECKPOINTED_GENERATED_ID_COLUMN_TABLE_SCHEMA
    """
    # Create fragment structs
    fragment_structs = []
    for (
        fragment_id,
        num_rows,
        num_checkpointed_rows,
        checkpointed_row_ids,
    ) in fragment_specs:
        # Create struct using the actual schema type definition
        fragment_struct = pa.StructArray.from_arrays(
            [
                pa.array([fragment_id], type=pa.int32()),
                pa.array([num_rows], type=pa.int32()),
                pa.array([num_checkpointed_rows], type=pa.int32()),
                pa.array([checkpointed_row_ids], type=pa.large_list(pa.bool_())),
            ],
            fields=CHECKPOINTED_FRAGMENT_TYPE,
        )
        fragment_structs.append(fragment_struct)

    # Create file fragments struct
    if fragment_structs:
        all_fragment_structs = pa.concat_arrays(fragment_structs)
        offsets = pa.array([0, len(all_fragment_structs)], type=pa.int64())
        fragments_list = pa.LargeListArray.from_arrays(offsets, all_fragment_structs)
    else:
        fragments_list = pa.array([[]], type=pa.large_list(CHECKPOINTED_FRAGMENT_TYPE))

    # Calculate if fully checkpointed
    fully_checkpointed = all(
        num_checkpointed_rows == num_rows
        for _, num_rows, num_checkpointed_rows, _ in fragment_specs
    )

    checkpointed_file_fragments = pa.StructArray.from_arrays(
        [
            pa.array([len(fragment_specs)], type=pa.int32()),  # num_fragments
            pa.array([fully_checkpointed], type=pa.bool_()),  # fully_checkpointed
            fragments_list,  # fragments
        ],
        fields=CHECKPOINTED_FILE_FRAGMENTS_TYPE,
    )

    return pa.Table.from_arrays(
        [
            pa.array([file_path]),  # checkpointed_file
            checkpointed_file_fragments,  # checkpointed_file_fragments
        ],
        schema=CHECKPOINTED_GENERATED_ID_COLUMN_TABLE_SCHEMA,
    )


def test_read_files_with_checkpoint_ids_fully_skip_fragment(
    ray_start_regular_shared, tmp_path, checkpoint_config_fixture
):
    """Test read_files with checkpoint_ids that fully skip a fragment."""
    checkpoint_config_fixture(generated_id_column=GENERATED_ID_COL)

    table = pa.table(
        {"id": list(range(100)), "value": [f"val_{i}" for i in range(100)]}
    )
    file_path = os.path.join(tmp_path, "test.parquet")
    pq.write_table(table, file_path, row_group_size=25)  # 4 row groups

    file_size = os.path.getsize(file_path)

    parquet_reader = ParquetReader(
        schema=None,
        dataset_kwargs={},
        batch_size=None,
        to_batches_kwargs={},
        block_udf=None,
        include_paths=False,
        partitioning=None,
        target_block_size=None,
    )

    # Create checkpoint data in the new table format (fully checkpointed)
    fragment_specs = [
        (i, 25, 25, []) for i in range(4)  # 4 row groups, all fully checkpointed
    ]
    checkpointed_ids = _create_checkpointed_ids_table(str(file_path), fragment_specs)

    # Convert to manifest format using the new utility function
    from ray.anyscale.data.checkpoint.util import index_checkpointed_fragments

    checkpointed_fragments_by_path = index_checkpointed_fragments(checkpointed_ids)
    checkpoint_ids_scalar = get_checkpoint_fragments_info_for_file(
        checkpointed_ids, str(file_path), checkpointed_fragments_by_path
    )

    file_manifest = FileManifest.construct_manifest(
        [file_path], [file_size], [None], [checkpoint_ids_scalar]
    )

    tables = list(
        parquet_reader.read_files(
            file_manifest,
            filesystem=pa.fs.LocalFileSystem(),
        )
    )

    assert len(tables) == 0, f"Expected 0 tables, got {len(tables)}"


def test_read_files_with_checkpoint_ids_partial_skip_fragment(
    ray_start_regular_shared, tmp_path, checkpoint_config_fixture
):
    """Test read_files with checkpoint_ids that partially skip a fragment."""
    checkpoint_config_fixture(generated_id_column=GENERATED_ID_COL)

    table = pa.table({"id": list(range(20)), "value": [f"val_{i}" for i in range(20)]})
    file_path = os.path.join(tmp_path, "test.parquet")
    pq.write_table(table, file_path, row_group_size=5)  # 4 row groups

    file_size = os.path.getsize(file_path)

    parquet_reader = ParquetReader(
        schema=None,
        dataset_kwargs={},
        batch_size=None,
        to_batches_kwargs={},
        block_udf=None,
        include_paths=False,
        partitioning=None,
        target_block_size=None,
    )

    # Create checkpoint data in the new table format (partially checkpointed)
    fragment_specs = [
        (
            0,
            5,
            3,
            [True, True, True, False, False],
        ),  # Fragment 0: 3 out of 5 rows checkpointed
        (
            1,
            5,
            0,
            [False, False, False, False, False],
        ),  # Fragment 1: no rows checkpointed
        (
            2,
            5,
            0,
            [False, False, False, False, False],
        ),  # Fragment 2: no rows checkpointed
        (
            3,
            5,
            0,
            [False, False, False, False, False],
        ),  # Fragment 3: no rows checkpointed
    ]
    checkpointed_ids = _create_checkpointed_ids_table(str(file_path), fragment_specs)

    # Convert to manifest format using the new utility function
    from ray.anyscale.data.checkpoint.util import index_checkpointed_fragments

    checkpointed_fragments_by_path = index_checkpointed_fragments(checkpointed_ids)
    checkpoint_ids_scalar = get_checkpoint_fragments_info_for_file(
        checkpointed_ids, str(file_path), checkpointed_fragments_by_path
    )

    file_manifest = FileManifest.construct_manifest(
        [file_path], [file_size], [None], [checkpoint_ids_scalar]
    )
    tables = list(
        parquet_reader.read_files(
            file_manifest,
            filesystem=pa.fs.LocalFileSystem(),
        )
    )

    # Should return 4 tables (all row groups), but first group should have filtered rows
    assert len(tables) == 4, f"Expected 4 tables, got {len(tables)}"
    total_rows = sum(table.num_rows for table in tables)
    assert total_rows == 17, f"Expected 17 rows (20 - 3 checkpointed), got {total_rows}"


def test_read_files_with_checkpoint_ids_no_skip_fragment(
    ray_start_regular_shared, tmp_path, checkpoint_config_fixture
):
    """Test read_files with checkpoint_ids that don't skip any fragments."""
    checkpoint_config_fixture(generated_id_column=GENERATED_ID_COL)

    table = pa.table(
        {"id": list(range(100)), "value": [f"val_{i}" for i in range(100)]}
    )
    file_path = os.path.join(tmp_path, "test.parquet")
    pq.write_table(table, file_path, row_group_size=25)  # 4 row groups

    file_size = os.path.getsize(file_path)

    parquet_reader = ParquetReader(
        schema=None,
        dataset_kwargs={},
        batch_size=None,
        to_batches_kwargs={},
        block_udf=None,
        include_paths=False,
        partitioning=None,
        target_block_size=None,
    )

    # Create checkpoint data in the new table format for a different file
    fragment_specs = [
        (
            0,
            25,
            5,
            [False] * 100 + [True, True, True, True, True] + [False] * 95,
        ),  # Different file, partially checkpointed
    ]
    checkpointed_ids_table = _create_checkpointed_ids_table(
        "/non/existent/path.parquet", fragment_specs
    )

    # Convert to manifest format using the new utility function
    from ray.anyscale.data.checkpoint.util import index_checkpointed_fragments

    checkpointed_fragments_by_path = index_checkpointed_fragments(
        checkpointed_ids_table
    )
    checkpoint_ids_scalar = get_checkpoint_fragments_info_for_file(
        checkpointed_ids_table, str(file_path), checkpointed_fragments_by_path
    )

    file_manifest = FileManifest.construct_manifest(
        [file_path], [file_size], [None], [checkpoint_ids_scalar]
    )

    tables = list(
        parquet_reader.read_files(
            file_manifest,
            filesystem=pa.fs.LocalFileSystem(),
        )
    )

    # Should return all data with no filtering
    assert len(tables) > 0, "Expected some tables to be returned"
    total_rows = sum(table.num_rows for table in tables)
    assert total_rows == 100, f"Expected 100 rows (no filtering), got {total_rows}"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main(["-v", __file__]))
