import functools
import re
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyarrow.fs import FileSystemHandler, LocalFileSystem, PyFileSystem
from ray.anyscale.data._internal.readers.parquet_reader import ParquetReader
from ray.data.context import MAX_SAFE_BLOCK_SIZE_FACTOR
from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    FileManifest,
)

import ray
from ray.data.tests.conftest import *  # noqa


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


def test_proper_projection_for_partitioned_datasets(temp_dir):
    ds = ray.data.read_parquet("example://iris.parquet").materialize()

    partitioned_ds_path = f"{temp_dir}/partitioned_iris"
    # Write out partitioned dataset
    ds.write_parquet(partitioned_ds_path, partition_cols=["variety"])

    partitioned_ds = ray.data.read_parquet(
        partitioned_ds_path, columns=["variety"]
    ).materialize()

    print(partitioned_ds.schema())

    assert [
        "sepal.length",
        "sepal.width",
        "petal.length",
        "petal.width",
        "variety",
    ] == ds.take_batch(batch_format="pyarrow").column_names
    assert ["variety"] == partitioned_ds.take_batch(batch_format="pyarrow").column_names

    assert ds.count() == partitioned_ds.count()


@pytest.mark.parametrize(
    "columns,expected_exception,expected_message",
    [
        ([], ValueError, "`columns` cannot be an empty list."),
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
    "num_rows,batch_size,row_group_size,target_block_size_bytes,data_config,expected_num_batches",
    [
        # Default batch size (None) - rowgroup-wise reading
        # Large row groups (row_group_size_bytes > MAX_SAFE_BLOCK_SIZE_FACTOR * target_block_size_bytes)
        (
            1000,
            None,  # Use automatic batch sizing
            1000,
            32 * 1024,  # 32KB target block size
            {
                "int_col": lambda i: i,
                "float_col": lambda i: float(i),
                "str_col": lambda i: "x" * 100000,
            },
            4,  # Expect 4 batches
        ),
        # Small row groups (row_group_size_bytes <= MAX_SAFE_BLOCK_SIZE_FACTOR * target_block_size_bytes)
        (
            1000,
            None,  # Use automatic batch sizing
            100,
            1024 * 1024,  # 1MB target block size
            {
                "int_col": lambda i: i,
                "float_col": lambda i: float(i),
            },
            10,  # 10 batches for 1000 total rows
        ),
        # Explicit batch size - batch-wise reading
        # Batch size aligned with row group size
        (
            1000,
            200,
            200,
            None,  # Not used when batch_size is specified
            {
                "int_col": lambda i: i,
                "float_col": lambda i: float(i),
                "str_col": lambda i: f"str_{i}",
            },
            5,  # 5 row groups of 200 rows, 5 batches of 200 rows
        ),
        (
            1000,
            100,
            200,
            None,  # Not used when batch_size is specified
            {
                "int_col": lambda i: i,
                "float_col": lambda i: float(i),
                "str_col": lambda i: f"str_{i}",
            },
            10,  # 5 row groups of 200 rows, 10 batches of 100 rows
        ),
        # Batch size not aligned with row group size
        (
            1000,
            80,
            200,
            None,  # Not used when batch_size is specified
            {
                "int_col": lambda i: i,
                "float_col": lambda i: float(i),
                "str_col": lambda i: f"str_{i}",
            },
            15,  # 5 row groups of 200 rows, 15 batches of 80 rows
        ),
        (
            1000,
            60,
            200,
            None,  # Not used when batch_size is specified
            {
                "int_col": lambda i: i,
                "float_col": lambda i: float(i),
                "str_col": lambda i: f"str_{i}",
            },
            20,  # 5 row groups of 200 rows, 20 batches of 60 rows
        ),
    ],
)
def test_read_parquet_batching(
    ray_start_regular_shared,
    tmp_path,
    num_rows,
    batch_size,
    row_group_size,
    target_block_size_bytes,
    data_config,
    expected_num_batches,
):
    """Test ParquetReader with both default batch size (None) and explicit batch sizes."""
    data = {
        col_name: [gen_func(i) for i in range(num_rows)]
        for col_name, gen_func in data_config.items()
    }
    table = pa.Table.from_pydict(data)

    file_path = os.path.join(tmp_path, "test.parquet")
    pq.write_table(table, file_path, row_group_size=row_group_size)

    target_block_size = (
        target_block_size_bytes
        if target_block_size_bytes is not None
        else ray.data.DataContext.get_current().target_max_block_size
    )

    reader = ParquetReader(
        schema=None,
        dataset_kwargs={},
        batch_size=batch_size,
        use_threads=True,
        to_batches_kwargs={},
        block_udf=None,
        include_paths=False,
        partitioning=None,
        target_block_size=target_block_size,
    )

    file_manifest = FileManifest.from_paths_and_sizes(
        [file_path], [0]
    )  # Use dummy file size for test
    tables = list(
        reader.read_files(
            file_manifest,
            filter_expr=None,
            columns=None,
            columns_rename=None,
            filesystem=pa.fs.LocalFileSystem(),
        )
    )

    assert len(tables) == expected_num_batches
    total_rows = sum(table.num_rows for table in tables)
    assert total_rows == num_rows

    if batch_size is None:
        # For automatic batch sizing, verify behavior based on row group size
        dataset = pa.dataset.dataset(file_path, format="parquet")
        fragment = list(dataset.get_fragments())[0]
        row_group_meta = fragment.metadata.row_group(0)
        row_group_size_bytes = sum(
            row_group_meta.column(col_idx).total_uncompressed_size
            for col_idx in range(row_group_meta.num_columns)
        )

        if row_group_size_bytes <= MAX_SAFE_BLOCK_SIZE_FACTOR * target_block_size:
            # For small row groups, verify each table has the same number of rows as the row group size
            for i, table in enumerate(tables):
                assert (
                    table.num_rows == row_group_size
                ), f"Table {i} has {table.num_rows} rows, expected {row_group_size}"
        else:
            # For large row groups, verify each table's size matches the calculated batch size
            average_row_size = row_group_size_bytes / row_group_size
            expected_batch_size = max(1, int(target_block_size / average_row_size))

            # Verify all but last batch have expected size
            for i, table in enumerate(tables[:-1]):
                assert (
                    table.num_rows == expected_batch_size
                ), f"Table {i} has {table.num_rows} rows, expected {expected_batch_size}"

            # Last batch should have remaining rows
            last_batch = tables[-1]
            expected_last_batch_size = row_group_size - (
                expected_batch_size * (len(tables) - 1)
            )
            assert (
                last_batch.num_rows == expected_last_batch_size
            ), f"Last batch has {last_batch.num_rows} rows, expected {expected_last_batch_size}"
    else:
        # For explicit batch size, verify no batch exceeds the batch size
        # (PyArrow may return smaller batches due to row group boundaries)
        for i, table in enumerate(tables):
            assert (
                table.num_rows <= batch_size
            ), f"Table {i} has {table.num_rows} rows, expected <= {batch_size}"
            assert (
                table.num_rows > 0
            ), f"Table {i} has {table.num_rows} rows, expected > 0"


@pytest.fixture
def test_schema():
    """Test data schema used across multiple tests."""
    return {
        "int_col": pa.int64(),
        "float_col": pa.float64(),
        "str_col": pa.string(),
    }


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


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main(["-v", __file__]))
