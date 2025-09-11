import pytest
import numpy as np
import pyarrow as pa

from ray.anyscale.data._internal.arrow_ops.transform_pyarrow import (
    _hash_partition_vectorized,
    hash_partition_optimized,
    deepcopy_array,
)


def test_row_partitions_simple_types():
    """Test that row_partitions works correctly with simple types."""
    # Create a simple table with basic types
    table = pa.Table.from_pydict(
        {
            "ints": [1, 2, 3, 4, 5],
            "strings": ["a", "b", "c", "d", "e"],
            "floats": [1.1, 2.2, 3.3, 4.4, 5.5],
        }
    )

    num_partitions = 3
    partitions = _hash_partition_vectorized(table, num_partitions)

    assert np.array_equal(partitions, [2, 0, 0, 0, 2])


def test_row_partitions_fallback_complex_types():
    """Test that row_partitions falls back to original implementation for complex types."""
    # Create a table with complex types (List)
    table = pa.Table.from_pydict(
        {
            "ints": [1, 2, 3],
            "lists": [[1, 2], [3, 4], [5, 6]],
        }
    )

    num_partitions = 2
    partitions = _hash_partition_vectorized(table, num_partitions)

    # Should still work, just using fallback implementation
    assert np.array_equal(partitions, [1, 0, 0])


def test_row_partitions_fallback_struct_types():
    """Test that row_partitions falls back to original implementation for struct types."""
    # Create a table with struct types
    table = pa.Table.from_pydict(
        {
            "ints": [1, 2, 3],
            "structs": [{"value": 1}, {"value": 2}, {"value": 3}],
        }
    )

    num_partitions = 2
    partitions = _hash_partition_vectorized(table, num_partitions)

    assert np.array_equal(partitions, [0, 1, 1])


def test_hash_partition_optimized_empty_table():
    """Test hash_partition_optimized with empty table."""
    empty_table = pa.Table.from_pydict({"idx": []})

    result = hash_partition_optimized(empty_table, hash_cols=["idx"], num_partitions=5)

    assert result == {}


def test_hash_partition_optimized_single_partition():
    """Test hash_partition_optimized with single partition."""
    table = pa.Table.from_pydict({"idx": [1, 2, 3, 4, 5]})

    result = hash_partition_optimized(table, hash_cols=["idx"], num_partitions=1)

    assert result == {0: table}


def test_hash_partition_optimized_multiple_partitions():
    """Test hash_partition_optimized with multiple partitions."""
    table = pa.Table.from_pydict(
        {
            "idx": list(range(20)),
            "values": [f"value_{i}" for i in range(20)],
        }
    )

    result = hash_partition_optimized(table, hash_cols=["idx"], num_partitions=3)

    # Expected result based on hash partitioning
    expected = {
        0: pa.Table.from_pydict(
            {"idx": [0, 5, 9], "values": ["value_0", "value_5", "value_9"]}
        ),
        1: pa.Table.from_pydict(
            {
                "idx": [3, 7, 8, 10, 11, 12, 13, 16],
                "values": [
                    "value_3",
                    "value_7",
                    "value_8",
                    "value_10",
                    "value_11",
                    "value_12",
                    "value_13",
                    "value_16",
                ],
            }
        ),
        2: pa.Table.from_pydict(
            {
                "idx": [1, 2, 4, 6, 14, 15, 17, 18, 19],
                "values": [
                    "value_1",
                    "value_2",
                    "value_4",
                    "value_6",
                    "value_14",
                    "value_15",
                    "value_17",
                    "value_18",
                    "value_19",
                ],
            }
        ),
    }

    assert result == expected


def test_hash_partition_optimized_with_complex_types():
    """Test that hash_partition_optimized works with complex types (using fallback)."""
    table = pa.Table.from_pydict(
        {
            "idx": [1, 2, 3, 4],
            "lists": [[1, 2], [3, 4], [5, 6], [7, 8]],
            "structs": [{"value": 1}, {"value": 2}, {"value": 3}, {"value": 4}],
        }
    )

    result = hash_partition_optimized(table, hash_cols=["idx"], num_partitions=2)

    # Expected result based on hash partitioning
    expected = {
        0: pa.Table.from_pydict(
            {
                "idx": [1, 3],
                "lists": [[1, 2], [5, 6]],
                "structs": [{"value": 1}, {"value": 3}],
            }
        ),
        1: pa.Table.from_pydict(
            {
                "idx": [2, 4],
                "lists": [[3, 4], [7, 8]],
                "structs": [{"value": 2}, {"value": 4}],
            }
        ),
    }

    assert result == expected


@pytest.mark.parametrize("num_partitions", [1, 2, 5, 10])
def test_hash_partition_optimized_partition_counts(num_partitions):
    """Test hash_partition_optimized with different partition counts."""
    table = pa.Table.from_pydict(
        {
            "idx": list(range(50)),
            "data": [f"item_{i}" for i in range(50)],
        }
    )

    result = hash_partition_optimized(
        table, hash_cols=["idx"], num_partitions=num_partitions
    )

    # Should have at most num_partitions partitions
    assert len(result) <= num_partitions

    # All partition IDs should be in valid range
    for partition_id in result.keys():
        assert 0 <= partition_id < num_partitions

    # Total rows should be preserved
    total_rows = sum(partition_table.num_rows for partition_table in result.values())
    assert total_rows == table.num_rows


@pytest.mark.parametrize(
    "chunked",
    [
        True,
        False,
    ],
)
def test_deepcopy_array(chunked):
    """Test that deepcopy_array creates a new copy of the array
    and that the original array buffers are not referenced by the copy."""
    array = (
        pa.array([1, 2, 3, 4, 5])
        if chunked
        else pa.chunked_array([pa.array([1, 2]), pa.array([3, 4, 5])])
    )
    copy = deepcopy_array(array)

    def _get_buffer_addresses(array):
        copy_buffer_addresses = []
        if isinstance(array, pa.ChunkedArray):
            for chunk in array.chunks:
                copy_buffer_addresses.extend(
                    [buf.address for buf in chunk.buffers() if buf is not None]
                )
        else:
            copy_buffer_addresses = [
                buf.address for buf in array.buffers() if buf is not None
            ]
        return copy_buffer_addresses

    copy_buffer_addresses = _get_buffer_addresses(copy)
    original_buffer_addresses = _get_buffer_addresses(array)

    # Make sure that none of the original buffer addresses are referenced by the copy.
    for original_buffer_address in original_buffer_addresses:
        assert original_buffer_address not in copy_buffer_addresses
