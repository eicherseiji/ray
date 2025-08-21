import logging
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa

from ray.data._internal.arrow_ops.transform_pyarrow import _hash_partition
from ray.util.debug import log_once

logger = logging.getLogger(__name__)


# Try to import numba, fallback to non-JIT implementation if not available
try:
    import numba as nb
    from numba import types, int64

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    if log_once("numba_not_available"):
        logger.warning(
            "Numba is not available. Falling back to slower Python implementation "
            "for hash partitioning operations."
        )


def _hash_partition_vectorized(
    projected_table: pa.Table, num_partitions: int
) -> np.ndarray:
    """
    For each row, calculates hash(row_values) % num_partitions in a vectorized manner using Polars.

    Args:
        projected_table: Arrow table containing rows to hash.
        num_partitions: Number of target partitions (must be > 0).

    Returns:
        np.ndarray: Array of hashed values for each row.
    """
    import polars as pl

    df: pl.DataFrame = pl.from_arrow(
        projected_table, rechunk=False
    )  # zero-copy wrapper

    for col_name in df.columns:
        dtype = df.schema[col_name]

        if dtype.is_nested():  # List, Array, Struct etc.
            # Fall back to original implementation for complex types
            return _hash_partition(projected_table, num_partitions=num_partitions)

    # Hash the entire row (now all columns are already hashed integers)
    return (df.hash_rows(seed=0).to_numpy() % num_partitions).astype(np.int64)


# Conditionally apply numba compilation if available
if _NUMBA_AVAILABLE:

    def _group_indices_numba(
        partition_mask: np.ndarray, counts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Group row indices by their partition assignment.

        Args:
            partition_mask: Array where partition_mask[i] indicates which partition row i belongs to
            counts: Histogram array where counts[j] is the number of rows assigned to partition j

        Returns:
            Tuple of:
            - grouped_partition_indices: Array of row indices, grouped by partition (all indices for partition 0 first,
                then partition 1, etc.). Maintains stable ordering within each partition.
            - offsets: Array where offsets[j] is the starting position of partition j's indices
                    in the out array (exclusive prefix sums of counts)

        Example:
            If partition_mask=[1,0,1,0] and counts=[2,2], returns:
            - grouped_partition_indices=[1,3,0,2] (indices for partition 0: [1,3], partition 1: [0,2])
            - offsets=[0,2] (partition 0 starts at index 0, partition 1 starts at index 2)
        """
        # Convert cumulative sums to exclusive start offsets for each partition.
        # cumsum() gives us "total items through each partition" but we need
        # "starting position of each partition" in the output array.
        # The shift transforms inclusive cumulative counts to exclusive start indices.
        # Example: counts=[2,3,1] -> cumsum()=[2,5,6] -> shift to [0,2,5]
        # This snippet replicates `np.concatenate((np.zeros(1, dtype=counts.dtype), counts)).cumsum()[:-1]`
        # We're doing this because the manual looping when compiled down to C is faster than the NumPy implementation, and results in fewer allocations.
        offsets = counts.cumsum()
        total_prev = 0
        for j in range(offsets.size - 1):
            tmp = offsets[j]
            offsets[j] = total_prev
            total_prev = tmp
        offsets[-1] = total_prev

        grouped_partition_indices = np.empty(partition_mask.size, dtype=np.int64)

        # The next part ensures that all indices for the same partition are contiguous.
        # This snippet replicates `np.argsort(arr, kind="stable")`. Numba doesn't support `kind="stable"` yet.
        # We're doing this because the manual looping when compiled down to C is faster than the NumPy implementation, and results in fewer allocations.
        # On testing the delta in performance between the two implementations is around 15%
        write = offsets.copy()
        for i in range(partition_mask.size):
            p = partition_mask[i]
            pos = write[p]
            grouped_partition_indices[pos] = i
            write[p] = pos + 1
        return grouped_partition_indices, offsets

    _group_indices = nb.njit(
        types.UniTuple(int64[:], 2)(int64[:], int64[:]),
        cache=True,
        nogil=True,
        fastmath=True,
    )(_group_indices_numba)
else:

    def _group_indices(
        partition_mask: np.ndarray, counts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Same definition as _group_indices_numba, but using NumPy. Used when Numba is not available.
        """
        offsets = np.concatenate((np.zeros(1, dtype=counts.dtype), counts)).cumsum()[
            :-1
        ]
        grouped_partition_indices = np.argsort(partition_mask, kind="stable")
        return grouped_partition_indices, offsets


def hash_partition_optimized(
    table: pa.Table,
    *,
    hash_cols: List[str],
    num_partitions: int,
) -> Dict[int, pa.Table]:
    """Optimized hash-partitions implementation using Polars."""
    assert num_partitions > 0

    if table.num_rows == 0:
        return {}
    elif num_partitions == 1:
        return {0: table}

    # NOTE: Subsequent `take` operation is known to be sensitive to the number of
    #       chunks w/in the individual columns, and therefore to improve performance
    #       we attempt to defragment the table to potentially combine some of those
    #       chunks into contiguous arrays.
    from ray.data._internal.arrow_ops.transform_pyarrow import (
        try_combine_chunked_columns,
    )

    table: pa.Table = try_combine_chunked_columns(table)

    projected_table = table.select(hash_cols)
    partitions_array = _hash_partition_vectorized(projected_table, num_partitions)

    # For every partition compile list of indices of rows falling under that partition
    counts = np.bincount(partitions_array, minlength=num_partitions)
    grouped_idx, start_offsets = _group_indices(partitions_array, counts)

    # take is an expensive operation, so we want to run it once.
    reordered = table.take(grouped_idx)

    # NOTE: Since some of the partitions might be empty, we're filtering out
    #       indices of the length 0 to make sure we're not passing around
    #       empty tables
    nz = np.nonzero(counts)[0]
    result = {p: reordered.slice(start_offsets[p], int(counts[p])) for p in nz}

    return result
