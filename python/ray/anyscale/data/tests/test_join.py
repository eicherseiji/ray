import ray
import pyarrow as pa
import numpy as np

from ray.data import DataContext


def test_streaming_join(restore_data_context):
    """Tests that join results are properly streamed for joins not fitting
    into memory.
    """

    DataContext.get_current().target_max_block_size = None

    N = 10000
    dupes = pa.table({"id": np.ones(N)})

    ds = ray.data.from_arrow(dupes)

    # Because we join 2 tables with duplicated id column, this join
    # will produce cartesian product (N^2) of 100M rows (~0.8Gb)
    joined_ds = ds.join(ds, join_type="full_outer", num_partitions=1)

    num_blocks = 0
    total_rows = 0

    for i, rb in enumerate(joined_ds.iter_internal_ref_bundles()):
        num_blocks += 1
        total_rows += rb.num_rows()

        print(f">>> Bundle {i}: {rb.num_rows()=}, {rb.size_bytes()=}")

    assert total_rows == N**2
    # There should be more than 1000 blocks streamed (Polars defaults to 100k
    # rows batches)
    assert num_blocks >= 1000
