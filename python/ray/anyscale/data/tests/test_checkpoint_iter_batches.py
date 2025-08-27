import collections
from pathlib import Path
import threading

import pytest
import pyarrow.parquet as pq

import ray.data
from ray.anyscale.data.checkpoint.data_iterator_checkpointer import (
    RowIDBasedDataIteratorCheckpointer,
)
from ray.train.v2._internal.execution.context import DistributedContext

from ray.tests.conftest import *  # noqa


def _read_checkpoint_files_for_state_dict(state_dict: dict, root_path: Path) -> list:
    checkpoint_idx = state_dict["checkpoint_idx"]
    epoch = state_dict["epoch_idx"]
    # Get all checkpoints up to and including checkpoint_idx
    checkpointed_row_ids = []
    for i in range(checkpoint_idx + 1):
        for rank_dir in root_path.glob("rank=*"):
            checkpoint_path = rank_dir / f"epoch={epoch}" / f"checkpoint={i}"
            for checkpoint_file in checkpoint_path.glob("*.parquet"):
                checkpointed_row_ids.extend(
                    pq.read_table(checkpoint_file).column("id").to_pylist()
                )
    return sorted(checkpointed_row_ids)


@pytest.mark.parametrize("reinit_iter", [True, False])
def test_multiple_checkpoints_per_epoch(
    ray_start_regular_shared, tmp_path, reinit_iter
):
    """Test that multiple checkpoints can be created per epoch, for several epochs.

    Checkpoints are created at 1/3 and 2/3 way through each epoch.
    A checkpoint is also created at the end of each epoch.
    """
    checkpointer = RowIDBasedDataIteratorCheckpointer(
        id_column="id", checkpoint_path=tmp_path
    )

    num_epochs = 2
    num_batches = 120
    batch_size = 10
    num_mid_epoch_checkpoints = 2
    checkpoint_at_batches = [
        i * (num_batches // (num_mid_epoch_checkpoints + 1))
        for i in range(1, num_mid_epoch_checkpoints + 1)
    ]

    # repartition(1) to ensure that batches are yielded in order.
    ds = ray.data.range(num_batches * batch_size).repartition(1)
    ds_iter = ds.iterator()
    ds_iter._enable_checkpointing(checkpointer)

    state_dict = ds_iter.state_dict()
    assert state_dict == {
        "epoch_idx": 0,
        "checkpoint_idx": -1,
    }
    # There should be no row_ids associated with a state dict initially.
    assert _read_checkpoint_files_for_state_dict(state_dict, tmp_path) == []

    batch_iter = None
    for epoch in range(num_epochs):
        checkpoint_idx = 0
        consumed_batches = 0
        if batch_iter is None or reinit_iter:
            batch_iter = ds_iter.iter_batches(batch_size=batch_size)
        for _ in batch_iter:
            consumed_batches += 1
            if consumed_batches in checkpoint_at_batches:
                state_dict = ds_iter.state_dict()
                assert state_dict == {
                    "epoch_idx": epoch,
                    "checkpoint_idx": checkpoint_idx,
                }
                assert _read_checkpoint_files_for_state_dict(
                    state_dict, tmp_path
                ) == list(range(consumed_batches * batch_size))
                checkpoint_idx += 1

        assert ds_iter.state_dict() == {
            "epoch_idx": epoch + 1,
            "checkpoint_idx": -1,
        }
        # There should be no row_ids associated with a state dict at an epoch boundary.
        assert (
            _read_checkpoint_files_for_state_dict(ds_iter.state_dict(), tmp_path) == []
        )


def test_streaming_split_iterator_checkpointing(ray_start_regular_shared, tmp_path):
    """Test that streaming split iterators can be checkpointed.

    This tests multiple workers iterating concurrently over the same dataset.

    We test that the mid-epoch and end-of-epoch checkpoints are correct.
    TODO: Just convert this to a e2e test in a follow-up PR.
    """
    world_size = 2
    num_epochs = 2
    num_batches_per_worker = 120
    batch_size = 10
    num_mid_epoch_checkpoints = 2
    checkpoint_at_batches = [
        i * (num_batches_per_worker // (num_mid_epoch_checkpoints + 1)) + 1
        for i in range(1, num_mid_epoch_checkpoints + 1)
    ]

    checkpointers = [
        RowIDBasedDataIteratorCheckpointer(
            checkpoint_path=str(tmp_path),
            id_column="id",
            distributed_context=DistributedContext(
                world_rank=i,
                world_size=world_size,
                local_rank=i,
                local_world_size=world_size,
                node_rank=0,
            ),
        )
        for i in range(world_size)
    ]
    ds = ray.data.range(num_batches_per_worker * world_size * batch_size)
    ds_iters = ds.streaming_split(world_size, equal=True)

    for ds_iter, checkpointer in zip(ds_iters, checkpointers):
        ds_iter._enable_checkpointing(checkpointer)

    state_dicts_per_worker = collections.defaultdict(list)

    def run_epoch(ds_iter, rank):
        consumed_batches = 0
        for _ in ds_iter.iter_batches(batch_size=batch_size):
            consumed_batches += 1
            if consumed_batches in checkpoint_at_batches:
                state_dict = ds_iter.state_dict()
                state_dicts_per_worker[rank].append(state_dict)
        state_dict = ds_iter.state_dict()
        state_dicts_per_worker[rank].append(state_dict)

    for epoch in range(num_epochs):
        consumers = [
            threading.Thread(target=run_epoch, args=(ds_iter, rank))
            for rank, ds_iter in enumerate(ds_iters)
        ]
        [consumer.start() for consumer in consumers]
        [consumer.join() for consumer in consumers]

        # All workers should have returned the same state dicts.
        rank_0_state_dicts = state_dicts_per_worker[0]
        for state_dicts in state_dicts_per_worker.values():
            assert all(
                state_dict == rank_0_state_dict
                for state_dict, rank_0_state_dict in zip(
                    state_dicts, rank_0_state_dicts
                )
            )

        # Check the mid-epoch checkpoints.
        for checkpoint_idx, consumed_batches in enumerate(checkpoint_at_batches):
            checkpointed_row_ids = _read_checkpoint_files_for_state_dict(
                rank_0_state_dicts[checkpoint_idx], tmp_path
            )
            # We just check that the number of checkpointed row ids is correct.
            # streaming_split(equal=True) splits blocks between workers,
            # so the the consumption is not in order.
            # For example, for 2 workers, the first batch would look like:
            # Worker 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            # Worker 1: [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009]
            assert (
                len(checkpointed_row_ids) == consumed_batches * world_size * batch_size
            )

        # Check the end of epoch checkpoints.
        for state_dicts in state_dicts_per_worker.values():
            last_state_dict = state_dicts[-1]
            assert last_state_dict == {"epoch_idx": epoch + 1, "checkpoint_idx": -1}
            assert (
                _read_checkpoint_files_for_state_dict(last_state_dict, tmp_path) == []
            )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
